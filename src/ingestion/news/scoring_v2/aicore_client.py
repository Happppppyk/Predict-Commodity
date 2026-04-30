"""SAP AI Core Orchestration v2 client for soybean_news scoring.

등록된 orchestration template (`news-sentiment-classifier` v0.0.4+) 을 config_ref 로 호출하고
모델 출력 문자열을 그대로 반환한다. 기존 codex_client.run 자리에 끼워넣는 per-item 호출 모듈.

Env 요구사항:
    AICORE_CLIENT_ID, AICORE_CLIENT_SECRET, AICORE_AUTH_URL, AICORE_API_URL,
    AICORE_RESOURCE_GROUP, DEPLOYMENT_ID
    그리고 다음 중 택1:
      ORCHESTRATION_CONFIG_ID
      ORCHESTRATION_CONFIG_NAME + ORCHESTRATION_CONFIG_SCENARIO + ORCHESTRATION_CONFIG_VERSION

재시도 정책 (tenacity):
  - 429 / 5xx / network / 빈응답(EmptyResponseError) 은 최대 4회 재시도, exponential backoff (1s ~ 30s, jitter)
  - 그 외 4xx 는 retry 없이 즉시 None 리턴 (caller 가 row 단위 fail 로 기록)
  - 401/403 / env 누락 은 AICoreConfigError raise (caller 가 batch 전체 abort 신호로 사용)
"""
from __future__ import annotations

import logging
import os
from functools import cache

import httpx
from gen_ai_hub.orchestration_v2.exceptions import (
    OrchestrationError,
    OrchestrationErrorList,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


log = logging.getLogger("aicore_client")

MAX_ATTEMPTS = 4
BACKOFF_MIN_S = 1.0
BACKOFF_MAX_S = 30.0


class AICoreConfigError(RuntimeError):
    """Auth(401/403) / env 누락 — caller 는 batch 를 abort 해야 한다."""


class EmptyResponseError(RuntimeError):
    """모델이 빈 content / 비정상 finish_reason 으로 응답.

    GPT-5 mini 같은 reasoning model 은 reasoning_tokens 가 가변적이라
    가끔 max_completion_tokens 한계에 걸려 실제 출력이 잘림. tenacity 가
    transient 로 보고 재시도하면 대부분 회복됨.

    finish_reason='length' 는 결정적 cap (입력 + 요청 max_tokens 한계) 이므로
    재시도해도 같은 결과 → not retryable. 다른 빈응답 케이스만 재시도.
    """

    def __init__(self, msg: str, finish_reason: str | None = None):
        super().__init__(msg)
        self.finish_reason = finish_reason


@cache
def _get_service():
    """OrchestrationService 를 lazy-init + 프로세스 수명 동안 재사용.

    @cache 인 이유: 내부에서 OAuth 토큰을 받아오는 비싼 호출. 매 row 마다
    재생성하면 첫 호출이 1~3초 + 토큰 캐시 무효화로 quota 낭비.
    테스트 시 강제 reset 필요하면 _get_service.cache_clear().
    """
    from gen_ai_hub.orchestration_v2.service import OrchestrationService

    deployment_id = os.environ.get("DEPLOYMENT_ID")
    if not deployment_id:
        raise AICoreConfigError("DEPLOYMENT_ID env missing")
    return OrchestrationService(deployment_id=deployment_id)


@cache
def _get_config_ref():
    """env 에서 config 참조 정보 → config_ref 객체.

    우선순위: ID > NAME+SCENARIO+VERSION. ID 가 안정적이라 운영에선 ID 권장.
    """
    from gen_ai_hub.orchestration_v2.models.config import (
        CompletionRequestConfigurationReferenceByIdConfigRef,
        CompletionRequestConfigurationReferenceByNameScenarioVersionConfigRef,
    )

    cid = os.getenv("ORCHESTRATION_CONFIG_ID")
    if cid:
        return CompletionRequestConfigurationReferenceByIdConfigRef(id=cid)

    name = os.getenv("ORCHESTRATION_CONFIG_NAME")
    scenario = os.getenv("ORCHESTRATION_CONFIG_SCENARIO")
    version = os.getenv("ORCHESTRATION_CONFIG_VERSION")
    if name and scenario and version:
        return CompletionRequestConfigurationReferenceByNameScenarioVersionConfigRef(
            name=name, scenario=scenario, version=version,
        )

    raise AICoreConfigError(
        "Orchestration config reference 가 env 에 없습니다. "
        "ORCHESTRATION_CONFIG_ID 또는 NAME+SCENARIO+VERSION 셋을 지정하세요."
    )


# --- retry classification --------------------------------------------------

NETWORK_EXC_TYPES = (
    ConnectionError,
    TimeoutError,
    httpx.NetworkError,
    httpx.TimeoutException,
)


def _status_code(exc: BaseException) -> int | None:
    """예외에서 HTTP status code 추출. SDK 별 노출 방식을 모두 처리."""
    if isinstance(exc, OrchestrationError):
        code = getattr(exc, "code", None)
        return code if isinstance(code, int) else None
    if isinstance(exc, OrchestrationErrorList):
        errors = getattr(exc, "errors", None) or []
        if errors:
            inner = getattr(errors[0], "code", None)
            return inner if isinstance(inner, int) else None
        return None
    if isinstance(exc, httpx.HTTPStatusError):
        resp = getattr(exc, "response", None)
        if resp is not None:
            code = getattr(resp, "status_code", None)
            return code if isinstance(code, int) else None
    return None


def _is_retryable(exc: BaseException) -> bool:
    """tenacity retry predicate — 429 / 5xx / network / 빈응답(length 제외) transient."""
    if isinstance(exc, EmptyResponseError):
        # length cap 은 deterministic — retry 무의미
        return exc.finish_reason != "length"
    code = _status_code(exc)
    if code is not None:
        return code == 429 or 500 <= code < 600
    return isinstance(exc, NETWORK_EXC_TYPES)


@retry(
    stop=stop_after_attempt(MAX_ATTEMPTS),
    wait=wait_exponential_jitter(initial=BACKOFF_MIN_S, max=BACKOFF_MAX_S),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
    before_sleep=before_sleep_log(log, logging.WARNING),
)
def _invoke(placeholder_values: dict[str, str]) -> str:
    # service / config_ref init 실패는 deterministic — retry 안 하고 AICoreConfigError 로 빠른 abort
    try:
        service = _get_service()
        config_ref = _get_config_ref()
    except AICoreConfigError:
        raise
    except Exception as e:  # noqa: BLE001
        # SDK 내부 (ai_api_client_sdk) 가 raise 하는 OAuth/network 예외는 httpx 가 아니라
        # 자체 클래스 — _status_code 가 인식 못 함. service init 단계 실패는 모두
        # config-tier 로 간주하고 batch abort 신호로 변환한다.
        raise AICoreConfigError(
            f"AI Core service init failed ({type(e).__name__}): {e}"
        ) from e

    resp = service.run(config_ref=config_ref, placeholder_values=placeholder_values)
    choice = resp.final_result.choices[0]
    content = choice.message.content or ""
    if choice.finish_reason == "length" or not content.strip():
        raise EmptyResponseError(
            f"empty/truncated response (finish_reason={choice.finish_reason!r}, "
            f"content_len={len(content)})",
            finish_reason=choice.finish_reason,
        )
    return content


_AUTH_KEYWORDS = ("401", "403", "unauthorized", "forbidden", "invalid_token", "expired_token")


def _looks_like_auth_failure(exc: BaseException) -> bool:
    """SDK 가 비-httpx 예외로 던질 때 메시지 휴리스틱으로 401/403 판별.

    OrchestrationError.code / httpx.HTTPStatusError 로 안 잡힌 경우의 안전망.
    오탐(401 토큰이 본문에 우연히 등장) 가능성은 낮음 — 인증 실패 메시지에는
    거의 항상 위 키워드 중 하나가 포함됨.
    """
    msg = str(exc).lower()
    return any(k in msg for k in _AUTH_KEYWORDS)


def run(placeholder_values: dict[str, str]) -> str | None:
    """orchestration 1회 호출. 모델 출력 문자열 또는 None.

    None 케이스 (row 단위 실패로 기록):
      - retry 4회 모두 실패한 transient (429/5xx/network/빈응답)
      - terminal 4xx (400/422 등 — 입력 형식 문제일 가능성)
      - finish_reason='length' (max_completion_tokens 한계, retry 안 됨)

    AICoreConfigError raise 케이스 (batch 전체 abort 권고):
      - 401/403 (인증 실패) — HTTP status 또는 메시지 키워드로 검출
      - service init 실패 (OAuth/네트워크/잘못된 deployment_id 등)
      - env 필수값 누락
    토큰 만료 의심 시 _get_service.cache_clear() 한 번 호출 후 1회 재시도.
    """
    try:
        return _invoke(placeholder_values)
    except AICoreConfigError:
        raise
    except KeyError as e:
        raise AICoreConfigError(f"required env missing: {e}") from e
    except Exception as e:  # noqa: BLE001
        code = _status_code(e)
        is_auth = code in (401, 403) or _looks_like_auth_failure(e)
        if is_auth:
            # OAuth 토큰 만료 가능성 — 캐시된 서비스 1회 폐기 후 재호출
            log.warning("auth failure detected (code=%s); clearing service cache and retrying once",
                        code)
            _get_service.cache_clear()
            try:
                return _invoke(placeholder_values)
            except AICoreConfigError:
                raise
            except Exception as e2:  # noqa: BLE001
                code2 = _status_code(e2)
                if code2 in (401, 403) or _looks_like_auth_failure(e2):
                    raise AICoreConfigError(
                        f"auth failed after token refresh (code={code2}): {e2}"
                    ) from e2
                log.warning("AI Core call failed after refresh (%s, code=%s): %s",
                            type(e2).__name__, code2, str(e2)[:200])
                return None
        log.warning("AI Core call failed (%s, code=%s): %s",
                    type(e).__name__, code, str(e)[:200])
        return None
