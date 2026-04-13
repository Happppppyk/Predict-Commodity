"""
역할: RAG로 검색한 근거 + 예측·XAI·RL 권고를 묶어 SAP Joule용 자연어 응답 페이로드를 만든다.
레이어: RAG + LLM (6) — Joule API 스키마에 맞춘 출력만 조립 (실제 호출은 연동 단계에서).
"""

# 핵심 import (구현 시 사용 예정)
# from typing import Any


def build_prompt(system_instructions: str, retrieved_docs: list[str], structured_facts: dict) -> str:
    """LLM에 넣을 최종 프롬프트 문자열을 조립한다."""
    pass


def call_llm(prompt: str, model_config: dict) -> str:
    """LLM 추론 호출 (스켈레톤)."""
    pass


def format_joule_response(natural_language: str, citations: list[dict], metrics: dict) -> dict:
    """Joule/챗봇이 소비할 JSON 형태 응답을 구성한다."""
    pass


def validate_joule_schema(payload: dict) -> bool:
    """필수 필드 존재 여부 등 스키마 검증."""
    pass
