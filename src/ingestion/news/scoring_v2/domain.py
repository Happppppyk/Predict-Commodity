"""Domain layer for scoring v2 — Category, Score, ScoringResult.

Light-weight DDD building blocks:
  - Category: 14 + other 의 type-safe enum (config.py CATEGORIES 와 1:1 동기화)
  - Score: sentiment / impact / certainty 의 VO. invariant 를 __post_init__ 에서 보장
  - ScoringResult: aggregate (Article + Category + Score + meta). DB row 직렬화 책임

검증 로직이 절차적으로 흩어져있던 `_validate_pass1/2`, `_validate_onepass`,
`compute_signal` 을 이 모듈로 흡수했다. rescore_v2 는 이 타입만 다룬다.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Category(str, Enum):
    """Scoring 카테고리. `str, Enum` 믹스인 — DB/JSON 직렬화 시 자동으로 value (str) 로 동작.

    config.py CATEGORIES 와 1:1 일치 필수. 변경 시 둘 다 수정.
    """

    # Supply-n-Demand
    SND_US = "snd_us"
    SND_BR = "snd_br"
    SND_PALM = "snd_palm"
    # Weather
    WEATHER_US = "weather_us"
    WEATHER_BR = "weather_br"
    WEATHER_GLOBAL = "weather_global"
    # Policy
    POLICY_US_EIA = "policy_us_eia"
    POLICY_RVO = "policy_rvo"
    POLICY_REBIO = "policy_rebio"
    POLICY_PALM = "policy_palm"
    # Market
    MARKET_GENERAL = "market_general"
    MARKET_TRADE = "market_trade"
    MARKET_CORPORATE = "market_corporate"
    # Catch-all
    OTHER = "other"

    @classmethod
    def parse(cls, raw: Any) -> Category | None:
        """문자열 → Category. 알 수 없는 값은 None (caller 가 fallback 처리)."""
        if raw is None:
            return None
        try:
            return cls(raw)
        except ValueError:
            return None


@dataclass(frozen=True)
class Score:
    """Tradeable 기사의 sentiment·impact·certainty.

    invariant:
      - sentiment ∈ {-2, -1, 0, 1, 2}
      - impact    ∈ {0, 1, 2, 3, 4, 5}   (0 은 not-tradeable sentinel)
      - certainty ∈ {0, 1, 2, 3, 4, 5}   (0 은 not-tradeable sentinel)

    not-tradeable 케이스는 Score.neutral() 로 생성 (모두 0).
    LLM 응답에서 tradeable 점수를 파싱할 땐 Score.from_llm() — 1..5 강제.
    """

    sentiment: int
    impact: int
    certainty: int

    def __post_init__(self) -> None:
        if not -2 <= self.sentiment <= 2:
            raise ValueError(f"sentiment out of range: {self.sentiment}")
        if not 0 <= self.impact <= 5:
            raise ValueError(f"impact out of range: {self.impact}")
        if not 0 <= self.certainty <= 5:
            raise ValueError(f"certainty out of range: {self.certainty}")

    @classmethod
    def neutral(cls) -> Score:
        """not-tradeable / category=other / weight=0 케이스. 모두 0."""
        return cls(sentiment=0, impact=0, certainty=0)

    @classmethod
    def from_llm(cls, obj: dict | None) -> Score | None:
        """LLM 응답 dict 에서 tradeable 점수 파싱. 1..5 강제. 검증 실패 시 None.

        not-tradeable 케이스는 호출자가 Score.neutral() 로 분기 — 이 메서드는
        오직 tradeable 점수만 다룬다.
        """
        if not isinstance(obj, dict):
            return None
        try:
            s = int(obj["sentiment"])
            i = int(obj["impact"])
            c = int(obj["certainty"])
        except (KeyError, ValueError, TypeError):
            return None
        if not (-2 <= s <= 2 and 1 <= i <= 5 and 1 <= c <= 5):
            return None
        return cls(sentiment=s, impact=i, certainty=c)

    def to_signal(self, weight: int) -> float:
        """final_signal ∈ [-2.0, +2.0]. weight=0 또는 점수=0 이면 0.0."""
        if weight == 0:
            return 0.0
        return self.sentiment * (self.impact / 5.0) * (self.certainty / 5.0) * (weight / 5.0)


@dataclass(frozen=True)
class ScoringResult:
    """한 기사의 scoring 결과 aggregate. raw_news_scored_v2 row 1:1 매핑.

    score=None 은 'tradeable 인데 점수 파싱 실패' 케이스로, DB 에 sentiment/impact/certainty
    NULL 로 기록된다. tradeable=False 면 score=Score.neutral() (0/0/0) 으로 채워진다.
    """

    article_id: int
    category: Category
    is_tradeable: bool
    skip_reason: str | None
    score: Score | None
    weight: int
    final_signal: float
    scoring_input_mode: str
    original_content_status: str | None
    processed_at: str

    def to_db_row(self) -> dict:
        """raw_news_scored_v2 INSERT 용 dict. None score → NULL 컬럼."""
        return {
            "id": self.article_id,
            "category": self.category.value,
            "is_tradeable": int(self.is_tradeable),
            "skip_reason": self.skip_reason,
            "sentiment": self.score.sentiment if self.score else None,
            "impact": self.score.impact if self.score else None,
            "certainty": self.score.certainty if self.score else None,
            "weight": self.weight,
            "final_signal": self.final_signal,
            "processed_at": self.processed_at,
            "scoring_input_mode": self.scoring_input_mode,
            "original_content_status": self.original_content_status,
        }
