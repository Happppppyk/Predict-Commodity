"""Category → Weight lookup table for scoring v2.

Weight = "이 카테고리 뉴스가 콩 시장에 얼마나 직접적인가" 의 구조적 속성.
LLM이 매번 추론하지 않고 카테고리 결정 시점에 결정론적으로 매핑.

Scale: 0 (제외) ~ 5 (직접)

조정 규칙:
- 6개월 단위 backtest로 카테고리별 평균 가격충격을 측정해 weight 재조정
- 단발 조정 금지. 변경은 git commit으로 추적.

Source-of-truth: 본 파일.
"""
from __future__ import annotations

from src.ingestion.news.scoring_v2.domain import Category


# Category → Weight (1~5, other=0)
# config.py CATEGORIES 와 동기화 필수.
WEIGHT_MAP: dict[Category, int] = {
    # Supply — 콩/팜 직접 공급
    Category.SND_US:           5,   # 미국은 콩 가격 1차 결정 요인
    Category.SND_BR:           5,   # 브라질도 글로벌 공급 1차
    Category.SND_PALM:         3,   # 팜유는 대두유 대체재 (간접)
    # Weather — 작황 매개로 가격 영향
    Category.WEATHER_US:       4,
    Category.WEATHER_BR:       4,
    Category.WEATHER_GLOBAL:   2,   # AR / 인도 / 인니 가뭄 등 (영향 약함)
    # Policy — 수요 측면 직접
    Category.POLICY_US_EIA:    4,   # RD 수요 통계 직접
    Category.POLICY_RVO:       5,   # RVO 변경은 BO 즉각 가격 충격
    Category.POLICY_REBIO:     3,   # 브라질 혼합비 (간접)
    Category.POLICY_PALM:      3,   # 인니/말 정책
    # Market
    Category.MARKET_GENERAL:   3,
    Category.MARKET_TRADE:     4,   # 관세는 즉시 가격 반영
    Category.MARKET_CORPORATE: 2,   # 개별 기업 (ABCD M&A 외 약함)
    # Other
    Category.OTHER:            0,   # 파이프라인 제외
}


def get_weight(category: Category | str | None) -> int:
    """Lookup weight. Category | str | None 모두 허용 — 모르는 값은 0.

    str 입력은 Category(str) 로 시도 후 실패 시 0 반환 (방어적).
    """
    if category is None:
        return 0
    if isinstance(category, str) and not isinstance(category, Category):
        try:
            category = Category(category)
        except ValueError:
            return 0
    return WEIGHT_MAP.get(category, 0)
