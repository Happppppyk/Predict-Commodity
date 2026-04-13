"""
matplotlib·seaborn 공통: 한글 글리프(□) 방지 및 마이너스 기호 깨짐 방지.
노트북·CLI 스크립트에서 import 후 configure_matplotlib_korean() 한 번 호출.
"""

from __future__ import annotations

import matplotlib as mpl

_KO_SANS = [
    "AppleGothic",
    "Apple SD Gothic Neo",
    "Malgun Gothic",
    "NanumGothic",
    "Noto Sans CJK KR",
]


def configure_matplotlib_korean() -> None:
    """sans-serif 앞에 한글 지원 폰트 후보를 두고, 유니코드 마이너스를 ASCII로 표시."""
    prev = list(mpl.rcParams["font.sans-serif"])
    mpl.rcParams["font.sans-serif"] = _KO_SANS + [x for x in prev if x not in _KO_SANS]
    mpl.rcParams["axes.unicode_minus"] = False
