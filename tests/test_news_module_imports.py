"""News 모듈의 import 회귀 검증 — 패키지 내부 import가 깨지지 않도록."""
import importlib

import pytest


@pytest.mark.parametrize("mod", [
    "src.ingestion.news",
    "src.ingestion.news.config",
    "src.ingestion.news.export_to_main",
    "src.ingestion.news.scoring_v2",
    "src.ingestion.news.scoring_v2.aicore_client",
    "src.ingestion.news.scoring_v2.domain",
    "src.ingestion.news.scoring_v2.rescore_v2",
    "src.ingestion.news.scoring_v2.weight_map",
])
def test_module_imports(mod):
    importlib.import_module(mod)
