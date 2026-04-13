# 대두유 원자재 구매 의사결정 PoC

8주 일정의 SQLite 기반 AI PoC이며, 최종적으로 SAP Joule 자연어 출력과 연동하는 것을 목표로 한다.

## 아키텍처 레이어

1. **데이터 수집** — `raw_*` 테이블 적재 (`src/ingestion/`)
2. **피처** — `master_daily` 빌드 (`src/features/`, `config/features.yaml`)
3. **예측** — XGBoost 베이스라인 → TFT (`src/models/`)
4. **강화학습** — DQN 검증 → PPO 최적화 (`src/models/`)
5. **XAI** — SHAP + Counterfactual (`src/xai/`)
6. **RAG + LLM** — 벡터 검색 후 Joule용 응답 조립 (`src/rag/`)

## 디렉터리

| 경로 | 용도 |
|------|------|
| `data/raw/` | 엑셀 등 수동 적재 원본 |
| `data/db/` | SQLite 파일 |
| `notebooks/` | 탐색·시각화 |
| `tests/` | 단위·통합 테스트 |
| `config/features.yaml` | 피처 버전(v1/v2/v3) 정의 |

## 상태

현재 저장소는 **스켈레톤**(역할 주석·import 힌트·빈 함수)만 포함하며, 비즈니스 로직은 이후 스프린트에서 구현한다.

## 로컬 설정 (예정)

```bash
cd soybean-oil-poc
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
