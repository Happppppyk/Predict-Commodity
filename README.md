# 대두유 원자재 구매 의사결정 PoC

## 아키텍처 레이어

1. 데이터 수집 — `raw_*` 테이블 적재 (`src/ingestion/`)
2. 피처 — `master_daily` 빌드 (`src/features/`, `config/features.yaml`)
3. 예측 — XGBoost 베이스라인 → TFT (`src/models/`)
4. 시나리오·권고 — TFT 예측 기반 조달 의사결정 (`src/rl/scenario_engine.py`)
5. XAI — SHAP + Counterfactual (`src/xai/`)

## 디렉터리

| 경로 | 용도 |
|------|------|
| `data/raw/` | 엑셀 등 수동 적재 원본 |
| `data/db/` | SQLite 파일 |
| `notebooks/` | 탐색·시각화 |
| `tests/` | 단위·통합 테스트 |
| `config/features.yaml` | 피처 버전(v1/v2/v3) 정의 |

작업 디렉터리는 저장소 루트(`soybean-oil-poc`)로 두고, DB 기본값은 `data/db/soybean.db`임.

## 실행 순서

### 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. 데이터 수집

```bash
python src/ingestion/run_ingestion.py
```

- Yahoo Finance: 대두유/원유/대두/대두박/환율/DXY/VIX
- FRED: 금리(FEDFUNDS)
- CFTC: 투기 포지션
- USDA: WASDE PSD 벌크 CSV (`data/raw/wasde_psd.csv`)
- 수동 CSV 필요: 팜유(CPOc1), 카놀라(RSc1)  
  → `data/raw/palm_oil_cpo.csv`  
  → `data/raw/canola_oil_rsc1.csv`

- 뉴스(GDELT) 단계를 켜려면 환경변수 `NEWS_INGESTION_API_KEY`를 비어 있지 않게 설정함. 비어 있으면 뉴스 적재는 스킵됨.
- WASDE는 `data/raw/wasde_psd.csv`가 있으면 CSV로 재적재하고, 없으면 USDA Open Data API(`.env`의 `USDA_OPEN_DATA_API_KEY` 등) 경로로 적재 시도함.
- SAP·World Bank 엑셀 등은 `data/raw/`에 파일이 있어야 해당 `raw_*`가 채워짐.
- raw + `master_daily`만 빠르게: `bash scripts/rebuild_pipeline.sh` — `run_ingestion.py` 다음 `build_master.py`를 연속 실행하고, 일부 테이블 행 수를 출력함.

### 2. master_daily 빌드

```bash
python src/features/build_master.py
```

- 모든 `raw_*` 테이블 통합
- 파생 피처 계산 (lag, MA, crush_spread 등)
- USD/톤 변환 컬럼 추가
- 결과: `master_daily` (4,092행, 91컬럼+)

### 3. XGBoost 앙상블 학습

```bash
python src/models/xgboost_model.py
```

- v3_clean t28 기준
- XGBoost + LightGBM + CatBoost 앙상블
- 저장: `models/ensemble_v3_clean_t28_final/`

### 4. TFT 멀티 시계열 학습

```bash
PYTHONPATH=src python3 src/models/tft_model.py --train
```

- 6개 품목 동시 학습 (USD/톤 단위)
- 저장: `models/tft_v3_multi.ckpt`

### 5. 시나리오 엔진 테스트

```bash
PYTHONPATH=src python3 tests/test_scenarios.py
```

- 현업 입력 3가지 기반 Buy/Split/Wait 권고
- `decision_log` 테이블에 결과 저장

데모(단일 JSON 출력 + DB 저장): `PYTHONPATH=src python3 scripts/run_demo.py`

### 6. SHAP 설명

```bash
python src/xai/shap_explainer.py
```

- DB와 저장된 XGBoost 모델을 전제로 함. XGB 파이프라인 이후 실행하는 편이 자연스러움.

### 데이터 소스별 수동 업데이트 주기

| 데이터 | 주기 | 방법 |
|--------|------|------|
| Yahoo Finance | 자동 (매일) | `run_ingestion.py` |
| FRED | 자동 (월별) | `run_ingestion.py` |
| CFTC | 자동 (주별) | `run_ingestion.py` |
| WASDE | 월 1회 수동 | CSV 교체 후 `run_ingestion.py` |
| 팜유 CPOc1 | 월 1회 수동 | CSV 교체 후 `run_ingestion.py` |
| 카놀라 RSc1 | 월 1회 수동 | CSV 교체 후 `run_ingestion.py` |

## Look-ahead 방지·조인 규칙

모듈별로 코드에 있던 전제를 여기에 모아 둠. `master_daily`·모델·적재 파이프라인을 바꿀 때 이 순서를 깨지 않도록 함.

### `build_master` (`master_daily`)

- CFTC·WASDE·월별 spot·달러지수·카놀라·해바라기유 등은 공표일·월초·직전 관측 시점 값을 ZL 거래일 뼈대에 맞춘 뒤, 과거→현재 방향의 forward-fill만 사용함.
- `report_date`처럼 as-of 시점이 애매한 키로 미래 정보를 조인하지 않음.
- 타깃 컬럼(`target_price_t1`, `target_price_t7`, `target_price_t28` 등)은 미래 가격을 담으므로 피처로 쓰지 않음.

### `price` (선물·팜유 등 OHLCV 적재)

- 일별 행에는 해당 거래일에 확정된 OHLCV만 넣음. t일 피처에는 t일 종가가 장 마감 후 확정된 뒤에만 쓰고, 장중·미확정 봉을 미래 정보처럼 쓰지 않음.
- `master_daily`와 라벨은 as-of t에서 t 및 과거만 조인하고, t+1 이후 시세를 섞지 않음.
- 동일 `date` 재적재는 `INSERT OR IGNORE` 등으로 기존 행을 덮어쓰지 않게 두어 과거 스냅샷이 바뀌지 않게 함. 정정이 필요하면 별도 백필·마이그레이션으로 처리함.

### `wasde` (`raw_wasde`)

- USDA PSD는 FAS Open Data API(`USDA_OPEN_DATA_API_KEY`, 프로젝트 루트 `.env` 또는 환경 변수) 또는 플랫 CSV·`data/raw/wasde_psd.csv` 형식 벌크로 적재 가능함.
- `raw_wasde`는 `(release_date, marketing_year)` 등 복합 PK일 수 있으므로, `master_daily` 조인 시 마케팅 연도·공표일 정의를 맞춤.
- `release_date` 기준으로만 `master_daily`에 조인함. 월중 forward-fill은 공표 이후 날짜에만 적용함.

### `cftc` (`raw_cftc`)

- 소스: CFTC 레거시 선물 ZIP(연도별 `deacot{연도}.zip` 등), 구 `deacot.zip`(404일 수 있음), 최신 `deafut.txt` 등. 이력 ZIP과 스냅샷은 병합·중복 제거함.
- `report_date` 기준 forward-fill은 금지함. `master_daily`에는 반드시 `release_date`(통상 금요일) 기준으로만 조인함. 화~목에 그 주 공표분을 쓰면 미래 정보 유출임.

### `macro` (환율·현물·Pink Sheet·달러지수·해바라기·EIA 등)

- Yahoo 등 일별 관측은 해당 일자 확정분으로 봄. 주말·휴장일은 직전 관측일로 forward-fill하며, 보간 행은 `is_interpolated=1` 등으로 구분 가능함(신규 시장 정보가 아님).
- `master_daily`에서는 관측일만 쓸지·보간 행을 제외할지 정책으로 통제함.
- forward-fill은 과거→현재만 사용함.
