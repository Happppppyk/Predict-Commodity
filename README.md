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

## 실행 순서

작업 디렉터리는 저장소 루트(`soybean-oil-poc`)로 두고, 아래는 모두 그 기준 경로임. DB 기본값은 `data/db/soybean.db`임.

### 전체 파이프라인(권장 순서)

1. 로컬 설정(venv, `pip install -r requirements.txt`) 후 활성화함.
2. 데이터 수집으로 `raw_*` 채움.
3. `master_daily` 빌드함.
4. 예측 모델(XGBoost, TFT 등) 실행함.
5. XAI·노트북은 예측·피처가 준비된 뒤 선택 실행함.
6. 조달 시나리오 엔진은 DB·TFT 예측 입력이 있으면 바로 실행 가능함.

### 데이터 수집(`src/ingestion/`)

- 한 번에 돌릴 때: `python src/ingestion/run_ingestion.py` (선물·매크로·CFTC·WASDE·뉴스·SAP 등 순서대로 호출됨).
- 뉴스(GDELT) 단계를 강제로 켜려면 환경변수 `NEWS_INGESTION_API_KEY`를 비어 있지 않게 설정함. 비어 있으면 뉴스 적재는 스킵됨.
- WASDE는 `data/raw/wasde_psd.csv`가 있으면 CSV로 재적재하고, 없으면 USDA Open Data API(`.env`의 `USDA_OPEN_DATA_API_KEY` 등) 경로로 적재 시도함.
- SAP·World Bank 엑셀 등은 `data/raw/`에 파일이 있어야 해당 `raw_*`가 채워짐.
- 개별 소스만 갱신할 때는 각 모듈의 `if __name__ == "__main__"` 블록을 참고함(`price.py`, `macro.py`, `cftc.py`, `wasde.py`, `news_scorer.py` 등).

### 피처(`master_daily`)

- `python src/features/build_master.py` — `raw_*`와 `config/features.yaml`을 읽어 `master_daily`를 갱신함.

### 예측 모델(`src/models/`)

- XGBoost 기본 학습·평가: `python src/models/xgboost_model.py` (옵션: `--version`, `--target` 등 `argparse` 도움말 참고).
- 추가 실험 플래그는 같은 파일에 정의됨(예: `--tuned`, `--walkforward`, `--rerun-v1-stationary`, `--v2-interview-exp`, `--challenge-070`, `--save-ensemble-final` 등).
- TFT(멀티 시계열): `python src/models/tft_model.py` — 기본은 저장된 체크포인트로 재평가함. 전체 재학습·체크포인트 덮어쓰기는 `--train` 사용함.

### XAI(`src/xai/`)

- TreeSHAP 등: `python src/xai/shap_explainer.py` — DB와 저장된 XGBoost 모델을 전제로 함. XGB 파이프라인 이후 실행하는 편이 자연스러함.

### 조달 시나리오(`src/rl/scenario_engine.py`)

- TFT 분위수 예측과 `master_daily` 맥락을 받아 Buy/Split/Wait 권고 JSON을 생성함. 테스트: `PYTHONPATH=src python3 tests/test_scenarios.py`.

### raw + `master_daily`만 빠르게

- `bash scripts/rebuild_pipeline.sh` — `run_ingestion.py` 다음 `build_master.py`를 연속 실행하고, 일부 테이블 행 수를 출력함.

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


## 로컬 설정 (예정)

```bash
cd soybean-oil-poc
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
