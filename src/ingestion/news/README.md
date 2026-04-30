# News Pipeline (v2)

GDELT GKG 15분 batch → soybean 필터 → 본문 추출 → SAP AI Core scoring → `data/db/soybean.db.raw_news_scored_v2`.

본 모듈의 정본 DB는 `data/db/news_v2.db`(별도 SQLite 파일, gitignored). ML 입력으로 쓰일 점수 + 메타 통합 테이블은 `data/db/soybean.db.raw_news_scored_v2`로 매 실행 마지막에 풀 dump 된다.

## 빠른 실행

```bash
# 환경변수 로드 (NEWS_INGESTION_API_KEY, AICORE_* 6개) 후
python src/ingestion/news/scripts/run_news_pipeline.py
```

옵션 없이 실행하면 `articles`의 가장 최근 `date` 다음 시점부터 현재까지 GDELT 슬롯을 자동 수집한다.

특정 구간 백필:
```bash
python src/ingestion/news/scripts/run_news_pipeline.py --from 2026-04-29T00:00 --to 2026-04-29T23:45
```

이미 raw가 있고 이후 단계만 다시 돌리려면:
```bash
python src/ingestion/news/scripts/run_news_pipeline.py --skip-raw
```

scoring 없이 DB append까지만:
```bash
python src/ingestion/news/scripts/run_news_pipeline.py --skip-score
```

## 단계별 실행

```bash
python src/ingestion/news/scripts/00_news_ingest.py --last 2h
python src/ingestion/news/scripts/01_news_filter.py
python src/ingestion/news/scripts/02_news_language.py
python src/ingestion/news/scripts/03_news_fetch.py
python src/ingestion/news/scripts/04_news_build_db.py
python src/ingestion/news/scripts/05_news_score.py
python -m src.ingestion.news.export_to_main
```

## 파일 역할

| 파일 | 목적 |
|---|---|
| `config.py` | 날짜 범위, 필터 키워드, 제외 도메인, 언어 정책, category taxonomy 등 공통 설정 |
| `schema.sql` | `news_v2.db`의 핵심 테이블과 VIEW 정의 |
| `scripts/00_news_ingest.py` | GDELT GKG 15분 batch zip을 받아 `data/news/gkg_raw/` parquet으로 저장 |
| `scripts/01_news_filter.py` | soybean 관련 기사 필터링, URL 정규화, 유사 제목 dedup, canonical 선택 |
| `scripts/02_news_language.py` | 언어 감지, domain fallback, `en/pt/es` 허용 언어 필터링 |
| `scripts/03_news_fetch.py` | canonical URL 본문 추출, `OK/PAYWALL/NOT_FOUND/TIMEOUT/SHORT/SCRIPT_ARTIFACT` 등 `body_status` 생성 |
| `scripts/04_news_build_db.py` | enriched parquet과 fetched jsonl을 읽어 `articles`에 신규 URL만 append |
| `scripts/05_news_score.py` | `scoring_v2/rescore_v2.py` 호출하는 scoring wrapper |
| `scripts/run_news_pipeline.py` | 전체 파이프라인 orchestration |
| `scoring_v2/rescore_v2.py` | SAP AI Core per-item scoring, body/title fallback, `raw_news_scored_v2` upsert |
| `scoring_v2/aicore_client.py` | SAP AI Core Orchestration v2 호출 wrapper (config_ref + tenacity retry) |
| `scoring_v2/weight_map.py` | category별 weight lookup |
| `export_to_main.py` | `news_v2.db` → `soybean.db.raw_news_scored_v2` 풀 dump 어댑터 |

## 운영 규칙

- `data/db/news_v2.db`는 정본 SQLite 파일(gitignored). 1.6GB 정본은 작업자가 직접 전달.
- `data/db/news_v2_seed.sql.gz`는 body 제외 dump(git commit). 새 환경에서 `python scripts/seed_news_db.py`로 점수만 복원 가능.
- `articles`는 단일 기사 원장. **운영 중 rebuild 금지** (raw_news_scored_v2.id가 articles.id를 참조).
- 신규 실행분은 신규 URL만 INSERT, 기존 row는 절대 수정/삭제 안 함.
- `is_tradeable=0` row의 score 차원(sentiment/impact/certainty/final_signal)은 의미 없는 값으로 간주.
- ML 입력에서는 `body` 원문이 아니라 `soybean.db.raw_news_scored_v2`의 점수 + 메타를 사용. 본문 분석이 필요할 때만 `ATTACH 'data/db/news_v2.db' AS news`로 직접 join.

## 점검 쿼리 (`data/db/soybean.db` 기준)

전체 커버리지:
```sql
SELECT COUNT(*) FROM raw_news_scored_v2;
SELECT scoring_input_mode, COUNT(*) FROM raw_news_scored_v2 GROUP BY scoring_input_mode;
SELECT is_tradeable, COUNT(*) FROM raw_news_scored_v2 GROUP BY is_tradeable;
```

발행 시점 분포:
```sql
SELECT substr(date,1,7) AS yyyymm, COUNT(*) FROM raw_news_scored_v2
GROUP BY yyyymm ORDER BY yyyymm DESC LIMIT 12;
```

본문 보유 정본 검증 (`data/db/news_v2.db` 기준):
```sql
SELECT COUNT(*) FROM articles WHERE body IS NOT NULL AND length(body) > 0;
```

## 참고
- 점수 정본은 `data/db/news_v2.db.raw_news_scored_v2`. `data/db/soybean.db.raw_news_scored_v2`는 매 실행 마지막에 정본을 그대로 옮긴 사본.
- look-ahead 정책 (`processed_at` vs `date`)은 ML 측 features 레이어 책임. 본 모듈은 객관적 사실만 적재.
