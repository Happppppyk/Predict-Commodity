#!/bin/bash
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== Step 1: 데이터 재수집 ==="
PYTHONPATH=src python src/ingestion/run_ingestion.py

echo "=== Step 2: master_daily 재빌드 ==="
PYTHONPATH=src python src/features/build_master.py

echo "=== Step 3: 결과 확인 ==="
python3 -c "
import sqlite3
conn = sqlite3.connect('data/db/soybean.db')
tables = [
    'raw_canola_oil', 'raw_dollar_index',
    'raw_sunflower_oil', 'raw_eia_biodiesel', 'master_daily'
]
print('=== 테이블별 행 수 ===')
for t in tables:
    try:
        n = conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
        print(f'{t:30s}: {n}행')
    except sqlite3.Error:
        print(f'{t:30s}: 테이블 없음')
conn.close()
"
