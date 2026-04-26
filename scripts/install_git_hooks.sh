#!/bin/bash
# 이 저장소의 Git 훅(.githooks)을 켭니다. clone 후 한 번 실행하면 됩니다.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
git config core.hooksPath "$ROOT/.githooks"
echo "core.hooksPath -> $ROOT/.githooks (이 저장소 전용)"
echo "훅 적용: pre-commit, commit-msg (기본 푸터/슬로건 자동 삽입 방지)"
