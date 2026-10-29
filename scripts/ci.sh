#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
for suite in test_algorithms.py test_remote_delta.py test_lifecycle.py test_memory_budget.py; do
  python3 -m unittest discover -s tests -p "$suite" -v
done
python3 tests/test_control.py
python3 tests/test_remote_protocol.py
python3 tools/check_results.py
python3 tools/check_round2.py
python3 scripts/check_release.py
for script in start.sh stop.sh status.sh verify.sh scripts/ci.sh; do bash -n "$script"; done
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT
"${CXX:-g++}" -std=c++17 -O1 -g -fsanitize=address,undefined -pthread tests/test_tiered_store.cpp -o "$build_dir/test-tiered-store"
"$build_dir/test-tiered-store"
