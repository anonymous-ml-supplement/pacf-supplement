#!/usr/bin/env bash
set -euo pipefail
python3 scripts/run_config.py --config configs/llm/llm_chat_r8_pacfcons_paper.yaml --dry_run
