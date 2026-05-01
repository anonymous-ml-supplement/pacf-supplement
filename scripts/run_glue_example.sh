#!/usr/bin/env bash
set -euo pipefail
python3 scripts/run_config.py --config configs/glue/glue_sst2_r16_pacfkl_paper.yaml --dry_run
