#!/usr/bin/env bash
set -euo pipefail
python3 scripts/run_config.py --config configs/clip/clip_cifar10_r16_pacfkl_paper.yaml --dry_run
