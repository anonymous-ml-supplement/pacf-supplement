#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run an experiment from a YAML config.

The YAML format is:
  domain: <glue|clip|llm|sdxl>
  entry:  path/to/script.py
  args:   {key: value, ...}

It simply converts args into CLI flags: --key value (booleans as flags).
"""

import os
import sys
import shlex
import argparse
import subprocess

try:
    import yaml
except Exception as e:
    raise RuntimeError("PyYAML is required. Please `pip install pyyaml`.") from e

def _to_cli(args: dict):
    out = []
    for k, v in args.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                out.append(flag)
            continue
        if v is None:
            continue
        out += [flag, str(v)]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    entry = cfg.get("entry", "")
    argd = cfg.get("args", {}) or {}
    if not entry:
        raise RuntimeError("Missing `entry` in config.")

    cmd = [sys.executable, entry] + _to_cli(argd)
    print("[CMD]", " ".join(shlex.quote(x) for x in cmd))
    if args.dry_run:
        return

    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
