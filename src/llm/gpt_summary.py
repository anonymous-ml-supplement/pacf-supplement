#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize scored JSONL files for LLM experiments.

Reads one or more `*_scored.jsonl` files and produces:
  - long-form CSV (per question)
  - summary CSV (mean/std per run)

This is used by `scripts/summarize_all.py`.
"""

import os, json, glob, argparse, csv, re
from collections import defaultdict

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"[JSON ERROR] {path}:{ln} cannot json.loads(): {e}")
    return rows

def parse_score(x):
    """
    Robust score parsing:
    - int/float -> float
    - string -> extract first number
    - list -> mean of numeric items
    - dict -> try common keys, else mean of numeric values
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        # extract first numeric token
        m = re.search(r"[-+]?\d*\.?\d+", x)
        return float(m.group(0)) if m else None
    if isinstance(x, list):
        vals = []
        for t in x:
            try:
                vals.append(float(t))
            except:
                if isinstance(t, str):
                    m = re.search(r"[-+]?\d*\.?\d+", t)
                    if m:
                        vals.append(float(m.group(0)))
        return (sum(vals) / len(vals)) if vals else None
    if isinstance(x, dict):
        for k in ["score", "final", "overall", "avg", "mean"]:
            if k in x:
                try:
                    return float(x[k])
                except:
                    pass
        vals = []
        for v in x.values():
            s = parse_score(v)
            if s is not None:
                vals.append(s)
        return (sum(vals) / len(vals)) if vals else None
    return None

def normalize_question_id(obj):
    """
    Prefer explicit ids if present.
    If missing, fall back to 'q' text (recommended for MT-Bench style).
    """
    qid = obj.get("question_id", obj.get("qid", None))
    if qid is not None:
        return str(qid)

    q_text = obj.get("q", None)
    if q_text is None:
        return None

    # Keep the full prompt text as ID (stable & human-debuggable).
    # If you worry about CSV bloat, you can hash it, but text is safest.
    return str(q_text).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_dir", required=True, help="folder containing scored *.jsonl (with q/answer/score/judge_model/rationale)")
    ap.add_argument("--out_dir", default=None, help="output folder for csv (default: <scored_dir>/csv)")
    ap.add_argument("--pattern", default="*.jsonl", help="glob pattern, default=*.jsonl")
    args = ap.parse_args()

    scored_dir = os.path.abspath(args.scored_dir)
    if not os.path.isdir(scored_dir):
        raise RuntimeError(f"[ERROR] scored_dir not found: {scored_dir}")

    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(scored_dir, "csv")
    safe_mkdir(out_dir)

    files = sorted(glob.glob(os.path.join(scored_dir, args.pattern)))
    print(f"[INFO] scored_dir = {scored_dir}")
    print(f"[INFO] out_dir    = {out_dir}")
    print(f"[INFO] pattern    = {args.pattern}")
    print(f"[INFO] found files = {len(files)}")
    for p in files[:10]:
        print(f"  - {os.path.basename(p)}")
    if len(files) == 0:
        raise RuntimeError("[FATAL] No jsonl files found. Check --scored_dir and --pattern")

    long_rows = []
    bad_rows = 0
    first_bad_example = None

    for fp in files:
        data = read_jsonl(fp)
        if len(data) == 0:
            print(f"[WARN] empty jsonl: {fp}")
            continue

        # file-level run_name fallback: use filename
        file_run = os.path.splitext(os.path.basename(fp))[0]

        for obj in data:
            qid = normalize_question_id(obj)
            run_name = obj.get("run_name", file_run)

            judge = obj.get("judge_model", obj.get("judge", obj.get("model", "")))
            score_raw = obj.get("score", obj.get("rating", obj.get("final_score", obj.get("scores", None))))
            rationale = obj.get("rationale", obj.get("reason", ""))

            # strict requirements: must have qid and score
            if qid is None or score_raw is None:
                bad_rows += 1
                if first_bad_example is None:
                    first_bad_example = {"file": os.path.basename(fp), "obj_keys": list(obj.keys()), "obj": obj}
                continue

            score_f = parse_score(score_raw)
            if score_f is None:
                bad_rows += 1
                if first_bad_example is None:
                    first_bad_example = {"file": os.path.basename(fp), "obj_keys": list(obj.keys()), "obj": obj}
                continue

            long_rows.append({
                "run_name": run_name,
                "question_id": qid,
                "judge_model": judge,
                "score": score_f,
                "rationale": rationale,
                "source_file": os.path.basename(fp),
            })

    print(f"[INFO] parsed rows = {len(long_rows)} (skipped={bad_rows})")
    if len(long_rows) == 0:
        if first_bad_example is not None:
            print("[DEBUG] first skipped example:")
            print("  source_file:", first_bad_example["file"])
            print("  keys:", first_bad_example["obj_keys"])
            # print small slice to avoid huge dumps
            obj = first_bad_example["obj"]
            slim = {k: obj[k] for k in list(obj.keys())[:10]}
            print("  obj_head:", slim)
        raise RuntimeError("[FATAL] Parsed 0 rows. Your jsonl likely lacks score or q/question_id fields, or score is non-parsable.")

    # --- write scores_long.csv ---
    long_path = os.path.join(out_dir, "scores_long.csv")
    with open(long_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        w.writeheader()
        w.writerows(long_rows)

    # --- summary per run ---
    by_run = defaultdict(list)
    for r in long_rows:
        by_run[r["run_name"]].append(r["score"])

    summary = []
    for run, scores in sorted(by_run.items(), key=lambda x: x[0]):
        n = len(scores)
        mean = sum(scores) / n
        # std (sample)
        if n > 1:
            var = sum((s - mean) ** 2 for s in scores) / (n - 1)
            std = var ** 0.5
        else:
            std = 0.0
        summary.append({"run_name": run, "n": n, "mean_score": mean, "std_score": std})

    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run_name", "n", "mean_score", "std_score"])
        w.writeheader()
        w.writerows(summary)

    # --- per-question wide table (one row per question_id, columns = run_name score) ---
    # if same (run, qid) appears multiple times (e.g., multiple judges), take average
    tmp = defaultdict(list)  # (qid, run) -> [scores]
    qids = set()
    runs = set()
    for r in long_rows:
        qids.add(r["question_id"])
        runs.add(r["run_name"])
        tmp[(r["question_id"], r["run_name"])].append(r["score"])

    runs = sorted(runs)
    qids = sorted(qids)

    wide_path = os.path.join(out_dir, "wide_scores.csv")
    with open(wide_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["question_id"] + runs
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for qid in qids:
            row = {"question_id": qid}
            for run in runs:
                ss = tmp.get((qid, run), [])
                row[run] = (sum(ss) / len(ss)) if ss else ""
            w.writerow(row)

    print("[SAVED]")
    print(f" - {long_path}")
    print(f" - {summary_path}")
    print(f" - {wide_path}")

if __name__ == "__main__":
    main()