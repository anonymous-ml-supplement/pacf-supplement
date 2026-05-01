#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM response evaluator (optional).

This script calls an external judge model (via the OpenAI API) to score model responses.
It is **optional** for paper reproduction: you can use pre-scored files in `runs/llm/*_scored.jsonl`
or skip judge calls entirely.

Compatibility:
  - Works with both legacy `openai` (0.x) and the newer OpenAI Python SDK (1.x).
  - Requires `OPENAI_API_KEY` in the environment.

Output format:
  Writes JSONL with added fields: `judge_model`, `score`, `rationale`.
"""

import os
import json
import argparse
import time
from tqdm import tqdm

SYSTEM = (
    "You are a strict evaluator for instruction-following responses.\n"
    "Score the response from 1 to 10 (integer).\n"
    "Criteria: correctness, helpfulness, clarity, and instruction following.\n"
    "Be consistent across all examples.\n"
    "Return ONLY valid JSON in one line with keys:\n"
    "{\"score\": <int 1-10>, \"rationale\": \"<=40 words\"}.\n"
)

def _call_openai_chat(model: str, messages, temperature: float = 0.0):
    """Call OpenAI chat completion using either SDK v1 or legacy v0."""
    try:
        # New SDK (openai>=1.0.0)
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content
    except Exception:
        # Legacy SDK (openai<1.0.0)
        import openai  # type: ignore
        openai.api_key = os.environ.get("OPENAI_API_KEY", "")
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp["choices"][0]["message"]["content"]

def judge_one(model, question, answer, max_retries=5):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Instruction:\n{question}\n\nResponse:\n{answer}\n"},
    ]

    for t in range(max_retries):
        try:
            text = (_call_openai_chat(model, messages, temperature=0.0) or "").strip()

            # Tolerant JSON extraction: take the first {...} span.
            l = text.find("{")
            r = text.rfind("}")
            if l == -1 or r == -1:
                return {"score": None, "rationale": text[:200]}

            try:
                return json.loads(text[l : r + 1])
            except Exception:
                return {"score": None, "rationale": text[:200]}

        except Exception as e:
            if t == max_retries - 1:
                return {"score": None, "rationale": f"ERROR: {repr(e)}"}
            time.sleep(2 ** t)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="input JSONL with fields like {question, answer}")
    ap.add_argument("--out_jsonl", required=True, help="output JSONL with added judge fields")
    ap.add_argument("--model", required=False, default=os.environ.get("OPENAI_JUDGE_MODEL", ""),
                    help="judge model name (or set OPENAI_JUDGE_MODEL).")
    ap.add_argument("--max_n", type=int, default=-1, help="optional cap for debugging")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY", ""):
        raise RuntimeError("OPENAI_API_KEY is not set")

    if not args.model:
        raise RuntimeError("Please provide --model or set OPENAI_JUDGE_MODEL")

    rows = []
    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if args.max_n > 0 and len(rows) >= args.max_n:
                break

    scores = []
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    with open(args.out_jsonl, "w", encoding="utf-8") as w:
        for ex in tqdm(rows, desc="Judge-eval"):
            q = ex.get("question") or ex.get("q") or ex.get("prompt") or ""
            a = ex.get("answer") or ex.get("response") or ""

            j = judge_one(args.model, q, a)

            out = dict(ex)
            out["judge_model"] = args.model
            out["score"] = j.get("score")
            out["rationale"] = j.get("rationale")
            w.write(json.dumps(out, ensure_ascii=False) + "\n")

            if isinstance(out["score"], int):
                scores.append(out["score"])

    if scores:
        print(f"[DONE] n_valid={len(scores)} avg_score={sum(scores)/len(scores):.3f}")
    else:
        print("[DONE] no valid scores parsed. Check out_jsonl for errors/rationale.")

if __name__ == "__main__":
    main()
