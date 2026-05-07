# PACF NeurIPS 2026 Anonymous Supplement, v8 for main16.tex

This repository is an anonymous supplemental artifact for a NeurIPS 2026 submission on PACF.

PACF studies robustness in the trainable LoRA coordinate space rather than in the full frozen parameter space. PACF-Cons is the primary consistency objective; it adds one perturbed forward pass and penalizes prediction-level disagreement under LoRA-subspace perturbations. PACF-KL is included as a lightweight LoRA-supported posterior and diagnostic regime with no additional perturbed forward pass. Under fixed variances, the optimized PACF-KL penalty is gradient-equivalent to LoRA-only L2 regularization, so this package does not present PACF-KL as a standalone optimizer-level novelty.

This v8 artifact aligns the supplemental table files to `main16.tex`. It updates the table mapping to the current 15 paper tables, adds the new Cars column and LoRA-variant context rows in Table 1, adds the new clean R-Drop comparison and alignment-control CSVs, and removes stale top-level v7 table-numbering files. Pretrained model weights, datasets, API credentials, and private run services are not redistributed.

## Directory structure

```text
supplemental_pacf_neurips2026_anonymous/
  README.md
  REPRODUCIBILITY.md
  ANONYMIZATION.md
  requirements.txt
  environment/
  configs/
  scripts/
  src/
  data/
  results/
  tables/
```

- `src/` contains domain-specific experiment code for GLUE, CLIP, LLM, and SDXL experiments.
- `configs/` contains YAML examples used by `scripts/run_config.py`.
- `tables/` contains metadata-free CSV exports and paper-table summaries aligned to `main16.tex`.
- `results/sdxl_qualitative/` contains qualitative PNG files for the qualitative SDXL example.
- `environment/` contains domain-specific dependency files.

## Environment setup

For a minimal setup, create a virtual environment and install the generic requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For full experiments, use the domain-specific files in `environment/`. GLUE and CLIP, LLM, and SDXL may require separate environments because their dependency versions differ.

## Running example configs

The following commands print the resolved command without launching a long run:

```bash
bash scripts/run_glue_example.sh
bash scripts/run_clip_example.sh
bash scripts/run_llm_example.sh
bash scripts/run_sdxl_example.sh
```

Remove `--dry_run` inside the scripts or call `scripts/run_config.py` directly to execute an experiment.

## Inspecting result tables

The `tables/` directory contains clean paper-table summaries and metadata-free CSV exports. For the current paper, use the `main16_tableXX_*.csv` files. The files `tables/table_coverage_audit_v8_main16.csv` and `tables/table_value_audit_v8_main16.csv` map the current paper tables to supporting supplemental files.

## External models and datasets

Users must download public datasets and pretrained models from their official sources and follow the corresponding licenses and access rules. This artifact does not redistribute pretrained weights or benchmark data.

## API-based evaluation

LLM judge evaluation is optional and requires user-provided credentials at runtime. No credentials are included in this artifact. Use a non-identifying environment when running optional API-based evaluation. Optional experiment logging is disabled by default in the supplied configs and scripts.

## Anonymity statement

All paths in the supplied configs are relative. No author-identifying information is intentionally included. Hidden system files, local spreadsheet metadata, and outdated draft-specific table mappings detected during the audit were removed or rewritten.
