# Table CSVs and Paper-Table Mapping, v8 for main16.tex

This directory contains clean CSV summaries aligned to `main16.tex`, the advisor-updated paper version. The recommended files for checking table values are the `main16_tableXX_*.csv` files and the two audit maps:

- `table_coverage_audit_v8_main16.csv`
- `table_value_audit_v8_main16.csv`

## Current main16 table mapping

- `main16_table01_10seed_controlled_audit.csv` matches Table 1 (`tab:ten_seed_main`).
- `main16_table02_rdrop_clean_cola.csv` matches Table 2 (`tab:rdrop_clean`).
- `main16_table03_alignment_control_cola.csv` matches Table 3 (`tab:alignment_control`).
- `main16_table04_glue_summary.csv` matches Table 4 (`tab:glue_full`).
- `main16_table05_vision_summary.csv` matches Table 5 (`tab:vision_full`).
- `main16_table06_llm_full.csv` matches Table 6 (`tab:llm_full`).
- `main16_table07_rank_objective_ablation.csv` matches Table 7 (`tab:rank_objective_ablation`).
- `main16_table08_rdrop_probe.csv` matches Table 8 (`tab:rdrop_probe`).
- `main16_table09_matched_baseline_grids.csv` matches Table 9 (`tab:matched_grid`).
- `main16_table10_flat_lora_schedule.csv` matches Table 10 (`tab:flat_lora_cfg`).
- `main16_table11_glue_pacf_hyperparameters.csv` matches Table 11 (`tab:glue_pacf_compact`).
- `main16_table12_vision_pacf_hyperparameters.csv` matches Table 12 (`tab:vision_compact`).
- `main16_table13_llm_task_data.csv` matches Table 13 (`tab:llm_task_data`).
- `main16_table14_llm_shared_optimization_lora_config.csv` matches Table 14 (`tab:llm_shared_train`).
- `main16_table15_llm_pacf_selected_hyperparameters.csv` matches Table 15 (`tab:llm_pacf_selected`).

## Supporting exports

- `main16_table01_cola_pacf_cons_10seed_raw.csv` contains the sanitized raw 10-seed CoLA PACF-Cons rows used for the PACF-Cons CoLA MCC entry in Table 1.
- `main16_table01_cola_pacf_cons_10seed_summary_from_raw.csv` records the summary derived from those raw rows.
- `glue/`, `clip/`, and `llm/` contain domain-level exports preserved from the previous supplemental artifact for traceability. The `main16_tableXX_*.csv` files are the paper-aligned source of truth for the current paper tables.

The previous v7 table-numbering files were removed from the top level to avoid stale mappings against earlier drafts.
