# Table CSVs and Paper-Table Mapping

This directory contains clean CSV summaries aligned to the current submitted paper table numbering. The recommended files for checking table values are the top-level table CSVs listed below and the audit maps `table_coverage_audit_v6.csv` and `table_value_audit_v6.csv`.

## Current mapping

- `main_table1_10seed_audit.csv` matches main Table 1.
- `main_table2_llm_summary.csv` matches main Table 2.
- `appendix_table3_glue_summary.csv` matches Appendix Table 3.
- `appendix_table4_clip_summary.csv` matches Appendix Table 4.
- `appendix_table5_same_setup_ablation_paper_reported.csv` records Appendix Table 5 values.
- `appendix_table6_rdrop_probe_paper_reported.csv` records Appendix Table 6 values.
- `appendix_table7_mtbench_gpt52_summary.csv` matches Appendix Table 7.
- `appendix_table8_10seed_checks_summary.csv` records Appendix Table 8.
- `appendix_table9_flat_lora_schedule_summary.csv` records Appendix Table 9.
- `appendix_table10_glue_pacf_hyperparameters.csv` records Appendix Table 10.
- `appendix_table11_vision_pacf_hyperparameters.csv` records Appendix Table 11.
- `appendix_table12_llm_task_data_summary.csv` records Appendix Table 12.
- `appendix_table13_llm_shared_optimization_lora_config.csv` records Appendix Table 13.
- `appendix_table14_llm_reg_space_summary.csv` records Appendix Table 14.
- `appendix_table15_llm_pacf_selected_hyperparameters.csv` records Appendix Table 15.
- `appendix_table16_llm_seed_variability_summary.csv` records Appendix Table 16.
- `appendix_table17_sdxl_lora_personalization_hyperparameters.csv` records Appendix Table 17.
- `appendix_table18_sdxl_regularization_knobs.csv` records Appendix Table 18.
- `table_coverage_audit_v6.csv` maps every current paper table and Figure 1 to supporting supplemental files.
- `table_value_audit_v6.csv` records the final manual value audit against the current paper.

## Domain-level exports

- `glue/` contains task-level GLUE result exports used by Appendix Table 3.
- `clip/` contains dataset-level CLIP result exports used by Appendix Table 4.
- `llm/` contains clean paper-aligned summaries for math, code, and chat evaluations used by main Table 2 and Appendix Table 7.

The LLM detail files were rewritten as clean paper-aligned summaries to avoid carrying stale detailed rows from older drafts. Table 5, Table 6, and Table 8 are included as paper-reported summaries where separate raw ablation or 10-seed logs were not present in the uploaded artifact.
