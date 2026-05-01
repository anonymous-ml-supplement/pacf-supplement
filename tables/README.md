# Tables Directory

This directory contains metadata-free CSV exports and clean paper-table summaries. Numeric values were not changed during anonymization.

## Clean paper-table summaries

- `main_table1_glue_summary.csv` matches Table 1 in the paper.
- `main_table2_clip_summary.csv` matches Table 2 in the paper.
- `main_table3_llm_summary.csv` matches Table 3 in the paper.
- `table4_same_setup_ablation_paper_reported.csv` records the paper-reported Table 4 values. The raw ablation logs were not present in the uploaded artifact.
- `table5_rdrop_probe_paper_reported.csv` records the paper-reported Table 5 values. The raw ablation logs were not present in the uploaded artifact.
- `appendix_table6_mtbench_gpt52_summary.csv` matches Appendix Table 6 in the paper.

## Domain-level CSV exports

- `glue/` contains task-level GLUE result exports.
- `clip/` contains dataset-level CLIP result exports. The `dtd.csv` file is the DTD export used by Table 2.
- `llm/` contains LLM result exports for math, code, and chat evaluations.

The detailed CSV exports are intended for inspection. Some exported files include per-seed rows and command arguments from the uploaded workbooks. These rows are retained for transparency, but clean paper-table summaries should be used when checking the submitted tables.
