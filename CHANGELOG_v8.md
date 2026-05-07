# v8 update for main16.tex

This update aligns the supplemental material with the advisor-updated `main16.tex` paper version.

- Rebuilt the top-level table CSV mapping around the 15 tables present in `main16.tex`.
- Updated Table 1 support to include CoLA MCC, Cars accuracy, CIFAR-10 accuracy, Cars paired-bootstrap/Wilcoxon values, and the additional PiSSA, CLoRA, and StelLA-style context rows.
- Added CSV support for the new clean CoLA R-Drop comparison (`tab:rdrop_clean`) and the 10-seed aligned-vs-misaligned perturbation control (`tab:alignment_control`).
- Updated GLUE, vision, LLM, ablation, R-Drop probe, matched-grid, Flat-LoRA schedule, and hyperparameter CSVs to match the current main16 table numbering.
- Removed stale top-level v7 table-numbering CSVs to avoid confusion with earlier drafts.
- Regenerated `table_coverage_audit_v8_main16.csv` and `table_value_audit_v8_main16.csv`.
