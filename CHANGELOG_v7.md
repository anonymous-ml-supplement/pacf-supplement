# v7 update

This update aligns the supplemental material with the current submitted paper version.

- Updated `tables/main_table1_10seed_audit.csv` so the PACF-Cons CoLA entry is `59.05±1.23`, with the paired-bootstrap delta `+0.26 [-0.34,+0.83]` matching the current Table 1.
- Added `tables/main_table1_cola_pacf_cons_10seed_raw.csv`, a sanitized per-seed support file for the updated PACF-Cons CoLA result.
- Added `tables/main_table1_cola_pacf_cons_10seed_summary_from_raw.csv`, which recomputes the PACF-Cons CoLA mean and sample standard deviation from the raw rows.
- Replaced the stale older Appendix Table 8 support file with `tables/appendix_table8_matched_baseline_coefficient_grids.csv`, matching the current Table 8.
- Regenerated the audit maps as `tables/table_coverage_audit_v7.csv` and `tables/table_value_audit_v7.csv`.
