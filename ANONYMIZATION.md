# Anonymization Notes

This artifact was prepared for double-blind review. Author names, institutional identifiers, personal paths, private URLs, Git metadata, API credentials, hidden system files, and local machine artifacts were checked before packaging.

## Removed or rewritten

- macOS metadata files such as `.DS_Store` and AppleDouble `._*` files.
- The `__MACOSX/` resource fork directory.
- Outdated venue-specific wording in README text and script docstrings.
- Office spreadsheet metadata by converting uploaded `.xlsx` result workbooks to CSV.
- Broken smoke-test config references that pointed to non-existing files.
- Non-portable or stale script references where detected.

## Checked identifier classes

The audit checked for author names, advisor names, lab names, institution names, email addresses, usernames, local absolute paths, cluster markers, job markers, Git metadata, private URLs, API credentials, notebook metadata, PDF or image metadata, and hidden files.

## Paths and metadata

All commands in the cleaned configs use relative paths. PNG qualitative files were checked for author-identifying text where possible. Spreadsheet result files are stored as CSV files and therefore do not retain Office document properties.

Readers should contact the authors through the submission system if a reproducibility issue occurs.
