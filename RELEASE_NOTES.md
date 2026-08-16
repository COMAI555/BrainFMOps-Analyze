# Release preparation notes — v1.0.0

## Included changes

- Replaced machine-specific Windows paths in source examples with relative paths.
- Replaced fixed notebook roots with paths resolved from the repository working directory.
- Removed all stored notebook execution outputs and execution counters.
- Removed two superseded notebooks:
  - `31A_IEEE_subject_level_results_figure_generator.ipynb`
  - `34_Table1_Table2_Table3_AutoGenerator.ipynb`
- Added a public dependency specification and repository README.

## Deliberately retained

- OASIS naming-pattern examples and regular expressions required to discover and
  normalize OASIS subject folders. These are structural examples, not released
  participant records.

## Still required before DOI deposit

- Confirm author order and affiliations.
- Add `CITATION.cff`.
- Confirm copyright ownership and add the final `LICENSE`.
- Capture an environment lock from the verified release environment.
- Add the GitHub release URL and Zenodo DOI after publication.
