# Release notes — v1.0.0

## Included changes

- Replaced machine-specific Windows paths in source examples with relative paths.
- Replaced fixed notebook roots with paths resolved from the repository working directory.
- Removed all stored notebook execution outputs and execution counters.
- Removed two superseded notebooks:
  - `31A_IEEE_subject_level_results_figure_generator.ipynb`
  - `34_Table1_Table2_Table3_AutoGenerator.ipynb`
- Added a public dependency specification and repository README.
- Added author and affiliation metadata in `CITATION.cff`.
- Added the MIT License under the confirmed copyright holder.

## Deliberately retained

- OASIS naming-pattern examples and regular expressions required to discover and
  normalize OASIS subject folders. These are structural examples, not released
  participant records.

## Known reproducibility limitations

- The recovered public evaluation table supports independent verification of
  the reported evaluation arithmetic, but the original optimizer state,
  complete development manifest, and full training provenance are unavailable.
- `requirements.txt` specifies supported dependency ranges rather than an
  environment lock from the archived training run.
- The version-specific Zenodo DOI will be available after Zenodo archives the
  GitHub release.
