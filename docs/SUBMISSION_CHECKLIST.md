# Submission checklist

This checklist contains publication decisions that cannot be made by source
code alone.

1. Select the target venue template and insert either the required anonymous
   author block or the final author, affiliation, and corresponding-author
   information.
2. Add a code-and-data availability statement appropriate to the venue.  Do
   not expose an author-identifying repository URL during double-blind review.
3. Choose and add a repository license.  This is a rights-holder decision and
   is intentionally not inferred by this release.
4. Run the locked public protocols, archive the required predictions and
   evaluator logs, then populate only the bracketed result cells in the paper.
5. Run the comparator protocol in Table 9.  Retain the exact baseline source,
   checkpoint, dataset version, evaluator command, and any conversion script.
6. For every confidence interval, provide one per-sequence evaluator CSV per
   trained seed and use the multi-seed command in `README.md`.
7. Retain the synthetic controller regression as an implementation check; do
   not present it as natural-image tracking evidence.
