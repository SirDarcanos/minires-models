# Expanded four-source dataset assembly

Issue #31 completed the private assembly of retained historical measurements and
the accepted records from the new STL collection prepared by issue #30.

## Execution summary

- The uniquely identified 34-row historical source was excluded completely.
- Historical measurements were reused without geometry probing, PrusaSlicer, or
  UVtools execution.
- Three retained historical anonymous source groups and one new anonymous source
  group contributed to the final partitions.
- Historical and new records were harmonized into canonical and legacy-compatible
  fields without rounding authoritative targets.
- Derived surface-to-volume ratios were recalculated from their underlying
  measurements.
- The approved deterministic source-balanced allocator produced the current
  70/15/15 package.
- The complete package was installed atomically beneath private storage.

## Accounting

| Item | Count |
| --- | ---: |
| Historical input rows | 12,922 |
| Explicit historical source exclusion | 34 |
| Unusable historical rows | 2 |
| New inventory entries | 1,931 |
| Accepted new rows | 1,924 |
| Rejected new rows | 7 |
| Eligible harmonized rows | 14,810 |
| Training rows | 10,368 |
| Validation rows | 2,222 |
| Test rows | 2,220 |
| Total unusable rows in rejection accounting | 9 |

Historical input reconciles to the explicit exclusion, unusable historical rows,
and retained historical rows. New inventory reconciles to accepted and rejected
outcomes. Eligible rows reconcile exactly to the three model partitions.

## Audit result

The private post-assembly audit passed all checks for input fingerprints, artifact
checksums, partition counts, disjointness and coverage, four-source representation,
duplicate confinement, ratio derivation, private-field removal, permissions, and
aggregate reconciliation.

The ignored `private/issue-31/` package contains the current train, validation,
and test artifacts, rejection accounting, provenance, reconciled manifest,
checksums, and audit result. No source identity, filename, raw path, row-level
measurement, target, or checksum is published here.

Evaluation using this package supports claims about performance on held-out STL
rows drawn from the retained sources. It does not establish unseen-source or
miniature-family-independent performance.
