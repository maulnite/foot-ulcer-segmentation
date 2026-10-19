# Local datasets

Store downloaded datasets and prepared splits here. Dataset contents are excluded from Git; this README is the only tracked file in this directory.

Before training, document sources, licenses, versions, counts, sample correspondence, and split manifests. Preserve original files and derive preprocessing outputs separately. Do not combine Chronic Wound and FUSeg samples without auditing their overlap.

Suggested local layout: `raw/`, `processed/`, and `splits/`. Share sanitized split metadata separately only after reviewing identifiers and redistribution terms.
