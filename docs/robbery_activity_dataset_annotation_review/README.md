# Human-verified annotation workflow

## Inventory inspected

Source ZIP: `C:\Users\admin\Downloads\robbery_activity_dataset-20261030T075950Z-1-001.zip`. Only the split manifest, annotation/review CSVs, and ten small robbery contact sheets were read or extracted. VideoMAE checkpoints and feature files were not extracted. All 400 original video files were found at `C:\Users\admin\OneDrive\Documents\crime_video_dataset` and SHA-256 hashed read-only.

The manifest contains 400 entries: 280 train, 59 validation, and 61 test. Category counts are Robbery 150, Normal 50, Assault 100, and Shooting 100. The provided action annotation CSV has 280 rows, all `verified_by_human=no`, with no action labels or boundaries. The event annotation CSV has 150 Robbery labels, all unverified and with blank start/end times. The review CSV contains one `needs_review` label with 37?44 second timestamps; it is unverified and is preserved only in prior-annotation context fields. No human annotation values were copied into the review fields.

## Review outputs

- `robbery_activity_dataset_human_review_worklist.csv`: all 400 manifest rows and blank human-review fields.
- `training_labels_review.csv`: train and validation entries after source-group leakage control.
- `heldout_test_labels_review.csv`: held-out test entries after the same control.
- `duplicate_and_conflict_groups.csv`: source re-encodes and exact-hash duplicate groups, split changes, and label conflicts.
- `contact_sheets/`: the ten contact sheets supplied in the ZIP; matching worklist rows link to these by relative path.
- `workflow_inventory.json`: machine-readable counts and integrity summary.

Human-review fields (`human_verified_category`, `action_label`, `start_timestamp_sec`, `end_timestamp_sec`, `reviewer`, `review_notes`) are intentionally blank. `verification_status` starts as `pending_review`. Contact sheets are navigation aids; reviewers must verify labels and exact boundaries against the original video.

## Duplicate and split integrity

Exact SHA-256 and normalized same-category source stems (for example, x264/x264A re-encodes) were grouped. There are 100 duplicate/source groups involving 201 rows. 39 groups crossed the supplied split assignments, so 50 rows have a `leakage_control_split` different from the manifest. The original `existing_split` is retained unchanged. Group members were assigned together using the most held-out original split in the group (test, then validation, then train), so duplicates cannot cross the output train/validation/test groups. This may change the effective split counts to train 236, validation 77, and test 87.

A cross-category exact-content duplicate group is flagged as a category-label conflict. Do not use that group for supervised training or evaluation until a human reviewer resolves its category. No source video, ZIP contents, model, checkpoint, Risk Score code, or tracking artifact was changed.

## Review procedure

1. Assign a reviewer who does not see classifier predictions before making the initial annotation. Record the reviewer identity and video SHA-256.
2. Review the original video. Use a linked contact sheet for orientation when present, then confirm the content and all event boundaries in the full-resolution source video.
3. Enter exactly one human-verified category: Assault, Robbery, Shooting, or Normal. A Normal category means no target activity was observed in this clip; it does not establish that no incident occurred outside the reviewed content.
4. Enter an action label and one row per distinct event interval. Use source-video seconds; use blank intervals only after the reviewer has checked the full video. Mark ambiguous cases for adjudication rather than guessing.
5. A second reviewer independently verifies the category and boundaries. Set `verification_status=independently_confirmed` only after agreement; otherwise use `needs_adjudication` or `ambiguous` and explain in notes.
6. Keep all duplicate-group members in their assigned `leakage_control_split`. Freeze the held-out test labels/split before model evaluation. Use `training_labels_review.csv` only for training/validation annotation and `heldout_test_labels_review.csv` only for final held-out evaluation.
