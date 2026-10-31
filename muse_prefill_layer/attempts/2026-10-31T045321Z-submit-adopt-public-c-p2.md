# muse_prefill_layer · 2026-10-31T045321Z · submit · adopt-public-c-p2

- **Job:** `8b89517bd87544b6` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` on master (295,944 B, sha256 e5558cfef075).
- **Authority:** user ruling 10-31 ~05:10Z (adopt the public board as the foundation) and the AskUserQuestion answer "Submit all 4 now".

## Goal
Scored run of the adopted foundation: public-C P2 (alygg77@6cae49a prefill, own FP8 attention with FP32 unrounded-P denominator, variants pinned).

## What changed, and where
The whole file is replaced by the public-based one. Line 2 is the attribution notice; notices are vendored in `third_party/okc-muse-spark-kernels/`. Our previous file is at tag `ours-final-1031`.

## Rules check (each must be true before the run)
- [x] Compliance (FA3-precision ruling; okc-stay-compliant): audit PASS (public-plus, 10-31); H100 8/8 rows, 5/5 tests.
- [x] Same body that passed judge checks: 1.7848 / 1.7634 (d114d2e8, c456c4b9; bytes identical below line 2).
- [x] Native code through torch load_inline/Triton only; under 1 MB; header kept.
- [x] No timing patches, harness/I-O/secrets, reference import, subprocesses or network; all work inside the timed call.
- [x] Licence: Apache-2.0 (plus MIT for the bf16 parts), notices kept; we have the right to submit under the challenge's Apache-2.0 terms.
- Reopen: public-base foundation, different structure.

## Risks / what could fail
Board = best scored run; one read can land anywhere in the ~0.5-1.2% A/A band.

## Result (after the run)
- pending
