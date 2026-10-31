# muse_moe_w4a16 · 2026-10-31T045322Z · submit · adopt-public-moe

- **Job:** `0882294eec144bea` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` on master (326,271 B, sha256 6aec980e1447).
- **Authority:** user ruling 10-31 ~05:10Z (adopt the public board as the foundation) and the AskUserQuestion answer "Submit all 4 now".

## Goal
Scored run of the adopted foundation: public alygg77@6cae49a moe, unmodified.

## What changed, and where
The whole file is replaced by the public-based one. Line 2 is the attribution notice; notices are vendored in `third_party/okc-muse-spark-kernels/`. Our previous file is at tag `ours-final-1031`.

## Rules check (each must be true before the run)
- [x] Compliance (FA3-precision ruling; okc-stay-compliant): clean under the compliance ruling (inventory/verifier wf_1949358b-dd0).
- [x] Same body that passed judge checks: 2.3648 / 2.3761 (M0 controls; bytes identical below line 2).
- [x] Native code through torch load_inline/Triton only; under 1 MB; header kept.
- [x] No timing patches, harness/I-O/secrets, reference import, subprocesses or network; all work inside the timed call.
- [x] Licence: Apache-2.0 (plus MIT for the bf16 parts), notices kept; we have the right to submit under the challenge's Apache-2.0 terms.
- Reopen: public-base foundation, different structure.

## Risks / what could fail
Board = best scored run; one read can land anywhere in the ~0.5-1.2% A/A band.

## Result (after the run)
- pending
