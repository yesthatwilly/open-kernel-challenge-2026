# muse_bf16_proj · 2026-10-31T045323Z · submit · adopt-public-bf16

- **Job:** `109f577bcfaf4d8a` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` on master (382,823 B, sha256 6811a3ae6f18).
- **Authority:** user ruling 10-31 ~05:10Z (adopt the public board as the foundation) and the AskUserQuestion answer "Submit all 4 now".

## Goal
Scored run of the adopted foundation: public alygg77@6cae49a bf16, unmodified (MIT notice kept).

## What changed, and where
The whole file is replaced by the public-based one. Line 2 is the attribution notice; notices are vendored in `third_party/okc-muse-spark-kernels/`. Our previous file is at tag `ours-final-1031`.

## Rules check (each must be true before the run)
- [x] Compliance (FA3-precision ruling; okc-stay-compliant): clean under the compliance ruling (inventory/verifier wf_1949358b-dd0).
- [x] Same body that passed judge checks: 1.5365 / 1.5442 (B0 controls; bytes identical below line 2).
- [x] Native code through torch load_inline/Triton only; under 1 MB; header kept.
- [x] No timing patches, harness/I-O/secrets, reference import, subprocesses or network; all work inside the timed call.
- [x] Licence: Apache-2.0 (plus MIT for the bf16 parts), notices kept; we have the right to submit under the challenge's Apache-2.0 terms.
- Reopen: public-base foundation, different structure.

## Risks / what could fail
Board = best scored run; one read can land anywhere in the ~0.5-1.2% A/A band.

## Result (after the run)
- **Passed**, scored. **Speedup 1.5657** [1.5422, 1.5892] on sfo3. Registered as BEST via churn.py register-landing.
