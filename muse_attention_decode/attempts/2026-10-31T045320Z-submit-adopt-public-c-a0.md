# muse_attention_decode · 2026-10-31T045320Z · submit · adopt-public-c-a0

- **Job:** `0ef88b074a4e4cd8` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` on master (156,629 B, sha256 ccfc129d7277).
- **Authority:** user ruling 10-31 ~05:10Z (adopt the public board as the foundation) and the AskUserQuestion answer "Submit all 4 now".

## Goal
Scored run of the adopted foundation: public-C A0 (alygg77@6cae49a attention + exact-max FA3 softmax, P<=256, FP32 unrounded denominator, R11).

## What changed, and where
The whole file is replaced by the public-based one. Line 2 is the attribution notice; notices are vendored in `third_party/okc-muse-spark-kernels/`. Our previous file is at tag `ours-final-1031`.

## Rules check (each must be true before the run)
- [x] Compliance (FA3-precision ruling; okc-stay-compliant): audit PASS (public-plus, 10-31); H100 24/24 rows, 16/16 tests.
- [x] Same body that passed judge checks: 2.4361 / 2.4377 (5b751676, 3ba805db; bytes identical below line 2).
- [x] Native code through torch load_inline/Triton only; under 1 MB; header kept.
- [x] No timing patches, harness/I-O/secrets, reference import, subprocesses or network; all work inside the timed call.
- [x] Licence: Apache-2.0 (plus MIT for the bf16 parts), notices kept; we have the right to submit under the challenge's Apache-2.0 terms.
- Reopen: public-base foundation, different structure.

## Risks / what could fail
Board = best scored run; one read can land anywhere in the ~0.5-1.2% A/A band.

## Result (after the run)
- **Passed**, scored. **Speedup 2.4337** [2.3972, 2.4702] on sfo3. Registered as BEST via churn.py register-landing.
