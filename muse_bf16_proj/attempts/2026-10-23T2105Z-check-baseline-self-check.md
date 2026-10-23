# muse_bf16_proj · 2026-10-23T2105Z · check · baseline-self-check

*Backfilled 2026-10-23T2135Z from `~/.config/okc/jobs.json` and git history; no explanation was
written before this run.*

- **Job:** `1ccc6ba13da94194`
- **File:** `muse_bf16_proj/baseline.py` at git `ab21d70` (the committed starter kit)

## Goal
Calibration: run SGLang's own kernel (baseline.py) through the judge to see a ~1.0x speedup and the noise interval, so later gains can be read against it. Profile saved as profiles/baseline.1ccc6ba13da94194.*

## What changed, and where
Nothing: the file as `okc init` wrote it (plus the commit's own edits, if any).

## Rules check
Not performed before the run. The starter is a plain PyTorch reference and uses no caching,
no reduced precision and no harness access, so no rule was at risk.

## Result (after the run)
Passed all 9 tests, no flags. Speedup 1.0028x, 95% CI [0.988, 1.018]: SGLang's own kernel measures as 1.0x against itself within the ±1.5% interval, so the harness and this account's setup are sound. SGLang step time 4830.8 µs; scored 4817.6 µs. Profile: profiles/baseline.1ccc6ba13da94194.txt.
