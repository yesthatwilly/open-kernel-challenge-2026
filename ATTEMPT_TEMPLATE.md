# <problem> · <YYYY-MM-DDTHHMMZ> · <check|submit> · <slug>

Copy this to `<problem>/attempts/<YYYY-MM-DDTHHMMZ>-<check|submit>-<slug>.md` (UTC, so the
timestamp sorts with the 00:00 UTC daily-limit reset) BEFORE running `okc check` or
`okc submit`. Fill in the job id and result afterwards.

- **Job:** `<job id>` (`okc status <job>`)
- **File:** `<problem>/submission.py` at git `<short sha or "uncommitted">`

## Goal
What this attempt hopes to gain, and the number it is chasing (which shapes/loads, roughly
how much time, from which profile line).

## What changed, and where
Which functions / kernels / launches were touched, added or removed. One bullet per part.

## Rules check (each must be true before the run)
- [ ] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints; no
      allocation or control flow on input values (shapes and `config` are fine).
- [ ] Nothing derived from `data` is cached across calls.
- [ ] Weights: repacked losslessly in `prepare()` only; NVFP4 stays NVFP4, BF16 stays BF16
      (router FP32 → BF16 hi+lo with FP32 accumulation is the one exception).
- [ ] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3.
- [ ] Attention sees every key (full window / full context); no top-k, sparse, block-skip,
      or weight-based skipping.
- [ ] No timing patches, harness frames/I/O/secrets, module edits, reference import,
      subprocesses or network.
- [ ] All work happens inside the timed call; no output caching or replay.

## Risks / what could fail
Correctness margins, shapes most likely to break, anything near a rule's edge.

## Result (after the run)
Pass/fail, first failing case if any, speedup with interval, profile file, what to try next.
