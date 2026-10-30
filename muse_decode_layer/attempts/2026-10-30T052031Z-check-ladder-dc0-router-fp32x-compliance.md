# muse_decode_layer · 2026-10-30T052031Z · check · ladder-dc0-router-fp32x-compliance

Hypothesis: `muse_decode_layer/hypotheses/queue/2026-10-30T050914Z-ladder-dc0-router-fp32x-compliance.md` (branch `ladder/dece2`).

- **Job:** `7f5e68e6fcfe402d` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `64a6861f (8a8a7dcf + w2 reduce pinned to FP32)` (ladder/dece2), 991,587 B

## Goal
Compliance (user 10-30: "fix it first ... STAY COMPLIANT"). R16 (campaign gates/R16.flipped) found every T <= 8 router launch
on the scored path (b12p at T=1 = decode 1 rows, dl_pooler_router at T=2..8 = verify 1 rows) consumed the BF16-rounded
FFN input, while SGLang (baseline.py:134 / :316-328) feeds F.linear the FP32 router input up to 8 tokens. DC0 makes the
router consume exactly SGLang's FP32 input with SGLang's op at T <= 8. Expected cost: decode 1 / verify 1 rows +3..6.5 us
each (public FACTS s5: separate FP32 router GEMV +5.5-6.5 us), headline about -1..-2%. Also carries the E2 size trim
(aac30051: 999,631 -> 991,352 B, dead Python / blank lines / anatomy diagnostics; outputs bitwise == master on every
T <= 32 row).

## What changed, and where
**Mechanism:** FP32 router input at T <= 8 (SGLang's F.linear on the FP32 router_input from FusedPostNormResidualPreNorm).
- `ROUTER_FP32_MAX_TOKENS = 8` (was 1): the no-dense-chain fallback (`_t128_side`) now also uses F.linear(router_input) at T <= 8.
- `kernel_fn`: `attn_boundary(..., router_dtype=None if split_router else torch.float32)` (was None whenever the dense chain
  ran, i.e. every T <= 32); passes `rx=(router_input, state["router"])` to `_dl_forward` at T <= 8.
- `_dl_forward(..., rx=None)`: after the pooler kernel (b12p at T=1 / dl_pooler_router), `logits = F.linear(*rx)` (FP32 x,
  FP32 weight, SGLang's op); the route kernel consumes those logits. The fused kernels' own hi/lo logits are discarded at
  T <= 8. T > 8 unchanged (BF16 hi + lo pair, FP32 accumulation, allowed by rules :107).
- E2 trim underneath: 4 commits ca82ba43..aac30051 (research/dece2-1030.md: removal table, AST proof, CUDA/Triton sources byte-identical).

Node validation (gpunode06 GPU 6, takemyjob hopper/ladder-dece2):
- DC0 gate, `runs/20261029-220843-gpunode06` dece2-dc0gate (kernel_fn hooked on the real scored path, judge's reference inputs,
  T <= 8 judge cases + decode T = 2..7, 48 seeds): 3,952 token rows; router input float32 and unrounded in every
  call at every T = 1..8 (T=1 via b12p 192/192); logits consumed == F.linear(router_input, W fp32) **bitwise (max diff 0)**;
  F.linear vs fp64 2.7e-6..1.04e-5 (SGLang's own op error; public: 7.2e-6); **top-8 identical 3,952/3,952** vs the fp64
  biased top-8; check_implementation 0 failures. (The driver's extra fp64 bar of 1e-5 trips at 1.04e-5 on T=5: that is
  F.linear itself, identical to SGLang; the pre-registered bar, |consumed - F.linear fp32| <= 1e-5, holds at 0.)
- Same run, nodejudge dc0-val (cold fresh builds, 300 s budget enforced, judge-like router ACCEPT): prepare 117.9 s
  (import 250 s; est. run 378 s / 900 s), 13/13 tests, 12 shapes x 2 reps every replay checked, passed=True.

## Rules check (each must be true before the run)
Checked against `git diff ae98caf0 64a6861f (8a8a7dcf + w2 reduce pinned to FP32) -- muse_decode_layer/submission.py`.
- [x] **Router precision (rules :107-108, :137): at T <= 8 the router consumes SGLang's FP32 router input (unrounded,
      from SGLang's own FusedPostNormResidualPreNorm with router_dtype=float32) and the FP32 weight via F.linear, exactly
      SGLang's op; at T > 8 the BF16 hi + lo weight pair with FP32 accumulation on the BF16 activation, as SGLang does
      above 8 tokens. No router path at any T consumes a lower-precision input than SGLang.**
- [x] CUDA-graph capturable: one F.linear on the same stream in kernel_fn; no sync / .item() / host branch on data.
- [x] Nothing cached across calls; all decisions in prepare from config shapes.
- [x] Weights unchanged (FP32 router used as is; NVFP4 / BF16 weights untouched).
- [x] Activations BF16 where SGLang keeps them BF16; FP8 KV cache unchanged; queries unchanged.
- [x] Attention untouched.
- [x] No timing patches, harness access, reference import, subprocess or network.
- [x] All work inside the timed call.
- [x] Size 991,465 B < 1,000,000; every compiled CUDA / Triton source byte-identical to master (E2 proof).

```
okc_lint: OK with warnings (pre-existing host-sync-in-prepare / uint8 warnings only; 0 errors)
```

## Risks / what could fail
low (correctness), certain cost: decode 1 / verify 1 rows slower by the F.linear (~3-6.5 us each); prepare races (b12p,
dense-chain, Engine A layer gates) now time layers that include it, so a T <= 8 route may flip where margins were thin.
The board would drop roughly 1-2%; this is the compliant baseline every later decode run must keep.
Next (speed, not compliance): DC0b overlap the FP32 router on the side stream with the pooler, or a fused FP32-x router
stage in b12p / dl_pooler_router (public: ~parity locally, judge charged ~3.8% on v1), both gated by the same dc0gate.

- [x] Split-K reduce: w2 pinned to SGLang's FP32 reduce (use_atomic_add=False, use_fp32_reduce=True) on every route; the BF16-atomic fallback at T=1 is removed (compliance audit 2026-10-30, research/compliance-audit-1030.md). Validated only by py_compile here: the scored routes (Engine A) never took that path per the audit.

## Result
- job `7f5e68e6fcfe402d` (check), passed 13/13, **1.645x [1.621, 1.670]** (compliant; the withdrawn non-compliant best was 1.7225). d1s 96.8, d1f 103.6, v1s 166.2, v1f 181.8 us: the F.linear FP32 router costs ~+9 us at d1 and ~+24-28 us at v1 vs the BF16-x router.
