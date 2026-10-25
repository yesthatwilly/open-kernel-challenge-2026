---
problem: muse_bf16_proj
slug: pooler-unpooler-m128-nn-layout-gated
created: 2026-10-25T011407Z
base: master
area: pooler GEMM + unpooler GEMM (m=65..128, cuBLAS NN layout)
change: >
  Built on master (blob a8bb42a). Narrow bf16-m128-nn-weight-layout-probe (done/neutral, 1.158x on
  the older dba4ffd master) to the only rows it won. Measured per shape in that run: pooler
  verify_16 m=128 25.21 -> 22.86 us (1.120x) and unpooler verify_16 m=128 19.65 -> 19.15 (1.043x);
  it LOST qkv_gate verify_32 m=256 (0.941x), and the other m=128/256 rows were flat. Master's
  incumbent on those two rows is cuBLAS TN through _bf16_gemm_dispatch_impl: pooler
  nvjet_sm90_tst_64x64_64x13_2x1 25.7 us kernel (row 25.55, ~2.6 TB/s against a 14.6 us floor),
  unpooler nvjet_sm90_tst_128x64_64x8_1x2 20.4 us (row 19.79). Change: in prepare(), for pooler
  and unpooler ONLY, store a lossless transposed copy state['pooler_t'] =
  W_pooler.t().contiguous() and state['unpooler_t'] = W_unpooler.t().contiguous() ([K, N] bf16
  row-major, same values and dtype; the originals stay for every other path; +134 MB once,
  untimed). In kernel_fn, add one branch placed just before master's cuBLAS fallback: if layer in
  {pooler, unpooler} and 65 <= cfg m <= 128 and cfg n equals the full weight width, return
  torch.mm(x, state[layer + '_t']) (a contiguous bf16 [m, n] with fp32 accumulation inside cuBLAS,
  the same math). Gate only on layer and cfg m/n. Everything else is byte-identical to master: m
  <= 32 Triton paths, the router, qkv_gate and o at every m, and pooler/unpooler at m >= 129 (so
  m=256 stays TN, where the probe showed nothing, and the m=130 and m=1000 unpooler tests stay on
  master). The attempt file records the nvjet kernel name and us for the two target rows from the
  profile.
expected:
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: NN probe measured 25.21 -> 22.86 us (1.12x); master row 25.55, nvjet 64x64 kernel 25.7 at ~2.6 TB/s vs 14.6 floor
  - shapes: layer=unpooler load=verify_16 m=128
    direction: faster
    why: NN probe measured 19.65 -> 19.15 (1.04x), small; needs the per-shape row, may read flat
  - shapes: layer=pooler load=verify_32 m=256
    direction: flat
    why: m >= 129 keeps TN
  - shapes: layer=unpooler load=verify_32 m=256
    direction: flat
    why: m >= 129 keeps TN
  - shapes: layer=pooler load=decode_*
    direction: flat
    why: m <= 32 untouched
  - shapes: layer=unpooler load=decode_*
    direction: flat
    why: m <= 32 untouched
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=o *
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: low
depends_on: bf16-m128-nn-weight-layout-probe
priority: 1
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T011846Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T011846Z
base_sha: a9625932047aa6e3c155977025168939699cb9b2
base_blob: a8bb42ac483318e9fbc84bc7826a27b7a704d1fb
branch: hyp/muse_bf16_proj/pooler-unpooler-m128-nn-layout-gated
worktree: .worktrees/muse_bf16_proj/pooler-unpooler-m128-nn-layout-gated
mode: null
restack_of: bf16-m128-nn-weight-layout-probe
stale_base: null
implemented_at: 2026-10-25T012003Z
impl_commit: 41146e726d7a3aa358d2ac097456e5d0182ed4c5
submission_sha256: 566e5e253ab0b470b429aaf95ad685a511645d457ab3f3f810b4238bcb010738
norm_sha256: 93c247c9b41768654a72c651c30e754e3d5accf1671e44ac6f9345621bb56e1e
submission_blob: 49c0a5c0d548f29014363a027a990ff274abd5de
lint_warnings: 0
---

# pooler-unpooler-m128-nn-layout-gated

## Change

Built on master (blob a8bb42a). Narrow bf16-m128-nn-weight-layout-probe (done/neutral, 1.158x on the older dba4ffd master) to the only rows it won. Measured per shape in that run: pooler verify_16 m=128 25.21 -> 22.86 us (1.120x) and unpooler verify_16 m=128 19.65 -> 19.15 (1.043x); it LOST qkv_gate verify_32 m=256 (0.941x), and the other m=128/256 rows were flat. Master's incumbent on those two rows is cuBLAS TN through _bf16_gemm_dispatch_impl: pooler nvjet_sm90_tst_64x64_64x13_2x1 25.7 us kernel (row 25.55, ~2.6 TB/s against a 14.6 us floor), unpooler nvjet_sm90_tst_128x64_64x8_1x2 20.4 us (row 19.79). Change: in prepare(), for pooler and unpooler ONLY, store a lossless transposed copy state['pooler_t'] = W_pooler.t().contiguous() and state['unpooler_t'] = W_unpooler.t().contiguous() ([K, N] bf16 row-major, same values and dtype; the originals stay for every other path; +134 MB once, untimed). In kernel_fn, add one branch placed just before master's cuBLAS fallback: if layer in {pooler, unpooler} and 65 <= cfg m <= 128 and cfg n equals the full weight width, return torch.mm(x, state[layer + '_t']) (a contiguous bf16 [m, n] with fp32 accumulation inside cuBLAS, the same math). Gate only on layer and cfg m/n. Everything else is byte-identical to master: m <= 32 Triton paths, the router, qkv_gate and o at every m, and pooler/unpooler at m >= 129 (so m=256 stays TN, where the probe showed nothing, and the m=130 and m=1000 unpooler tests stay on master). The attempt file records the nvjet kernel name and us for the two target rows from the profile.

## Expected

- `layer=pooler load=verify_16 m=128` faster: NN probe measured 25.21 -> 22.86 us (1.12x); master row 25.55, nvjet 64x64 kernel 25.7 at ~2.6 TB/s vs 14.6 floor
- `layer=unpooler load=verify_16 m=128` faster: NN probe measured 19.65 -> 19.15 (1.04x), small; needs the per-shape row, may read flat
- `layer=pooler load=verify_32 m=256` flat: m >= 129 keeps TN
- `layer=unpooler load=verify_32 m=256` flat: m >= 129 keeps TN
- `layer=pooler load=decode_*` flat: m <= 32 untouched
- `layer=unpooler load=decode_*` flat: m <= 32 untouched
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=router *` flat: untouched

## Risk

Low. Pure dispatch, no kernel. The probe's gain came from one run against an older master; LESSONS says cuBLAS rows at m>=128 swing between runs (pooler m=128 nvjet 23.9 vs 25.7 us across runs), so a per-shape gain under ~7% needs a repeat before it counts. The headline effect is small (~+0.5-0.7%: one load of six), so judge on the two rows.

## Adversarial review

Reviewed 2026-10-25T011846Z by an adversarial subagent (model opus).

**Verdict: proceed.** A narrow, lossless, dispatch-only follow-up to bf16-m128-nn-weight-layout-probe on the two rows where it won. Those rows are measured paired against SGLang's TN kernel in the same run, so one run gives a readable per-shape answer. Rules are clean: a transpose of a weight in prepare() is a BF16 repack, not caching derived from inputs.

### Justification
Plausible. On master, pooler verify_16 m=128 runs nvjet_sm90_tst_64x64_64x13_2x1 at 24.5 us (row 25.0 vs SGLang 24.9, SoL 14.6), and unpooler m=128 runs nvjet 128x64 at 20.3 us (row 20.0, SoL 14.6). Both are about 1.0x SGLang and 1.4-1.7x off the floor. An NN operand layout makes cuBLAS take a different heuristic entry, so it picks a different nvjet tile. The probe's result.json gives the paired per-shape ratios vs SGLang: pooler m=128 1.092x (22.86 vs 24.96 us, err 54 ns) and unpooler m=128 1.051x (19.15 vs 20.13, err 45 ns). Note that the brief quotes 1.120x/1.043x against master rows, not the paired ratio. Expected directions are consistent with TASK weights: verify_16 is one load of six, so the headline effect is about +0.3-0.6% and judging on the two rows is correct. WEAKEST LINK: the gain rests on one run, and the brief's own LESSONS cite cross-run nvjet swings (pooler m=128 23.9 vs 25.7 us). That swing concerns the absolute us. The paired ratio vs SGLang (same run, same GPU, alternating replays) largely cancels it, so read the paired speedup of the two rows, not the change in absolute us. A second, smaller point: the probe's pooler verify_32 m=256 was also 1.034x paired, which is not 'nothing', and the gate leaves it on TN. That is fine for attribution, but note it as a follow-up.

### Confounders on the current kernel
- None of the landed hypotheses touch m=65..128 for pooler/unpooler. sglang-gemm-dispatch-parity only made the TN incumbent the same cuBLAS call SGLang makes, which is why the paired ratio is the clean read. The swap-AB/Triton paths (bf16-shape-gated-triton-winners-combo, bf16-m32-swapab-wgmma-triton, the restack) gate at m<=32 and are not on these rows. The router changes are a different layer.
- Not landed, but in flight on the same rows: bf16-verify-deepgemm-bf16-gemm-nt-route also routes pooler/unpooler at m=65..256 just before the cuBLAS fallback. If both are ever stacked, branch order decides which runs, so give each its own gate constant (LESSONS 2026-10-25 stacking note).

### Interference and A/B needs
No companion. A gain or a loss on the two m=128 rows is unambiguous: the only change on those rows is TN to NN on the same cuBLAS, measured paired against SGLang's TN in the same run, and nothing landed shares the path. A control would just be master. If the paired ratio comes back within ~2% of 1.0, the probe's win did not replicate on the current image/run, so close it as neutral rather than spending a companion.

### Open questions
- Record the NN nvjet kernel names and us for both m=128 rows in the attempt file. If they match the probe's kernels, the gain is a deterministic heuristic pick, and a repeat is less necessary.
- Pooler m=256 was 1.034x paired in the probe. If this lands, a follow-up should widen the pooler gate to 129..256 (unpooler m=256 was 1.012x, about noise; qkv_gate m=256 lost at 0.941x, so keep it excluded).
- Do the m=130 and m=1000 unpooler tests and any 65..127 test m stay on paths whose correctness was already exercised? NN at m in 65..127 is new for unpooler/pooler. The judge's checks cover correctness, but list the test m values in the gate's range in the attempt file.


## Implementation notes

Restack of bf16-m128-nn-weight-layout-probe: the staged 3-way-applied patch applied the NN path to all four bf16 layers at m >= 128. Narrowed in `muse_bf16_proj/submission.py` (worktree) only:
- Constants `NN_GATED_LAYERS = ("pooler", "unpooler")`, `NN_GATED_MIN_M = 65`, `NN_GATED_MAX_M = 128` replace the probe's `NN_LAYERS`/`NN_MIN_M` (own gate, so it stacks cleanly with bf16-verify-deepgemm-bf16-gemm-nt-route).
- `prepare()`: lossless `state["pooler_t"]` and `state["unpooler_t"]` = `W.t().contiguous()` ([K, N] bf16); qkv_gate and o copies from the probe removed (no +memory for them).
- `kernel_fn`: inside the `layer != "router"` fallback, before master's cuBLAS call: if layer in the gated set and 65 <= cfg m <= 128 and cfg n == transposed weight's N, return `torch.mm(x, state[layer + "_t"])`. Gate reads only layer and cfg m/n.
- Module docstring updated. Everything else equals master: m <= 32 Triton paths, router, qkv_gate and o at all m, pooler/unpooler at m >= 129 (TN).
- Deviation: none. Note the earlier Triton routes (pooler/unpooler shape gates, swap-AB) still take precedence where they apply, as on master.
- Not run (no GPU). For the attempt file: record NN nvjet kernel names/us for the two m=128 rows and list the test m values in 65..128.

## Rules check

- [x] CUDA-graph capturable: diff adds only a cfg-gated torch.mm; no .item()/.cpu()/sync, no input-value control flow.
- [x] Nothing derived from `data` is cached across calls: the only new state is weight transposes made in prepare().
- [x] Weights: repacked losslessly in prepare() only; BF16 stays BF16 (transpose copy, same values/dtype).
- [x] Activations BF16 where SGLang keeps BF16: x passed unchanged to torch.mm, bf16 output with fp32 accumulation inside cuBLAS.
- [x] n/a: attention not touched.
- [x] No timing patches, harness edits, reference import, subprocesses or network in the diff.
- [x] All work happens inside the timed call; no output caching or replay.

okc_lint output:
```
CLEAN: muse_bf16_proj/submission.py
```
py_compile: OK; __pycache__ removed.
