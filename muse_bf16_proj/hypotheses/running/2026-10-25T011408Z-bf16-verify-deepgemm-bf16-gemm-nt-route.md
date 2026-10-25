---
problem: muse_bf16_proj
slug: bf16-verify-deepgemm-bf16-gemm-nt-route
created: 2026-10-25T011408Z
base: master
area: bf16 GEMM dispatch (m=65..256, DeepGEMM bf16)
change: >
  Built on master (blob a8bb42a). At m=128/256 all four bf16 layers still run cuBLAS nvjet and sit
  at ~1.0x SGLang (master profile 1602a4e947e64090): qkv_gate m=128 nvjet 96x64 splitK 13.0 +
  cublasLt splitKreduce 3.0 us (two nodes, row 16.81, floor 8.4), m=256 nvjet 96x64 19.4 (row
  19.62, 498 TFLOP/s, floor 9.8); pooler m=128 25.7 (row 25.55, floor 14.6), m=256 29.5 (583
  TFLOP/s, floor 17.4); o m=128 5.8, m=256 8.3; unpooler m=128 20.4, m=256 24.6 (~700 TFLOP/s).
  Our own kernels have lost here three ways (Triton 128x128 split-K, Triton TMA+warp-spec, CUTLASS
  stream-K: qkv m=256 28.5 us, every m=256 row worse), and prepare-time algo picking is a dead
  knob. Untried: DeepGEMM, which the image ships (its include dir was found by the stream-K build:
  deep_gemm/include with CUTLASS 4.2.1) and whose recent releases have dense BF16 GEMMs
  (deep_gemm.bf16_gemm_nt and friends) with analytic block/multicast/SM-count heuristics tuned for
  exactly these skinny-M, deep-K Hopper shapes, no timing search. Change: at import, try `import
  deep_gemm`; print deep_gemm.__version__ (or __file__) and whether hasattr(deep_gemm,
  'bf16_gemm_nt'); if the module or the function is missing print '[deepgemm] unavailable, master
  kept' and change nothing. In prepare(), for each of qkv_gate, o, pooler, unpooler and m in {128,
  256}: allocate out = torch.empty(m, n, bf16), call deep_gemm.bf16_gemm_nt(x, W, out) on a
  synthetic bf16 x (this also triggers DeepGEMM's JIT compile, untimed; print the seconds),
  torch.cuda.synchronize(), then self-check against x.double() @ W.double().T: rel L2 <= 3e-3, a
  one-hot power-of-two probe row within 1e-4, finite. Wrap each layer in try/except; a failure
  prints '[deepgemm] <layer> DISABLED: <reason>' and that layer keeps master. Also run m=130 for
  unpooler and m=200 for each layer through the self-check so the 65..256 route is covered. Record
  state['dg_ok'][layer]. In kernel_fn add one branch just before master's cuBLAS fallback: if
  state['dg_ok'][layer] and 65 <= cfg m <= 256 and cfg n is the full width and x is contiguous
  bf16 [m, k], out = torch.empty(m, n, bf16); deep_gemm.bf16_gemm_nt(x, W, out); return out. If
  the signature differs in the installed version (e.g. it takes (a, b, d, c=None) or tuples),
  adapt to it and print the signature used. Weights are read in place (W [n, k] row-major is
  DeepGEMM's NT 'b'), no repack, no precision change. m <= 32 paths, the router, and the
  m=512/1000 tests are unchanged. Print '[deepgemm] PATH CHOSEN <layer>: deepgemm|master'. At
  harvest, read the PATH CHOSEN lines and confirm a deep_gemm kernel in top_kernels on the
  m=128/256 rows before reading any number. Check DeepGEMM's launch is CUDA-graph capturable (no
  host sync, no allocation inside it once JIT-compiled): run one torch.cuda.graph capture + replay
  per layer in prepare and disable the layer if capture raises.
expected:
  - shapes: layer=qkv_gate load=verify_16 m=128
    direction: faster
    why: one node instead of nvjet splitK 13.0 + splitKreduce 3.0 us; row 16.81 vs floor 8.4
  - shapes: layer=qkv_gate load=verify_32 m=256
    direction: faster
    why: nvjet 19.4 us at 498 TFLOP/s (36-72 tiles on 132 SMs); floor 9.8
  - shapes: layer=pooler load=verify_*
    direction: faster
    why: m=128 nvjet 25.7 at ~2.6 TB/s (floor 14.6); m=256 29.5 at 583 TFLOP/s (floor 17.4)
  - shapes: layer=o load=verify_*
    direction: flat
    why: o is short-K (1024); a gain is possible at m=256 (8.3 us vs 4.5 floor) but not expected
  - shapes: layer=unpooler load=verify_*
    direction: flat
    why: m=256 nvjet already ~700 TFLOP/s; m=128 20.4 vs 14.6 floor may move
  - shapes: layer=* load=decode_*
    direction: flat
    why: m <= 32 untouched
  - shapes: layer=* load=verify_1 m=8
    direction: flat
    why: m <= 32 untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T011846Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T011846Z
base_sha: 9bb477f684086cd51b09c1e8a08fa8d50ae04e59
base_blob: a8bb42ac483318e9fbc84bc7826a27b7a704d1fb
branch: hyp/muse_bf16_proj/bf16-verify-deepgemm-bf16-gemm-nt-route
worktree: .worktrees/muse_bf16_proj/bf16-verify-deepgemm-bf16-gemm-nt-route
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T012031Z
impl_commit: ba330227271a61b5b66b2561ea2072550c888afe
submission_sha256: 0fb340d50b20a772f3b4d35da138746ed8a68dfa6132fa6be2da5fa424b42d9f
norm_sha256: ee29894b95224268b60cded8c47c7166f746b89f606ac3de3882987fb074fc1b
submission_blob: 94b37346a8f849f5cc5a87d48977b9ff1b4f513b
lint_warnings: 3
attempt_file: 2026-10-25T012031Z-check-bf16-verify-deepgemm-bf16-gemm-nt-route.md
prepared_at: 2026-10-25T012031Z
job: 2d522d47ca5c4fc9
dispatched_at: 2026-10-25T012103Z
---

# bf16-verify-deepgemm-bf16-gemm-nt-route

## Change

Built on master (blob a8bb42a). At m=128/256 all four bf16 layers still run cuBLAS nvjet and sit at ~1.0x SGLang (master profile 1602a4e947e64090): qkv_gate m=128 nvjet 96x64 splitK 13.0 + cublasLt splitKreduce 3.0 us (two nodes, row 16.81, floor 8.4), m=256 nvjet 96x64 19.4 (row 19.62, 498 TFLOP/s, floor 9.8); pooler m=128 25.7 (row 25.55, floor 14.6), m=256 29.5 (583 TFLOP/s, floor 17.4); o m=128 5.8, m=256 8.3; unpooler m=128 20.4, m=256 24.6 (~700 TFLOP/s). Our own kernels have lost here three ways (Triton 128x128 split-K, Triton TMA+warp-spec, CUTLASS stream-K: qkv m=256 28.5 us, every m=256 row worse), and prepare-time algo picking is a dead knob. Untried: DeepGEMM, which the image ships (its include dir was found by the stream-K build: deep_gemm/include with CUTLASS 4.2.1) and whose recent releases have dense BF16 GEMMs (deep_gemm.bf16_gemm_nt and friends) with analytic block/multicast/SM-count heuristics tuned for exactly these skinny-M, deep-K Hopper shapes, no timing search. Change: at import, try `import deep_gemm`; print deep_gemm.__version__ (or __file__) and whether hasattr(deep_gemm, 'bf16_gemm_nt'); if the module or the function is missing print '[deepgemm] unavailable, master kept' and change nothing. In prepare(), for each of qkv_gate, o, pooler, unpooler and m in {128, 256}: allocate out = torch.empty(m, n, bf16), call deep_gemm.bf16_gemm_nt(x, W, out) on a synthetic bf16 x (this also triggers DeepGEMM's JIT compile, untimed; print the seconds), torch.cuda.synchronize(), then self-check against x.double() @ W.double().T: rel L2 <= 3e-3, a one-hot power-of-two probe row within 1e-4, finite. Wrap each layer in try/except; a failure prints '[deepgemm] <layer> DISABLED: <reason>' and that layer keeps master. Also run m=130 for unpooler and m=200 for each layer through the self-check so the 65..256 route is covered. Record state['dg_ok'][layer]. In kernel_fn add one branch just before master's cuBLAS fallback: if state['dg_ok'][layer] and 65 <= cfg m <= 256 and cfg n is the full width and x is contiguous bf16 [m, k], out = torch.empty(m, n, bf16); deep_gemm.bf16_gemm_nt(x, W, out); return out. If the signature differs in the installed version (e.g. it takes (a, b, d, c=None) or tuples), adapt to it and print the signature used. Weights are read in place (W [n, k] row-major is DeepGEMM's NT 'b'), no repack, no precision change. m <= 32 paths, the router, and the m=512/1000 tests are unchanged. Print '[deepgemm] PATH CHOSEN <layer>: deepgemm|master'. At harvest, read the PATH CHOSEN lines and confirm a deep_gemm kernel in top_kernels on the m=128/256 rows before reading any number. Check DeepGEMM's launch is CUDA-graph capturable (no host sync, no allocation inside it once JIT-compiled): run one torch.cuda.graph capture + replay per layer in prepare and disable the layer if capture raises.

## Expected

- `layer=qkv_gate load=verify_16 m=128` faster: one node instead of nvjet splitK 13.0 + splitKreduce 3.0 us; row 16.81 vs floor 8.4
- `layer=qkv_gate load=verify_32 m=256` faster: nvjet 19.4 us at 498 TFLOP/s (36-72 tiles on 132 SMs); floor 9.8
- `layer=pooler load=verify_*` faster: m=128 nvjet 25.7 at ~2.6 TB/s (floor 14.6); m=256 29.5 at 583 TFLOP/s (floor 17.4)
- `layer=o load=verify_*` flat: o is short-K (1024); a gain is possible at m=256 (8.3 us vs 4.5 floor) but not expected
- `layer=unpooler load=verify_*` flat: m=256 nvjet already ~700 TFLOP/s; m=128 20.4 vs 14.6 floor may move
- `layer=* load=decode_*` flat: m <= 32 untouched
- `layer=* load=verify_1 m=8` flat: m <= 32 untouched
- `layer=router *` flat: untouched

## Risk

Medium. (a) The installed DeepGEMM may predate the BF16 GEMMs (then the run measures master: read the [deepgemm] lines first). (b) Its JIT needs nvcc and a writable cache dir at prepare time; load_inline builds have run on the judge, so nvcc exists, but print the JIT seconds against the 10-minute budget. (c) DeepGEMM's heuristics favour large M; at m=128/256 with M on the token side it may choose block_m 64/128 and lose like CUTLASS did; per-shape rows decide, gate per layer at harvest. (d) If it allocates or syncs inside the call, graph capture fails in prepare and the layer is disabled. Rules: an in-image library (rules list DeepGEMM explicitly), no timing-based selection.

## Adversarial review

Reviewed 2026-10-25T011846Z by an adversarial subagent (model opus).

**Verdict: proceed.** DeepGEMM is present in the judge image: the stream-K run's log (archive f5b1361aaa0e48b5.log) found it via importlib find_spec at /opt/sglang/lib/python3.12/site-packages/deep_gemm with bundled CUTLASS 4.2.1, and the technical rules name DeepGEMM. Whether that version exports bf16_gemm_nt is unverified, but the change already prints a loud availability line and PATH CHOSEN per layer, so a no-op run cannot be misread. Proceed. Pre-warm every m in the gate (or narrow the gate) so no JIT compile can happen inside a timed or captured call.

### Justification
The mechanism is plausible but overclaimed. At m=128/256, master runs cuBLAS nvjet at about 1.0x SGLang and 1.4-2x off the floor: qkv_gate m=128 is splitK 12.6 + splitKreduce 3.1 us (row 17.0, SoL 8.4); m=256 is 19.5 (SoL 9.8); pooler m=128 is 24.5 (SoL 14.6) and m=256 is 30.7 (SoL 17.4). Three in-house attempts lost here (Triton split-K, TMA+warp-spec, CUTLASS stream-K), so a vendor-tuned SM90 BF16 GEMM is the right next thing to try. WEAKEST LINK: the claim that DeepGEMM's heuristics are 'tuned for exactly these skinny-M, deep-K shapes' is not supported. DeepGEMM's normal (non-grouped) GEMMs have no split-K, and its block/multicast heuristic is designed around large-M training and prefill. At m=128 with n=2304 (qkv_gate), one m-block gives only 2304/block_n output tiles (18-72), and with no split-K each tile walks the whole K=8192, which is the same occupancy problem nvjet solved with splitK. A loss at qkv_gate m=128 is therefore as likely as a gain. pooler/unpooler (n=4096/8192) have more tiles and are the likelier winners. Directions in `expected` are reasonable apart from qkv_gate m=128, which should read 'uncertain'. The headline effect is bounded by verify_16/verify_32 being two loads of six.

### Confounders on the current kernel
- None of the landed hypotheses touch m=65..256 on the four bf16 layers. The swap-AB/Triton paths gate at m<=32. sglang-gemm-dispatch-parity made master's incumbent the same TN cuBLAS call SGLang makes, so the per-shape paired ratio vs SGLang measures DeepGEMM vs nvjet directly.
- In flight, not landed: pooler-unpooler-m128-nn-layout-gated routes pooler/unpooler at 65..128 just before the same cuBLAS fallback. The two hypotheses must not be stacked without separate gate constants and an explicit branch order. Each is measured against master, so neither confounds the other's single run.

### Interference and A/B needs
No companion. Each layer×m row is either DeepGEMM (confirmed by PATH CHOSEN plus a deep_gemm kernel in top_kernels) or master, and is measured paired against SGLang's identical cuBLAS. A gain or a loss per row is attributable. If [deepgemm] prints 'unavailable, master kept' or every layer is DISABLED, churn must classify the run as VOID (measured master), not neutral, and must not record it as a DeepGEMM result. Hold the line only after that happens, not before.

### Open questions
- Does the installed deep_gemm export bf16_gemm_nt (DeepGEMM added dense BF16 GEMMs in its 2.x line; the bundled CUTLASS 4.2.1 suggests a late-2025 build, but sglang may pin an older fork)? The first [deepgemm] line answers it. If absent, consider whether any other in-image BF16 path (e.g. a FlashInfer or sgl_kernel bf16 GEMM) is worth the same scaffold.
- Is the minimum n alignment satisfied for qkv_gate n=2304 at DeepGEMM's chosen block_n? The self-check and try/except cover it, but print the chosen block config per layer×m so a loss can be traced to the tile choice.
- Use per-layer harvest gating: promote only the layer×m rows whose paired ratio beats master by more than the CI. Expect pooler/unpooler to be the candidates and qkv_gate m=128 to be at risk.


## Implementation notes

Edited only `muse_bf16_proj/submission.py` in the worktree (+157 lines, nothing removed).
- Module-level guarded `import deep_gemm`; `_dg_find_fn()` prints module path, version and every bf16/gemm candidate name, then picks `bf16_gemm_nt` (fallbacks `gemm_bf16_nt`, `bf16_gemm_nt_contiguous`). A loud `[deepgemm] AVAILABLE` or `[deepgemm] unavailable, master kept` line results. Signature is printed via inspect and the call is `fn(x, W, out)`.
- `prepare()` (last step, `_dg_prepare`): for pooler, qkv_gate, unpooler, o and EVERY routed m in (128, 130, 200, 256): eager call (JIT), fp32 matmul self-check (rel L2 <= 3e-3), one-hot probe (<= 1e-4), then torch.cuda.graph capture + replay checked again. Per (layer, m) failure prints `DISABLED` and only that m is dropped. Elapsed time is tracked from prepare() start; past 200 s remaining (layer, m) are skipped (gate off). Prints `PATH CHOSEN <layer> m=<m>: deepgemm|master`.
- `kernel_fn`: one branch just before master's cuBLAS fallback, gated on the exact verified m set (not 65..256, per the review), cfg n == full weight rows, contiguous bf16 x. Router, m<=32 paths and m=512/1000 are unchanged.
- Deviation: gate is the warmed set {128,130,200,256}, not the whole 65..256, so no JIT can happen in a timed or captured call. Chosen tile config is not introspectable from here; not printed. Nothing could be run on this node (no GPU, no deep_gemm).

## Rules check

- [x] CUDA-graph capturable: branch has only torch.empty + the DeepGEMM launch, no sync/.item()/value-dependent control flow; capture tested per (layer, m) in prepare. Syncs are prepare-only.
- [x] Nothing derived from `data` is cached: state holds only weight-derived/verified-m sets built from synthetic x.
- [x] Weights: read in place, no repack; BF16 stays BF16.
- [x] Activations BF16, output BF16; no precision change.
- [x] Attention: n/a, dense projections only.
- [x] No timing patches, harness access, subprocess or network added by me (DeepGEMM's own JIT is a library compile; no timing-based selection).
- [x] All work inside the timed call; no output caching or replay.

okc_lint output:
```
warn  host-sync    L713: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L736: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L741: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: submission.py
```
py_compile OK; __pycache__ removed.
