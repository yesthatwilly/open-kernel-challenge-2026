---
problem: muse_attention_decode
slug: full-decode1-sk-pdl-from-preattn
created: 2026-10-25T042335Z
base: master
area: launch overhead (full decode_1, pre-attention -> stream-K kernel)
change: >
  Built on master (stream-K two-level ticket at full decode B=1, landed c3c1b01). Full decode_1 is
  22.6 us shape vs pre-attn 1.4 + sk kernel 18.6 us of kernels, so ~2.6 us is launch gap plus the
  serial ring fill. At batch 1 only (config): launch splitkv_attn_sk_kernel with
  cudaLaunchKernelEx and cudaLaunchAttributeProgrammaticStreamSerialization = 1 after master
  Triton _pre_attention_kernel (which needs no change: without an explicit trigger the dependent
  may launch as the primary drains). In the kernel, before cudaGridDependencySynchronize()
  (griddepcontrol.wait) do everything that does not read pre-attention outputs: smem/ticket setup,
  the tile-range computation, page_table index loads, and cp.async issue of the prologue ring
  stages for tiles whose keys are all < seq_b - 1 (old keys; the pre-attention kernel writes only
  key seq_b - 1 at cache_loc[b]). After the wait: load the FP8 q and issue any tile containing key
  seq_b - 1 (only the last tile of the request, normally the final CTA). Workspace and ticket
  writes stay after the wait. Batch > 1 unchanged (the running
  full-decode-splitkv-s3-fused-preattn targets B>1 pre-attention). Self-check: the landed B=1
  cases plus two graph replays with PDL on; fallback to plain launch on any error (PATH CHOSEN:
  full decode_1 two-level + PDL / no PDL (reason)). Print a prepare CUDA-event time of
  pre-attn+kernel with and without PDL and assert PDL is not slower.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: overlaps the sk kernel launch, index loads and ring fill with the 1.4 us pre-attention kernel; shape 22.6 -> ~20.5 us (-1 to -2.5 us)
  - shapes: "*layer=full load=decode_16* *layer=full load=decode_32* *layer=sliding* *load=verify_*"
    direction: flat
    why: outside the batch-1 gate
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T043143Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T043143Z
base_sha: df59fe97803906261a926deeb694b808107ab115
base_blob: d8769ef25caf845a0e207538ccc5c62394b6ee3b
branch: hyp/muse_attention_decode/full-decode1-sk-pdl-from-preattn
worktree: .worktrees/muse_attention_decode/full-decode1-sk-pdl-from-preattn
mode: null
restack_of: null
stale_base: null
implemented_at: 2026-10-25T043825Z
impl_commit: d4a9f927d2040769856e3b4c95ad0a76644d58ce
submission_sha256: 540e2541d3af9d26c354b2bc2c562bf6db9d00c87a5ca22f9f10c3165b096f77
norm_sha256: 1b64ad9ae82a4f1b4883a816c4c4ef5555fcb06e8136faa09903742c59489842
submission_blob: c91fe482d3d19c0c9785cc932c7db5732076adbf
lint_warnings: 18
---

# full-decode1-sk-pdl-from-preattn

## Change

Built on master (stream-K two-level ticket at full decode B=1, landed c3c1b01). Full decode_1 is 22.6 us shape vs pre-attn 1.4 + sk kernel 18.6 us of kernels, so ~2.6 us is launch gap plus the serial ring fill. At batch 1 only (config): launch splitkv_attn_sk_kernel with cudaLaunchKernelEx and cudaLaunchAttributeProgrammaticStreamSerialization = 1 after master Triton _pre_attention_kernel (which needs no change: without an explicit trigger the dependent may launch as the primary drains). In the kernel, before cudaGridDependencySynchronize() (griddepcontrol.wait) do everything that does not read pre-attention outputs: smem/ticket setup, the tile-range computation, page_table index loads, and cp.async issue of the prologue ring stages for tiles whose keys are all < seq_b - 1 (old keys; the pre-attention kernel writes only key seq_b - 1 at cache_loc[b]). After the wait: load the FP8 q and issue any tile containing key seq_b - 1 (only the last tile of the request, normally the final CTA). Workspace and ticket writes stay after the wait. Batch > 1 unchanged (the running full-decode-splitkv-s3-fused-preattn targets B>1 pre-attention). Self-check: the landed B=1 cases plus two graph replays with PDL on; fallback to plain launch on any error (PATH CHOSEN: full decode_1 two-level + PDL / no PDL (reason)). Print a prepare CUDA-event time of pre-attn+kernel with and without PDL and assert PDL is not slower.

## Expected

- `*layer=full load=decode_1 *` faster: overlaps the sk kernel launch, index loads and ring fill with the 1.4 us pre-attention kernel; shape 22.6 -> ~20.5 us (-1 to -2.5 us)
- `*layer=full load=decode_16* *layer=full load=decode_32* *layer=sliding* *load=verify_*` flat: outside the batch-1 gate

## Risk

Kill: decode_1 full shape not >= 1 us faster. splitkv-pdl-attn-to-combine (A->B) was neutral; this boundary differs because the dependent here can do useful gather work before the wait.

## Adversarial review

Reviewed 2026-10-25T043143Z by an adversarial subagent (model opus).

**Verdict: proceed.** PDL from the Triton pre-attention kernel into the B=1 stream-K kernel is rules-safe: only key seq_b-1 is written by the primary, and the hypothesis defers that tile plus q until after griddepcontrol.wait. With no explicit trigger in the primary, though, the overlap is limited to the launch gap and roughly meets the 1 us kill bar. Time an explicit early-trigger variant too, and split the q load out of the first tile's cp.async group.

### Justification
Mechanism: full decode_1 is 22.6 us of shape against 1.4 + 18.6 us of kernels, leaving ~2.6 us of launch and dependency gap. The sk kernel's prologue is a serial chain: load_prefix(seqlens) -> issue_meta (page_table) -> cp.async of K/V tiles, two dependent DRAM round trips (~1-1.5 us) before the first byte of KV arrives. Running that chain under pre-attention is a real saving. WEAKEST LINK: without griddepcontrol.launch_dependents in the primary, the dependent is released only when every primary CTA has exited. The overlap is then the dependent's launch latency and the primary's tail/flush (~0.5-1 us), not the 1.4 us body. splitkv-pdl-attn-to-combine (neutral, same environment) is the precedent for a sub-noise PDL gain. An explicit trigger at the top of _pre_attention_kernel (Triton gdc_launch_dependents if this Triton has it, else tl.inline_asm_elementwise('griddepcontrol.launch_dependents;')) lets the index chain overlap the body. It is a no-op for any dependent launched without the PDL attribute, so every other shape is unaffected.

### Confounders on the current kernel
- full-decode1-streamk-s3-two-level-ticket (landed c3c1b01): the dependent is its variant-3 kernel at grid SM. The change touches only the B=1 launch, so read the full decode_1 row and the sk kernel row. The kernel row should not change; the shape row should drop.
- fused-preattention-triton (landed): this is the primary. If an explicit trigger is added, the kernel changes for every shape (benign no-op). Check that sliding decode rows (which do not run it) and the other full rows stay flat.
- In flight: full-decode1-sk4-cluster-dsmem-merge (queued) changes the same B=1 launch to a cluster launch. Run against master each is attributable on its own; a later stack needs both attributes on one cudaLaunchKernelEx. full-decode-splitkv-s3-fused-preattn (running) removes pre-attn at B>1 only, so it does not overlap.

### Interference and A/B needs
No companion. The prepare prints pre-attn+kernel with and without PDL (and with the explicit-trigger variant, see open_questions) on the same input. PATH CHOSEN names the variant, so a gain or flat on full decode_1 is attributable. A separate control check would only repeat the prepare.

### Open questions
- BUILDER: add the variant 'PDL + explicit griddepcontrol.launch_dependents at the top of _pre_attention_kernel' to the prepare timing, alongside no-PDL and PDL-implicit. Print all three and choose the fastest that is not slower than no-PDL.
- BUILDER: report the shape-row delta against the 1 us kill bar from the profile. If only the implicit variant is available and the delta is < 1 us, record it as a stacking item for sk4 rather than a dead end.

## Implementation notes

Edited only `muse_attention_decode/submission.py` in the worktree (194 insertions, 20 deletions). No GPU on this node: nothing was compiled or run, only `py_compile` and `okc_lint`.

- Kernel: new template flag `PDL` on `splitkv_attn_sk_kernel` (variant 4 = `<4, ticket, two-level, PDL>`; variants 0-3 compile to the same code as before). Launched via `cudaLaunchKernelEx` with `cudaLaunchAttributeProgrammaticStreamSerialization = 1`. Before `griddepcontrol.wait` the kernel does only `load_prefix` (cache_seqlens), the tile-range math, `issue_meta` (page_table) and the cp.async of ring tiles that are not the last tile of a request. The primary writes only q and K/V row seq_b-1, which lives in the request's last tile.
- Review fix applied (the review wins over the spec): q is no longer in the first tile's cp.async group. `issue_tile(nn, withq)`: the PDL prologue passes `withq=false`; after the wait, threads `tid < 64` copy the prologue segment heads' q with plain 16 B loads into the q slots, published by the `__syncthreads` at the top of the main loop. Main-loop tiles (after the wait) keep the cp.async q. The wait sits just before the first prologue tile that is last-of-request (found with `find_req` on smem values, so it is CTA-uniform), else right after the prologue; tiles after it are issued after the wait. Workspace, ticket, gticket and output writes are all later than the wait.
- Variant (b): `_pre_attention_kernel` gets `TRIGGER: tl.constexpr = False`; when True it runs `tl.inline_asm_elementwise("{ griddepcontrol.launch_dependents; mov.u32 $0, $1; }", ...)` (is_pure=False) on `tl.arange` at the top and uses the result as `i`, so the asm is kept and ordered first. `kernel_fn` passes `TRIGGER=True` only when `splitkv and batch == 1 and _B1_PDL == 2`; every other shape compiles the TRIGGER=False specialization, identical to before.
- Selection (`_sk_pdl_select`, import time, after the B=1 validation): variant 4 attributes (cubin 90); 7 cases x trigger {off, on} with the two kernels back to back and no sync (B=1 80000 / 4097 / 64 / 63 / 1 keys, grids SM / 7 / 3, plus B=16 ragged); then `kernel_fn` captured into three CUDA graphs (mode 0 landed plain launch, 1 PDL attribute only, 2 PDL + trigger), each replayed twice with NaN-poisoned output against the FP32 reference; then 40 interleaved cold-L2 (256 MB flush) graph replays per mode at synthetic B=1 x 80000. The fastest mode is chosen only if its median beats plain by more than 0.05 us; otherwise, and on any error, mode 0 (landed). Prints per-mode median/p10 and `PATH CHOSEN: full decode_1 streamk two-level + PDL ...` or `... no PDL (...)`, plus the final `... launch plain / (a) / (b)` line.
- Deviation: the prepare timing is graph-replay of pre-attention + kernel (as the brief asks), not eager CUDA events. The hypothesis asked to "assert PDL is not slower": implemented as a selector (never ships a PDL mode that is not faster), not a hard assert.
- Not guarded by `__CUDA_ARCH_FEAT_SM90_ALL`: `griddepcontrol` is plain sm_90 PTX, and a guard with an `#else` that skipped the wait would be a hazard. The module still builds with `TORCH_CUDA_ARCH_LIST=9.0a` (existing `_build_sk`).
- Biggest risks: (1) first compile of the Triton `inline_asm_elementwise` with a no-op `griddepcontrol.launch_dependents` (falls back to mode 0/1 only via the exception handler if compile raises at warm-up; the selector catches that and returns 0); (2) PDL edges captured in a CUDA graph need CUDA >= 12.3, else capture raises and the landed path is kept; (3) the gain may be below the 1 us kill bar (see open questions: record as a stacking item for sk4 if only the implicit variant helps).

## Rules check

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()` in kernel_fn; the diff adds a constexpr kwarg and a variant index in kernel_fn/_full_decode_sk only; syncs are in prepare-time `_sk_pdl_select`.
- [x] Nothing derived from `data` is cached across calls (`_B1_PDL` is a global chosen from synthetic data at import; tickets are device-only as before).
- [x] Weights: n/a: attention problem, no weights touched.
- [x] Activations BF16 / KV cache FP8 / queries at most FP8 E4M3: unchanged (the q is the same FP8 query).
- [x] Attention sees every key (full context): unchanged; the deferred last tile is still read in full after the wait.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocess: checked the diff; none. No child process anywhere.
- [x] All work happens inside the timed call; no output caching or replay: the PDL launch is still part of kernel_fn; import-time work is self-check and timing selection on synthetic data only.
- [x] sm_90a: n/a for arch guard (griddepcontrol is sm_90); build env unchanged (TORCH_CUDA_ARCH_LIST=9.0a around load_inline, restored).
- [x] Failure handling: any build error, self-check miss, graph capture error or raise lands on mode 0 (landed path) with a PATH CHOSEN line; never raises.

okc_lint:
```
warn  host-sync lines (18, all `torch.cuda.synchronize()` / `.item()` in prepare-time self-checks and diagnostics, "outside kernel_fn's call graph; fine in prepare()")
OK with warnings: submission.py
```

## Pre-flight audit

Audited 2026-10-24 by an opus subagent (read-only reasoning; no GPU on this node). **Verdict: dispatch-with-fixes.**

- PDL hazard: OK. In the judge's generate_input (page 1) cache_loc = page_table[b, seq_lens] and cache_seqlens = seq_lens + 1, so the primary (`_pre_attention_kernel`, unpaired at B=1) writes only q and K/V row of key seq_b-1. Before `griddepcontrol.wait` the kernel reads only cache_seqlens, page_table, k_scale/v_scale (read-only in the primary) and cp.asyncs tiles that are not the last of a request (keys < (ntiles-1)*64 <= seq_b-1). The wait point is CTA-uniform (smem values); early-exit CTAs never read primary output. q is read after the wait and published by the `__syncthreads` at the top of the main loop; workspace, ticket and output writes all follow the wait. No other stream op sits between the two launches (only torch.empty).
- Trigger semantics: OK. `griddepcontrol.launch_dependents` only allows the dependent to be scheduled; `griddepcontrol.wait` still blocks until the primary grid has completed and its memory is visible. TRIGGER is a constexpr set from config + the import-time global, compiled before capture.
- Selector: kernel_fn reads only cfg and the module global `_B1_PDL` (set at import, reset in `finally`); no per-call state, prints or try/except in kernel_fn. Tickets are re-zeroed in the import `finally` after the selector and diagnostics. Nothing input-dependent is computed on the host.
- 0.05 us margin on 40 cold-L2 medians is below the noise, but all three modes are correct and PDL is not slower in principle, so a wrong pick costs about noise. Acceptable.
- Fix: the post-wait prologue q copy used a plain deref of a `const __restrict__` pointer, which nvcc may emit as `ld.global.nc` (non-coherent path) for data the primary has just written. Changed it to `__ldcg` (L2, coherent with the primary's flushed stores), as the cp.async.cg q loads in the main loop already are. Same cost. The self-check replays one fixed input twice, so it would not catch stale q; the judge's refilled-input repetitions would.
- okc_lint OK with warnings (prepare-time syncs only); py_compile OK.

Residual risk: Triton `inline_asm_elementwise` compile, or capture of a PDL edge (CUDA < 12.3), can fail; both land on mode 0 with a PATH CHOSEN line, and the gain may still miss the 1 us kill bar.
