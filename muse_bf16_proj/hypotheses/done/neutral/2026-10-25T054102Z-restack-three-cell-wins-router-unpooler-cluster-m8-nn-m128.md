---
problem: muse_bf16_proj
slug: restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128
created: 2026-10-25T054102Z
base: master
area: "shape-gated restack: router GEMM m<=8 + unpooler-slice GEMM m<=8 + pooler/unpooler GEMM m=65..128"
change: >
  Built on current master (b99b1dd, bf16-restack-m16-swapab-router-swapab-m32, check
  1e90292c7b954d96, 1.2215). Stack the three row wins that were each checked alone on b99b1dd,
  each gated to exactly the (layer, m) cells it won; nothing else changes and no constant is
  retuned. Patches live in muse_bf16_proj/hypotheses/done/neutral/. (1) Unpooler-slice cluster at
  m<=8: churn 3-way-applies restack-unpooler-slice-cluster-dsmem-m8.patch (the depends_on parent,
  job e557d92969864449) as is: ouns_kernel NT=8 via load_inline (ext name
  muse_bf16_unpool_slice_cluster_v3, TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, -std=c++20), self-check
  at m {1,5,8}, the cold-flush graph-replay gate (41 reps, >=1.02x) that sets state['ouns_cl'],
  and the kernel_fn route "layer == 'unpooler' and n == 1024 and 1 <= m <= 8 and state['ouns_cl']"
  placed before the swapab16 check. o is NOT routed. (2) Router cluster at m<=8: then apply
  restack-router-cluster-dsmem-m8.patch (job d323ce8ef6a54fde) with git apply -3 and resolve by
  hand: the router_kernel<8,8> source and _RC_EXT build (its own ext name, distinct from the
  unpooler one), _router_cluster_prepare(state) called after _router_prepare in prepare() with its
  self-check at m {1,5,8} setting state['router_cluster_ok'], and the kernel_fn router branch
  "state['router_cluster_ok'] and 1 <= m <= 8 and n == 256 -> cluster forward" inserted before
  path = _router_path(...); m 9..32 keeps swapab, m > 32 keeps hilo. (3) NN layout at m=65..128:
  then apply restack-pooler-unpooler-nn-layout-m128.patch (job 92cd4e758a924ca0): prepare stores
  state['pooler_t'] and state['unpooler_t'] = W.t().contiguous() (lossless [K, N] bf16 copies;
  originals stay), and kernel_fn, just before the "if layer != 'router'" cuBLAS fallback, returns
  torch.mm(x, state[layer + '_t']) when layer in {pooler, unpooler}, 65 <= m <= 128 and n is the
  full width; m >= 129 keeps TN, m <= 32 keeps the swap-AB paths. The three edits touch disjoint
  (layer, m) cells; the only expected conflicts are adjacent insertions in prepare() and
  kernel_fn. Keep all three, in the order unpooler-cluster check, router-cluster check, NN branch,
  then master's existing fallbacks. STATE RULE (the judge prepares two states, setup copies 2, and
  rotates them): every device buffer, flag and handle lives in the state dict built by prepare().
  Neither component patch holds device pointers in module or C++ static globals: the only C++
  statics are host-side "static bool done" one-shot cudaFuncSetAttribute flags, and _RC_EXT /
  _OUNS_EXT are compiled-module handles. Keep it that way: if the merge introduces any
  module-level or static global holding a device pointer, tensor, workspace or TMA descriptor (the
  "static CUtensorMap g_tb" pattern from the DeepGEMM rewrite), move it into state and pass it as
  a kernel argument. Gate and self-check scratch (the 256 MB flush buffer, synthetic x) is local
  to prepare and released before it returns. Hard rules: kernel_fn stays CUDA-graph capturable,
  with no host syncs, no allocation keyed on input values beyond torch.empty of the output, no
  data-dependent control flow, no prints, no try/except, no per-call mutable state and nothing
  cached from inputs; routing reads only shapes and constants fixed in prepare; bf16 output with
  fp32 accumulation as master; every key attended (n/a, dense GEMM); no child processes
  (load_inline in-process only). Self-check before dispatch: prepare two states, capture a graph
  of kernel_fn per state at m in {1, 8, 128} for router, unpooler and pooler, and replay them
  alternately (A, B, A, B) with x refilled in place each time, comparing every output to a
  torch.mm reference; on any miss the affected component's flag falls back to master's path. At
  harvest, confirm in the profile: router_kernel<8,8> on router decode_1/verify_1, ouns_kernel on
  unpooler decode_1/verify_1, nvjet *_NNT on pooler/unpooler verify_16, and master's
  _bf16_swapab_kernel / _router_swapab_kernel / hilo on every other row (the 03:41Z harvest found
  runs that were really master).
expected:
  - shapes: layer=router load=decode_1 m=1
    direction: faster
    why: "router cluster (d323ce8ef6a54fde): 5.38 -> 4.98 us (-7.5%, pr 1.078), memset node gone"
  - shapes: layer=router load=verify_1 m=8
    direction: faster
    why: "router cluster (d323ce8ef6a54fde): 5.47 -> 4.93 us (-9.8%, pr 1.099)"
  - shapes: layer=unpooler load=decode_1 m=1
    direction: faster
    why: "unpooler-slice cluster ouns_kernel (e557d92969864449): 6.15 -> 5.34 us (-13.2%, pr 1.136); master's 6.15 is a high reading (5.34-6.15 over four runs), so judge on kernel name plus pr"
  - shapes: layer=unpooler load=verify_1 m=8
    direction: faster
    why: "unpooler-slice cluster ouns_kernel (e557d92969864449): 6.00 -> 5.41 us (-9.8%, pr 1.124)"
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: "NN layout (92cd4e758a924ca0): nvjet 128x64 splitK NNT, 25.00 -> 22.59 us (-9.6%, pr 1.117)"
  - shapes: layer=unpooler load=verify_16 m=128
    direction: faster
    why: "NN layout (92cd4e758a924ca0): nvjet NNT, 19.96 -> 18.99 us (-4.9%, pr 1.047)"
  - shapes: layer=router load=decode_16 m=16
    direction: flat
    why: m 9..32 keeps master swapab
  - shapes: layer=router load=decode_32 m=32
    direction: flat
    why: keeps master swapab
  - shapes: layer=router load=verify_16 m=128
    direction: flat
    why: m>32 keeps hilo
  - shapes: layer=router load=verify_32 m=256
    direction: flat
    why: keeps hilo
  - shapes: layer=unpooler load=decode_16 m=16
    direction: flat
    why: full width, untouched
  - shapes: layer=unpooler load=decode_32 m=32
    direction: flat
    why: untouched
  - shapes: layer=pooler load=decode_*
    direction: flat
    why: m<=32 keeps swap-AB
  - shapes: layer=pooler load=verify_1 m=8
    direction: flat
    why: untouched
  - shapes: layer=*pooler load=verify_32 m=256
    direction: flat
    why: m>=129 keeps TN
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=o *
    direction: flat
    why: o not routed; +4-8% drift on o verify_16/decode_32 seen in all three solo runs on unchanged nvjet
risk: medium
depends_on: restack-unpooler-slice-cluster-dsmem-m8
priority: 1
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-25T054315Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T054315Z
base_sha: 84c6b217af909ca0534b49bdd5a8072e10b692c5
base_blob: 970b415b746fa03cb92e419d270be92588273178
branch: hyp/muse_bf16_proj/restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128
worktree: .worktrees/muse_bf16_proj/restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128
mode: check
restack_of: restack-unpooler-slice-cluster-dsmem-m8
stale_base: null
implemented_at: 2026-10-25T054957Z
impl_commit: 83a1fd1fa0f27bfd4412272e1200cdf490d81514
submission_sha256: 2992aaa93918145ccaf9fc61b4d483791a6ced5868af9fc95a60e7a7b198c014
norm_sha256: 106a7bd065a4fafa020632207718ac6db927671a1993641c51e86bc123a48c70
submission_blob: 353e3b19a7edd5cb879e202d78097f474bcce207
lint_warnings: 13
attempt_file: 2026-10-25T054957Z-check-restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128.md
prepared_at: 2026-10-25T054957Z
job: 84f17adc3f074a33
dispatched_at: 2026-10-25T055005Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.236629518166488
speedup_ci: [1.2180800753939907, 1.2551789609389852]
harvested_at: 2026-10-25T084157Z
best_at_harvest: 1.2215006063849365
promotion: merged
hand_landed: true
promotion_commit: 040fc7e18f4d9c17ea9fc4a4b9a4cae9216a2335
---

# restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128

## Change

Built on current master (b99b1dd, bf16-restack-m16-swapab-router-swapab-m32, check 1e90292c7b954d96, 1.2215). Stack the three row wins that were each checked alone on b99b1dd, each gated to exactly the (layer, m) cells it won; nothing else changes and no constant is retuned. Patches live in muse_bf16_proj/hypotheses/done/neutral/. (1) Unpooler-slice cluster at m<=8: churn 3-way-applies restack-unpooler-slice-cluster-dsmem-m8.patch (the depends_on parent, job e557d92969864449) as is: ouns_kernel NT=8 via load_inline (ext name muse_bf16_unpool_slice_cluster_v3, TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, -std=c++20), self-check at m {1,5,8}, the cold-flush graph-replay gate (41 reps, >=1.02x) that sets state['ouns_cl'], and the kernel_fn route "layer == 'unpooler' and n == 1024 and 1 <= m <= 8 and state['ouns_cl']" placed before the swapab16 check. o is NOT routed. (2) Router cluster at m<=8: then apply restack-router-cluster-dsmem-m8.patch (job d323ce8ef6a54fde) with git apply -3 and resolve by hand: the router_kernel<8,8> source and _RC_EXT build (its own ext name, distinct from the unpooler one), _router_cluster_prepare(state) called after _router_prepare in prepare() with its self-check at m {1,5,8} setting state['router_cluster_ok'], and the kernel_fn router branch "state['router_cluster_ok'] and 1 <= m <= 8 and n == 256 -> cluster forward" inserted before path = _router_path(...); m 9..32 keeps swapab, m > 32 keeps hilo. (3) NN layout at m=65..128: then apply restack-pooler-unpooler-nn-layout-m128.patch (job 92cd4e758a924ca0): prepare stores state['pooler_t'] and state['unpooler_t'] = W.t().contiguous() (lossless [K, N] bf16 copies; originals stay), and kernel_fn, just before the "if layer != 'router'" cuBLAS fallback, returns torch.mm(x, state[layer + '_t']) when layer in {pooler, unpooler}, 65 <= m <= 128 and n is the full width; m >= 129 keeps TN, m <= 32 keeps the swap-AB paths. The three edits touch disjoint (layer, m) cells; the only expected conflicts are adjacent insertions in prepare() and kernel_fn. Keep all three, in the order unpooler-cluster check, router-cluster check, NN branch, then master's existing fallbacks. STATE RULE (the judge prepares two states, setup copies 2, and rotates them): every device buffer, flag and handle lives in the state dict built by prepare(). Neither component patch holds device pointers in module or C++ static globals: the only C++ statics are host-side "static bool done" one-shot cudaFuncSetAttribute flags, and _RC_EXT / _OUNS_EXT are compiled-module handles. Keep it that way: if the merge introduces any module-level or static global holding a device pointer, tensor, workspace or TMA descriptor (the "static CUtensorMap g_tb" pattern from the DeepGEMM rewrite), move it into state and pass it as a kernel argument. Gate and self-check scratch (the 256 MB flush buffer, synthetic x) is local to prepare and released before it returns. Hard rules: kernel_fn stays CUDA-graph capturable, with no host syncs, no allocation keyed on input values beyond torch.empty of the output, no data-dependent control flow, no prints, no try/except, no per-call mutable state and nothing cached from inputs; routing reads only shapes and constants fixed in prepare; bf16 output with fp32 accumulation as master; every key attended (n/a, dense GEMM); no child processes (load_inline in-process only). Self-check before dispatch: prepare two states, capture a graph of kernel_fn per state at m in {1, 8, 128} for router, unpooler and pooler, and replay them alternately (A, B, A, B) with x refilled in place each time, comparing every output to a torch.mm reference; on any miss the affected component's flag falls back to master's path. At harvest, confirm in the profile: router_kernel<8,8> on router decode_1/verify_1, ouns_kernel on unpooler decode_1/verify_1, nvjet *_NNT on pooler/unpooler verify_16, and master's _bf16_swapab_kernel / _router_swapab_kernel / hilo on every other row (the 03:41Z harvest found runs that were really master).

## Expected

- `layer=router load=decode_1 m=1` faster: router cluster (d323ce8ef6a54fde): 5.38 -> 4.98 us (-7.5%, pr 1.078), memset node gone
- `layer=router load=verify_1 m=8` faster: router cluster (d323ce8ef6a54fde): 5.47 -> 4.93 us (-9.8%, pr 1.099)
- `layer=unpooler load=decode_1 m=1` faster: unpooler-slice cluster ouns_kernel (e557d92969864449): 6.15 -> 5.34 us (-13.2%, pr 1.136); master's 6.15 is a high reading (5.34-6.15 over four runs), so judge on kernel name plus pr
- `layer=unpooler load=verify_1 m=8` faster: unpooler-slice cluster ouns_kernel (e557d92969864449): 6.00 -> 5.41 us (-9.8%, pr 1.124)
- `layer=pooler load=verify_16 m=128` faster: NN layout (92cd4e758a924ca0): nvjet 128x64 splitK NNT, 25.00 -> 22.59 us (-9.6%, pr 1.117)
- `layer=unpooler load=verify_16 m=128` faster: NN layout (92cd4e758a924ca0): nvjet NNT, 19.96 -> 18.99 us (-4.9%, pr 1.047)
- `layer=router load=decode_16 m=16` flat: m 9..32 keeps master swapab
- `layer=router load=decode_32 m=32` flat: keeps master swapab
- `layer=router load=verify_16 m=128` flat: m>32 keeps hilo
- `layer=router load=verify_32 m=256` flat: keeps hilo
- `layer=unpooler load=decode_16 m=16` flat: full width, untouched
- `layer=unpooler load=decode_32 m=32` flat: untouched
- `layer=pooler load=decode_*` flat: m<=32 keeps swap-AB
- `layer=pooler load=verify_1 m=8` flat: untouched
- `layer=*pooler load=verify_32 m=256` flat: m>=129 keeps TN
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: o not routed; +4-8% drift on o verify_16/decode_32 seen in all three solo runs on unchanged nvjet

## Risk

Headline: substituted estimate ~1.238-1.243 vs best 1.2215 (CI +-1.5%), so promotion rests on the six target rows' paired ratios plus kernel names, not the headline (REWRITE_READ 04:36Z). Landing still needs the user's approval (PENDING_USER 04:36Z). depends_on names only the unpooler-slice parent because hyp.py takes one slug; the router-cluster and NN-layout patches (restack-router-cluster-dsmem-m8, restack-pooler-unpooler-nn-layout-m128, both done/neutral) are applied by hand. Risks: two load_inline builds in one prepare (compile time adds up; keep distinct ext names); merge conflicts at adjacent insertions; the unpooler gate times candidates on synthetic data in untimed prepare, the same pattern as the open cuBLASLt algo-pick rules heads-up. It already ran once in e557d929. If the user holds that pattern, replace the gate with a constant route: it won 1.10x in prepare and 1.12-1.14x on the judged rows.

## Adversarial review

Reviewed 2026-10-25T054315Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a clean restack of three components. Each was checked alone on the same base (b99b1dd) and wins on its own disjoint (layer, m) cells, so per-row attribution plus kernel names makes one combined run readable without companions, and it is the best use of the last bf16 check. The headline will probably sit at or just past the top of best's CI (about 1.238-1.243 against 1.2398), so promotion depends on the six target rows. The risks are prepare-time budget and the unpooler gate flipping off silently, not a design flaw.

### Justification
A: The mechanism is plausible and the profile supports it. Router m=1/8 runs at 5.4-5.5 us against a 1.8 us floor, and about 0.7-0.8 us of that is a vectorized_elementwise memset node, which the cluster kernel removes (solo: 5.38->4.98, 5.47->4.93). Unpooler slice m=1/8 runs at 6.0-6.2 us against a 1.7-1.8 us floor; ouns_kernel gave 5.34/5.41 solo, and in the prepare gate it won about 1.10x net of an empty graph on both m. At m=128, pooler 25.0 us and unpooler 20.0 us sit well above their 14.6 us floor, and NN-layout nvjet gave 22.6/19.0 solo. The expected directions are consistent: six faster rows on cells no other component touches, flat elsewhere. The summed row savings are about 0.4+0.5+0.8+0.6+2.4+1.0 us, and in each solo run they moved the headline +0.0 to +0.7%, so +1.3-1.8% combined is a fair estimate. That is roughly the CI half-width, so the headline alone may stay 'unproven-vs-best'. Weakest link: the unpooler cell. Master's own unpooler decode_1 reading spans 5.34-6.15 us across four runs, so its 'faster' is partly noise. The solo NN-layout run also logged an unpooler decode_1 gain with NO change on that row, which shows the noise is as large as the claimed effect. Also, ouns_cl is decided per prepare() by a timed gate (both m=1 and m=8 must reach >=1.02 net), so the cluster path can silently turn off in one or both rotated states. Judge that row only on the printed '[ouns] PATH CHOSEN' line for each state plus ouns_kernel in the profile. A second weak point: the earlier NN-layout narrowing on the older master (pooler-unpooler-m128-nn-layout-gated, a8bb42a) was a 1.1953 regression with many off-target losses. The b99b1dd restack run (1.2282) suggests those losses were drift, but the NN gain is still only two solo readings.

### Confounders on the current kernel
- bf16-restack-m16-swapab-router-swapab-m32 (base b99b1dd): it put router m<=32 on _router_swapab_kernel + memset. The router-cluster win at m=1/8 is measured against that kernel, and most of it is the deleted memset node. If the swap-AB router were retuned or reverted, the cluster margin would change. This is not a confound for this run: the base is fixed and the same base was used in all three solo runs.
- bf16-restack-m16-swapab-router-swapab-m32 / bf16-m16-swapab-wgmma-extend: ouns_kernel competes with the swapab16 unpooler-slice path (4.3-4.4 us kernel). The in-prepare gate compares against exactly that incumbent, so the comparison is internally controlled. No confound beyond the gate's noise.
- router-triton-hilo-fused-control-splitk-all-m / router-hilo-at-m8: these set the hi+lo BF16 router numerics that router_kernel<8,8> must reproduce with fp32 accumulation, a correctness coupling rather than a timing one. The self-check at m {1,5,8} covers it.
- sglang-gemm-dispatch-parity: it fixes cuBLAS dispatch above m=32. The NN branch is inserted before that fallback for pooler/unpooler at 65..128 only, so a pooler/unpooler verify_16 change is the NN layout itself. Confirm with an *_NNT nvjet name in the profile.
- Shared drift (not a landed hypothesis): o verify_16/decode_32 got +4-8% slower in all three solo runs with unchanged kernels, and router verify_16 lost in two of three with hilo unchanged. Expect it again and do not attribute it to the restack.

### Interference and A/B needs
No companions. The three components touch disjoint (layer, m) cells and each has already been measured alone on the identical base (d323ce8ef6a54fde, e557d92969864449, 92cd4e758a924ca0). Each would-be control already exists, so a gain or loss on a target row reads directly against both master and that component's solo number. The only real interaction channels are prepare-side: two load_inline builds plus two prepared states against the 300 s prepare budget, ext-name collisions, and the GPU state left by one component's gate or self-check while the other times. None of these is something an A/B run would resolve better than the run's own log lines. An ab-revert of a landed hypothesis is not meaningful because every component targets cells that landed work owns, and companions cannot run today anyway. The one ambiguity a companion would resolve, whether a combined headline below the summed estimate comes from interaction or noise, is better answered by per-row paired ratios against the three solo runs.

### Open questions
- Does prepare() fit 300 s with two cold load_inline builds (if the torch extension cache is not warm on the judge), two states, the unpooler gate and the new A/B/A/B graph self-check? Print per-phase timings. If the router build fails or times out it falls back silently, and the run becomes a two-component restack.
- Did both prepared states choose the same paths? Require the '[ouns] PATH CHOSEN' and router-cluster self-check lines twice in the log before reading the target rows.
- If the headline lands inside best's CI but all six target rows show the expected kernel names and paired ratios above 1, is per-row promotion enough under the current promotion rule, or does landing need the user's approval (PENDING_USER 04:36Z already says it does)?
- If the combined run shows the router or unpooler m<=8 gains smaller than solo, is that because the second cluster extension changes co-resident cluster occupancy, or is it noise? The ouns cudaOccupancyMaxActiveClusters line (45 vs 16 needed) should be re-printed to rule out the first.


## Implementation notes

All edits in the worktree's muse_bf16_proj/submission.py (diff vs master blob: +1226/-6).
- Unpooler-slice cluster: as 3-way-applied by the claim (ext muse_bf16_unpool_slice_cluster_v3, 9.0a, -lineinfo, gate sets state['ouns_cl'], route before swapab16).
- Router cluster: hand-merged from restack-router-cluster-dsmem-m8.patch as section (8): router_kernel<8,8> source, _RC_EXT (ext muse_bf16_router_cluster_m8_v1, distinct), _router_cluster_prepare after the ouns prepare, route before `_router_path`.
- NN layout: state['pooler_t'/'unpooler_t'] in prepare, branch just before the cuBLAS fallback; constants NN_GATED_*.
- Review fixes applied (review wins over spec where they differ):
  - distinct load_inline ext names (see above).
  - per-phase prepare timings printed (`[prepare-time] ...` lines per state, plus totals).
  - dispatch flags (`ouns_cl`, `router_cluster_ok`, new `nn_ok`) are read from `state` only; the `_OUNS_EXT is None` / `_RC_EXT is None` module checks were removed from the route functions (flags are already off when a module is missing).
  - `[ouns] PATH CHOSEN` and a single `[router_cluster] PATH CHOSEN` line print once per prepare() (the second A/B state inherits flags and prints none). The router patch's per-m PATH CHOSEN lines were collapsed to one.
  - ouns gate: finally block drops the 256 MB flush buffer, baseline graph and scratch; prepare() ends with synchronize + empty_cache.
  - ouns cudaOccupancyMaxActiveClusters (cl 8 and 4) and the router occupancy (needs 16) are reprinted at the end of prepare, after both modules are loaded.
  - kernel_fn has no try/except, print or sync.
- Added (spec + brief): `_ab_selfcheck` in prepare: builds a second state (`_prepare_state(weights, flags=...)`, inherits A's flags, skips gates), captures kernel_fn per state at router/unpooler/pooler x m {1,8,128} (unpooler m<=8 at n=1024, m=128 at n=8192), replays A,B,A,B with x refilled in place, compares to fp64 (router, 5e-5) / fp32 (bf16 layers, 6e-3) references, flushed prints around each capture. A miss or exception switches off that cell's component (router_cluster_ok / ouns_cl / nn_ok) in both states. Cells served by master's path are checked but have nothing to switch off. prepare() was split into `_prepare_state` + `prepare`.
- Deviations: no -std=c++20 added to either build (the ouns build compiled without it in e557d929 and the flags do not override gencode); no sm_90a-only PTX is used (mma.sync, cp.async.cg, cluster, mapa), so no __CUDA_ARCH_FEAT_SM90_ALL guard; arch list still pinned to 9.0a around both builds and restored. Router self-check keeps the patch's own m {1,5,8} check in addition to the A/B check.
- Not run (no GPU on this node). Main risks: prepare time (two cold load_inline builds + second state build + 18 graph captures against 300 s), second state memory.

## Rules check

- [x] first line `# okc: problem=muse_bf16_proj` kept
- [x] no child processes (grep for subprocess/fork/multiprocessing/Popen/os.system: none); load_inline in-process only
- [x] no host sync, allocation on input values, print or try/except in kernel_fn (diff checked; only torch.empty of the output and cfg/shape branches)
- [x] no device pointers, tensors, TMA maps or workspaces in module/C++ globals: only `static bool done` flags and _RC_EXT/_OUNS_EXT module handles; everything else in state
- [x] nothing derived from inputs cached; routing reads cfg, shapes, strides, alignment and prepare() constants
- [x] precision: router fp32 output from hi+lo bf16 with fp32 accumulation as master; bf16 layers bf16 out, fp32 accumulation; NN copies are lossless
- [x] weights not stored at lower precision (pooler_t/unpooler_t are exact bf16 transposes)
- [x] CUDA-graph capturable kernel_fn; self-check captures graphs of it per state
- [x] every key attended: n/a, dense GEMM
- [x] sm_90a PTX guard: n/a, no wgmma/setmaxnreg in the CUDA code; arch list pinned anyway
- [x] mbarrier/spin waits: n/a, none in either CUDA kernel (cluster barriers only)
- [x] build error / self-check miss / unsupported shape falls to master's path without raising

okc_lint output: `OK with warnings: muse_bf16_proj/submission.py` (13 warnings: host-sync in prepare-only code, uint8 flush buffer in the gate; no errors). py_compile: ok, __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 by an opus subagent, by reading the diff against master blob 970b415 (no GPU on this node).

**Verdict: dispatch** (no fixes needed).

- Outside the gated cells it is master. kernel_fn adds three branches, each gated on cfg/shape plus a flag from `state`. (1) ouns runs first, only for unpooler, n 1024, k 4096, 1 <= m <= 8, `state["ouns_cl"]`. (2) NN runs inside `layer != "router"` just before the cuBLAS fallback, only for pooler/unpooler, 65 <= m <= 128, n equal to the full width, `state["nn_ok"]`. The test shapes m 130 and m 1000 (unpooler) and m 512 keep TN. (3) The router cluster runs before `_router_path`, only for 1 <= m <= 8, n 256, k 8192, `state["router_cluster_ok"]`. `_prepare_state` is master's prepare body copied verbatim, with the NN copies, the ouns prepare and the router-cluster prepare added.
- Globals: at module level only `_OUNS_EXT` / `_RC_EXT` (module handles) plus constants. In C++, the only statics are host `static bool done` flags for cudaFuncSetAttribute, which first fires outside capture (self-check / router_info). Nothing holds a device pointer, tensor or TMA map.
- `_ab_selfcheck` builds a real second state through `_prepare_state(weights, flags=...)`. B shares the read-only weight tensors and has its own hi/lo, NN copies, workspaces and tickets. It captures kernel_fn once per state for each of the 9 cells. It replays A,B,A,B and refills x in place with `copy_` before every replay. It compares against an fp64 reference (router, 5e-5) or an fp32 reference (bf16 layers, 6e-3). Both bounds have margin over the expected error, about 5e-6 and 2e-3. Each cell sits inside try/except, so a miss (including a failed capture) switches off only that cell's component in both states. A failed B build switches off all three in A, which is conservative and printed loudly. `[prepare] FINAL FLAGS` gives the final decision.
- kernel_fn has no sync, print, try/except or value-dependent branch. Its only allocations are torch.empty of the output, inside the forwards, and torch.mm's output.
- Budget: the two cold load_inline builds run at import, about 37 s each (37.3 s in e557d929, 36.9 s in d323ce8e). In the log, prepare runs once per run. B rebuilds no Triton kernels, because they are cached in-process, so it costs warm-up launches only. The 18 captures and 36 replays take well under 10 s. The ouns gate takes about 2 s. Estimated prepare is about 60 s or less, against 300 s, and the whole run about 170 s against 600 s. Nothing was cut.
- okc_lint: OK with warnings (13, all host syncs in prepare plus the uint8 flush buffer). py_compile OK. __pycache__ removed.

Residual risk: the ouns graph-replay gate can still turn the unpooler m<=8 cluster off on noise (it needs >=1.02x at both m), and the same prepare-time timing pattern is still the open rules heads-up. Read `[ouns] PATH CHOSEN` and `[prepare] FINAL FLAGS` before reading the six target rows.

## Result

- job `84f17adc3f074a33` (check), status done, passed True, tests 9/9
- speedup 1.2366x, 95% CI [1.2181, 1.2552], score 3906.4246235707496 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2215006063849365, "ci": [1.2031780972891624, 1.2398231154807104], "job": "1e90292c7b954d96"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-restack-m16-swapab-router-swapab-m32.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128.result.json`, `muse_bf16_proj/hypotheses/done/neutral/restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128.patch`; branch `hyp/muse_bf16_proj/restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 12.18 | 12.38 | 0.9929 | flat |
| layer=o load=decode_1 m=1 | 62 | 5.89 | 5.89 | 1.0022 | flat |
| layer=pooler load=decode_1 m=1 | 62 | 18.84 | 18.74 | 0.9958 | flat |
| layer=unpooler load=decode_1 m=1 | 62 | 6.15 | 5.27 | 1.1662 | gained |
| layer=router load=decode_1 m=1 | 62 | 5.38 | 5.1 | 1.0722 | gained |
| layer=qkv_gate load=decode_16 m=16 | 62 | 13.15 | 13.14 | 0.9999 | flat |
| layer=o load=decode_16 m=16 | 62 | 6.23 | 6.27 | 0.9804 | flat |
| layer=pooler load=decode_16 m=16 | 62 | 19.47 | 18.72 | 1.0082 | flat |
| layer=unpooler load=decode_16 m=16 | 62 | 18.51 | 18.57 | 0.9814 | flat |
| layer=router load=decode_16 m=16 | 62 | 5.47 | 5.58 | 0.9806 | flat |
| layer=qkv_gate load=decode_32 m=32 | 62 | 14.19 | 13.93 | 0.9972 | flat |
| layer=o load=decode_32 m=32 | 62 | 6.34 | 6.07 | 0.9943 | flat |
| layer=pooler load=decode_32 m=32 | 62 | 19.73 | 19.84 | 0.9938 | flat |
| layer=unpooler load=decode_32 m=32 | 62 | 18.08 | 17.99 | 0.9724 | flat |
| layer=router load=decode_32 m=32 | 62 | 6.2 | 5.73 | 1.0518 | gained |
| layer=qkv_gate load=verify_1 m=8 | 62 | 12.5 | 12.45 | 1.0154 | flat |
| layer=o load=verify_1 m=8 | 62 | 6.13 | 6.13 | 1.0013 | flat |
| layer=pooler load=verify_1 m=8 | 62 | 18.89 | 18.91 | 0.9928 | flat |
| layer=unpooler load=verify_1 m=8 | 62 | 6.0 | 4.99 | 1.1542 | gained |
| layer=router load=verify_1 m=8 | 62 | 5.47 | 4.97 | 1.1074 | gained |
| layer=qkv_gate load=verify_16 m=128 | 62 | 17.05 | 16.58 | 1.01 | flat |
| layer=o load=verify_16 m=128 | 62 | 6.98 | 7.35 | 0.9819 | flat |
| layer=pooler load=verify_16 m=128 | 62 | 25.0 | 22.94 | 1.1171 | gained |
| layer=unpooler load=verify_16 m=128 | 62 | 19.96 | 18.95 | 1.0297 | flat |
| layer=router load=verify_16 m=128 | 62 | 7.28 | 7.38 | 0.9865 | flat |
| layer=qkv_gate load=verify_32 m=256 | 62 | 19.69 | 19.71 | 0.9901 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.26 | 9.61 | 0.9636 | lost |
| layer=pooler load=verify_32 m=256 | 62 | 31.19 | 31.33 | 0.9856 | flat |
| layer=unpooler load=verify_32 m=256 | 62 | 24.42 | 24.55 | 0.987 | flat |
| layer=router load=verify_32 m=256 | 62 | 9.6 | 9.31 | 1.0218 | flat |

gained on unpooler decode_1, verify_1; router decode_1, decode_32, verify_1; pooler verify_16; lost on o verify_32. Expected hit: 15/17. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

