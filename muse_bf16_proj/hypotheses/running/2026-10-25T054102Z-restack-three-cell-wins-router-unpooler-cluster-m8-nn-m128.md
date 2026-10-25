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
status: running
review_at: 2026-10-25T054315Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T054315Z
base_sha: 84c6b217af909ca0534b49bdd5a8072e10b692c5
base_blob: 970b415b746fa03cb92e419d270be92588273178
branch: hyp/muse_bf16_proj/restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128
worktree: .worktrees/muse_bf16_proj/restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128
mode: null
restack_of: restack-unpooler-slice-cluster-dsmem-m8
stale_base: null
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

