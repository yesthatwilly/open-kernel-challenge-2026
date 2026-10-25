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
review: required
status: queued
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
