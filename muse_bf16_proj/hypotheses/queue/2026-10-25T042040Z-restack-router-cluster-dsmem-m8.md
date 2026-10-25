---
problem: muse_bf16_proj
slug: restack-router-cluster-dsmem-m8
created: 2026-10-25T042040Z
base: master
area: router GEMM (m<=8)
change: >
  Built on master (b99b1dd, the landed restack bf16-restack-m16-swapab-router-swapab-m32, 1.2215x;
  check 1e90292c7b954d96). Port ONLY the router cluster kernel from branch
  hyp/muse_bf16_proj/router-cuda-cluster-dsmem-splitk-rewrite (job dc5d03c4: 16 clusters x 8 CTAs,
  swap-AB mma.sync, split-K reduced through DSMEM, torch.empty output, no memset, one node): its
  CUDA source string, the load_inline build helper (keep TORCH_CUDA_ARCH_LIST=9.0a around the
  build and -lineinfo; if extra_cuda_cflags are overridden they must contain -std=c++20, replacing
  any other -std), the prepare() self-check and the state['router_cluster_ok'] constant. Compile
  only the NT=8 instance. In the restack's kernel_fn router branch, insert before `path =
  _router_path(...)`: if state['router_cluster_ok'] and 1 <= x.shape[0] <= 8 and cfg n == 256 ->
  return the cluster forward. m 9..32 keeps the restack's _router_forward (swapab; it beat the
  cluster at m=16, 5.47 vs 5.87), m > 32 keeps _router_hilo_forward. Self-check at m in {1, 5, 8}
  only (same bounds as the source branch). Everything else byte-identical to the restack. At
  harvest confirm the cluster kernel name on router decode_1/verify_1 in the profile and the
  restack's _bf16_swapab_kernel/_router_swapab_kernel on the other rows (the 03:41Z harvest found
  nine runs that were really master).
expected:
  - shapes: layer=router load=decode_1 m=1
    direction: faster
    why: restack row 5.38 = swapab 3.1 + memset 0.8 + node; cluster read 5.24 on master (-8.7%, z -16)
  - shapes: layer=router load=verify_1 m=8
    direction: faster
    why: restack row 5.47; cluster read 5.08 on master (-14%, z -31)
  - shapes: layer=router load=decode_16 m=16
    direction: flat
    why: keeps restack swapab
  - shapes: layer=router load=decode_32 m=32
    direction: flat
    why: keeps restack swapab
  - shapes: layer=router load=verify_* m=12*
    direction: flat
    why: m=128 keeps hilo
  - shapes: layer=router load=verify_32 m=256
    direction: flat
    why: keeps hilo
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=o *
    direction: flat
    why: untouched
  - shapes: layer=pooler *
    direction: flat
    why: untouched
  - shapes: layer=unpooler *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# restack-router-cluster-dsmem-m8

## Change

Built on master (b99b1dd, the landed restack bf16-restack-m16-swapab-router-swapab-m32, 1.2215x; check 1e90292c7b954d96). Port ONLY the router cluster kernel from branch hyp/muse_bf16_proj/router-cuda-cluster-dsmem-splitk-rewrite (job dc5d03c4: 16 clusters x 8 CTAs, swap-AB mma.sync, split-K reduced through DSMEM, torch.empty output, no memset, one node): its CUDA source string, the load_inline build helper (keep TORCH_CUDA_ARCH_LIST=9.0a around the build and -lineinfo; if extra_cuda_cflags are overridden they must contain -std=c++20, replacing any other -std), the prepare() self-check and the state['router_cluster_ok'] constant. Compile only the NT=8 instance. In the restack's kernel_fn router branch, insert before `path = _router_path(...)`: if state['router_cluster_ok'] and 1 <= x.shape[0] <= 8 and cfg n == 256 -> return the cluster forward. m 9..32 keeps the restack's _router_forward (swapab; it beat the cluster at m=16, 5.47 vs 5.87), m > 32 keeps _router_hilo_forward. Self-check at m in {1, 5, 8} only (same bounds as the source branch). Everything else byte-identical to the restack. At harvest confirm the cluster kernel name on router decode_1/verify_1 in the profile and the restack's _bf16_swapab_kernel/_router_swapab_kernel on the other rows (the 03:41Z harvest found nine runs that were really master).

## Expected

- `layer=router load=decode_1 m=1` faster: restack row 5.38 = swapab 3.1 + memset 0.8 + node; cluster read 5.24 on master (-8.7%, z -16)
- `layer=router load=verify_1 m=8` faster: restack row 5.47; cluster read 5.08 on master (-14%, z -31)
- `layer=router load=decode_16 m=16` flat: keeps restack swapab
- `layer=router load=decode_32 m=32` flat: keeps restack swapab
- `layer=router load=verify_* m=12*` flat: m=128 keeps hilo
- `layer=router load=verify_32 m=256` flat: keeps hilo
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=pooler *` flat: untouched
- `layer=unpooler *` flat: untouched

## Risk

Kill: router decode_1 and verify_1 rows not both below the restack's 1e90292c rows (5.38/5.47) with the cluster kernel in the profile. Expected headline ~1.224 (03:41Z substitution). Risk: the load_inline build adds ~40 s and an illegal-instruction fault would poison the context; the kernel already ran clean on the judge, so run as a check.
