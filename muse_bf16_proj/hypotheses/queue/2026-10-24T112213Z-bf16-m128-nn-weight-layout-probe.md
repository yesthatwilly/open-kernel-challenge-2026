---
problem: muse_bf16_proj
slug: bf16-m128-nn-weight-layout-probe
created: 2026-10-24T112213Z
base: master
area: bf16 GEMM dispatch (m>=128, weight layout)
change: >
  Built on master (dba4ffd; router untouched; m <= 32 paths untouched). The m=128/256 shapes are
  where cuBLAS sits furthest from the floor (pooler verify_16 25.2 us vs 14.6,
  nvjet_sm90_tst_64x64_64x13_2x1 with no split-K at ~2.7 TB/s; qkv_gate verify_16 17.0 vs 8.4,
  nvjet 96x64 splitK tile that pads 128 rows to 192 + a 3.1 us splitKreduce; o verify_32 9.5 vs
  4.5; unpooler verify_16 19.7 vs 14.6), and the Triton wgmma replacement lost there
  (pooler-unpooler-m128-wgmma-splitk-semaphore: wgmma emitted, 154 regs, no spills, but its main
  loop alone was 26 us warm-L2 plus a 4-6.5 us semaphore-reduce tail, 29.4/28.7 us vs nvjet
  25.2/20.0). This is a cheap dispatch probe that hands cuBLAS a different heuristic entry instead
  of writing a kernel: in prepare(), for qkv_gate, o, pooler and unpooler build a lossless
  transposed copy state[layer + "_t"] = W.t().contiguous() ([K, N] bf16 row-major; the original
  stays for the m <= 32 paths). In kernel_fn, for those four layers when cfg m >= 128 only,
  replace _bf16_gemm_dispatch_impl(x, W[:n], None) with torch.mm(x, state[layer + "_t"][:, :n]) (n
  is the full width at m >= 128, so the slice is a no-op view; contiguous [m, n] bf16 result, fp32
  accumulation inside cuBLAS as before). cuBLAS then sees an NN problem instead of TN, the weight
  becomes an N-major A operand, and nvjet picks from a different tile/split-K table. No Triton, no
  new kernel; m <= 32 and the router are byte-identical to master. Print once in prepare() the
  chosen kernel is not observable host-side, so read top_kernels in the profile: the attempt file
  must record, per m=128/256 shape, the nvjet kernel name and time under both layouts (master from
  FINDINGS). Gate only on cfg layer and m. Tests m=130 unpooler, m=512 qkv_gate, m=1000 unpooler
  take the NN path (same math).
expected:
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: "largest gap on the board (25.2 us vs floor 14.6): master nvjet 64x64 non-split tile runs 2.7 TB/s; the NN table at m=128 should pick a larger N tile or split-K (as TN does at m=32: 128x32 splitK, 3.3 TB/s). Target <= 22 us"
  - shapes: layer=qkv_gate load=verify_16 m=128
    direction: faster
    why: "17.0 vs 8.4 us (2.0x, largest ratio on a heavy shape): the 96-row TN tile wastes a third of the MMA at m=128 and needs a 3.1 us splitKreduce; any 64/128-row NN tile avoids the pad"
  - shapes: layer=unpooler load=verify_16 m=128
    direction: faster
    why: 19.7 vs 14.6 us, nvjet 128x64 at ~3.4 TB/s; probe
  - shapes: layer=o load=verify_16 m=128
    direction: flat
    why: 7.0 vs 4.0 us; k=1024 is short, either layout is prologue-bound; probe
  - shapes: layer=* load=verify_32 m=256
    direction: flat
    why: near compute floor for pooler/qkv (17.4/9.8 us at 989 TF; nvjet ~580 TF); o 9.5 vs 4.5 is the one that could move; probe, may lose
  - shapes: layer=* load=decode_* m=*
    direction: flat
    why: m <= 32 untouched
  - shapes: layer=* load=verify_1 m=8
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# bf16-m128-nn-weight-layout-probe

## Change

Built on master (dba4ffd; router untouched; m <= 32 paths untouched). The m=128/256 shapes are where cuBLAS sits furthest from the floor (pooler verify_16 25.2 us vs 14.6, nvjet_sm90_tst_64x64_64x13_2x1 with no split-K at ~2.7 TB/s; qkv_gate verify_16 17.0 vs 8.4, nvjet 96x64 splitK tile that pads 128 rows to 192 + a 3.1 us splitKreduce; o verify_32 9.5 vs 4.5; unpooler verify_16 19.7 vs 14.6), and the Triton wgmma replacement lost there (pooler-unpooler-m128-wgmma-splitk-semaphore: wgmma emitted, 154 regs, no spills, but its main loop alone was 26 us warm-L2 plus a 4-6.5 us semaphore-reduce tail, 29.4/28.7 us vs nvjet 25.2/20.0). This is a cheap dispatch probe that hands cuBLAS a different heuristic entry instead of writing a kernel: in prepare(), for qkv_gate, o, pooler and unpooler build a lossless transposed copy state[layer + "_t"] = W.t().contiguous() ([K, N] bf16 row-major; the original stays for the m <= 32 paths). In kernel_fn, for those four layers when cfg m >= 128 only, replace _bf16_gemm_dispatch_impl(x, W[:n], None) with torch.mm(x, state[layer + "_t"][:, :n]) (n is the full width at m >= 128, so the slice is a no-op view; contiguous [m, n] bf16 result, fp32 accumulation inside cuBLAS as before). cuBLAS then sees an NN problem instead of TN, the weight becomes an N-major A operand, and nvjet picks from a different tile/split-K table. No Triton, no new kernel; m <= 32 and the router are byte-identical to master. Print once in prepare() the chosen kernel is not observable host-side, so read top_kernels in the profile: the attempt file must record, per m=128/256 shape, the nvjet kernel name and time under both layouts (master from FINDINGS). Gate only on cfg layer and m. Tests m=130 unpooler, m=512 qkv_gate, m=1000 unpooler take the NN path (same math).

## Expected

- `layer=pooler load=verify_16 m=128` faster: largest gap on the board (25.2 us vs floor 14.6): master nvjet 64x64 non-split tile runs 2.7 TB/s; the NN table at m=128 should pick a larger N tile or split-K (as TN does at m=32: 128x32 splitK, 3.3 TB/s). Target <= 22 us
- `layer=qkv_gate load=verify_16 m=128` faster: 17.0 vs 8.4 us (2.0x, largest ratio on a heavy shape): the 96-row TN tile wastes a third of the MMA at m=128 and needs a 3.1 us splitKreduce; any 64/128-row NN tile avoids the pad
- `layer=unpooler load=verify_16 m=128` faster: 19.7 vs 14.6 us, nvjet 128x64 at ~3.4 TB/s; probe
- `layer=o load=verify_16 m=128` flat: 7.0 vs 4.0 us; k=1024 is short, either layout is prologue-bound; probe
- `layer=* load=verify_32 m=256` flat: near compute floor for pooler/qkv (17.4/9.8 us at 989 TF; nvjet ~580 TF); o 9.5 vs 4.5 is the one that could move; probe, may lose
- `layer=* load=decode_* m=*` flat: m <= 32 untouched
- `layer=* load=verify_1 m=8` flat: untouched
- `layer=router *` flat: untouched

## Risk

A probe, like bf16-cublaslt-backend-toggle: direction is genuinely unknown per shape, and it may lose on some (cuBLAS tuned TN most for nn.Linear). It is attributable per shape because only m >= 128 bf16 rows change, and it costs no engineering. Its value is the per-shape table: any m=128/256 shape where NN beats TN by more than drift becomes one more component of a later shape-gated combo (like bf16-shape-gated-triton-winners-combo); shapes where it loses are simply not routed. Also informs the held qkv-gate-o-m128-wgmma-splitk-semaphore (if cuBLAS NN already closes the m=128 qkv gap, that rewrite is not worth it). Extra memory: ~189 MB of transposed bf16 copies, lossless, same precision (allowed repack). Risk: torch.mm may internally detect a transposed-contiguous operand and re-normalize to the same TN call; the implementer must ensure the [K, N] copy is truly contiguous (not a .t() view of W) and confirm from top_kernels that kernel names changed (NN suffix); if every name is unchanged the run is void, not neutral.
