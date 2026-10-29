---
problem: muse_bf16_proj
slug: b12-batched-escape-m8-32-race
created: 2026-10-29T200122Z
base: master
mint_blob: "a36f33927c06500cbba267d7e0be2235eb8cd716"
best_at_mint: 1.2794
area: "bf16 dense GEMM cells at m=8..32 (pooler, qkv_gate, o): bulk12 kernels"
change: >
  Add bulk12 kernels for the m=8/16/32 cells of pooler, qkv_gate (and o if it fits), each raced
  per cell in prepare against master's incumbent (swap-AB Triton / cuBLAS paths), routed only if
  bit-exact to its own fp32-accumulate contract within the problem tolerance as master's b12 m=1
  cells are, and <= 0.97x the incumbent in every round of a read-flushed interleaved race (reuse
  master's bulk12 read-flush gate machinery, CLAIMS:59). Kernels from takemyjob hopper/b12tma-1029
  d95798e (probes/b12tma; research/h100-b12tma-1029.md): (a) the cp.async bridge b12x with the
  end-of-kernel serial escape-x gather replaced by batched loads, and (b) uc1 = TMA-box b12
  without clusters, same batched escape gather, plain mbarrier.arrive.shared::cluster (no
  arrive.release.cluster / MEMBAR.ALL.GPU). Reuse master's lossless 12-bit codes where R matches
  (pooler R=32, qkv R=16 are already encoded for m=1) to save prepare time and memory. Print
  [b12m] RACE/ROUTE lines per cell (incumbent us, each arm us and ratio per round) in a <= 2 KB
  summary at the end of prepare.
expected:
  - shapes: layer=pooler load=verify_1*
    direction: faster
    why: "node: batched bridge 21.7 / uc1 20.7 us vs node cuBLAS 26.1 at pooler m8; projected judge 0.89-0.94"
  - shapes: layer=qkv_gate load=verify_1*
    direction: faster
    why: node qkv m8 bridge 22.9->17.6 us batched; uc1 0.922 of the bridge
  - shapes: layer=pooler load=decode_16*
    direction: faster
    why: pooler m16 bridge 44.0->27.4 batched; uc1 0.887
  - shapes: layer=* load=decode_32*
    direction: faster
    why: "m32 cells, same escape-gather fix (untested on node; N>=32 mma.sync trap per DISTILL 3.4: route only on the race)"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "b12mc-multicast-x-pooler-qkv-m8-readflush-diag: that measured multicast on a TMA build stalled by MEMBAR.ALL.GPU and kept the serial escape gather; this fixes both and races per cell with routing"
target: check
value_gain: 2.0
value_p: 0.4
value_ev: 0.8
value_cost: 400k tokens, 1 judge run
evidence: >
  H100 b12tma (takemyjob hopper/b12tma-1029 d95798e; research/h100-b12tma-1029.md): serial escape
  gather ~10 us/8 tokens; batched bridge 29.2->21.7 us (pooler m8), 44.0->27.4 (m16), 22.9->17.6
  (qkv m8); uc1 20.7 us vs cuBLAS 26.1 (pooler m8); judge b12x qkv m8 x0.769 (CLAIMS), board
  1.2794 (3e3ffc5a); triage shares pooler/qkv m8-32 ~25%
user_requested: true
review: required
status: queued
---

# b12-batched-escape-m8-32-race

## Change

Add bulk12 kernels for the m=8/16/32 cells of pooler, qkv_gate (and o if it fits), each raced per cell in prepare against master's incumbent (swap-AB Triton / cuBLAS paths), routed only if bit-exact to its own fp32-accumulate contract within the problem tolerance as master's b12 m=1 cells are, and <= 0.97x the incumbent in every round of a read-flushed interleaved race (reuse master's bulk12 read-flush gate machinery, CLAIMS:59). Kernels from takemyjob hopper/b12tma-1029 d95798e (probes/b12tma; research/h100-b12tma-1029.md): (a) the cp.async bridge b12x with the end-of-kernel serial escape-x gather replaced by batched loads, and (b) uc1 = TMA-box b12 without clusters, same batched escape gather, plain mbarrier.arrive.shared::cluster (no arrive.release.cluster / MEMBAR.ALL.GPU). Reuse master's lossless 12-bit codes where R matches (pooler R=32, qkv R=16 are already encoded for m=1) to save prepare time and memory. Print [b12m] RACE/ROUTE lines per cell (incumbent us, each arm us and ratio per round) in a <= 2 KB summary at the end of prepare.

## Expected

- `layer=pooler load=verify_1*` faster: node: batched bridge 21.7 / uc1 20.7 us vs node cuBLAS 26.1 at pooler m8; projected judge 0.89-0.94
- `layer=qkv_gate load=verify_1*` faster: node qkv m8 bridge 22.9->17.6 us batched; uc1 0.922 of the bridge
- `layer=pooler load=decode_16*` faster: pooler m16 bridge 44.0->27.4 batched; uc1 0.887
- `layer=* load=decode_32*` faster: m32 cells, same escape-gather fix (untested on node; N>=32 mma.sync trap per DISTILL 3.4: route only on the race)

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** H100 b12tma (takemyjob hopper/b12tma-1029 d95798e; research/h100-b12tma-1029.md): serial escape gather ~10 us/8 tokens; batched bridge 29.2->21.7 us (pooler m8), 44.0->27.4 (m16), 22.9->17.6 (qkv m8); uc1 20.7 us vs cuBLAS 26.1 (pooler m8); judge b12x qkv m8 x0.769 (CLAIMS), board 1.2794 (3e3ffc5a); triage shares pooler/qkv m8-32 ~25%
- **Gain if it works:** 2.0% headline; **p** 0.4; **EV** 0.8% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (gain 2.0% < 2.5%; EV 0.8% < 1.5%), minted because the user described it.
- **Cost:** 400k tokens, 1 judge run.

## Risk

Earlier judged b12x at m=8 (x0.769 qkv, x0.984 pooler) carried the serial escape gather; the H100 lane found it costs ~10 us per 8 tokens (25-38% of the kernel) and matches the judge's unexplained ~1 us per token row. Rows touched sum to ~25% of the headline; ~8% per cell is ~2%. HBM rates do not transfer H100->H200, so the race decides; a cell that misses keeps the incumbent.
