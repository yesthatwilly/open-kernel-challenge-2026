---
problem: muse_prefill_layer
slug: experts-turn4-ssdq-s2-tma-raw-weight-ring
created: 2026-10-25T110612Z
base: master
area: "w13 + w2 expert GEMMs at 1024 < T <= 4096 (turn_4): ssdq producer weight feed, TMA raw-code smem ring instead of register __ldg prefetch"
change: >
  Stage 2 of the ssdq line, built on master (experts-turn4-ssdq-s1b-dataptr-symbol-fix landed:
  ssdq_kernel serves 1024 < T <= 4096). Only the PRODUCER's weight feed changes; tiles (BM 128, BN
  256, BKT 64), the two SS-wgmma consumer warpgroups, the A-tile TMA, the epilogues,
  align/gather/_silu_rows/weighted sum and every other T class stay as master has them. EVIDENCE:
  master's own counters (archive ef281fb8b7304240.log) say the kernel is producer-bound: w13
  T=3012, 1024 CTAs, 157242 cycles per CTA, consumers wait on full stages 132587 cycles (84%)
  while the producer waits on empty stages only 6353. Per CTA that is 576 KB of packed
  codes+scales in 157k cycles, about 0.85 TB/s chip-wide: the producer warpgroup (4 warps, one per
  scheduler) holds only one 128-k chunk (cur/nxt, 16 KB per CTA) in flight through __ldg into
  registers, so the weight stream is latency-bound, and each dequant waits on its own global load.
  CHANGE: in a new __global__ ssdq2_kernel (same file, same load_inline module, renamed
  SSDQ_EXT_NAME 'muse_prefill_ssdq_v2'), replace the cur/nxt/cs/ns register prefetch with an smem
  RAW ring of RAW_STAGES = 4 entries, each one 128-k chunk of the CTA's 256 weight rows: 16 KB of
  packed E2M1 codes loaded by ONE 2-D TMA (a second CUtensorMap in Maps over bank['w13'|'w2']
  viewed as uint8 [E*N, K/2], box {64 B, 256 rows}, SWIZZLE_NONE, encoded per launch on the host
  exactly like the A map) plus 2 KB of e4m3 scales (8 B per row) copied by the producer threads
  with cp.async.ca 8-byte copies (2 rows per thread) followed by cp.async.mbarrier.arrive.noinc on
  the entry's raw_full mbarrier (init count 1 + 128; thread 0 does expect_tx(16384) and the TMA).
  Keep bf16 STAGES = 3 (3 x 48 KB) so smem = 1024 + 144 KB + 4 x 18 KB = 217 KB (<= 227 KB opt-in;
  print it). Producer loop per chunk c: mbar_wait(raw_full[c % 4]); dequantize both halves from
  smem (thread p reads 16 B at row (it*32 + p>>2) * 64 + (p&3)*16, conflict-free, same
  nibble/perm/dq_pair math and the same swizzled st.shared into the bf16 stage, same empty/full
  protocol with the consumers); then bar.sync on a producer-only named barrier (id 1, 128 threads)
  so every producer thread has finished reading entry c % 4, and refill it with chunk c + 4 (TMA
  by thread 0, scales by cp.async from all) if c + 4 < K/128. Prologue issues chunks 0..3. That
  keeps up to 3 chunks (54 KB) in flight per SM while one is dequantized, against 16 KB today, and
  removes 96 prefetch registers from the producer (print regs/local bytes: master has 168 regs and
  a 104 B spill). Keep master's ssdq_kernel compiled in the same module unchanged. prepare(): the
  canary, self-check (T=1100 one, 1506 random, 3012 random, 4096 random and skewed; rel-L2 and
  worst-row vs the fp32 dequant reference with master's bounds) and the two-state replay check
  (capture once per state, refill inputs in place, A/B/A/B; never gate on data_ptr equality: the
  judge clones tensors between copies, LESSONS) run for ssdq2 exactly as for ssdq; print the
  producer/consumer wait counters for ssdq2 at T=3012 next to master's. GATE (three-way, at T=3012
  soft routing, cold-L2 graph replays, >= 9 interleaved alternations, empty-graph baseline
  subtracted): time Marlin pair, master's ssdq block and the ssdq2 block; route (1024, 4096] to
  ssdq2 only if ssdq2 <= 0.98x master's ssdq median (it must WIN against the incumbent, which is
  now ssdq, not Marlin), else to master's ssdq if it still wins its own 0.95x gate vs Marlin, else
  Marlin. Print all three medians and one PATH CHOSEN line per T class as the last lines of
  prepare(). RULES: kernel_fn stays graph-capturable (no host syncs, prints, try/except,
  allocation sized by values, or data-dependent host branching; dispatch only on T and prepare()
  flags); NVFP4 weights stay NVFP4 in global memory, dequantized in-kernel only (the per-call BF16
  weight copy family stays held); no module/static/C++ globals holding device pointers, maps or
  buffers (both judge states); build with TORCH_CUDA_ARCH_LIST='9.0a' and extra_cuda_cflags
  exactly ['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a'], wgmma under
  __CUDA_ARCH_FEAT_SM90_ALL with the error-word #else, reinterpret_cast<T*>(t.data_ptr()) only;
  keep SSDQ_BUDGET_S. First measurement is an okc check.
expected:
  - shapes: "*load=turn_4*"
    direction: faster
    why: "master's ssdq w13+w2 is 1878/1865 us in the profile (1256 us in the prepare gate) against a ~306 us math floor and ~224 us byte floor (1.07 GB NVFP4 at 4.8 TB/s), and its consumers wait 84% of each CTA on the producer. Freeing the producer from global-load latency should put each bf16 stage at ~1000 cycles (consumer MMA time) instead of ~2457: target ssdq pair <= 1000 us profiled (<= 700 us gate). Sliding turn_4 3012 -> ~2200-2400 us, full turn_4 4686 -> ~3900-4100 us; milestone: consumer wait share < 40% in the printed counters"
  - shapes: "*load=turn_1*"
    direction: unchanged
    why: T=753 stays on Marlin
  - shapes: "*load=first_turn*"
    direction: unchanged
    why: T=16384 stays on the Triton NVFP4 GEMM
risk: high
depends_on: experts-turn4-ssdq-s1b-dataptr-symbol-fix
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: ssdq_kernel x2 (w13, w2) at 1024 < T <= 4096 (turn_4, T=3012)
review: required
status: queued
---

# experts-turn4-ssdq-s2-tma-raw-weight-ring

## Change

Stage 2 of the ssdq line, built on master (experts-turn4-ssdq-s1b-dataptr-symbol-fix landed: ssdq_kernel serves 1024 < T <= 4096). Only the PRODUCER's weight feed changes; tiles (BM 128, BN 256, BKT 64), the two SS-wgmma consumer warpgroups, the A-tile TMA, the epilogues, align/gather/_silu_rows/weighted sum and every other T class stay as master has them. EVIDENCE: master's own counters (archive ef281fb8b7304240.log) say the kernel is producer-bound: w13 T=3012, 1024 CTAs, 157242 cycles per CTA, consumers wait on full stages 132587 cycles (84%) while the producer waits on empty stages only 6353. Per CTA that is 576 KB of packed codes+scales in 157k cycles, about 0.85 TB/s chip-wide: the producer warpgroup (4 warps, one per scheduler) holds only one 128-k chunk (cur/nxt, 16 KB per CTA) in flight through __ldg into registers, so the weight stream is latency-bound, and each dequant waits on its own global load. CHANGE: in a new __global__ ssdq2_kernel (same file, same load_inline module, renamed SSDQ_EXT_NAME 'muse_prefill_ssdq_v2'), replace the cur/nxt/cs/ns register prefetch with an smem RAW ring of RAW_STAGES = 4 entries, each one 128-k chunk of the CTA's 256 weight rows: 16 KB of packed E2M1 codes loaded by ONE 2-D TMA (a second CUtensorMap in Maps over bank['w13'|'w2'] viewed as uint8 [E*N, K/2], box {64 B, 256 rows}, SWIZZLE_NONE, encoded per launch on the host exactly like the A map) plus 2 KB of e4m3 scales (8 B per row) copied by the producer threads with cp.async.ca 8-byte copies (2 rows per thread) followed by cp.async.mbarrier.arrive.noinc on the entry's raw_full mbarrier (init count 1 + 128; thread 0 does expect_tx(16384) and the TMA). Keep bf16 STAGES = 3 (3 x 48 KB) so smem = 1024 + 144 KB + 4 x 18 KB = 217 KB (<= 227 KB opt-in; print it). Producer loop per chunk c: mbar_wait(raw_full[c % 4]); dequantize both halves from smem (thread p reads 16 B at row (it*32 + p>>2) * 64 + (p&3)*16, conflict-free, same nibble/perm/dq_pair math and the same swizzled st.shared into the bf16 stage, same empty/full protocol with the consumers); then bar.sync on a producer-only named barrier (id 1, 128 threads) so every producer thread has finished reading entry c % 4, and refill it with chunk c + 4 (TMA by thread 0, scales by cp.async from all) if c + 4 < K/128. Prologue issues chunks 0..3. That keeps up to 3 chunks (54 KB) in flight per SM while one is dequantized, against 16 KB today, and removes 96 prefetch registers from the producer (print regs/local bytes: master has 168 regs and a 104 B spill). Keep master's ssdq_kernel compiled in the same module unchanged. prepare(): the canary, self-check (T=1100 one, 1506 random, 3012 random, 4096 random and skewed; rel-L2 and worst-row vs the fp32 dequant reference with master's bounds) and the two-state replay check (capture once per state, refill inputs in place, A/B/A/B; never gate on data_ptr equality: the judge clones tensors between copies, LESSONS) run for ssdq2 exactly as for ssdq; print the producer/consumer wait counters for ssdq2 at T=3012 next to master's. GATE (three-way, at T=3012 soft routing, cold-L2 graph replays, >= 9 interleaved alternations, empty-graph baseline subtracted): time Marlin pair, master's ssdq block and the ssdq2 block; route (1024, 4096] to ssdq2 only if ssdq2 <= 0.98x master's ssdq median (it must WIN against the incumbent, which is now ssdq, not Marlin), else to master's ssdq if it still wins its own 0.95x gate vs Marlin, else Marlin. Print all three medians and one PATH CHOSEN line per T class as the last lines of prepare(). RULES: kernel_fn stays graph-capturable (no host syncs, prints, try/except, allocation sized by values, or data-dependent host branching; dispatch only on T and prepare() flags); NVFP4 weights stay NVFP4 in global memory, dequantized in-kernel only (the per-call BF16 weight copy family stays held); no module/static/C++ globals holding device pointers, maps or buffers (both judge states); build with TORCH_CUDA_ARCH_LIST='9.0a' and extra_cuda_cflags exactly ['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a'], wgmma under __CUDA_ARCH_FEAT_SM90_ALL with the error-word #else, reinterpret_cast<T*>(t.data_ptr()) only; keep SSDQ_BUDGET_S. First measurement is an okc check.

## Expected

- `*load=turn_4*` faster: master's ssdq w13+w2 is 1878/1865 us in the profile (1256 us in the prepare gate) against a ~306 us math floor and ~224 us byte floor (1.07 GB NVFP4 at 4.8 TB/s), and its consumers wait 84% of each CTA on the producer. Freeing the producer from global-load latency should put each bf16 stage at ~1000 cycles (consumer MMA time) instead of ~2457: target ssdq pair <= 1000 us profiled (<= 700 us gate). Sliding turn_4 3012 -> ~2200-2400 us, full turn_4 4686 -> ~3900-4100 us; milestone: consumer wait share < 40% in the printed counters
- `*load=turn_1*` unchanged: T=753 stays on Marlin
- `*load=first_turn*` unchanged: T=16384 stays on the Triton NVFP4 GEMM

## Risk

CUDA rewrite stage: build, mbarrier-count or proxy-ordering bugs (TMA write after generic smem reads of the same RAW entry; the producer-only bar.sync must precede the refill) can hang or fault; the error word and mbarrier timeout from s1b catch hangs, and a fault poisons the context, so the canary must launch ssdq2 only after master's ssdq has been canaried, and a failure leaves master's ssdq ON. If the counters still show consumer wait >70% after the change, the producer is dequant-ALU/latency-bound rather than load-bound: that is the stage-3 signal for a second producer warpgroup (setmaxnreg 40/216). smem 217 KB leaves 1 CTA/SM as today.

## Milestones

Rewrite stage 2 of `ssdq_kernel x2 (w13, w2) at 1024 < T <= 4096 (turn_4, T=3012)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
