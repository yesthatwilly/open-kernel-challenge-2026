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
review: proceed
status: queued
review_at: 2026-10-25T111131Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-25T111131Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound stage-2 rewrite step. Master's own counters show ssdq is producer-starved (consumers wait 84% of each CTA), the change touches only the producer's weight feed, the three-way gate measures ssdq2 directly against the incumbent ssdq, and the separately named ssdq2_kernel makes the milestone readable from top_kernels. No companions are needed. Two corrections should go to the builder: master runs bf16 STAGES = 4, not 3, and the producer should print how long it waits on the raw ring, so a loss can be attributed to load latency or to dequant ALU.

### Justification
The mechanism is plausible. At w13 T=3012, master's ssdq takes 157242 cycles per CTA, about 2457 per bf16 stage (KT=64). The SS-wgmma work for one stage is 128x256x64 MACs, about 1000-1050 cycles at about 2048 bf16 MAC/clk/SM, and the consumers wait on full stages 84% of the time while the producer waits on empty stages only about 4%. The producer is therefore the bottleneck. Holding one 16 KB chunk in flight per SM through __ldg-to-register (cur/nxt) puts the weight stream at about 0.85 TB/s chip-wide, which fits a latency-bound producer. A 4-entry TMA ring with up to 3 chunks (54 KB) in flight is the standard fix. It also removes 96 prefetch registers from a kernel that runs at the 168-register cap with a 104 B spill. The profile supports the target: ssdq_kernel is 1878 us (59%) of sliding turn_4 at 3012 us and 1865 us (39%) of full turn_4 at 4686 us, against a ~306 us math floor and a ~224 us byte floor, so there is about 6x headroom. The expected directions agree with TASK: only turn_4 (1024 < T <= 4096) moves, turn_1 stays on Marlin and first_turn on Triton. WEAKEST LINK: the 84% consumer wait shows that the producer is the bottleneck, not why. Three things are mixed together. (a) Global-load latency, which the ring fixes. (b) Dequant issue rate: one warp per scheduler runs about 55 dependent instructions per row-iteration, about 440-600 issue cycles per stage, and that already uses half of the 1000-cycle MMA budget with no other warp to hide its cvt/st.shared latency. (c) The 104 B local spill inside the producer loop. If (b) dominates, the ring buys little and the <40% consumer-wait milestone is missed. Second risk, shared-memory bandwidth: per stage the two SS consumer warpgroups read about 80 KB, the producer stores 32 KB of bf16, the A TMA writes 16 KB, and the ring adds about 9 KB written plus 9 KB read. That is about 146 KB per stage, roughly 1140 cycles at 128 B/clk, so the ~1000-cycle-per-stage target (pair <= 1000 us profiled) is optimistic. A realistic outcome is about 1200-1500 cycles per stage, a pair of roughly 1000-1200 us, which is still a large win over 1878 us. The design itself is sound. The TMA box {64 B, 256 rows} is within limits (box dim <= 256, inner bytes a multiple of 16), the global strides K/2 are multiples of 16, cp.async 8 B is a legal size, and an mbarrier count of 1+128 with cp.async.mbarrier.arrive.noinc is the correct accounting. smem is 1024 + 3x48 KB + 4x18 KB = 222208 B, which fits 227 KB with the 80 B of static mbarriers.

### Confounders on the current kernel
- experts-turn4-ssdq-s1b-dataptr-symbol-fix (direct parent, the incumbent being replaced): not a confounder. ssdq2 sits beside it in the same module, and the in-prepare three-way gate times ssdq2 against master's ssdq on the same inputs in the same process. The hypothesis has to keep master's ssdq_kernel byte-identical for that gate to be honest (see open questions on STAGES).
- stack-tonight-five-prefill (router/align/zero-store/Marlin silu-mul): no interaction. The ssdq path uses _align_parallel at block 128 and _silu_rows, and neither changes. The router GEMM runs upstream of the experts and is identical in both arms.
- experts-hybrid-marlin-turn-triton-first-turn / experts-triton-w4a16-first-turn-blockt256: these only affect T > 4096 (first_turn) and the Marlin fallback. They are not on the turn_4 path, so no interaction.
- sglang-pipeline-parity: this is the surrounding pipeline. It is identical in both arms and has no effect on the replaced kernel's own time.

### Interference and A/B needs
No companion is warranted. This is kind: rewrite, judged on the replaced kernel's own time, and ssdq2_kernel gets its own name in profile top_kernels, so its microseconds can be read per shape against ssdq's 1878/1865 us. The in-prepare three-way gate is already the A/B against the incumbent. There is one hidden sub-change. The hypothesis says 'keep bf16 STAGES = 3', but master's ssdq uses STAGES = 4 (4 x 48 KB). Going to 3 is forced by the smem budget (4 x 48 KB + 4 x 18 KB > 227 KB), so it cannot be avoided with RAW_STAGES = 4. A control at STAGES = 4 with RAW_STAGES = 1 or 2 would not fit, or would gut the change. The printed counters already settle it: if ssdq2's producer empty-stage wait rises sharply above master's ~6353 cycles, the shallower bf16 ring is binding, and if it stays small, it is not. Ask for one extra counter instead of a companion run: the producer's cycles waiting on raw_full, next to its empty-stage wait and its total. Then a single run separates load-latency-bound (raw_full wait large before the change, small after) from dequant-ALU-bound (consumers still waiting while the producer barely waits on anything). That reading decides stage 3: a second producer warpgroup or setmaxnreg if ALU-bound, a deeper ring if still load-bound.

### Open questions
- STAGES/SMEM_BYTES/THREADS are namespace constexprs shared by ssdq_kernel, ssdq_info and the static_assert. ssdq2 needs its own constants (STAGES2 = 3, RAW_STAGES = 4, SMEM2 = 222208) and its own static bars array. Master's ssdq_kernel must keep STAGES = 4, or the 'master ssdq' arm of the three-way gate is no longer master. Does the builder know the hypothesis text misstates master's STAGES as 3?
- ssdq_info only calls cudaFuncSetAttribute(MaxDynamicSharedMemorySize) for ssdq_kernel. Does ssdq2 get its own opt-in to 222208 B before the canary? Without it the launch fails with invalid-argument, which the canary would report as a build or launch failure, not as a performance result.
- Proxy ordering on refill: the producer reads entry c%4 with generic ld.shared, then bar.sync on named barrier 1 (128 threads), then thread 0 issues a TMA into that entry, and all threads issue cp.async (generic proxy) for the scales. Is a fence.proxy.async needed between the generic reads and the async-proxy TMA write? CUTLASS relies on mbarrier ordering alone, but the bisect in LESSONS (full-verify-wgmma-s4) shows how expensive a silent ring race is to find. Put the fully synchronised variant behind a flag in the canary.
- The TMA tensor over bank['w13'|'w2'] viewed as [E*N, K/2] uint8: does every 256-row box start at a 256-row boundary inside one expert? N % 256 == 0 is TORCH_CHECKed, so yes. Is the bank tensor's base 16 B aligned after the nibble reorder (TMA requires it)?
- Will the profile show ssdq2_kernel with the producer's raw_full wait counter, so the 'consumer wait share < 40%' milestone and the load-vs-ALU split can be read from one run?

