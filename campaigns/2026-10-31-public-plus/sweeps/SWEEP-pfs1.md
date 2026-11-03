# SWEEP-pfs1: S1/A2 small-tile co-schedule on PFC2L (muse_prefill_layer), design 2026-11-02 ~02:40Z

Read-only design. No H100 run, no okc call, no repo write. Files written: this document and two analysis scripts in
`/tmp/claude-1000/-home-user-git-musespark/08efaca7-cd01-4c08-87f6-da5f73e85a3c/scratchpad/pfs1/` (`rho_pfs1.py`,
`floors_pf.py`). Numbers are re-read from the raw records in §1.1; anything derived from the shared-memory/tensor
budget is marked **model**.

Base: PFC2L-r1 (`.worktrees/public-plus/campaigns/2026-10-31-public-plus/files/combos/PFC2L-r1.py`, sha256 00e1e775…;
body without line 2 sha256 c696b8a9d24a8b46f016f779a2823f007b3563aaf930afb0d2753597a1bf8e20, identical for r1/r2/r3, so
a landed PFC2L master is the same base).

## 0. Answer

1. **S1/A2 is still the only prefill lever worth more than +1% headline if it works, and it now has a build that fits
   today's kernel**: 384 threads, setmaxnreg 40/232/232 unchanged, no 4th warpgroup, a separate kernel template
   (`moe_fused_s1<FINE, KP>`) routed by T only, so every incumbent kernel stays SASS-identical. **EV today is low:
   about +0.09% headline** (gain if it works ~+1.2% central × p ~0.07 end to end, the 11:00Z H100 window included).
2. **The producer-converter form named in the task does not pass the survival gate on the shared-memory budget.**
   - WG0's idle warps 2-3 dequantize small tiles into smem; the consumers run SS-wgmma between host stages.
   - Each converted small k-block then moves 53 KB through shared memory: the bf16 A operand is written once (16 KB)
     and read once by wgmma (16 KB), on top of the incumbent small path's 21 KB.
   - With that cost, an N=128 host gains nothing (host stretch ≈ the small k-block's own cost), so only N=256 items
     can host. The judge-replica plan at T=753 (d0) has 2×256 and 6×128 big tiles against 10 small tiles: N=256
     hosting covers 20% of the small k-blocks, and NSLOT 3 (needed for the converted ring) costs t1s big +1.95% (S0).
   - **Model: +0.27% H; H100 t1s/t4s combined ≈ +0.5%, below the 1.5% survival bar.** It stays in the sweep
     (#4/#5) because M0 measures the one thing that would rescue it: wgmma A-operand reads costing less smem than
     modelled.
3. **The same kernel with a consumer-fold engine (CF) is the variant to build.**
   - The consumers convert the paired small k-block in registers and issue RS wgmma right after the host's k-block
     group. There is no converter and no bf16 copy in smem, so the smem traffic is the incumbent small path's 21 KB.
   - NSLOT 4 is kept by trimming the slot to 41,984 B and the staging buffer to 34,816 B.
   - N=128 items can host: 60% coverage at t1s d0 and 100% at t4s.
   - **Model, central: +1.19% H (t1s +3.2%, t1f +2.2%, t4s +0.9%, t4f +0.6%), H100 combined ≈ +2.2%.**
   - Risks: (a) registers: the pair path sits at ~170-215 against the 232 cap, and there is a C7512 history whenever
     small-path code was added; (b) consumer-instruction latency: public FACTS §7 says the T=753 expert kernel is
     consumer-instruction bound. M0 measures both.
   - **CF is not barred by any reopen rule.** It is not a 4th WG (wg4, CLAIMS:94), not real-row N (PF-RR, CLAIMS:93),
     and not a separate small kernel (SK2, CLAIMS:95; SK2's reopen says "inside a co-resident S1/A2 design").
   - S0's "folding into the consumers is register-blocked" was inferred from the kernel-wide max R219/R229, not
     measured on a pair path. It deviates from the form the task named, so the main thread can veto it; #4/#5 are the
     named form.
4. **Diagnostic rung first (M0, §3, pre-registered).**
   - M0a: CPU-only compile matrix on gpunode06's build host (wg4cc pattern), zero GPU.
   - M0b: H100 oracle on GPU 6 (~40 GPU-min). Dummy small k-blocks go into the host k-blocks of the D1 big arm, per
     engine. It measures the host stretch per co-run small k-block (Δpair), the converter rate and small-stream
     latency hiding.
   - Build only on M0 GO, otherwise KILL with a CLAIMS line. No other form fits this window.
5. **Ranked variants.** Each is base + the S1 build + named knobs; generator `mk_pf-s1.py` (§2.11). "Survives" is the
   model's H100 verdict against the task's bar.

   | # | name | knobs (S1_ENGINE / hosts T753 / hosts T3012 / scope) | judge H central [range] | survives? | p | gain × p |
   |---|---|---|---:|---|---:|---:|
   | 1 | **S1CF-128** | cf / {128} / {256} / G1+G2 | **+1.19%** [+0.6, +1.8] | yes (H100 comb ≈ +2.2%) | 0.074 | **0.088%** |
   | 2 | S1CF-WIDE | cf / {128, 256} / {256} / G1+G2 | +1.43% [+0.7, +2.1] | yes (≈ +2.6%) | 0.057 | 0.082% |
   | 3 | S1CF-G1 | cf / {128} / {256} / G1 only | +0.79% [+0.4, +1.2] | marginal (≈ +1.4%) | 0.060 | 0.047% |
   | 4 | S1SS-256 (task's form) | ss / {256} / {256} / G1+G2, NSLOT 3 | +0.27% [0, +0.9 if smem cheap] | no (≈ +0.5%) | 0.015 | ~0.01% |
   | 5 | S1LR-256 (task's form, LDS-fed) | lr / {256} / {256} / G1+G2, NSLOT 3 | +0.25% | no | 0.012 | ~0.01% |

   - p = P(M0a clean) × P(M0b GO) × P(build + correct + H100 survive + audit by 11:00Z) × P(judge keeps ≥ half).
   - #1: 0.45 × 0.55 × 0.35 × 0.85. #2: 0.35 × 0.55 × 0.35 × 0.85.
   - #1 and #2 share one source and differ by one `#define`, so **build #1, and screen #2 too if it compiles
     clean**. #3 is #1 with one define. Up to 3 configurations get screened: {#1, #2 or #3, and #4 only if M0b shows
     SS Δ ≤ 0.5 × c_solo}.
6. **Timeline (written 02:45Z; window ends 11:00Z).**
   - M0 code + runs: 02:45-04:45Z.
   - Build: 04:45-08:45Z.
   - H100 screen: 08:45-09:45Z.
   - Audit H100 reruns: 09:45-10:45Z.
   - Checks go out after the audit (main thread).
   - If M0 has no verdict by ~05:30Z, there is no build today: write the CLAIMS line with the M0 numbers.

## 1. Evidence base

### 1.1 Measured (sources)

**Prize and per-item costs.**
- Judge diag GD-PF-D1 af53426a (H200, master P2; `campaigns/2026-10-31-public-plus/runs/af53426ac0af4eeb.GD-PF-D1.log`):

  | row | all | big | small | prize |
  |---|---:|---:|---:|---:|
  | t1s | 281.6 | 210.9 | 97.4 | 70.7 |
  | t1f | 246.8 | 176.8 | 98.2 | 70.0 |
  | t4s | 794.8 | 725.1 | 107.0 | 69.6 |

- Per-item µs, all arm, t1s:
  - GEMM1: r32 11.4, r64 16.8, r128 24.2, r256 42.5;
  - GEMM2: r32 11.6, r64 16.2, r128 22.5, r256 41.0.
- t4s is within ±1 µs of t1s per width.
- So one k-block costs about **0.18 µs at N=32, 0.258 at N=64, 0.36 at N=128 and 0.65 at N=256** (64 k-blocks per
  item).
- H100 D1 (wg4 baseline 20261101-004410, PF-RR -221340, S0 -182655), µs:

  | row | big | small | all |
  |---|---:|---:|---:|
  | t1s | 216.0 | 109.8-110.8 | 295.6-297.7 |
  | t1f | 184.1 | 110.1 | 262.0 |
  | t4s | 758.2 | 120-122.6 | 824.8 |

**Tensor and issue budget.**
- S0 (`…/47d181e3-…/scratchpad/pf-s0.md`): the big phase is tensor-bound. Tensor pipe busy 81.3 / 91.3 / 84.9%
  (t1s / t4s / t1f); issue-active 34 / 28 / 32%.
- Tensor work is additive (big 256.2k + small 59.6k = all 315.8k cycles per SM at t1s).
- Small needs 1.01× / 0.65× / 1.56× of big's idle tensor time.
- NSLOT 3 vs 4, big-only: t1s +1.95%, t4s +0.33%, t1f +1.04%; small items +6.7..+7.8%.
- Consumer SASS max R219 (<0>) / R229 (<1>); launch 168 registers, 0 local.

**What is closed.**
- PF-RR (CLAIMS:93): adding any wgmma width to the fused kernel serializes it (C7512); a runtime width branch inside
  the k-loop gives ~100 × C7519. Small-only is not tensor-bound in time.
- wg4 (CLAIMS:94): 512 threads cap ptxas at 128 registers per instruction (C7602 on m64n256k16).
- SK2 (CLAIMS:95): per-SM small throughput is fixed (true load-free t1s small 97.6 µs). A sequential split costs
  +17..31 µs. Any fix needs co-residency.

**Plans on judge-like draws.** From takemyjob `.takemyjob-wt/pf-kperm/…/runs/20261031-101308-gpunode06/out/pfk2-gatet1.txt`
and siblings (tiles; items = tiles × 96):

| row | d0 plan | d1 plan |
|---|---|---|
| t1s (T=753, fine0, KPERM) | {256:2, 128:6, 64:3, 32:7} | {256:5, 128:1, 32:11} |
| t1f | {256:2, 128:5, 64:2, 32:6} | {256:2, 128:8, 64:3, 32:4} |
| t4s (T=3012, fine1) | {256:20, 128:4, 64:4, 32:5} | {256:18, 192:1, 160:2, 128:4, 64:2, 32:6} |
| t4f | {256:18, 192:2, 128:3, 64:2, 32:8} | {256:18, 160:3, 128:3, 64:3, 32:4} |

**Public facts** (`~/git/okc-muse-spark-kernels/docs/H200_MEASURED_FACTS.md` §7): "The T = 753 expert kernel is
consumer-instruction bound. Removing loads gains 3%. Removing 12 instructions per k-block gains 5.8%." This is the
main risk for CF, which adds consumer instructions inside host k-blocks.

**PFC2L on the judge** (checks 56bfbb47 1.7871 / 7827cda6 1.7967; recomputed by `pfs1/rho_pfs1.py`):
- Row means: t1s 1.9340, t1f 1.6169, t4s 1.8656, t4f 1.5411, ftf0 1.8683, fts 1.8916, ftf32 1.6528, ftf64 1.5654.
- Our time µs: t1s 557.9, t1f 921.8, t4s 1800.3, t4f 3253.2.
- moe_fused share: t1s 284-287 µs (49%), t1f 250-291, t4s 777-781, t4f 729-813.
- prepare_s 47-51 s.

### 1.2 Model: shared-memory and tensor cost of one paired small k-block

Shared-memory traffic per k-block, KB (128 weight rows × 64 K; 2 consumer WGs; B is read once per WG):

| path | TMA write X+W | LDS W | bf16 A write | wgmma A read | wgmma B read | total |
|---|---:|---:|---:|---:|---:|---:|
| incumbent small RS, N=32 | 8.6 | 4.5 | 0 | 0 (regs) | 8 | **21.1** |
| CF (pair), N=32 / N=64 | 8.6 / 12.6 | 4.5 | 0 | 0 | 8 / 16 | **21.1 / 33.1** |
| SS (producer-converter), N=32 / N=64 | 8.6 / 12.6 | 4.5 | 16 | 16 | 8 / 16 | **53.1 / 65.1** |
| LR (converter + consumer LDS), N=32 | 8.6 | 4.5 + 16 | 16 | 0 | 8 | 53.1 |
| host k-block N=256 / N=128 | 36.6 / 20.6 | 4.5 | – | – | 64 / 32 | 105.1 / 57.1 |

**Assumptions, all stated:**
- Shared memory sustains 128 B/clk/SM, about 90% achievable in a mixed stream.
- wgmma operand fetch goes through that same port. *M0b tests this assumption.*
- Host in-loop time is ~0.62 µs (N=256) and ~0.34 µs (N=128) per k-block at ~1.83 GHz, about 72% shared-memory busy.
- Δpair = pair-step time minus host time, where pair-step time = max(tensor / 0.95, smem bytes / (0.9 × 128)).

| host | small | Δpair, CF | saving s, CF | Δpair, SS | saving s, SS | c_solo |
|---|---|---:|---:|---:|---:|---:|
| N=128 | N=32 | 0.03 µs | 0.15 (83%) | 0.18 µs | ≈ 0 | 0.18 |
| N=128 | N=64 | 0.10 | 0.16 (61%) | 0.26 | ≈ 0 | 0.258 |
| N=256 | N=32 | 0.04-0.09 | 0.09-0.14 | 0.12-0.13 | 0.05-0.06 | 0.18 |
| N=256 | N=64 | 0.16-0.19 | 0.07-0.10 | 0.17-0.19 | 0.07-0.08 | 0.258 |

The variant rows below use CF at an N=256 host s = 0.087 / 0.072, the conservative end. For SS they use s = 0.063 /
0.083, slightly above the table's range: that flatters #4/#5, and they still fail.

**Coverage** (paired small items / all small items; G1+G2 pairing, R=1 small k-block per host k-block):

| host set at T=753 | t1s d0 | t1s d1 | t1f d0 | t1f d1 |
|---|---:|---:|---:|---:|
| {128} | 60% | 9% | 62.5% | 100% |
| {128, 256} | 80% | 55% | 87.5% | 100% |
| {256} | 20% | 45% | 25% | 29% |

T=3012 with {256} hosts: 100% on every draw. Draws are weighted d0-like 0.75 / d1-like 0.25, as in SWEEP-pf §1.

**Per-row µs saved (judge).** central = 0.5 × model saving − measured costs (NSLOT 3: t1s 3.2, t1f 1.5, t4s/t4f 2.2 µs
judge-scaled from S0). Headline sensitivity %H/µs (PFC2L times, 3 loads): t1s 0.0379, t1f 0.0132, t4s 0.0114,
t4f 0.0040, ft rows ≤ 0.0025.

| variant | t1s | t1f | t4s | t4f | H |
|---|---:|---:|---:|---:|---:|
| #1 S1CF-128 | 17.2 (+3.2%) | 20.0 (+2.2%) | 16.8 (+0.94%) | 19.5 (+0.60%) | **+1.19%** |
| #2 S1CF-WIDE | 22.6 (+4.2%) | 23.0 (+2.5%) | 16.8 | 19.5 | +1.43% |
| #3 S1CF-G1 | 11.5 (+2.1%) | 13.4 (+1.5%) | 11.2 (+0.62%) | 13.0 (+0.40%) | +0.79% |
| #4 S1SS-256 | 1.1 (+0.2%) | 1.8 (+0.2%) | 12.9 (+0.72%) | 14.1 (+0.43%) | +0.27% |

- Ranges in §0 are 0.25×..0.75× of the model saving.
- The unhaircut t1s and t4s figures stay under S0's tensor-bound prize (judge-scaled ~63 µs t1s, ~37 µs t4s). t4f
  has no prize measurement.

### 1.3 Unmeasured rates (what M0 measures)

1. Δpair per engine, host width and small width. The model says CF Δ ≈ 0.03-0.16 µs and SS Δ ≈ 0.12-0.26 µs per
   small k-block; the incremental solo cost is c_solo ≈ 0.18 / 0.258 µs (N=32 / N=64). This is the whole verdict.
2. Converter throughput: two warps at 40 registers, LDS → convert8 → STSM, while hosts run. Need ≥ 1 k-block per
   host k-block (~300 warp-instructions per warp per 0.62 µs, IPC ~0.25).
3. Whether a 2-slot small ring (TMA X+W per k-block, small W L2-prefetched ≥ 4 k-blocks ahead) hides latency at
   0.34-0.65 µs pair steps.
4. Register feasibility of each pair path: host NACC 1 vs 2 at N=128; small N=32 / N=64; hosts {128}, {128, 256} or
   {256}.
5. Whether SS m64nNk16 gives bit-identical output to RS (the same bf16 operands, k16 order and accumulator).

## 2. The S1 kernel (shared by every variant; builder spec)

### 2.1 Routing and instantiations
- New `__global__ moe_fused_s1<FINE, KP>`. The engine, host sets and scope are `#define`s that the generator bakes in.
  It is launched by `k_moe_fused` when `fa.flags` bit 10 is set.
- Python routes by T only: `S1_T = (753, 3012)`.
- T=753 → `moe_fused_s1<0, 1>` (KPERM consumer, as PFC2L's `<0, 1>`).
- T=3012 → `moe_fused_s1<1, 0>` (fine plan).
  - Fallback knob `S1_FINE3012 = 0` uses the fine0 plan and `moe_fused_s1<0, 0>`. That means fewer solo widths and
    less C7512 risk, at the cost of the fine plan's ~+0.06% (FACTS §7). M0a compiles both.
- T=16384 keeps `moe_fused_persistent<1, 0>` with order 2 (PFC2L's rider) unchanged.
- The incumbent `<0>`, `<1>`, `<0, 1>` and every attention kernel must be SASS byte-identical between the S1 .so and
  the PFC2L .so (sasscmp.py from the SK2 lane).

### 2.2 Roles (384 threads, `__launch_bounds__(384, 1)`, setmaxnreg 40 / 232 / 232 as today)
- WG0 warp 0 lane 0: the TMA producer. It now drives two rings (§2.5).
- WG0 warp 1: the publisher, unchanged. It also copies small-item tiles out.
- WG0 warps 2-3: **converters (ss/lr engines only)**; they exit at once under cf, as today.
- WG1-2: consumers. They run solo items with the incumbent `consumer_item<N>` code, unchanged, and paired items with
  `consumer_item_pair<NB, NS, KP>`.

### 2.3 Work queues and pairing (deadlock-free by construction)
- **Plan.** It is order 1 (biggest first: tiles [0, B) are the big ones, N ≥ 96, and [B, B+S) the small ones, N ≤ 64,
  with 64 before 32) plus `ntiles[1] = B`. This needs a 2-int `ntiles` and `order |= 1 << 10` read in `plan_warp`.
  Change `fine = order >> 8` to `(order >> 8) & 1`.
- **Four counters** in zb[0..3]: big GEMM1 (B×64 items), small GEMM1 (S×64), big GEMM2 (B×32) and small GEMM2 (S×32).
  `done` moves to zb[4 : 4 + maxtiles], and kernel_fn's zlen grows by 3. The boundary kernel already zeroes all of zb.
- **Producer order:**
  1. Big G1 item. If its width is a host width, pull one small G1 item and pair them.
  2. When big G1 is exhausted: the remaining small G1 items run solo.
  3. Big G2 item, paired with a small G2 item (G1+G2 scope only).
  4. The remaining small G2 items run solo.
- **Pairs never cross GEMM phases.** Any G2 item waits only on G1 items. A CTA moves to G2 only after both G1 counters
  are exhausted, and G1 work never waits. So there is no wait cycle. (Pairing big G1 with small G2 can deadlock:
  X(big G1)+Y(small G2) waits on W(small G1), and W is paired with Z(big G2), which waits on X.)
- **Spread (default on, the order-2 lesson).** When (small left) > (hosts left), from relaxed reads of the counters,
  the producer pulls one extra solo small item. Otherwise all leftovers queue up as a small-item tail at each phase end.
- A pair's G2 dependency spin waits on both tiles' `done`, as the incumbent does per item.

### 2.4 Shared-memory map (227 KB request unchanged)
- `SLOT_S1` / `STG_S1` are S1-only constants. The incumbent's `SLOT` 43,008 and `STG_BYTES` 45,056 stay as they are,
  so its SASS stays byte-identical.
- cf: base 1024 + 4 × SLOT 41,984 (fits every incumbent stage: N=128 KPS2 41,984; N=256 37,376; N=32 KPS4 34,816;
  fine 160/192 25,088/29,184) + STG 34,816 (the fused kernel's largest use: 128 × 272 B GEMM2 half) + small ring
  2 × 13,312 (N=64 X 8,192 + W 4,608, rounded up to a multiple of 1,024 B) = 230,400 B + pad ≤ 1,023 → ≤ 231,423 ≤
  232,448.
- ss/lr: NSLOT 3 + converted ring 2 × 16,384 (two m64 halves of 8 KB, SW128 K-major for ss, fragment order for lr) →
  221,184 B. That pays S0's NSLOT-3 cost.
- The barriers fit in base[96, 240): small full/empty ×2, converted full/empty ×2, small-staging-free, and a
  converter job ring.

### 2.5 Producer scheduler
- It must never block on one ring while the other has a free slot. It polls both empty barriers with
  `mbarrier.test_wait` and issues in need order.
- The host and small streams are consumed in lockstep (R = 1 small k-block per host k-block).
- It issues `cp.async.bulk.prefetch.L2` for the small W 4 k-blocks ahead (the incumbent's MOE_L2PF covers 1 stage).
- It refills the small ring for a new pair only after `small-staging-free` (§2.8).

### 2.6 Consumer pair path `consumer_item_pair<NB, NS, KP>`
- **Host N=256** (non-GROUP4, per-k16 commit): the incumbent loop.
  - After each host k-block's four k16 steps, issue the paired small k-block as one commit group of 4 small wgmmas,
    then continue. Every wait in the loop is wait<1>.
  - Small buffers are released (empty arrive) after the next wait<1>.
- **Host N=128** (GROUP4, KPS 2): per host k-block: [host group of 4] [commit] [wait<1>: the previous small group is
  done] [small group] [commit] [wait<1>: the host group is done] [convert next host k-block].
  - **Host NACC = 1 in pair mode**: acc 64 instead of 64 + 64. NACC 2 plus the small state would reach ~222-238
    registers. M0a tries both.
- **Small engine:**
  - **cf**: 4 convert8 into a single-buffered 16-register `safrag`, then 4 × `wgmma_rs<NS>`.
    - This reuses the incumbent RS instruction, with LDS of the small slot's codes/scales via `ld_sc<KP>`.
    - Accumulator `sacc[NS/2]` with NACC = 1.
  - **ss**: wait `converted-full[c]`, then 4 × `wgmma.mma_async…m64nNSk16.f32.bf16.bf16 {sacc}, descA(conv[c] + wg·8 KB)
    + 2j, descB(small X) + 2j, p, 1, 1, 0, 0`. No `safrag`.
  - **lr**: 4 × LDS.128 from the converted slot, then `wgmma_rs<NS>`.
- **Register estimate:**

  | pair path | ~registers |
  |---|---:|
  | host 256 + cf, N=32 / N=64 | 200 / 216 |
  | host 128 (NACC1) + cf, N=32 / N=64 | 170 / 186 |
  | host 256 + ss | 190 / 206 |

  The cap is 232. History says ptxas may still serialize (C7511/C7512), so M0a decides.
- **New code paths per kernel:**
  - #1: 2 pairs per kernel: <128, 32>, <128, 64> at T=753 and <256, 32>, <256, 64> at T=3012.
  - #2: 4 at T=753.
- No runtime branch around a wgmma inside a k-loop (C7519). Pair vs solo is chosen per item, outside the loop, like
  today's switch on `ti.z`.

### 2.7 Converter warps (ss / lr only)
- Each converter warp owns one WG half (warp 2 → rows 0-63, warp 3 → rows 64-127).
- Per small k-block:
  1. wait `small-full[s]` and `converted-empty[c]`;
  2. per row-group q: LDS.128 codes (lane offset `(q·32 + lane)·16`) and KPERM scales, then 4 × (`scale_pair_sel` +
     `convert8` → `stmatrix.sync.aligned.m8n8.x4` to the SW128 address `(row/8)·1024 + (row%8)·128 +
     ((kchunk ^ row%8)·16)` (ss), or STS.128 in fragment order (lr));
  3. `fence.proxy.async.shared::cta` (ss only: generic writes, then async-proxy wgmma reads);
  4. arrive `converted-full[c]` and the W share of `small-empty[s]`.
- 40 registers is enough: ~25-30 live.
- The converter job sequence comes from a 2-entry converter job ring the producer writes, with its own mbarrier.

### 2.8 Small epilogue and staging
- At pair end (R = 1, so both items end together): `wait<0>`, then the **small epilogue first** (sacc dies), then the
  host epilogue (incumbent code). This keeps the peak register count at the host epilogue's.
- The small staging is the small ring, which is free after the last small k-block: GATED N ≤ 64 is 8 KB, GEMM2 N=64
  is 64 × 272 = 17,408 B, and the ring has 26,624 B.
- The publisher copies it out (kind 0 bulk / kind 1 token rows, as today) and releases `small-staging-free`. The
  producer waits on that before the next pair's first small TMA.

### 2.9 Numerics and the identity test
- FP32 accumulation and k16 order are unchanged.
- Paired small items and N=128 hosts in pair mode use one accumulator (NACC = 1), which is the incumbent's
  `MOE_NACC2 = 0` association.
- The incumbent itself is not bitwise run to run: atomic gather placement moves a row between tile sizes, and so
  between NACC paths.
- **Bit identity, the strongest clean test.** Build S1 and PFC2L both with `#define MOE_NACC2 0`. Then y must be
  bitwise equal on:
  - every D1 arm;
  - the real routings of all 8 judge rows (identical p / lid / pos, as in PF-RR);
  - 20 repeats per routing, so that different dynamic pairings are exercised.
- **Production bytes:** RELDIFF against PFC2L within PFC2L's own run-to-run spread, plus check_implementation.
- SS ≡ RS per element: a one-kernel bitwise unit check, part of M0b.

### 2.10 Python-side edits (anchored; each must match once in PFC2L-r1)
- After line 131 `MOE_ORDER_LARGE = 2  # [SW] was 1: order 2 also at T >= MOE_LARGE_T`, insert:
  `S1_T = (753, 3012)  # [S1] token counts routed to moe_fused_s1 (small-tile co-schedule; shape-only); () = off`
- kernel_fn: `zlen = LOCAL_EXPERTS + 1 + maxtiles + ((maxtiles + 4) if use_cit else 0)` → append
  `+ (3 if T in S1_T else 0)`.
- `_moe_forward`:
  - `s1 = T in S1_T`;
  - `ntiles = torch.empty((2 if s1 else 1,), …)`;
  - with s1: `order = 1 | (fine << MOE_FINE_ORDER_SHIFT) | (1 << 10)`;
  - pass `zb[0:4] if s1 else zb[0:1]` and `zb[4:4 + maxtiles] if s1 else zb[1:1 + maxtiles]`;
  - flags `| (int(s1) << 10)`.
- `assert not (s1 and cit is not None)`: CIT is off.
- prepare:
  - `k_moe_init` sets `MaxDynamicSharedMemorySize` for the S1 instantiations.
  - **Validation gate**: run `_moe_forward` with S1 on a skewed synthetic routing (the CLAIMS:83 rule) at T=753 and
    T=3012 and compare y to the incumbent within 2× PFC2L's run-to-run RELDIFF. On a miss, set `S1_T = ()` for the
    process. Shape-only, and nothing is timed.
- The SELFTEST block (off) needs no change.

### 2.11 Generator
- `python3 -I campaigns/2026-10-31-public-plus/files/combos/mk_pf-s1.py <PFC2L-r1.py> <ARM> <n> <out.py>`.
- ARM ∈ {S1CF128, S1CFWIDE, S1CFG1, S1SS256, S1LR256}.
- It asserts that the input's body sha256 (line 2 removed) is c696b8a9…, and adds the line-2 arm text + nonce.
- It inserts the Python edits above. It inserts `s1_cuda.cu` (the kernel, with that arm's `#define S1_ENGINE`,
  `S1_HOST128`, `S1_HOST256_T753`, `S1_SCOPE` and `S1_FINE3012`) after `moe_plan_kernel`, plus the GLUE dispatch and
  init lines.
- Every anchor is asserted to match once; `compile()` must pass.
- Branch `campaign/pf-s1` (`.worktrees/pf-s1`, from campaign/public-plus); never master.
- The M0 oracle reuses the same file with `#define S1_ORACLE <mode>` (§3.2).

## 3. M0: diagnostic rung (pre-registered; nothing is built past it without GO)

### 3.1 M0a: compile matrix (CPU only)
- Pattern: takemyjob `hopper/h100-pf-s1-wg4-oracle` `probes/gmpf/wg4cc.py` (macro-override variants → .cu → `nvcc
  -Xptxas -v`, `cuobjdump -res-usage`, `sassreg.py`) on the gpunode06 build host. No local nvcc exists on this
  workstation.
- Matrix: engine {cf, ss, lr} × kernel {s1<0, 1> hosts {128} / {128, 256}; s1<1, 0> and s1<0, 0> hosts {256}} × small
  pairs {N=32 only, N=32 + 64} × host NACC {1, 2} (N=128 hosts only). About 30 compiles, minutes.
- **Pass per cell:** 0 spill; no C7507, C7510, C7511, C7512, C7519 or C7602; SASS max register ≤ R231; incumbent
  kernels byte-identical.
- **KILL S1:** cf fails in every T=753 cell, AND ss fails in every T=3012 cell. The CLAIMS line costs no GPU time.

### 3.2 M0b: H100 oracle (gpunode06 GPU 6 only, hopper-probe rules, ~40 GPU-min)
- **Kernel.** `moe_fused_s1o` = the S1 kernel's real host loops, smem map and register footprint, with dummy small
  k-blocks. It is built from the same `s1_cuda.cu` with `S1_ORACLE`.
- **Dummy small k-blocks.** Inserted after every host k-block (hosts per the arm):
  - cf: LDS the host slot's own W lane word + scales into separate registers, then convert8 ×4 and `wgmma_rs<NS>` ×4
    into `sacc`, with B = the host slot's first NS X rows.
  - ss: `wgmma_ss<NS>` ×4 with A from a 16 KB dummy region.
  - lr: LDS.128 ×4 from that region, then `wgmma_rs<NS>`.
  - `sacc` is folded into a per-CTA dummy sum written to global (no dead-code elimination).
  - The dummy regions fit at NSLOT 4 (SLOT 41,984 + STG 34,816 frees 27.6 KB), so no NSLOT confound.
- **Converters (ss/lr arms).** Warps 2-3 free-run LDS → convert8 → STSM / STS into the dummy region until the
  consumers set an exit flag, and count k-blocks per CTA.
- **Arms**, on the D1 big-only arm at t1s and t4s (t1f reported) and the D1 all arm for the reference:

  | arm | content |
  |---|---|
  | A0 | PFC2L incumbent |
  | A1 | oracle off: SLOT/STG trim, separate instantiation (control) |
  | A1b | A1 with N=128 host NACC 1 (cost of the pair-mode host NACC) |
  | A2 | cf, N=128 host, NS 32 |
  | A3 | cf, N=128 host, NS 64 |
  | A4 | cf, N=256 host, NS 32 |
  | A5 | cf, N=256 host, NS 32, with a real small stream: a 2-slot small ring loading real small-expert W/X by TMA + L2 prefetch, consumed in lockstep (latency hiding) |
  | A6 | ss, N=256 host, NS 32, with converters |
  | A7 | ss, N=128 host, NS 32, with converters |
  | A8 | lr, N=256 host, NS 32, with converters |
  | U | SS ≡ RS bitwise unit check (m64n32k16 and m64n64k16, random bf16, both orders) |

  Drop A7/A8 if time is short. A0-A6 decide.
- **Protocol (mandatory):**
  - ≥ 4 byte-identical module copies per arm (2 processes × 2 copies);
  - load order shuffled per process from a seeded permutation, with process B reversed;
  - copy-pooled statistics: each copy's same-round ratio against the mean of A1's copies, median + 95% bootstrap CI;
  - GPU-bound CUDA-graph replay with the host enqueuing the whole round ahead;
  - a 256 MiB read flush before every call;
  - n ≥ 28 per row and copy;
  - A/A: A1's copy spread within ±0.25%, else extend the rounds.
  - Placement noise is up to ~1 µs (0.17% at t1s), far below the effects M0 has to resolve.
- **Outputs:**
  - Δ_x,r = (T(Ax) − T(A1)) / (dummy small k-blocks per SM, from the per-CTA counts);
  - c_solo,r = (T_all − T_big)(A0) / (small k-blocks per SM), per width, from per-item stamps;
  - the converter rate in k-blocks/µs per SM against the demand;
  - A5 − A4: the small-stream penalty;
  - the plan histogram of each routing;
  - ncu for one t1s run each of A1 / A2 / A6 (tensor %, `lts__t_bytes`, shared-pipe %).

### 3.3 Decision rule (fixed now)
- **Predicted H100 gain per row:** G_x,r = Σ_widths n_pair,r · (c_solo,r − Δ_x,r)
  − (NSLOT-3 cost for ss/lr, from S0) − (A1b − A1 host-NACC cost, cf at N=128) − (A5 − A4 penalty).
  - n_pair comes from the arm's host set on d0 (0.75) and d1 (0.25) plans.
- **GO (build engine x):**
  - M0a passes for the needed instantiations;
  - geomean(1 + G_x,t1s / T_t1s, 1 + G_x,t4s / T_t4s) ≥ 1.025 (H100);
  - G_x,t1f ≥ −0.3%;
  - converter rate ≥ 1.25 × demand (ss/lr).
  - The 1.0-point margin over the 1.5% survival bar covers what the oracle cannot see: sync, epilogue
    serialization and the leftover tail.
- **KILL:** no engine reaches 1.010, or M0a fails for every engine. The CLAIMS line: [refuted] S1-PC / S1-CF, with
  the Δ, c_solo and M0a numbers.
- **SHELF:** the best engine predicts 1.010-1.025. No build today; record the numbers.

## 4. Variants (ranked by expected judge gain × p; all on the PFC2L-r1 body, all via `mk_pf-s1.py`)

**Rows that must hold their floors (every variant).**
- Touched: t1s, t1f (T=753) and t4s, t4f (T=3012). The 2-check and scored-read floors are in §5.
- Untouched, SASS-identical: ftf0, fts, ftf32, ftf64. These are a no-harm guard: ≥ 2-check floors.
- The H100 survival bar applies as well (§6).

### #1 S1CF-128: `mk_pf-s1.py … S1CF128 <n>`
- **Knobs:** `S1_ENGINE cf`, `S1_HOST128 1`, `S1_HOST256_T753 0`, `S1_SCOPE 2`, NSLOT 4, R = 1, host NACC 1 in pair
  mode.
- **Judge prediction (central, µs saved):** t1s −17.2 (+3.2%), t1f −20.0 (+2.2%), t4s −16.8 (+0.94%), t4f −19.5
  (+0.60%), ft 0. **H +1.19%** (→ ~1.813 on PFC2L's 1.7919 mean).
  - I_S1 ≈ +1.9% (§5 defines I_S1).
  - Predicted rows: t1s ~1.996, t1f ~1.652, t4s ~1.883, t4f ~1.550.
- **H100 prediction:** t1s +3.3%, t4s +1.0%, combined ≈ **+2.2%** (passes the 1.5% bar); t1f +2.2%, t4f +0.6%.
- **Watch rows:**
  - t1s depends on the routing draw: coverage is 60% at d0 but 9% at d1 (one N=128 tile), so a d1-like judge draw
    gives ~0.
  - t1f carries the host-NACC change.

### #2 S1CF-WIDE: `S1CFWIDE` = #1 + `S1_HOST256_T753 1` (one define)
- N=256 items also host at T=753. That is 4 pair paths in `s1<0, 1>`.
- Coverage: t1s d0 80%, d1 55%.
- **Prediction:** t1s −22.6 (+4.2%), t1f −23.0 (+2.5%), t4 as #1. **H +1.43%**; H100 combined ≈ +2.6%.
- Lower M0a pass odds (more instantiations). Screen it only if every s1<0, 1> cell for it passes M0a.

### #3 S1CF-G1: `S1CFG1` = #1 + `S1_SCOPE 1`
- GEMM1 items pair only. That is simpler: no G2 pair epilogue (kind 1 token rows) and no paired dependency spin.
- **Prediction:** t1s +2.1%, t1f +1.5%, t4s +0.62%, t4f +0.40%. **H +0.79%**; H100 combined ≈ +1.4% (marginal against
  the 1.5% bar).
- It is the fallback if the G2 pair epilogue cannot be made correct in time.

### #4 S1SS-256 (the task's producer-converter form): `S1SS256`
- **Knobs:** `S1_ENGINE ss` (WG0 warps 2-3 convert to SW128 bf16, consumers SS-wgmma), hosts {256} at both T,
  `S1_SCOPE 2`, NSLOT 3 + converted ring.
- **Prediction:** t1s +0.2%, t1f +0.2%, t4s +0.72%, t4f +0.43%. **H +0.27%**; H100 combined ≈ +0.5%. **Fails the
  survival bar on the model.**
- Screen it only if M0b shows Δ_ss ≤ 0.5 × c_solo at an N=128 host (A7), that is, wgmma A reads much cheaper than
  128 B/clk. With N=128 hosting allowed, its model rises to ≈ #3.

### #5 S1LR-256: `S1LR256` = #4 with `S1_ENGINE lr`
- The converter writes fragment-order bf16, and consumers LDS.128 it into the incumbent RS wgmma. No proxy fence;
  bit identity is trivial.
- The same shared-memory traffic as #4 plus consumer LDS issue. **H +0.25%.** It is screened only on the same M0b
  condition as #4.

Not in this sweep: R = 2 small k-blocks per host k-block (t1s coverage → 100%, H ≈ +1.67% model). cf needs +16-32
pinned `safrag` registers (≈ 225+), and ss/lr become smem-bound (Δ > c_solo). Reopen it only if M0a shows ≥ 24 free
registers on the #1 pair paths.

## 5. Floors (improved floor forecasts) and the judge gate

**Gate metric (every gate below):** rho_row = the geomean over this variant's check reads of that row's
**`paired_ratio`** from `okc status <job> --json` (`result.records[]` with `event == "benchmark"`), divided by R0_row.
It is **NOT the median of the per-rep `ratios`**. Rho sets (all8, t1, t4, ft) are geomeans of rows.
- The S1 index: I_S1 = Σ w_r ln(rho_r) / Σ w_r over the touched rows, with w = 46 (t1s, t4s) and 16 (t1f, t4f).
- R0 = PFC2L's reads (now 2 checks).
- **Recompute R0 and every floor with `pfs1/floors_pf.py` at dispatch time**, adding PFC2L's 3 queued scored reads and
  PFC2's 2 checks (identical code at T < 8192).
- Floors are noise-scaled with z = 2.5, the SWEEP-dec2 §1.3 method, using pooled same-code sd (master 4, PFC1 3,
  PFC2L 2 reads; df = 6):
  - 2-check floor = R0 exp(−2.5 sd √(1/2 + 1/n));
  - scored-read floor = R0 exp(−2.5 sd √(1 + 1/n)).

| row | R0 PFC2L (n=2) | sd/read | 2-check floor | scored-read floor | old KILL (master min / 1.025) | PFC2L µs | %H per µs |
|---|---:|---:|---:|---:|---:|---:|---:|
| t1s | 1.9339 | 2.16% | 1.8325 (−5.2%) | 1.8104 (−6.4%) | 1.8142 | 557.9 | 0.0379 |
| t1f | 1.6168 | 1.39% | 1.5618 (−3.4%) | 1.5497 (−4.2%) | 1.5501 | 921.8 | 0.0132 |
| t4s | 1.8656 | 0.77% | 1.8299 (−1.9%) | 1.8220 (−2.3%) | 1.8047 | 1800.3 | 0.0114 |
| t4f | 1.5411 | 0.79% | 1.5109 (−2.0%) | 1.5043 (−2.4%) | 1.4814 | 3253.2 | 0.0040 |
| ftf0 | 1.8683 | 0.84% | 1.8293 (−2.1%) | 1.8207 (−2.5%) | 1.7877 | 9742.2 | 0.0003 |
| fts | 1.8916 | 0.69% | 1.8591 (−1.7%) | 1.8519 (−2.1%) | 1.8284 | 8933.6 | 0.0025 |
| ftf32 | 1.6528 | 0.93% | 1.6149 (−2.3%) | 1.6065 (−2.8%) | 1.5987 | 12755.3 | 0.0003 |
| ftf64 | 1.5654 | 0.52% | 1.5454 (−1.3%) | 1.5409 (−1.6%) | 1.5064 | 15969.8 | 0.0003 |

- Set references (PFC2L): rho_all8 1.7352, rho_t1 1.7683, rho_t4 1.6956, rho_ft 1.7389.
- Pooled sd per read: all8 0.74%, t1 1.70%, t4 0.49%, ft 0.57%.
- I_S1 sd per read ≈ 0.875% (weighted), so the 2-read I_S1 against an n=2 R0 has sd ≈ 0.88%.
- t1s is the noisiest row (2.16%: routing redraws move moe_fused 230-290 µs). A real +3% t1s is only ~1.4σ on 2 reads.
  The gate therefore keys on I_S1 and the landing rule, not on t1s alone.

**Judge gate (2 checks, same hour; metric = geomean of per-row `paired_ratio`, not the median of `ratios`):**
- **Route check** (profiles/submission.<job>.txt):
  - t1s and t1f run `moe_fused_s1<0, 1>`;
  - t4s and t4f run `moe_fused_s1<1, 0>` (or `<0, 0>` if S1_FINE3012 = 0);
  - the ft rows run `moe_fused_persistent<1, 0>`;
  - the stderr has no S1 validation-fallback line.
  - Anything else is KILL.
- **KILL** if any of these holds:
  - a failed test or timed check, a flag or a rejection;
  - prepare_s > 200;
  - mean2 I_S1 < −0.5%;
  - any touched row's mean2 below its 2-check floor;
  - 2+ ft rows below their 2-check floors.
- **GO** if all of these hold, then one scored run of the r1 bytes:
  - mean2 I_S1 ≥ +1.0%;
  - mean2 rho_all8 ≥ 1.7352 (the R0 mean);
  - every row at or above its 2-check floor.
  - P(GO | #1 as predicted) ≈ 0.84; at half the predicted effect ≈ 0.5.
- **HOLD** otherwise: one more check (r3), then the same rule on the 3-read pool, else shelf.
- **Landing** (standing user rule 11-02 00:00Z, "land anything that beats the board"): the scored read passes 5/5
  tests and every timed check, its headline is above the current board (1.7836, or a landed PFC2L read), and every row
  is at or above its scored-read floor.

## 6. H100 screen (survival; gpunode06 GPU 6 only)

- **Harness:** takemyjob `~/git/.takemyjob-wt/sw-pf/prototypes/hopper-probe/probes/pfsweep` (`combo.py`,
  `run-pfsweep.sh`; untracked on hp-pfsweep). It already has the full-layer per-row protocol.
  - Arms: PFC2L-r1 (control) + up to 3 S1 configurations (#1, then #2 or #3, then #4 only on its M0b condition).
- **Protocol:** as in §3.2 (≥ 4 copies, shuffled/reversed, copy-pooled, graph replay, 256 MiB flush).
  - Draws: judge-like d0, s11, s22, s33 plus adverse d1.
  - t1s/t1f/t4s/t4f at n = 28 on all 5 draws; ftf0/fts/ftf32/ftf64 at n = 14 on d0 and s33 (an A/A guard: code
    identical).
- **Survival (fixed now):** rho_H100(row) = the copy-pooled median of t_PFC2L / t_arm, pooled over the 4 judge-like
  draws.
  - **Pass if:**
    - geomean(rho_H100(t1s), rho_H100(t4s)) ≥ 1.015 with its bootstrap CI lower bound ≥ 1.005;
    - every one of the 8 rows ≥ 0.995;
    - d1 t1s and t1f ≥ 0.990 (adverse-draw guard);
    - the control A/A is within ±0.25%.
  - Otherwise the configuration dies (CLAIMS shelf or refuted with numbers).
- **Correctness:**
  - the bit-identity test of §2.9 (NACC2 = 0 builds; 20 repeats per routing);
  - check_implementation eager + graph on all 8 rows at d0 and d1; the 5 TASK tests × 2 seeds;
  - 16k graph replays at t1s and t4s (pair sync stress) and 4k at t1f and t4f;
  - `compute-sanitizer --tool racecheck` and `synccheck` on one t1s and one t4s call (new rings and proxy fences);
  - SASS byte-identity of every incumbent kernel; ptxas: 0 spill, no C75xx.
- **Cost:** ~35-45 GPU-min for 4 arms; correctness ~20 min.
- Then an independent adversarial audit (same GPU, before 11:00Z), then attempt files with the §5 gate verbatim.

## 7. Compliance (S1 against the rules checklist)

- **FA3-precision ruling:** attention code is byte-identical to PFC2L, and FP32 softmax denominators are untouched.
- **Lossless weights:** the same NVFP4 bytes and the same bit-exact convert8. ss/lr hold transient bf16 copies of one
  k-block in smem; nothing is re-encoded or approximated.
- **Shape-only routing:** S1 is chosen by T ∈ S1_T. Pairing is on-device queue scheduling from the plan, the same
  class as today's persistent queue. There is no host sync and no data-dependent host branch.
- **No timing or caching in kernel_fn:** the prepare validation only compares outputs and falls back. Nothing derived
  from `data` persists. It stays CUDA-graph capturable: zb and ntiles sizes are functions of T.
- **load_inline only:** the new code lives in `_CUDA_SRC` / GLUE. No subprocess, network, harness access or reference
  import.
- **Numerics:** FP32 accumulation is kept. NACC = 1 on paired paths is the incumbent's own `MOE_NACC2 = 0`
  association. Bit identity is proven against NACC2 = 0 builds, and RELDIFF stays within the incumbent's run-to-run
  spread.
- **Size:** 305,858 B + ~30-40 KB, well under 1 MB. Lines 1 and 3-4 are unchanged (asserted).
- **Not a re-roll:** new code. Never time another problem's kernels. Never run okc check/submit from a builder.

## 8. Dispatch (for the main thread)

1. **Now:** one builder (effort high) for M0. It writes `s1_cuda.cu` (the S1 kernel with `S1_ORACLE`, §3.2 arms),
   runs M0a on the build host, then M0b on GPU 6.
   - Branches: takemyjob `hopper/pf-s1` (probe under `probes/gmpf`, from hopper/gd-pf b1cb10a plus the sk2pf/rrpf
     cand machinery from hopper/h100-pf-sk2-small-kernel d838f2c), and musespark `campaign/pf-s1`.
   - Report: scratchpad `pf-s1-m0.md`, with the §3.3 verdict computed literally.
2. **On M0 GO:** the same builder completes the production paths for the M0-best engine (#1's code; #2/#3 are
   defines), writes `mk_pf-s1.py`, runs §6, and hands over to an independent audit.
3. **On audit PASS:** attempt files (rules checklist, the §5 gate verbatim, quota from the latest okc record), 2 checks
   of the top survivor, then one scored run on GO.
4. **On KILL/SHELF:** CLAIMS prefill line, appended on master in the main checkout. The slug is `pf-s1-cf` /
   `pf-s1-pc`, with the Δ, c_solo, M0a cells and the reopen condition.

## 9. Not proposed (closed or out of reach today)

| idea | why not |
|---|---|
| 4th converter warpgroup | CLAIMS:94: C7602 at the 512-thread cap; 2×n128 + WG4 → C7507/C7511 |
| real-row N for small tiles | CLAIMS:93: any extra width → C7512 in both kernels; only a tensor-budget enabler, and S1 v1 is not tensor-capped at t4 |
| a separate small kernel or an SM split | CLAIMS:95 (SK2), GAPMAP R5: per-SM small throughput is fixed; a sequential split costs +17..31 µs |
| ss/lr with N=128 hosts or R ≥ 2 | model: smem-bound, Δ ≥ c_solo (§1.2). Reopened by M0b only if wgmma A reads prove cheap |
| pairing across GEMM phases | deadlock (§2.3) |
| an in-prepare timing race S1 vs incumbent | transfer risk (CLAIMS:83, skewed-routing rule); fixed T routing plus a correctness fallback is enough |
| S1 at T = 16384 | 4-8 small tiles among ~130, so the prize is ~0; it would also re-touch PFC2L's ft rider |
| knob-only hedges on PFC2L | SWEEP-pf §5: everything else is at its optimum or adverse; PFC2F/G ≈ PFC2L inside placement noise (PFC2L audit) |

## 10. Files

- This design: `/tmp/claude-1000/-home-user-git-musespark/08efaca7-cd01-4c08-87f6-da5f73e85a3c/scratchpad/SWEEP-pfs1.md`
- R0/rho per group: `/tmp/claude-1000/-home-user-git-musespark/08efaca7-cd01-4c08-87f6-da5f73e85a3c/scratchpad/pfs1/rho_pfs1.py`
- Noise-scaled floors (rerun at dispatch with every PFC2L read added):
  `/tmp/claude-1000/-home-user-git-musespark/08efaca7-cd01-4c08-87f6-da5f73e85a3c/scratchpad/pfs1/floors_pf.py`
- Inputs read:
  - PFC2L-r1 and its attempt file;
  - CLAIMS.md (prefill) :91-96;
  - LOG.md on master (11-01);
  - GAPMAP-1031 §2.3;
  - SWEEP-pf.md;
  - pf-s0.md and pf-rr.md (session 47d181e3);
  - lane-pf-s1-wg4-oracle.md and lane-pf-sk2-small-kernel.md;
  - the GD-PF-D1 log;
  - pfk2-gatet1 ROUTING plans;
  - public H200_MEASURED_FACTS §7;
  - the PFC1/PFC2L/master okc JSONs and profiles.
