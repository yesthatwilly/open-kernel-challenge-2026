---
problem: muse_prefill_layer
slug: fa3-num-splits-cap-gated-turn
created: 2026-10-25T152235Z
base: master
area: FA3 call (num_splits cap on the paged path at q_len <= 4096, per-class timing gate)
change: >
  Built on master (commit 1932a9c4). Only the num_splits argument of master's paged
  flash_attn_with_kvcache call changes, and only at q_len <= 4096 (turn_1 T=753, turn_4 T=3012,
  and the tests). first_turn keeps master's call byte-for-byte. With scheduler_metadata None on
  prefill, FA3 sizes the split on the GPU (varlen dynamic split), and num_splits > 0 acts as a cap
  on it (LESSONS, Attention). The bff8e36788a04917 profile shows FlashAttnFwdCombine (6.9 us) on
  both turn_1 shapes, so FA3 splits there. Sliding turn_1 pays 25.3 us of FA3 plus 6.9 us of
  combine for a ~6.4 us exponential floor: about 64 PackGQA m-blocks of 192 rows over a 2048-key
  band of ~17 key tiles, with a partial write and a combine re-read in fp32. Full turn_1 pays
  417.5 + 6.9 us. Candidate caps per class: 0 (master's heuristic), 1 (no split: no fp32 partials,
  no combine launch), 2 and 4. PREPARE-TIME HARD GATE: for each class (layer sliding and full,
  times turn_1 b1 L753 ctx~80000 and turn_4 b4 L753 ctx~80000), build synthetic FP8 caches with a
  random page-size-1 page table at the scored widths. CUDA-graph capture master's exact call (same
  FP8 q, window, scale) once per cap. Before every replay write a flush buffer of at least 2x L2.
  Alternate the order, take the median of at least 25 replays, and pick the fastest cap only if it
  is <= 0.97 x the cap-0 time; otherwise 0. Key by config only: (window, batch, q_len,
  round((page_table width - q_len) / 16384)). Unknown keys use 0, which is master. SELF-CHECK: the
  chosen cap's output against cap 0 on coherent synthetic data (rel L2 <= 1e-2, all finite); a
  failure sets that class to 0. Any exception leaves every class at 0. Log '[fa3 splits] GATE
  <class>: cap0 X us, cap1 Y us, cap2 Z us, cap4 W us -> N' and one 'PATH CHOSEN' line per routed
  shape on first sight (de-dup set for printing only). kernel_fn passes
  num_splits=state['fa3_splits'].get(key, 0), a Python int from config: graph-safe, no host sync,
  no branching on values. Everything else is master's.
expected:
  - shapes: "*layer=sliding*load=turn_1*"
    direction: faster
    why: "FA3 25.3 + combine 6.9 = 32.2 us of 747.1 us (4.3%). Cap 1 drops the combine launch and the fp32 partial round trip while 64 CTAs walk ~17 key tiles each. Target ~22-26 us: -6 to -10 us, ~1% of the shape at weight 46, ~0.7% of the turn_1 load."
  - shapes: "*layer=full*load=turn_1*"
    direction: flat
    why: 417.5 + 6.9 us over 80K keys. Cap 1 would halve the CTAs, so the gate should keep the heuristic or a cap of 2. At most a few us.
  - shapes: "*load=turn_4*"
    direction: flat
    why: Sliding FA3 61.8 us and full FA3 1643 us, with 4x the m-blocks, are already near a full grid. The gate is expected to keep cap 0.
  - shapes: "*load=first_turn*"
    direction: flat
    why: "Untouched: q_len 16384 keeps master's call byte-for-byte."
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
review: hold
status: hold
review_at: 2026-10-25T152509Z
review_companions: null
hold_reason: "adversarial review: Hold this as a scored submit; the cause is the budget, not the idea. The mechanism is plausible and rules-clean, but the best case is about +0.2% on a headline "
---

# fa3-num-splits-cap-gated-turn

## Change

Built on master (commit 1932a9c4). Only the num_splits argument of master's paged flash_attn_with_kvcache call changes, and only at q_len <= 4096 (turn_1 T=753, turn_4 T=3012, and the tests). first_turn keeps master's call byte-for-byte. With scheduler_metadata None on prefill, FA3 sizes the split on the GPU (varlen dynamic split), and num_splits > 0 acts as a cap on it (LESSONS, Attention). The bff8e36788a04917 profile shows FlashAttnFwdCombine (6.9 us) on both turn_1 shapes, so FA3 splits there. Sliding turn_1 pays 25.3 us of FA3 plus 6.9 us of combine for a ~6.4 us exponential floor: about 64 PackGQA m-blocks of 192 rows over a 2048-key band of ~17 key tiles, with a partial write and a combine re-read in fp32. Full turn_1 pays 417.5 + 6.9 us. Candidate caps per class: 0 (master's heuristic), 1 (no split: no fp32 partials, no combine launch), 2 and 4. PREPARE-TIME HARD GATE: for each class (layer sliding and full, times turn_1 b1 L753 ctx~80000 and turn_4 b4 L753 ctx~80000), build synthetic FP8 caches with a random page-size-1 page table at the scored widths. CUDA-graph capture master's exact call (same FP8 q, window, scale) once per cap. Before every replay write a flush buffer of at least 2x L2. Alternate the order, take the median of at least 25 replays, and pick the fastest cap only if it is <= 0.97 x the cap-0 time; otherwise 0. Key by config only: (window, batch, q_len, round((page_table width - q_len) / 16384)). Unknown keys use 0, which is master. SELF-CHECK: the chosen cap's output against cap 0 on coherent synthetic data (rel L2 <= 1e-2, all finite); a failure sets that class to 0. Any exception leaves every class at 0. Log '[fa3 splits] GATE <class>: cap0 X us, cap1 Y us, cap2 Z us, cap4 W us -> N' and one 'PATH CHOSEN' line per routed shape on first sight (de-dup set for printing only). kernel_fn passes num_splits=state['fa3_splits'].get(key, 0), a Python int from config: graph-safe, no host sync, no branching on values. Everything else is master's.

## Expected

- `*layer=sliding*load=turn_1*` faster: FA3 25.3 + combine 6.9 = 32.2 us of 747.1 us (4.3%). Cap 1 drops the combine launch and the fp32 partial round trip while 64 CTAs walk ~17 key tiles each. Target ~22-26 us: -6 to -10 us, ~1% of the shape at weight 46, ~0.7% of the turn_1 load.
- `*layer=full*load=turn_1*` flat: 417.5 + 6.9 us over 80K keys. Cap 1 would halve the CTAs, so the gate should keep the heuristic or a cap of 2. At most a few us.
- `*load=turn_4*` flat: Sliding FA3 61.8 us and full FA3 1643 us, with 4x the m-blocks, are already near a full grid. The gate is expected to keep cap 0.
- `*load=first_turn*` flat: Untouched: q_len 16384 keeps master's call byte-for-byte.

## Risk

Low risk: one Python int argument of the same FA3 call, chosen by a prepare-time timing gate on synthetic inputs, with master's value as the default and the fallback. Same keys, same precision. Small stake: about +0.2% on the headline if sliding turn_1 moves as expected; read the GATE line and the FA3 + FlashAttnFwdCombine rows of the turn_1 profiles. It also answers, as a measurement, whether the combine is worth keeping at turn_1. FA3 already launches its combine only when the dynamic split is > 1, so 'combine only when splits > 1' is already master's behaviour; the only lever is the cap. pack_gqa is not a lever on this path: FA3 forces PackGQA for paged non-TMA KV (page size 1) at compile time. Independent of fa3-full-contiguous-kv-triton-gather-gated in shapes: that one gates full layers only and expects turn_1 full to stay paged. If both land, restack by hand: they edit the same call site.

## Adversarial review

Reviewed 2026-10-25T152509Z by an adversarial subagent (model opus).

**Verdict: hold.** Hold this as a scored submit; the cause is the budget, not the idea. The mechanism is plausible and rules-clean, but the best case is about +0.2% on a headline whose CI is about +/-1.5%. In roughly 60% of outcomes the gate keeps cap 0 everywhere, so the run would re-measure master. Run it on a check when checks reset, or add its GATE timing as a log-only probe (cap forced to 0) to the next prefill check, which costs nothing.

### Justification
Profile bff8e36788a04917 matches the hypothesis. On sliding turn_1, FA3 is 25.3 us and FlashAttnFwdCombine is 6.9 us, on a 747.09 us shape. Full turn_1 is 417.5 + 6.9 us. Sliding turn_4 has 61.8 us of FA3 with no combine row in its top kernels. The mechanism is that num_splits > 0 caps FA3's varlen dynamic split, which LESSONS (Attention) confirms. Cap 1 makes FA3 skip the split and the combine launch. This is the only real lever: caps 2 and 4 cannot raise the split above the heuristic, so any cap at or above FA3's chosen split is a no-op. The real candidate set is therefore {heuristic, 1}, maybe plus 2 if the heuristic picked more than 2. Full turn_1 is compute-bound at about 590 TFLOP/s (753 x 16 heads x 80K keys x 64 x 4 flops / 417.5 us), matching LESSONS' ~600 TFLOP/s ceiling, so cap 1 there can only lose. The gate will keep 0, as the spec expects. Turn_4 and first_turn being flat is consistent. Stake: 6-10 us x weight 46 = 276-460 us on a turn_1 load of about 52.5 ms (46 x 747 + 16 x 1137). That is 0.5-0.9% of one load and +0.17-0.29% on the 3-load geomean, roughly 5-10x below the CI half-width of master's score (1.426-1.470). WEAKEST LINK: whether cap 1 beats 32.2 us at all. With no split, about 64 PackGQA m-block CTAs sit on 132 SMs and each walks the whole ~17-tile band serially. If the heuristic split was 2, per-CTA work roughly doubles, while the 6.9 us combine is mostly fixed launch and tail cost. If FA3's prologue and epilogue are about 8-10 us, the unsplit kernel lands near 33-40 us, which is a wash or a loss, and the 0.97 gate correctly keeps 0. I put a cap-1 selection at about 35-40%. It is a coin flip whether the run executes anything different from master.

### Confounders on the current kernel
- sglang-pipeline-parity: it defined this exact FA3 call (scheduler_metadata None on prefill, so FA3 uses the dynamic varlen split). The change depends on that path but is not confounded by it; it is the base this cap acts on.
- No other landed hypothesis (stack-tonight-five, the rega s4 ports, align-two-phase, the Triton/Marlin expert stages) touches attention or runs concurrently with FA3. The router side stream forks after the o GEMM, so FA3 and the combine run alone on the GPU. Neither a gain nor a loss on the FA3 and combine rows could come from an interaction.
- Not landed, but the same call site: fa3-full-contiguous-kv-triton-gather-gated (queued) edits this flash_attn_with_kvcache call on full layers. The two are shape-disjoint in intent (turn_1 full stays paged there), but they need a hand restack if both land, as the spec already says.

### Interference and A/B needs
No companion is warranted. Attribution is direct, not inferred from the headline. The '[fa3 splits] GATE' line prints the per-cap timings and the chosen cap for each class, and the turn_1 profile rows (FA3 main kernel plus whether FlashAttnFwdCombine is still present) show the effect in place. Neither a gain nor a loss is ambiguous at the kernel level. At the headline, any reading is ambiguous whatever companion you add, because the effect is far below the CI. A second scored run would buy nothing. That ambiguity is the argument against a scored run in the first place, not for a companion. If the gate logs cap 0 for every class, the run has measured master again and should be filed neutral with the GATE line as the finding (the combine is worth keeping at turn_1).

### Open questions
- Is spending the scored submit on this worth it, or on a rega/experts hypothesis (rega_kernel is 32-36% of every turn shape)? If the slot can go elsewhere, this belongs on a check, or as a log-only probe inside another prefill check: print the GATE timings but force cap 0, so the other run's attribution is untouched.
- What split did FA3's heuristic pick on sliding turn_1 (2, or more)? If it is 2, caps 2 and 4 are no-ops and the only candidate is cap 1. Print FA3's chosen split (the dynamic split count is readable in prepare from the synthetic capture) alongside the GATE line.
- FINDINGS lists sliding turn_1 at 704.4 us while the bff8e profile table shows 747.09 us, two readings of the same master. The spec uses 747.1. That does not change the conclusion, but it shows run-to-run noise on this shape is about 40 us, several times the 6-10 us target, which reinforces reading the kernel rows rather than the shape total.
- Does FA3's prepare_varlen_num_blocks / scheduler kernel still launch at cap 1? If it does, the saving is the combine plus the fp32 partial round trip only, and the prepare kernel's few us remain.

