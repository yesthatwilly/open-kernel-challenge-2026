---
problem: muse_attention_decode
slug: merge-perkvhead-c320-sv32-tiled
created: 2026-10-25T190812Z
base: master
area: full decode_1 split-KV + sliding verify_32 pre-attention
change: >
  Hand-merge two gated pieces of combo-tiled-preattn-neighbour-regate-c320 (branch
  hyp/muse_attention_decode/combo-tiled-preattn-neighbour-regate-c320, impl 5add0d92, scored
  1.5199 in job 52cbfaebab9646f2) onto current attention master 5d528bf3
  (sliding-decode1-cluster-per-kvhead, 1.5120). That branch was built on master 7e128915, before
  per-kvhead landed; a 3-way merge (base 7e128915) has 8 conflicts. Port exactly: (1) the chunk320
  full decode_1 path: the separate muse_full_decode_c320 module, its sequential main-thread build
  phase (flags -O3 -arch=sm_90a), GATE A correctness and timing, and the route flag that replaces
  master's two-level split-KV at full decode_1 only when GATE A passes (ratio <= 0.98); (2) tiled
  pre-attention at SV32 only (sliding verify_32), with its GATE B exactness and timing gate. Leave
  every other tiled class, the neighbour regate of sliding decode B=2..32, and the dropped skF
  regate out; those classes keep master's per-kvhead and sk-window routing unchanged. Per-kvhead's
  sliding decode_1 cluster path stays as on master. Every ported phase runs after master's own
  setup, with its own deadline and the 480 s import start rule, and falls back to master on
  failure or overrun with one PATH CHOSEN line per piece.
expected:
  - shapes: "*layer=full*load=decode_1*"
    direction: faster
    why: c320 measured 0.894-0.892 of master two-level in b9d062aa and 52cbfaeb
  - shapes: "*layer=sliding*load=verify_32*"
    direction: faster
    why: tiled pre-attention at SV32 0.964 in b9d062aa; regate gained sliding verify_32
  - shapes: "*layer=sliding*load=decode_1*"
    direction: same
    why: "keeps per-kvhead's +9.9% that regate's branch lacked"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T191118Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T191118Z
base_sha: 8eca21f5b391cbe0335e7b32676b0f35d9c5ff9e
base_blob: 6d6afa4722731ceb9cb8216d0334a03f1a96ad2b
branch: hyp/muse_attention_decode/merge-perkvhead-c320-sv32-tiled
worktree: .worktrees/muse_attention_decode/merge-perkvhead-c320-sv32-tiled
mode: null
restack_of: null
stale_base: null
---

# merge-perkvhead-c320-sv32-tiled

## Change

Hand-merge two gated pieces of combo-tiled-preattn-neighbour-regate-c320 (branch hyp/muse_attention_decode/combo-tiled-preattn-neighbour-regate-c320, impl 5add0d92, scored 1.5199 in job 52cbfaebab9646f2) onto current attention master 5d528bf3 (sliding-decode1-cluster-per-kvhead, 1.5120). That branch was built on master 7e128915, before per-kvhead landed; a 3-way merge (base 7e128915) has 8 conflicts. Port exactly: (1) the chunk320 full decode_1 path: the separate muse_full_decode_c320 module, its sequential main-thread build phase (flags -O3 -arch=sm_90a), GATE A correctness and timing, and the route flag that replaces master's two-level split-KV at full decode_1 only when GATE A passes (ratio <= 0.98); (2) tiled pre-attention at SV32 only (sliding verify_32), with its GATE B exactness and timing gate. Leave every other tiled class, the neighbour regate of sliding decode B=2..32, and the dropped skF regate out; those classes keep master's per-kvhead and sk-window routing unchanged. Per-kvhead's sliding decode_1 cluster path stays as on master. Every ported phase runs after master's own setup, with its own deadline and the 480 s import start rule, and falls back to master on failure or overrun with one PATH CHOSEN line per piece.

## Expected

- `*layer=full*load=decode_1*` faster: c320 measured 0.894-0.892 of master two-level in b9d062aa and 52cbfaeb
- `*layer=sliding*load=verify_32*` faster: tiled pre-attention at SV32 0.964 in b9d062aa; regate gained sliding verify_32
- `*layer=sliding*load=decode_1*` same: keeps per-kvhead's +9.9% that regate's branch lacked

## Risk

User-requested (2026-10-25 ~19:05Z: 'do the attention hand-merge and submit it'). Runs as a scored submit; checks are spent. Expected to stack per-kvhead (1.5120) with regate's two live wins (1.5199 without per-kvhead). ## Conjecture: per-kvhead's sliding decode_1 clusters and c320's full decode_1 kernel never ran in the same layer graph; they touch different layers, so L2 working sets should not interact, and the gain should be additive. Read the PATH CHOSEN lines for full decode_1 [A chunk320], SV32 tiled and sliding decode_1 per-kvhead.

## Adversarial review

Reviewed 2026-10-25T191118Z by an adversarial subagent (model opus).

**Verdict: proceed.** Justified, safe merge of two independently reproduced row wins (c320 at full decode_1, about -3.7 us, and tiled pre-attention at SV32, about -1.0 to -1.4 us) onto master's per-kvhead (sliding decode_1 10.87 us). Prediction is about 1.53x (range 1.51-1.55), roughly 1% over the 1.5199 board entry, which is inside the CI. The per-shape rows and PATH CHOSEN lines show attribution directly, so no companion is needed. The one real hazard is a merge slip that silently brings back pre-per-kvhead sliding routing.

### Justification
A. The mechanism is measured, not inferred. Full decode_1, weight 16: master per-kvhead (4b608df5) is 22.70 us per shape (sk two-level 19.1 us plus pre-attention 1.4 us). c320 was 19.00 us in 52cbfaeb and 19.76 us in b9d062aa (splitkv_attn_kernel<320> 10.9 us, combine 4.4 us, pre-attention 1.3 us), and GATE A ratios of 0.894 and 0.892 reproduce across two builds. Per-kvhead touched only sliding code (sliding_cluster_kernel PERHEAD, _slide_*, and the _sd_setup_body route.pop(1)), and GATE A compares against the same two-level kernel, so the c320 ratio should carry over. SV32, weight 46: master is 28.83 us with untiled pre-attention at 4.2 us. Tiled runs measured 27.42 and 27.92 us, with _pre_attention_tiled_kernel at 2.3-2.6 us. Sliding decode_1, weight 46: per-kvhead is 10.87 us against 12.04 and 12.62 us in the two branches without it, and the merge keeps it. Arithmetic: the decode_1 load goes from 863.2 to about 804 weighted us (+7.4%), and the verify_32 load from 6167 to about 6102 (+1.06%). The geomean over 6 loads gives 1.5120 x 1.0120 x 1.0018, about 1.533x. The expected shapes and directions match the weights. Full decode_1 is still 4.4x its 4.3 us floor, so there is room left. The conjecture's L2 argument is moot rather than wrong: every shape is its own captured graph, timed from cold L2, so the two layers never share a graph. Weakest link: the SV32 piece is worth about 0.18% of headline, too small to show in the headline and only readable in its row (expect about 27.4-27.9 us against 28.83). The headline gain also rests almost entirely on c320, which is about 1.2%, so a run of 1.51-1.52 would be ambiguous at headline level without the rows. The second weak link is merge fidelity: 8 conflicts in setup and route orchestration.

### Confounders on the current kernel
- sliding-decode1-cluster-per-kvhead (master 5d528bf3): disjoint shapes, but the same import orchestration. Its _sd_setup_body does route.pop(1, None) before freezing _SD_ROUTE, and _slide_init sets _SLIDE_PH. The regate branch snapshots and republishes _SD_ROUTE (_NB_SNAP, MappingProxyType(new_sd)). If that publish or restore is ported, it can overwrite or restore a table built without per-kvhead's changes. Since the neighbour regate is dropped, the port must not touch _SD_ROUTE at all. A wrong merge would show as sliding decode_1 at about 12 us instead of about 10.9 us, or as a missing 'PERHEAD ON' line.
- full-decode1-streamk-s3-two-level-ticket (two-level sk kernel at FD1): this is the incumbent c320 replaces and the GATE A comparator. It is unchanged by per-kvhead, so the 0.89 ratio is not confounded. If GATE A falls back, the row stays at about 22.7 us.
- fused-preattention-triton / paired-heads-launch-cleanup (master _pre_attention_kernel): the tiled kernel replaces this first node only at SV32 (unpaired, sliding). GATE B compares [tiled + rest] against [master + rest] on the merged master, so the gain is self-attributed.
- restack-sliding-decode-sk-window-on-cluster-master (sk window at sliding decode B=17..32): regate's GATE B also probed SD32 tiled and kept master. Leaving every non-SV32 tiled class out keeps this untouched. Any SD32 movement would be noise or a merge error.

### Interference and A/B needs
No companion is warranted. Both ported pieces are shape-gated to rows that per-kvhead does not touch (FD1, SV32), and per-kvhead's only row (SD1) is kept unchanged. Each piece also prints its own PATH CHOSEN line with an in-run A/B ratio, and the profile shows which kernels ran per row. A gain or a loss can therefore be read row by row against three existing references: master 4b608df5 (FD1 22.70, SV32 28.83, SD1 10.87), regate 52cbfaeb (19.00 / 27.42 / 12.04) and naive b9d062aa (19.76 / 27.92 / 12.62). An ab-revert of per-kvhead would only reproduce 52cbfaeb. A control for c320 alone or SV32 alone would cost a scored run to learn what the row already shows.

### Open questions
- Builder checklist for the 8-conflict merge: (1) do not port the _SD_ROUTE publish or snapshot entries, or the neighbour-regate phases P4 and P6. (2) Confirm the log shows '[slide] PATH CHOSEN: sliding B=1 PERHEAD ON', 'PATH CHOSEN: full decode_1 [A chunk320]: route (... ratio <= 0.98)', and 'GATE B SV32: tiled first node', with every other GATE B class on master. (3) Run okc-style local lint, and if possible a local import, to confirm the 480 s start rule and fallback paths.
- Readout rule for the scored run: success means FD1 at about 19-19.8 us, SV32 at about 27.4-27.9 us, and SD1 at about 10.9 us. If the headline lands inside [1.497, 1.543] but those rows hold, the stacking conjecture is confirmed even though the headline is not a statistically proven best. If SD1 is about 12 us, the merge reverted per-kvhead routing; that is a build error, not a result.
- Bookkeeping: BEST.json lists score_us 2490.98 and CI [1.4893, 1.5347], which are master 4b608df5's numbers, against job 52cbfaeb, whose own values are 2477.92 and [1.4971, 1.5427]. Fix this before comparing the merge to the board.
- No scored test covers full decode at B=1 (the tests cover B=4 full decode and B=1/2 full verify). The c320 path is exercised only by GATE A and by the scored shape's per-call reference check. Both of those passed twice, but a c320 regression would surface only at the scored shape.

