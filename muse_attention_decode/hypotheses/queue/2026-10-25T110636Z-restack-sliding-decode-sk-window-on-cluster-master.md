---
problem: muse_attention_decode
slug: restack-sliding-decode-sk-window-on-cluster-master
created: 2026-10-25T110636Z
base: master
area: sliding-layer attention (decode, stream-K windowed kernel) restack
change: >
  Restack sliding-decode-sk-window-route (done/neutral 1.4973x [1.475,1.520], job
  118c6de9704146f3; built on the unlanded s1 branch, base blob 76492911) onto current master
  eb1b68b (restack-sk-pseudo-cluster-plus-v3-fused-preattn: sliding verify B=1 pseudo-decode,
  sliding cluster C=16 at decode B=1 and C=4 at decode B=17..32, full-decode FUSED_PRE skF).
  Parent's measured win: sliding decode_32 23.85 -> 17.38 us shape (profile: sk TOK=1 kernel 13.9
  us + _pre_attention_kernel 1.9 us), routed by PATH CHOSEN at B=17..32 only (<4 stages, ticket,
  PSEUDO TOK=1> grid p4 = 4B CTAs, 1.313x vs Triton); B=2..16 fell back to Triton (1.006x < 1.05x)
  and B=1 is record-only. Master's B=17..32 incumbent is now cluster C=4 at 20.6 us shape (kernel
  18.1 us), so the pair should still win by ~1.15x. MERGE (git apply of the parent patch onto
  eb1b68b checked in a scratch copy: 10 of 14 hunks apply cleanly, 4 reject): CLEAN, keep
  verbatim: the MappingProxyType import; SK_MAX_VARIANT 5 -> 7 under SK_PSEUDO; load_prefix_pseudo
  gains template <int TOK> (TOK 8 keeps the >>3 / &7 / B*8 forms, byte-identical for variants 4/5
  that master routes at sliding verify B=1); splitkv_attn_sk_kernel gains trailing int TOK = 8;
  sk_pick variants 6/7 (<4,true,false,true,1> / <4,true,true,true,1>); sk_prepare done[8]; sk_attn
  tok / R / two-level check for variant 7; _SK_LABELS 6/7; _ps_synthetic(..., L=8). Note _SKF_HEAD
  slices _SK_CUDA up to "template <int STAGES, bool TICKET", so the skF TU now carries the
  templated load_prefix_pseudo uninstantiated; it must still compile (confirm the skF PATH CHOSEN
  line). REJECTED, merge by hand: (1) kernel_fn sliding-decode branch. Master has `if
  _slide_gate(cfg): return _slide_forward(data, _slide_clusters(cfg))` then `return
  _sliding_forward(data)`; insert the parent's `if _sd_gate(cfg): return _sliding_decode_sk(data)`
  FIRST, above _slide_gate, so a band present in the frozen _SD_ROUTE takes [pre-attn + sk TOK=1],
  and every other sliding decode band keeps master exactly (cluster C=16 at B=1, Triton at
  B=2..16, cluster C=4 at B=17..32 when not routed). (2) The _SD_ block (_SD_ROUTE, _sd_gate,
  _sd_grid, _sd_sk_launch, _sd_run, _sliding_decode_sk, _sd_case, _sd_validate, _sd_interleaved,
  _sd_gate_band, _sd_replay_check, _sd_setup): place at module level after master's _ps_setup
  (context differs, content unchanged except the gate below). (3) The _sd_setup() call: master
  builds the cluster module LAST (_slide_init() at the end of the import block). Call _sd_setup()
  AFTER _slide_init(), in its own try/except that sets _SD_ROUTE = MappingProxyType({}) and prints
  PATH CHOSEN on a raise, so _SLIDE_MOD is final when the gate times the incumbent and when
  _sd_replay_check captures kernel_fn. _sd_setup already returns early (Triton/cluster kept) when
  not _PS_BUILT or _EXT2 is None. (4) Final PATH CHOSEN wording, trivial. GATE CHANGE (the only
  behavioural edit): _sd_gate_band must time the band's real master incumbent, not Triton alone:
  incumbent = _slide_forward(data, _slide_clusters(cfg)) when _slide_gate(cfg) else
  _sliding_forward(data), in the same interleaved cold-L2 graph replay with the empty graph
  subtracted, and also print Triton. Route B=17..32 only if the best pair beats the incumbent
  (cluster C=4) by >= 1.03x; keep the parent's >= 1.05x vs Triton for B=2..16; B=1 stays
  record-only (cluster C=16 kept). Replace the _SD_CLUSTER_C4_US 18.0 constant print with the
  measured cluster number. PATH CHOSEN prints pair, cluster and Triton. Everything else, including
  the two-state replay check, the self-resetting _ps_ticket/_ps_gticket and the TOK=1 self-check
  cases, stays as the parent ran it. RULES (every path): CUDA-graph capturable; kernel_fn does no
  host sync, no allocation sized by input values, no data-dependent Python control flow, no
  prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q
  at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child
  processes; two states (copies: 2) with no module-global device pointers into a state.
expected:
  - shapes: "*layer=sliding load=decode_32 *"
    direction: faster
    why: "parent routed B=17..32 to [pre-attn + sk TOK=1] p4: 17.38 us shape (kernels 13.9 + 1.9) vs master cluster C=4 20.6 us shape (kernel 18.1); ~-15% on a weight-46 row, ~+2% headline"
  - shapes: "*layer=sliding load=decode_16 *"
    direction: flat
    why: "parent's gate kept Triton at B=2..16 (pair 16.2 vs 16.3 us net, 1.006x < 1.05x); byte-identical master Triton"
  - shapes: "*layer=sliding load=decode_1 *"
    direction: flat
    why: record-only band; master cluster C=16 kept
  - shapes: "*layer=sliding load=verify_*"
    direction: flat
    why: "TOK=8 variants 4/5 constant-fold to master's code; FA3 / pseudo routes unchanged"
  - shapes: "*layer=full *"
    direction: flat
    why: PSEUDO=false and skF instantiations untouched
risk: low
depends_on: sliding-decode-sk-window-route
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# restack-sliding-decode-sk-window-on-cluster-master

## Change

Restack sliding-decode-sk-window-route (done/neutral 1.4973x [1.475,1.520], job 118c6de9704146f3; built on the unlanded s1 branch, base blob 76492911) onto current master eb1b68b (restack-sk-pseudo-cluster-plus-v3-fused-preattn: sliding verify B=1 pseudo-decode, sliding cluster C=16 at decode B=1 and C=4 at decode B=17..32, full-decode FUSED_PRE skF). Parent's measured win: sliding decode_32 23.85 -> 17.38 us shape (profile: sk TOK=1 kernel 13.9 us + _pre_attention_kernel 1.9 us), routed by PATH CHOSEN at B=17..32 only (<4 stages, ticket, PSEUDO TOK=1> grid p4 = 4B CTAs, 1.313x vs Triton); B=2..16 fell back to Triton (1.006x < 1.05x) and B=1 is record-only. Master's B=17..32 incumbent is now cluster C=4 at 20.6 us shape (kernel 18.1 us), so the pair should still win by ~1.15x. MERGE (git apply of the parent patch onto eb1b68b checked in a scratch copy: 10 of 14 hunks apply cleanly, 4 reject): CLEAN, keep verbatim: the MappingProxyType import; SK_MAX_VARIANT 5 -> 7 under SK_PSEUDO; load_prefix_pseudo gains template <int TOK> (TOK 8 keeps the >>3 / &7 / B*8 forms, byte-identical for variants 4/5 that master routes at sliding verify B=1); splitkv_attn_sk_kernel gains trailing int TOK = 8; sk_pick variants 6/7 (<4,true,false,true,1> / <4,true,true,true,1>); sk_prepare done[8]; sk_attn tok / R / two-level check for variant 7; _SK_LABELS 6/7; _ps_synthetic(..., L=8). Note _SKF_HEAD slices _SK_CUDA up to "template <int STAGES, bool TICKET", so the skF TU now carries the templated load_prefix_pseudo uninstantiated; it must still compile (confirm the skF PATH CHOSEN line). REJECTED, merge by hand: (1) kernel_fn sliding-decode branch. Master has `if _slide_gate(cfg): return _slide_forward(data, _slide_clusters(cfg))` then `return _sliding_forward(data)`; insert the parent's `if _sd_gate(cfg): return _sliding_decode_sk(data)` FIRST, above _slide_gate, so a band present in the frozen _SD_ROUTE takes [pre-attn + sk TOK=1], and every other sliding decode band keeps master exactly (cluster C=16 at B=1, Triton at B=2..16, cluster C=4 at B=17..32 when not routed). (2) The _SD_ block (_SD_ROUTE, _sd_gate, _sd_grid, _sd_sk_launch, _sd_run, _sliding_decode_sk, _sd_case, _sd_validate, _sd_interleaved, _sd_gate_band, _sd_replay_check, _sd_setup): place at module level after master's _ps_setup (context differs, content unchanged except the gate below). (3) The _sd_setup() call: master builds the cluster module LAST (_slide_init() at the end of the import block). Call _sd_setup() AFTER _slide_init(), in its own try/except that sets _SD_ROUTE = MappingProxyType({}) and prints PATH CHOSEN on a raise, so _SLIDE_MOD is final when the gate times the incumbent and when _sd_replay_check captures kernel_fn. _sd_setup already returns early (Triton/cluster kept) when not _PS_BUILT or _EXT2 is None. (4) Final PATH CHOSEN wording, trivial. GATE CHANGE (the only behavioural edit): _sd_gate_band must time the band's real master incumbent, not Triton alone: incumbent = _slide_forward(data, _slide_clusters(cfg)) when _slide_gate(cfg) else _sliding_forward(data), in the same interleaved cold-L2 graph replay with the empty graph subtracted, and also print Triton. Route B=17..32 only if the best pair beats the incumbent (cluster C=4) by >= 1.03x; keep the parent's >= 1.05x vs Triton for B=2..16; B=1 stays record-only (cluster C=16 kept). Replace the _SD_CLUSTER_C4_US 18.0 constant print with the measured cluster number. PATH CHOSEN prints pair, cluster and Triton. Everything else, including the two-state replay check, the self-resetting _ps_ticket/_ps_gticket and the TOK=1 self-check cases, stays as the parent ran it. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values, no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes; two states (copies: 2) with no module-global device pointers into a state.

## Expected

- `*layer=sliding load=decode_32 *` faster: parent routed B=17..32 to [pre-attn + sk TOK=1] p4: 17.38 us shape (kernels 13.9 + 1.9) vs master cluster C=4 20.6 us shape (kernel 18.1); ~-15% on a weight-46 row, ~+2% headline
- `*layer=sliding load=decode_16 *` flat: parent's gate kept Triton at B=2..16 (pair 16.2 vs 16.3 us net, 1.006x < 1.05x); byte-identical master Triton
- `*layer=sliding load=decode_1 *` flat: record-only band; master cluster C=16 kept
- `*layer=sliding load=verify_*` flat: TOK=8 variants 4/5 constant-fold to master's code; FA3 / pseudo routes unchanged
- `*layer=full *` flat: PSEUDO=false and skF instantiations untouched

## Risk

Approved by the user as a scored submit (attention has no checks left today). The CUDA is the parent's, already run on the judge (16/16, no fault); the merge is Python routing and setup order. Main risk: the gate against cluster C=4 lands under 1.03x in prepare and B=17..32 stays on cluster, so the run measures master (neutral). Read the B=32 [sd-gate] lines (pair vs cluster vs Triton) and the decode_32 kernel rows, not the headline alone. Confirm the skF PATH CHOSEN line so full decode rows are not on the v2 fallback.
