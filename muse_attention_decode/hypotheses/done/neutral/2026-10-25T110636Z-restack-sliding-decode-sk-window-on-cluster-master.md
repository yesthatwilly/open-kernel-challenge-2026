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
review: proceed
status: done
review_at: 2026-10-25T111218Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T111218Z
base_sha: 8bbf53baef204e1a7c7f2d0b36f6da1996b27db4
base_blob: 84b54ae2c9800a7b9bf96692557dd390904ed09b
branch: hyp/muse_attention_decode/restack-sliding-decode-sk-window-on-cluster-master
worktree: .worktrees/muse_attention_decode/restack-sliding-decode-sk-window-on-cluster-master
mode: submit
restack_of: sliding-decode-sk-window-route
stale_base: null
implemented_at: 2026-10-25T111803Z
impl_commit: 79b7a1b88e9e662b9104a7006cad727ebb252e4b
submission_sha256: ae4af530224f3622f69a1551409a80f6342d9186a7104c609da6ebe969e7f2c3
norm_sha256: 313cc0c77b7e6b68a9111bdcc7ac717c75b52d01e9473448bdff11367615515b
submission_blob: 5cf8fae31e5b84c36efeeb4569763eb9b06e1c8a
lint_warnings: 57
attempt_file: 2026-10-25T111803Z-submit-restack-sliding-decode-sk-window-on-cluster-master.md
prepared_at: 2026-10-25T111803Z
job: 252abb7f11ec48c3
dispatched_at: 2026-10-25T111810Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.5016522435853301
speedup_ci: [1.47912745993155, 1.530423257736894]
harvested_at: 2026-10-25T115709Z
best_at_harvest: 1.4919706177473366
promotion: null
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

## Adversarial review

Reviewed 2026-10-25T111218Z by an adversarial subagent (model opus).

**Verdict: proceed.** Safe as a scored submit. It reuses the parent's judge-proven CUDA (16/16 tests, no fault) and changes only Python routing, with one hard per-band gate whose result is printed. The win is real but small: about -3 us on the weight-46 sliding decode_32 row, worth about +0.5% headline (not the +2% claimed). That is inside the ±1.5% CI, so judge the run on the decode_32 sliding row and the [sd-gate] B=32 lines, never on the headline.

### Justification
Mechanism is plausible and already measured. On the judge (job 118c6de9704146f3) the pair [pre-attn + sk TOK=1, ticket, grid p4=128] ran sliding decode_32 at 17.38 us shape (sk 13.9 + pre-attn 1.9 us kernels). Master's cluster C=4 runs it at 20.64 us shape (kernel 18.1 us) in cbd49dc94be548e7 and 20.40 us in restack-sk-pseudo-b1-cluster-c16-c4, so the expected row gain is about 1.17-1.19x. The floor is 3.3 us, so the pair (16.8 MB in a 13.9 us kernel, about 1.2 TB/s) is still about 5x off SoL; there is headroom but no physics problem. The other expected directions are consistent: decode_16 stays on Triton (parent's gate measured 1.006x < 1.05x), decode_1 is record-only (cluster C=16 kept), sliding verify keeps TOK=8 variants 4/5 (they constant-fold to master), and full rows are untouched. ERROR IN THE HYPOTHESIS: the '~+2% headline' figure is wrong. Master's decode_32 load is 46*20.64 + 16*227.88 = 4595 us. With 17.38 us it becomes 4446 us, which is 1.034x on one of six loads, so the geomean moves by 1.034^(1/6) = +0.55% (about 1.492 -> 1.500). That is inside the ±1.5% CI. WEAKEST LINK: the in-prepare 1.03x gate against cluster C=4. The parent's gate measured the pair at 19.3 us net (first pass) and 18.8 us (second pass), about 1.5-2 us above its judge shape time, because the gate charges the two-node boundary and launch differently from the judge. Cluster C=4 has never been timed in this gate harness. If it is inflated less than the pair (it is one node), the gate ratio may come in nearer 1.05-1.10x than the judge's 1.18x. It should still clear 1.03x. The risk is that it lands below 1.03x and the scored run silently measures master. A secondary weakness: the parent picks the best of 12 candidates from one noisy pass (winner's curse) and computes its ratio from that same pass. Routing on that against a thin 1.03x margin is biased toward routing.

### Confounders on the current kernel
- restack-sk-pseudo-cluster-plus-v3-fused-preattn (master eb1b68b): its cluster C=4 is now the incumbent this route must beat at B=17..32. The parent only ever gated against Triton (25.4 us net), so the parent's 1.313x does not carry over. Expect about 1.1-1.2x against cluster.
- restack-sk-pseudo-cluster-plus-v3-fused-preattn (FUSED_PRE skF gate): on master's own job the skF gate REJECTED FUSED_PRE at B=16 (0.996x) and B=32 (1.015x vs the 1.02x gate). The full decode rows on master therefore ran pre-attn + v2, not skF. The hypothesis's instruction to 'confirm full decode rows are not on the v2 fallback' is misinformed: v2 is master's normal outcome. B=32 sits 0.5% under its gate, so it can flip between runs and move full decode_32 (weight 16, 228 us) by about 1.5% with no connection to this change.
- full decode_32 row noise (independent of any landed change): the same kernel measured 212.8 us in the parent's job and 227.9 us on master. That swing is about 3.3% of the decode_32 load, about 0.55% headline, the same size as this hypothesis's whole expected gain. The parent's 1.4973 headline was partly this noise.
- restack-sk-pseudo-cluster-plus-v3-fused-preattn (sliding verify B=1 pseudo route, variants 4/5): it shares _ps_ticket/_ps_gticket and load_prefix_pseudo with the new TOK=1 variants 6/7. Templating must leave TOK=8 byte-identical, so print the regs/smem line for variants 4/5 and compare it with master's. A sliding verify_1 move would come from the template, not from the route.
- fused-preattention-triton (landed): now runs as a separate node on sliding decode B=17..32 (PAIRED=False, ROPE on), and it alone must store the new K/V row before the sk window reads it. Stream order guarantees this, and the parent passed. The pre-attn node costs about 1.9 us on the profile and 3.8 us net in the gate, which is part of why the gate ratio is thinner than the kernel ratio.

### Interference and A/B needs
No companion is warranted, and none is affordable (0 attention checks today). Each sliding decode band runs either the routed pair, named in PATH CHOSEN with pair, cluster and Triton numbers, or byte-identical master code. A gain or a neutral result is therefore read directly from the decode_32 sliding row's top_kernels (splitkv_attn_sk_kernel<4,tru...> + _pre_attention_kernel vs sliding_cluster_kernel) and from the [sd-gate] B=32 lines. An ab-revert of the cluster restack would only re-measure the parent's run vs Triton, which already exists. The confounders that matter (full decode_32 noise, the skF gate flip) are headline-level. They are resolved by reading the rows and the skF PATH CHOSEN lines, not by another run.

### Open questions
- BUILDER: make the routing decision on a head-to-head interleaved pass (best pair vs cluster C=4 vs Triton, re-timed after the candidate is picked), not on the first-pass minimum of 12 candidates. Print both passes. The 1.03x margin is too thin to absorb winner's-curse bias.
- BUILDER: consider skipping the B=1 (record-only) and B=16 (1.006x last time) gate bands in this restack. That trims prepare time on a run that already took about 458 s wall on master, plus the extra variant-6/7 nvcc work against the 15-minute cap. It also removes a small chance of a noisy B=16 flip that would confound decode_16. If they are kept, they must stay record-only or keep the 1.05x margin as specified.
- BUILDER: confirm that _SKF_HEAD (sliced before 'template <int STAGES, bool TICKET') still compiles once load_prefix_pseudo becomes template<int TOK>. Without SK_PSEUDO its body is empty, so an uninstantiated template is fine. Confirm the [skF] and skF-gate lines appear as on master.
- HARVEST: compare the sliding decode_32 row with 20.4-20.6 us (master's two cluster C=4 runs). A pass is <= about 18 us with the sk + pre-attn kernels in top_kernels. Ignore the headline unless it is outside the noise of full decode_32. Record the skF PATH CHOSEN outcome alongside the result.
- Is a scored submit of eb1b68b's lineage already on the board? BEST.json is a check run. If no scored run carries master's 1.492x, this submit also posts master's gains, so a 'neutral' outcome still has board value.



## Implementation notes

Built in `.worktrees/muse_attention_decode/restack-sliding-decode-sk-window-on-cluster-master/muse_attention_decode/submission.py` (base 8bbf53ba; its submission.py is identical to current master dfc610f4, so no stale-base merge was needed). `git apply --reject` of `done/neutral/sliding-decode-sk-window-route.patch`: hunks 1-9 and 11 applied clean (MappingProxyType import, SK_MAX_VARIANT 7, `load_prefix_pseudo<TOK>`, kernel `TOK = 8` trailing param, sk_pick 6/7, sk_prepare done[8], sk_attn tok/R/two-level check, `_SK_LABELS`, `_ps_synthetic(..., L=8)`). Rejects merged by hand with a Python script:
1. kernel_fn: `if _sd_gate(cfg): return _sliding_decode_sk(data)` inserted FIRST, above master's `_slide_gate` branch.
2. The `_SD_*` block placed at module level just before the import-time `if torch.cuda.is_available():` block (after `_ps_setup`/`_skf`/`_slide_*` definitions, all defined before the import-time calls run).
3. `_sd_setup()` called AFTER `_slide_init()`, in its own try/except (`_SD_ROUTE = MappingProxyType({})` + PATH CHOSEN on a raise).
4. Final "every other shape" PATH CHOSEN wording updated.

Adversarial review items applied:
- Gate decision: `_sd_gate_band` now has pass 1 (all candidates + Triton + cluster, interleaved, median 21, empty subtracted) and pass 2 (picked pair vs cluster vs Triton + pre/sk rows, fresh graphs, median 31). Both passes are printed; routing uses ONLY the pass-2 ratio, vs the real incumbent (`_slide_forward(data, _slide_clusters(cfg))` if `_slide_gate(cfg)` else Triton), threshold 1.03x (1.05x if the incumbent falls back to Triton). `_SD_CLUSTER_C4_US` removed; measured cluster number printed; PATH CHOSEN prints pair, cluster and Triton.
- Skip extra bands: only (3, 32) is gated; B=1 and B=2..16 print "master, not gated in this restack" and keep master (cluster C=16 / Triton). The replay check covers band 3 only. `_sd_validate` still runs the parent's TOK=1 cases (B=1/3/16/32) as a kernel self-check.
- Ticket rezero: `_sd_setup` is a wrapper whose `finally` always calls `_rezero_tickets()` and prints the ticket sum (success, fallback or raise); the body keeps the parent's post-replay ticket check that drops the route if nonzero.
- skF build: `_SKF_HEAD` slices before `template <int STAGES, bool TICKET`, so it includes the new `template <int TOK> load_prefix_pseudo`; without SK_PSEUDO its body is empty, an uninstantiated template with unused params, which is valid. CANNOT BE COMPILED HERE (no GPU/nvcc); confirm the `[skF]` / skF PATH CHOSEN lines in the run log.
- Variants 4/5: `_sd_validate` prints regs/local/dyn smem/CTAs per sk_attrs for variants 4,5,6,7 (4/5 tagged "must equal sliding-verify line"); compare with master's `[ps]`/sk lines for byte-identity.
- Review's headline correction (+0.55%, not +2%) noted; read the decode_32 sliding row and `[sd-gate]` B=32 lines.

Deviation: none in the CUDA. Headline-level confounders (full decode_32 noise, skF gate flip) are unchanged and not addressed by the build.

## Rules check

- [x] CUDA-graph capturable: kernel_fn additions are a config-only predicate + `_sd_run` (pre-attn + sk launch, self-resetting tickets); all sync/.item()/prints live in prepare-time `_sd_*` setup only (checked in diff).
- [x] Nothing derived from `data` cached across calls: `_SD_ROUTE` holds (variant, grid-spec) from config bands only; workspaces allocated per call from shapes; tickets are the existing self-resetting arrays, not state pointers.
- [x] Weights: n/a, attention problem, no weights touched.
- [x] Activations/KV/q: KV FP8, q e4m3 via `_pre_attention_kernel`, fp32 softmax/merges, bf16 out (unchanged parent kernel).
- [x] Attention sees every key: sk kernel windows lo = max(0, sp - 2048) with the window including the new key; self-check vs FP32 reference at lens 0/1/63/2047/2048/2049/80000.
- [x] No timing patches, harness I/O, module edits, reference import: none (diff is submission-only; no subprocess/fork).
- [x] All work inside the timed call; no output caching or replay: graph replays only in prepare-time gate/check, kernel_fn recomputes.
- [x] Two states (copies: 2): no module-global device pointers into a state; `_sd_replay_check` captures kernel_fn on two states and alternates refilled replays.

okc_lint: `OK with warnings` (only host-sync warnings in prepare-time code outside kernel_fn's call graph). undef.py: `[]`. py_compile: ok.

## Pre-flight audit

Audited 2026-10-25 (opus, read-only emulation; no GPU here). **Verdict: dispatch.** No fixes applied.

- Merge: the CUDA hunks match the parent patch line for line. The `_SD_*` block matches the parent except for the specified gate change (incumbent = `_slide_forward(C=4)` when `_slide_gate`, else Triton; pass-2 head-to-head re-time; 1.03x / 1.05x thresholds) and the B=32-only band list. In kernel_fn, `_sd_gate` comes first, then `_slide_gate`, then `_sliding_forward`. That branch has no try/except, no prints and no per-call state; its allocations are sized by config only. undef.py returned `[]`, ast/py_compile passed, and okc_lint printed OK with only prepare-time host-sync warnings.
- Fallback: `_sd_setup()` runs after `_slide_init()` inside its own try/except. It rebinds `_SD_ROUTE = MappingProxyType({})` on a raise, and an inner `finally` re-zeroes the tickets. It only reads `_SLIDE_MOD`, `_PS_ROUTE`, `_FUSED_ROUTE` and `_EXT2` and never writes them, so master's routes survive any Python raise. The one exception is a sticky CUDA fault, which the parent's judge run (16/16, no fault) did not show.
- skF: `_SKF_HEAD` now carries `template <int TOK> load_prefix_pseudo`. Its body is empty without SK_PSEUDO, and the skF TU never calls it (the only call is in the sk kernel template, which the slice excludes). It is an uninstantiated, valid template.
- TOK=1 indexing: sp = cache_seqlens[b] = seq_len + 1 and lo = max(0, sp - 2048), so the kernel covers positions p-2047..p, which equals reference.py's window. Page-table, q, qkv and out rows are b. Ticket (256) and gticket (8192) cover R=32 and R*32.
- State: the tickets are module-global and keyed by device index, not pointers into a state. `_SD_ROUTE` holds config bands only. Nothing is keyed on data_ptr or id.
- Synthetic data for the incumbent timing has the dtypes slide_attn checks (positions int64, seq_lens int32, 4-D FP8 caches).

Residual risk: the in-prepare 1.03x gate against cluster C=4 may land below the threshold, in which case B=17..32 stays on cluster and the scored run measures master. That outcome is neutral, not a failure.

## Result

- job `252abb7f11ec48c3` (submit), status done, passed True, tests 16/16
- speedup 1.5017x, 95% CI [1.4791, 1.5304], score 2508.1040008355194 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.4919706177473366, "ci": [1.4695910584811265, 1.5143501770135466], "job": "cbd49dc94be548e7"}; base result: muse_attention_decode/hypotheses/done/gain/restack-sk-pseudo-cluster-plus-v3-fused-preattn.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/restack-sliding-decode-sk-window-on-cluster-master.result.json`, `muse_attention_decode/hypotheses/done/neutral/restack-sliding-decode-sk-window-on-cluster-master.patch`; branch `hyp/muse_attention_decode/restack-sliding-decode-sk-window-on-cluster-master`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 11.89 | 11.88 | 0.9738 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 22.48 | 22.4 | 1.0101 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 15.41 | 15.51 | 0.9965 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 111.34 | 111.12 | 1.0035 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 20.64 | 17.68 | 1.1582 | gained |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 227.88 | 227.87 | 1.0004 | flat |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 13.33 | 13.17 | 1.0141 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 30.27 | 30.48 | 1.016 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 23.1 | 22.83 | 1.006 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 163.37 | 163.0 | 1.0068 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 28.93 | 29.21 | 0.999 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 303.64 | 302.31 | 1.0038 | flat |

gained on sliding decode_32. Expected hit: 5/5. Noise ±1.7%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: restack-sliding-decode-sk-window-on-cluster-master`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

