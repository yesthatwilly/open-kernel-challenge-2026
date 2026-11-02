# Waiting on the user

churnsubs never blocks on these: the loop keeps dispatching everything else and lists open
items here and in each tick's summary. Resolved items move to the log at the bottom.

## Open

- 2026-11-02 12:4xZ decode + bf16 (decision, not blocking): **land verified gains that read below the board?** Two candidates passed their pre-registered gates on 3 judge reads each but cannot beat a board set by an older high read: (1) decode DD-W2 (T=17..32 router folded into the pooler launch): d32 rows +1.06%, rho +0.36% vs master, headline reads 2.198 vs board 2.2220 (DPP); (2) bf16 P1F (see 11:2xZ item): v16 unpooler +1.8%, headline 1.5331 vs board 1.5657. Landing makes master faster in expectation (and the next carriers build on it) without moving the board; final placement reruns the board entry's code, so this also ties into the 07:3xZ board-entry hygiene item. Default: hold both.
- 2026-11-02 11:2xZ bf16 (decision, not blocking): **land P1F on master below the board?** P1F (static cuBLASLt plan for the v16 unpooler) is a verified row gain: three judge reads 1.0535 / 1.0543 / 1.0535 vs R0 1.0352 (+1.8% on the row, ~+0.1% headline), harm guards pass. Its scored read 752054d5 is 1.5331 (board 1.5657 is a high read of master; master's 6-read rho mean 1.5785). Standing rule only lands what beats the board, so it is HELD. Landing would make master faster in expectation without moving the board. File: public-plus campaigns/2026-10-31-public-plus/files/combos/P1F-r3.py (scored sha). Default: hold.
- 2026-11-02 07:3xZ all problems (decision before the 11-04 close, not blocking): **board-entry hygiene.** The board keeps each team's best read and final placement reruns that entry's code, so a lucky read of weaker code becomes what is rerun. Now: moe shows T116HF a0588c87 2.3914 (refuted; ~2.368 pooled vs M1a/ZL ~2.38) and attention shows W313P 2ef42c8f 2.5647 (pre-SN code; its own d1f row 0.989, the excess is on untouched v1s/d1s). Option: before the close, `okc withdraw` the noise entries so the best-true-level code's best read is the entry (each withdraw is permanent; the next-best run takes the row). Default: hold until the end game (11-04), when the final master per problem is known, then propose an exact withdraw list.
- 2026-11-02 06:20Z moe (decision, not blocking): **withdraw T116HF scored a0588c870a714314 (2.3914)?** It is now our moe board entry (rank 151), above M1a's 2.3867, but it is a noise-high read of refuted code: T116HF reads 2.3509 / 2.3694 (checks), 2.362, 2.3914 (scored), ~2.368 pooled vs master M1a's pooled R0 ~2.385; on this read its own target rows are flat/worse vs R0 (d32 0.999, v16 0.994, v32 1.003) and the gain sits on untouched d1/d16. Final placement uses rerun medians of the board entry, so leaving it costs ~0.7% on moe at final review. `okc withdraw a0588c870a714314` is permanent (next-best M1a run takes the row; the run isn't refunded). Default: hold, because a real moe gain from today's moe-wgs / moe-skinny H100 lanes would replace it anyway; NOT landed on master (master stays M1a).
- 2026-10-25 15:20Z moe (FYI, not blocking): the moe mint agent flagged the w13 Marlin atomic-add flag at T=1. Two done-neutral runs (w13-marlin-atomic-add-on-master-small-m 1.766x, w13-marlin-atomic-add-t1-only 1.754x on decode_1) beat master-equivalent runs (1.70-1.72x on that row). Landing one of them by hand would need a merge onto moe master; say if you want it.
- 2026-10-24 rules heads-up (not blocking): bf16 **bf16-cublaslt-algo-pick-in-prepare-m128-m256** times cuBLASLt algorithm candidates on synthetic data inside the untimed prepare() and replays the winner in kernel_fn (no timing or autotune in kernel_fn). The Opus reviewer read the rules as allowing it, and decode's pooler-gemm-splitk-autotune used the same pattern. It runs on tomorrow's quota unless you say to hold it.

## Standing answers

- Churn paused (2026-10-26 ~04:40Z; the user approved campaign 2026-10-26-decode-first-rewrite with "execute the plan", having chosen "Pause churn"): no /loop churnsubs, no minting, all quota to the campaign; unspent checks accepted. Campaign state lives in campaigns/2026-10-26-decode-first-rewrite/LOG.md. Scored-submit discretion below still applies to campaign runs.

- Focus (2026-10-25 ~15:30Z, the user: "the most amount of focus, at this moment, to be on prefill"): prefill first in minting, review, build and spend; other problems only harvest and use leftover capacity.

- Scored runs: Claude may send any scored submit at its discretion (2026-10-25 ~15:00Z, the user: "submit any scored runs at your discretoin"). Landings that only serve a submit go with it.

- Scored runs for experiments: when a problem's checks are spent, churnsubs may dispatch reviewed hypotheses as scored submits (never risk: high), since the board shows only the best run. (2026-10-24, the user: "yes, do all three")

- The 40% rule: 40% of each problem's daily checks go to ambitious rewrites (whole-subsystem or
  whole-layer overhauls), the rest to incremental hypotheses. Unused shares are released so
  nothing idles. Mint big rewrites so the share is always fed. (2026-10-24, the user: "yes, add
  the 40% rule and mint the big rewrites")

- Scored submits (promotions with CI low above 1.0): ask each time, but never hold the loop for
  the answer. Leave the planned `okc submit` command here and keep going.
- Prefill expert-dequant family (per-call BF16 copy of NVFP4 expert weights): keep held. Pursue
  in-kernel dequant only. (2026-10-24)
- Attention-decode base: fused-preattention-triton with no parent; the 12 parity dependents
  re-pointed to it. (2026-10-24)
- Adversarial review model: Opus, not Fable. (2026-10-24)
- Autonomy: run the loop all day, every day. Mint hypotheses when queues run low, and mint next
  rewrite stages. Submits are still asked as above. (2026-10-24)

## Log

- 2026-10-24: bf16 router-triton-hilo-fused-control-splitk-all-m (check 1.151x) submitted by
  Claude with the user's approval, job 240eb22ea0b6487e.
- 2026-10-24: decode marlin-block-m-cap-16 (check 1.062x) submitted at the user's request, job 2b1f82408ab64116.
- 2026-10-24: attention_decode fused-preattention-triton (check 1.127x) submitted by the user, job c297440e53334f34.
- 2026-10-24: the user said "submit all five promotions"; Claude ran them: attention full-decode-cuda-gather-splitkv (check 1.347x) de486093311d4bb9, decode post-expert-norm-triton-small-t (1.164x) a017e10aec544c55, moe triton-map-align-on-block-m-restack (1.312x) 3cabc739f1af493e, prefill experts-hybrid-marlin-turn-triton-first-turn (1.060x) ba9c16b9cc6b4605, bf16 bf16-m32-swapab-wgmma-triton (1.198x) eeb9985396764ac2.
- 2026-10-24: bf16-lossless-exp4-weight-stream-rewrite heads-up closed: TASK.md says weights may be compressed losslessly ("BF16 weights stay BF16, or smaller, losslessly"). The review also had cuobjdump removed from its diagnostics, since it starts a process.
- 2026-10-24: moe align-parallel-per-expert-verify (check 1.3249x, hand-landed on per-shape rows) submitted at the user's request ("sure promote moe i guess"), job 1947e3ec2abb4bdf.
- 2026-10-25 ~04:20Z: the user said "approve all four landings". Landed on master: decode full-decode-splitkv-restack-on-pdl (0117ef5), moe w2-marlin-atomic-add-small-m (9080a71), bf16 bf16-restack-m16-swapab-router-swapab-m32 (b99b1dd), attention full-decode1-streamk-s3-two-level-ticket (c3c1b01; supersedes the s2 entry). moe scored submit 7a0d900a1dd34666 sent; attention/decode scored submits await approval (Open).
- 2026-10-25: the user said "send the scored submits"; Claude ran them: attention master (full-decode1-streamk-s3-two-level-ticket) 341623fa53d94cef, decode master (full-decode-splitkv-restack-on-pdl) 7edbcb33c98349cb.
- 2026-10-25: the user said "land the decode w2 atomic port"; landed w2-marlin-atomic-add-t1-port on decode master (00e36e3). No scored submit sent (not requested).
- 2026-10-25: the user said "send the decode scored submit too"; Claude ran it for decode master with w2-marlin-atomic-add-t1-port, job 3fd3fa1f96d045dd.
- 2026-10-25 ~05:40Z: the user said "lift the pause and mint for the remaining checks". Pause lifted; minting 25 (attention 8, decode 3, moe 5, prefill 8, bf16 1 = the three-row restack). Board after harvest: attention scored 1.4441 (341623fa), decode 1.2726 (3fd3fa1f).
- 2026-10-25 ~11:00Z: the user approved: (1) attention scored submit of master (restack-sk-pseudo-cluster-plus-v3-fused-preattn, check 1.4920) sent, job d672f69ea69b4396; (2) prefill experts-turn4-ssdq-s1b-dataptr-symbol-fix landed on master + scored submit 810b5f7decc84206; (3) sliding-decode-sk-window-route restack onto attention master, to run as a scored submit (minting); (4) bf16 restack-three-cell-wins landed on master + scored submit ee4f89f2f62c42d3. Then /hypothesize all -n 2.
- 2026-10-25 ~15:00Z: the user said "submit any scored runs at your discretion". Landed prefill align-two-phase-all-t-rega (master now rega first-turn + align2) and sent the prefill scored submit bff8e36788a04917; landed decode dense-chain-s3 as base (no submit: board 1.3056 already above its 1.3017); moe bucket-gate land refused (stale base), its branch code sent unchanged as scored submit 4e4a0e26c85944f2.
- 2026-10-25 ~19:45Z: the user said "do the attention hand-merge and submit it". merge-perkvhead-c320-sv32-tiled built, reviewed, audited and sent as scored submit a78f29aeab9b47cb.
- 2026-10-25 ~20:00Z (offer, not blocking): attention combo-b1-chain (1.5295, board) reached full decode_1 0.791 with its V3 chain but lacks per-kvhead; the in-flight hand-merge a78f29ae carries only c320 V1 (0.892). A per-kvhead + B1-chain V3 hand-merge is the next stack (one scored submit left today). moe occ2 (1.4021, board) and two-stream are likewise unmerged onto toprow master (6-7 conflicts).

## 2026-10-28 open items (findings-model review; not blocking)

- 2026-10-28 **memcheck VOID**: compute-sanitizer memcheck of the moe board's timed path is VOID, not a pass. Under the sanitizer the board's prepare() fills the 80 GB H100 (79.2 GiB), so ea4 / wg fail tests on OOM and ea_combine / ea_w abort; the filtered families never ran their judged paths (0 errors reported). Options: a lighter prepare under the sanitizer (diagnostics off, fewer state copies), or the 141 GB H200. The current moe board 54068229 (after L3 / L4 / fc / X1v6) has not been re-proven on the node either. LOG:1539-1540; research/h100-findings.md:161-175; moe CLAIMS.md node-only line.
- 2026-10-28 **DEC-02 FAIL**: decode node work stopped by its registered rule (V2 and V3 fail on all four bytes: the nsys wrapper forces incumbents, the T=1 Engine A gate falls to MASTER on 2/4 draws, H100 timing races, the norm_residual shim is +1.2-1.8 us). Reopening needs a new registration (your call): forced judge picks via E1 arms, a profiled pass whose prepare is not under nsys, a faster boundary shim. Until then decode stays on judge checks. LOG:1726-1735; research/h100-findings.md:887-906.
- 2026-10-28 **two H200 jobs held** (with the two moe fuzz fillers) since ~0045Z; "H200 stays held" (LOG:1522-1527). The H100 window closed at 13:37Z (LOG:2099-2100). Release, re-register or drop them?
- 2026-10-28 FYI for USER: gpunode06 GPU0 idles at 86 C / 229 W with SW thermal slowdown (78-79 C on 10-27); the runs never queue on it (LOG:1533-1535).
- 2026-10-28 **findings-model changes landing today**: hypotheses/FINDINGS-MODEL-REVIEW-2026-10-28.md §3 changes 1-7 (freshness gate at queue and claim, `<p>/hypotheses/CLAIMS.md`, `register-landing`, triage on the scoring formula (done, 4b45b873), plumbing, mechanism verdicts, dated DIAGNOSIS labels). The five CLAIMS.md files are seeded (attention 38, bf16 33, decode 51, moe 43, prefill 29 lines), including today's 30 screen verdicts, so they survive the hold expiry around 10-29T15:00Z. Deferred per §4 (needs your go): the full ledger (CLAIMS.jsonl / claims.py), CELLS and route flags, H100 registration through hyp.py, a claim check in okc-attempt-gate.py, a diagnose_due rewrite.

### 2026-10-28 ~20:50Z: decode final-rerun routing tail (needs your call after tonight's check)
- Decode check b47f8f73 failed a verify_32-full routing recheck: one expert flipped, +0.0321 against the 0.03 tolerance.
- The diagnosis (/home/user/git-scratch-c24a6e/decode-recheck-diag.md) rules out the probe. Our FP8-q attention against the judge's BF16-q reference flips a clear-margin expert about 1.6% of runs (1 of 61; 95% upper bound ~8%). The board 83f0c00e is exposed in the final held-out rerun.
- Settling check dec-rt-tail is being built for tonight. If the q pre-scale fix is a lead, it costs a decode rebuild plus a scored run before 11-04.
- 21:1xZ update: dec-rt-tail 0635f30d found the decode routing tail is INHERENT to FP8-q attention. Master FA3 has it too, and a q pre-scale does not help. The options are to accept the ~1.6%/run final-rerun risk, or to measure a BF16-q full-attention path at T=256 (verify_32) and pay its speed cost. That is your call before 11-04.

### 2026-10-31 20:31Z: GAPMAP-1031 open items (not blocking; defaults apply)
- **Prefill S1 in phase A?** The only gap-map item near the value bar (EV 1.3-1.7%, +3.8-4.2% if the H100 prize holds; prefill rank 187 -> ~50 at +3.5%). It does not count on the overall board, and phase A's standing scope is decode + moe. Gated by tonight's PF-D1 check (GO at prize >= 75 us at t1s and t4s). Default: the standing order holds (prefill stays last in phase B). Source: campaigns/2026-10-31-public-plus/GAPMAP-1031.md §2.1, §2.3.
- **Prefill exp-split** (about 15% of exp2 on the FMA pipe as a polynomial, ~+1.1%): it changes the exp implementation, so it needs your ruling before any build. Default: not built. GAPMAP-1031 §2.5, §4 Q10.
- **RESOLVED ~21:01Z (user: "if d0fix improves the robustness of the kernel, keep it"): D0FIX lands after scored run 545b80bb.** ~~2026-10-31 21:00Z: land the decode sk2 race fix (D0FIX) although it missed its speed guard?~~ Tonight's 2 checks read 2.1923 / 2.1866 (mean 2.18945, guard 2.19) with d1s/v1s over their caps. ET3FIX, which carries the same fences, read d1s/v1s inside the caps, and tonight's reads of master-equivalent code were also low, so the fence's H200 cost is somewhere between ~0.2% (H100) and ~1% (tonight's 4 reads vs DPP). The race is real (forced 15-25 bad outputs per 16,000 replays in a variant), the judge's tolerance does not catch it, and it has never fired on master in 100k+ replays. Note: the board keeps our best scored run, so a lower-scoring landing changes master's file, not the board entry. Default: not landed; tomorrow's new sk2 code paths keep the fences (CLAIMS:116).
- **RESOLVED ~21:04Z (user: "Land it (robustness)"): BF16FIX landed, master 8d0df028.** ~~2026-10-31 21:03Z: land BF16FIX (bf16 sk2 ring-slot hardening)?~~ Scored 1.5522 (checks 1.5510 / 1.5434) vs the 1.5657 board; H100 cost ~+0.3%. Same race pattern as D0FIX (71 SASS hazards -> 0), but no failure was ever forced in bf16 (no PDL neighbour, narrower window). Default: HOLD (not landed).

### RESOLVED 02:05Z (user: "set up the allow rule for hopper-probe runs"; settings.local.json allow rules + autoMode note): ~~2026-11-01 01:41Z: an H100 run was denied by the permission check~~
- The attention S2 builder's run `hopper-probe run gpunode06 --gpus 5 atts2-stamps` was denied by Claude Code's permission check as "Interfere With Workloads". The takemyjob queue STOP file is present.
- Today's other manual runs went through with STOP in place: the moe/prefill/attention/decode diag audits, S3 and S0.
- Nothing was retried, and the main thread will not run it on the builder's behalf.
- Ready to run: takemyjob `hopper/att-s2` 9062e3e, `probes/atts2`. Rebuild the bundle, then `HP_BUNDLE=atts2 bin/hopper-probe run gpunode06 --gpus 5 --tmax 85 atts2-stamps`.
- **Your options:**
  - (a) Confirm that manual hopper-probe runs are fine while queue/STOP is set. You can run it yourself with `! <command>`, or approve it when prompted.
  - (b) Leave S2 unmeasured.
- Default: S2 waits. Attention phase B has no other item.

### 2026-11-01 03:03Z: moe C1 M0 read BETWEEN; continue to a TINY-only M1? (not blocking; default applies)
- The H100 M0 (workflow wf_855a9495-ef4, independently verified) put the C1-12 engine at **1.239x** consumer weights/us at N=1 (whole-kernel 1.248-1.251) against the pre-registered GO-TINY line of 1.25x, and 1.14x at N=8 (GO needs 1.25 at N=1/8/16; KILL needs < 1.15 at N=1 and N=8). The rule says BETWEEN: no M1.
- Value cap: on the H200, d1 is bandwidth plus consumer bound, so a 1.24x consumer only recovers the HBM headroom (0-14% of d1 stream time); design EV ~+0.25% moe headline; the decode port +0.5-1.2% only if moe lands.
- Caveats from the verifier: C1's FP32 order is farther from fp64 on 30/31 cancellation elements (compliant); one per-refill proxy fence was dropped in campaign/c1-moe 016e301e and needs a memory-order review before any M1.
- **Your options:** (a) a TINY-only M1 anyway (C1-12 at d1; ~2-3 h build + audit, then moe checks); (b) hold C1 (CLAIMS [unresolved]).
- Default: (b) hold. The bit-identical diet rider D1+D6 (1.05-1.07x consumer) is queued as a moe phase-B item after the ~04:19Z usage-window reset.
