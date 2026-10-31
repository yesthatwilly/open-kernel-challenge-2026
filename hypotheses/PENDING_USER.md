# Waiting on the user

churnsubs never blocks on these: the loop keeps dispatching everything else and lists open
items here and in each tick's summary. Resolved items move to the log at the bottom.

## Open

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
