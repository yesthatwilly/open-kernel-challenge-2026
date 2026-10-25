---
problem: muse_attention_decode
slug: sliding-decode-paired-kvhead-rows
created: 2026-10-25T003735Z
base: master
area: sliding-layer attention (batched decode, KV row gather)
change: >
  One change inside master _sliding_fused_kernel and its launch in _sliding_forward, sliding
  decode at batch > 1 only (q_len 1; config-only predicate pair = length == 1 and batch > 1 and
  HKV == 2; batch 1 and the S == 1 path keep today code). Today program (b, g, s) gathers kv head
  g alone: a 64-byte K row and a 64-byte V row per token (kvoff = (row * HKV + g) * D + d), and
  its M = 16 query tile holds only GROUP = 8 real rows (8 padding rows). LESSONS: page-size-1 row
  gather is the limit of these kernels, and FA3 head pairing (one 128-byte row per token for both
  kv heads) gained 14-24% on full batch-16/32 shapes. New constexpr PAIR: grid (B, 1, S) with S =
  16 unchanged; the program serves BOTH kv heads: rows r = 0..15 are query heads 0..15 (head = r,
  kv head r // 8, so the M = 16 tile has no padding rows at L = 1); the query is laid out as [16,
  2*D] with each row values in its own kv head half and zeros in the other half (same trick as
  master paired FA3 query, exact: the zero half adds 0 to every logit); each old-key tile loads
  one contiguous 128-byte K row and one 128-byte V row per token: offset row * HKV * D +
  arange(2*D); qk = dot(q[16,128], K^T[128,BLOCK_N]); PV gives acc[16,128] and each row keeps only
  its own half (select by r // 8 when writing the partial and in the epilogue, so ws_acc stays
  [16, D] per partial). The last split builds both kv heads new k/v (norm + rope + E4M3, stored at
  cache_loc for g = 0 and 1 exactly as today) and attends them from registers in the same 128-wide
  form. Ticket counters indexed by b (one per request instead of b*HKV + g; _counters(batch, 1,
  device)). Workspace sized (B * S, 16, D) and (B * S, 2, 16). _gate_epilogue unchanged per row
  (head = r). BLOCK_N 64, num_warps 4, num_stages 2 unchanged. Everything else (full layers,
  sliding verify, sliding decode_1) byte for byte.
expected:
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: "master _sliding_fused_kernel 12.9 us (floor 1.65): half the gather instructions per token (one 128 B K and V row instead of two 64 B rows from two programs) and no padding query rows; expect 10-11.5 us"
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: master 22.1 us (floor 3.29), 1024 programs -> 512 with full 128-byte rows; expect 16-19 us
  - shapes: "*layer=sliding load=decode_1 * *layer=sliding load=verify_* *layer=full*"
    direction: flat
    why: outside the batch > 1 decode predicate
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T004021Z
review_companions: [sliding-decode-paired-kvhead-rows-control-unpaired-s8]
hold_reason: null
stage: dispatched
claimed_at: 2026-10-25T014325Z
base_sha: 0c91821ce96e550a59594394591699d20286eb84
base_blob: c6854f093588da524e63657576b27ed720756e32
branch: hyp/muse_attention_decode/sliding-decode-paired-kvhead-rows
worktree: .worktrees/muse_attention_decode/sliding-decode-paired-kvhead-rows
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T014526Z
impl_commit: 990710b28b9dcf8dcc5513dd19981907c15e460d
submission_sha256: 60a19c2fe163ce14ab00717c6cced9a3c35e9ec417b511b162052a4daf3353b0
norm_sha256: fdda24a3a51127ec55fb90dbbb77e4a02e88e0859debf40544fb2b9ff47723fa
submission_blob: af33f9553ee353859b9026c4071c9d22288fd453
lint_warnings: 0
attempt_file: 2026-10-25T014526Z-check-sliding-decode-paired-kvhead-rows.md
prepared_at: 2026-10-25T014526Z
job: 48ffbe5690cc4ab4
dispatched_at: 2026-10-25T014607Z
---

# sliding-decode-paired-kvhead-rows

## Change

One change inside master _sliding_fused_kernel and its launch in _sliding_forward, sliding decode at batch > 1 only (q_len 1; config-only predicate pair = length == 1 and batch > 1 and HKV == 2; batch 1 and the S == 1 path keep today code). Today program (b, g, s) gathers kv head g alone: a 64-byte K row and a 64-byte V row per token (kvoff = (row * HKV + g) * D + d), and its M = 16 query tile holds only GROUP = 8 real rows (8 padding rows). LESSONS: page-size-1 row gather is the limit of these kernels, and FA3 head pairing (one 128-byte row per token for both kv heads) gained 14-24% on full batch-16/32 shapes. New constexpr PAIR: grid (B, 1, S) with S = 16 unchanged; the program serves BOTH kv heads: rows r = 0..15 are query heads 0..15 (head = r, kv head r // 8, so the M = 16 tile has no padding rows at L = 1); the query is laid out as [16, 2*D] with each row values in its own kv head half and zeros in the other half (same trick as master paired FA3 query, exact: the zero half adds 0 to every logit); each old-key tile loads one contiguous 128-byte K row and one 128-byte V row per token: offset row * HKV * D + arange(2*D); qk = dot(q[16,128], K^T[128,BLOCK_N]); PV gives acc[16,128] and each row keeps only its own half (select by r // 8 when writing the partial and in the epilogue, so ws_acc stays [16, D] per partial). The last split builds both kv heads new k/v (norm + rope + E4M3, stored at cache_loc for g = 0 and 1 exactly as today) and attends them from registers in the same 128-wide form. Ticket counters indexed by b (one per request instead of b*HKV + g; _counters(batch, 1, device)). Workspace sized (B * S, 16, D) and (B * S, 2, 16). _gate_epilogue unchanged per row (head = r). BLOCK_N 64, num_warps 4, num_stages 2 unchanged. Everything else (full layers, sliding verify, sliding decode_1) byte for byte.

## Expected

- `*layer=sliding load=decode_16*` faster: master _sliding_fused_kernel 12.9 us (floor 1.65): half the gather instructions per token (one 128 B K and V row instead of two 64 B rows from two programs) and no padding query rows; expect 10-11.5 us
- `*layer=sliding load=decode_32*` faster: master 22.1 us (floor 3.29), 1024 programs -> 512 with full 128-byte rows; expect 16-19 us
- `*layer=sliding load=decode_1 * *layer=sliding load=verify_* *layer=full*` flat: outside the batch > 1 decode predicate

## Risk

Risk: per-program work doubles while program count halves (B=16: 256 programs of 4 warps on 132 SMs, under one wave per 2 CTAs/SM); if occupancy hurts decode_16, an ab-tune with S=32 at batch 16 is the companion. The 128-wide dot doubles MMA work, but these shapes are gather/latency bound (the fp8-qk mma companion on the CUDA path was flat).

## Adversarial review

Reviewed 2026-10-25T004021Z by an adversarial subagent (model opus).

**Verdict: proceed-with-companions.** Mechanism plausible, but most of the predicted decode_32 gain is already explained by the smaller grid: unpaired S=8 at B=32 (sliding-decode32-splits-8) reached 17.3 us, inside this hypothesis's 16-19 us target. Run it with an unpaired same-grid control, and only after sliding-cluster-s2-decode-c4-pdl-inproc reports: that check serves exactly these shapes, and if it meets its gate this Triton path stops running on any scored shape.

### Justification
_sliding_fused_kernel takes 12.9 us at decode_16 (floor 1.65) and 22.1 us at decode_32 (floor 3.29): latency/per-program-overhead bound. Pairing saves one page-table load per token, the 8 padding rows of M=16 (MMA, exp2, fp32 partials; half of 4 MB workspace at B=32), one new-key build and one combine per request, and halves the programs. Weakest link: 'half the gather instructions' is false in Triton (16 B per thread per load; same instruction count and DRAM sectors grid-wide), so the FA3 head-pairing precedent (issue-bound 80K gathers) does not carry over. Halving programs partly undoes deep-splits (which gained decode_16 by 128->512 programs), so decode_16 'faster' is the least supported prediction. Sliding is ~21-23% of the decode_16/32 loads; a 2-5 us kernel gain moves the headline ~0.5-0.8%, inside CI. Judge on _sliding_fused_kernel time per shape.

### Confounders on the current kernel
- sliding-deep-splits-on-cuda-splitkv (landed): pairing halves the grid, so program count and pairing are confounded at both shapes.
- sliding-fused-triton-decode1-gate (landed): counter key changes (B,2)->(B,1) and the combine becomes per-b; recheck the zero-reset invariant.
- fa3-hdim128-head-pairing / paired-heads-launch-cleanup: full layers only; their gain is not evidence for this regime.
- full-decode-cuda-gather-splitkv: does not touch sliding shapes.

### Interference and A/B needs
One companion: an unpaired S=8 same-grid control (B x 2 x 8 programs, 16 KB each) so paired minus control isolates pairing. Interaction: sliding-cluster-s2-decode-c4-pdl-inproc (running) gates all sliding decode 1<=B<=32 onto a CUDA cluster kernel with better targets (decode_16 <=10, decode_32 <=12 us); mutually exclusive, not stackable. WAIT for that result before dispatching this pair. If it lands, drop both; if it fails, ties, or lands only at one batch, run this pair.

### Companions queued
- `sliding-decode-paired-kvhead-rows-control-unpaired-s8`: Separates pairing from the grid halving that pairing also brings; pairing's effect = paired minus control kernel time per shape.

### Open questions
- Does the cluster stage 2 meet its targets and land? If so drop this and its control.
- If pairing beats the control, next step S=32 at B=16 or S=8 at B=32?
- Print n_regs/n_spills of the paired kernel.
- Row-half selection: reshape acc to [16,2,64] and masked-sum over axis 1; check no large smem shuffle.
- Credit pairing at decode_32 only if clearly below 17.3 us.


## Implementation notes

- Worktree `muse_attention_decode/submission.py`, +172 lines, nothing removed. `_sliding_fused_kernel` and all other paths are untouched (byte-identical).
- Deviation: a new kernel `_sliding_paired_kernel` instead of a PAIR constexpr inside `_sliding_fused_kernel`, so the unpaired kernel's code cannot change. Gate in `_sliding_forward` (config only): `length == 1 and batch > 1 and hkv == 2 and hq == 16 and d == 64`. Grid is (B, 16) with S = 16, M = 16, BLOCK_N 64, warps 4, stages 2.
- Rows r = 0..15 are query heads (kv head r // 8, no padding). The query is loaded as its 64 values twice into 128 columns, normed and roped, then zeroed outside its own half.
- Review item 1: `_norm_rope_halves` takes the RMSNorm per 64-wide half, for the query and for the new k. RoPE freq indexes `(d % 64) // 2`. The new k/v is stored contiguously at `loc * 128 + d2`, which is `(loc * 2 + g) * 64 + d` for g = 0, 1.
- Review item 2: `_own_half` reshapes acc to [16, 2, 64] and takes a masked sum over axis 1. The partial write and the epilogue both use the result, so ws_acc stays [B*S, 16, 64]. Whether Triton lowers this without a large smem shuffle is unchecked (no GPU here).
- Review item 3: counters are `_counters(batch, 1, device)`, one per request. Each of the 16 programs of request b draws one ticket, the last one combines and resets it to 0, so the zero-reset invariant holds across graph replay. The (B, 1) buffer is a different dict key from the old (B, 2) buffer, so they cannot alias.
- Review item 4: the submission has no `prepare()`. The kernel's n_regs and n_spills are printed once, on the first (eager) launch, from the launch handle, with no sync. Check the log_tail for `[paired] ...`.
- Weakest point: untested on GPU. The query norm over duplicated columns is mathematically equal to the 64-wide norm, but the sum order may differ in the last ulp.

## Rules check

- [x] CUDA-graph capturable: the only host work is the one-time print of n_regs and n_spills (host metadata); the branch is on config only; workspaces are allocated from shapes.
- [x] Nothing derived from `data` is cached: counters are zero-by-invariant and carry no data; `_inv_freq` is unchanged.
- [x] n/a: weights untouched, no repacking.
- [x] KV cache stays FP8 E4M3; the query is bf16 as in master.
- [x] Every key is attended: old window keys in all 16 splits, plus the new key from registers.
- [x] No timing patches, subprocesses or network: the diff only adds a Triton kernel and a gated launch.
- [x] All work is inside the timed call; no output caching or replay.

okc_lint: `CLEAN: muse_attention_decode/submission.py`. `python3 -I -m py_compile` passed and `__pycache__` removed.
