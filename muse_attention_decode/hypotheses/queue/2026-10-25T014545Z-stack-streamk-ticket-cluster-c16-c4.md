---
problem: muse_attention_decode
slug: stack-streamk-ticket-cluster-c16-c4
created: 2026-10-25T014545Z
base: master
area: full + sliding decode attention (stack of two pending kernels)
change: >
  Stack the two pending (user-held, not on master) decode wins on current master; they touch
  disjoint shapes and code paths (REWRITE_READ_2026-10-25 '01:36Z harvest', 'Do they stack?'). (1)
  full-decode-splitkv-s2-streamk-ticket-inproc (check e592b1f5d3db4f83, 1.4171x): as depends_on,
  claim restacks its patch (done/neutral/full-decode-splitkv-s2-streamk-ticket-inproc.patch) onto
  master automatically; keep it unchanged: TU muse_full_decode_splitkv_v2, splitkv_attn_sk_kernel
  with 4 stages + ticket combine, routed only for full layer, decode, 1<B<=64, page 1, with its
  canary and 8 self-checks. (2) sliding-cluster-s2-decode-c4-pdl-inproc (check 1990ce53e9d34073,
  1.4001x): claim restacks only ONE parent patch, so the implementer must 3-way merge this second
  branch's diff BY HAND inside the worktree: git apply -3
  --include=muse_attention_decode/submission.py on
  muse_attention_decode/hypotheses/done/neutral/sliding-cluster-s2-decode-c4-pdl-inproc.patch
  (equivalently, the diff of branch
  hyp/muse_attention_decode/sliding-cluster-s2-decode-c4-pdl-inproc against its base 593859d3),
  resolving conflicts so both TUs, both prepare() blocks and both dispatch branches survive; each
  path keeps its own gate constant (LESSONS stack note). Then switch the cluster C=8 band OFF:
  sliding decode B=2..16 goes back to master's Triton path; keep C=16 at B=1 and C=4 at B=17..32
  (fp8qk sliding_cluster_kernel); master keeps B>32, every verify and page 64. Do not build or
  self-check the C=8 instantiation. Keep both prepare() self-checks and canaries; print both PATH
  CHOSEN lines (streamk: full decode 1<B<=64; cluster: C=16 at B=1, C=4 at B=17..32, Triton at
  B=2..16) compactly and late (archived log is a ~15 KB tail). Run as an okc check. This check
  also RE-MEASURES streamk on top of current master (its 1.4171 CI lower bound 1.396 only just
  cleared master's 1.3949), so read the streamk kernel rows and the cluster kernel rows from the
  profile separately.
expected:
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: "streamk+ticket: splitkv A 111.8 + combine 8.2 = 120.0 us -> sk 108.9 us (-9.3%); shape 122.8 -> 110.9 us (-9.7%)"
  - shapes: "*layer=full load=decode_32 *"
    direction: faster
    why: "streamk+ticket: 254.3 -> 226.8 us kernel (-10.8%); shape 241.9 -> 226.8 us (-6.2%)"
  - shapes: "*layer=sliding load=decode_32 *"
    direction: faster
    why: "cluster C=4: Triton 22.1 -> cluster 18.3 us kernel; shape 24.2 -> 20.2 us (-16.8%)"
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "cluster C=16: shape 13.2 -> 12.5 us (-5.0%), kernel flat"
  - shapes: "*layer=sliding load=decode_16 *"
    direction: flat
    why: "C=8 band gated off: master Triton 12.9 us kernel (C=8 cluster was 13.8, shape +5.7%)"
  - shapes: "*load=verify_*"
    direction: flat
    why: untouched FA3 control
  - shapes: "*layer=full load=decode_1 *"
    direction: flat
    why: B=1 full decode not routed (FA3)
risk: medium
depends_on: full-decode-splitkv-s2-streamk-ticket-inproc
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# stack-streamk-ticket-cluster-c16-c4

## Change

Stack the two pending (user-held, not on master) decode wins on current master; they touch disjoint shapes and code paths (REWRITE_READ_2026-10-25 '01:36Z harvest', 'Do they stack?'). (1) full-decode-splitkv-s2-streamk-ticket-inproc (check e592b1f5d3db4f83, 1.4171x): as depends_on, claim restacks its patch (done/neutral/full-decode-splitkv-s2-streamk-ticket-inproc.patch) onto master automatically; keep it unchanged: TU muse_full_decode_splitkv_v2, splitkv_attn_sk_kernel with 4 stages + ticket combine, routed only for full layer, decode, 1<B<=64, page 1, with its canary and 8 self-checks. (2) sliding-cluster-s2-decode-c4-pdl-inproc (check 1990ce53e9d34073, 1.4001x): claim restacks only ONE parent patch, so the implementer must 3-way merge this second branch's diff BY HAND inside the worktree: git apply -3 --include=muse_attention_decode/submission.py on muse_attention_decode/hypotheses/done/neutral/sliding-cluster-s2-decode-c4-pdl-inproc.patch (equivalently, the diff of branch hyp/muse_attention_decode/sliding-cluster-s2-decode-c4-pdl-inproc against its base 593859d3), resolving conflicts so both TUs, both prepare() blocks and both dispatch branches survive; each path keeps its own gate constant (LESSONS stack note). Then switch the cluster C=8 band OFF: sliding decode B=2..16 goes back to master's Triton path; keep C=16 at B=1 and C=4 at B=17..32 (fp8qk sliding_cluster_kernel); master keeps B>32, every verify and page 64. Do not build or self-check the C=8 instantiation. Keep both prepare() self-checks and canaries; print both PATH CHOSEN lines (streamk: full decode 1<B<=64; cluster: C=16 at B=1, C=4 at B=17..32, Triton at B=2..16) compactly and late (archived log is a ~15 KB tail). Run as an okc check. This check also RE-MEASURES streamk on top of current master (its 1.4171 CI lower bound 1.396 only just cleared master's 1.3949), so read the streamk kernel rows and the cluster kernel rows from the profile separately.

## Expected

- `*layer=full load=decode_16 *` faster: streamk+ticket: splitkv A 111.8 + combine 8.2 = 120.0 us -> sk 108.9 us (-9.3%); shape 122.8 -> 110.9 us (-9.7%)
- `*layer=full load=decode_32 *` faster: streamk+ticket: 254.3 -> 226.8 us kernel (-10.8%); shape 241.9 -> 226.8 us (-6.2%)
- `*layer=sliding load=decode_32 *` faster: cluster C=4: Triton 22.1 -> cluster 18.3 us kernel; shape 24.2 -> 20.2 us (-16.8%)
- `*layer=sliding load=decode_1 *` faster: cluster C=16: shape 13.2 -> 12.5 us (-5.0%), kernel flat
- `*layer=sliding load=decode_16 *` flat: C=8 band gated off: master Triton 12.9 us kernel (C=8 cluster was 13.8, shape +5.7%)
- `*load=verify_*` flat: untouched FA3 control
- `*layer=full load=decode_1 *` flat: B=1 full decode not routed (FA3)

## Risk

Estimated headline ~1.440x (the read's load-level recompute; drift not removed) vs master 1.3949. Risks: the hand 3-way merge of the cluster patch over the restacked streamk patch (both edit prepare() and dispatch in submission.py); both parents are pending the user, so a win here is a landing candidate only with user approval; full decode_32 shows a known 242 vs 256-260 us bimodality on identical kernels, so judge streamk from kernel rows; the cluster kernel spills 32 B local.
