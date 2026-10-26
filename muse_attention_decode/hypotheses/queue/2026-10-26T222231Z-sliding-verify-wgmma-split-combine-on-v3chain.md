---
problem: muse_attention_decode
slug: sliding-verify-wgmma-split-combine-on-v3chain
created: 2026-10-26T222231Z
base: master
area: sliding-layer attention (verify B=2..32, wgmma window kernel merge) on the v3chain base
change: >
  Board-candidate twin of sliding-verify-wgmma-window-s2-split-combine, built on the best code
  instead of s1's base. Base: merge-perkvhead-v3chain-sv32-tiled (claim restacks its patch onto
  master 5d528bf3). Then (1) apply
  muse_attention_decode/hypotheses/done/neutral/sliding-verify-wgmma-window-s1.patch with git
  apply -3 inside the worktree (both patches are against master 5d528bf3 and touch disjoint code
  except kernel_fn and the end-of-import setup calls): in kernel_fn keep v3chain's _C320_FN line
  and tiled pre-attention branch AND s1's _sw_gate sliding-verify route; at the end of import run
  s1's _sw_setup() after v3chain's piece B under the same 480 s start rule (s1's _SW_*_BY
  deadlines measured from v3chain's _IMPORT_T0 clock; v3chain ended its phases at 212 s, s1
  started its gate at 250 s on its own base). (2) Make exactly the change specified in
  sliding-verify-wgmma-window-s2-split-combine (route slidewin_verify_wgmma_kernel<false> plus the
  one-pass parallel sw_combine_kernel, grid (B * QL, HKV), with the shared seq_s / tile_prefix
  helper; module renamed _v2; gate >= 1.05x in both final rounds, unchanged checks); if that
  hypothesis is already implemented on its branch, copy its diff. Nothing else changes: per-kvhead
  SD1, the c320 chain, SV32 tiled GATE B and every master route stay as on v3chain.
expected:
  - shapes: "*layer=sliding load=verify_16 *"
    direction: faster
    why: "as the s2 split-combine: kernel-only 14.9-16.1 us vs master path 24.4 us at B=16; shape 22.84 -> ~20 us at 0.21%/us"
  - shapes: "*layer=sliding load=verify_32 *"
    direction: faster
    why: band B=17..32, same mechanism; 29.07 us shape at 0.124%/us
  - shapes: "*layer=full load=decode_1 *"
    direction: flat
    why: "v3chain's c320 chain kept (17.77 us)"
  - shapes: "*layer=sliding load=decode_1 *"
    direction: flat
    why: per-kvhead kept (10.81 us)
risk: medium
depends_on: merge-perkvhead-v3chain-sv32-tiled
priority: 3
kind: primary
controls_for: null
prior: "merge-perkvhead-v3chain-sv32-tiled: same base (1.5360 check); this adds s1's sliding-verify wgmma module with the s2 split-combine fix so a winning SV16/SV32 route lands on the best code in one check"
review: required
status: queued
---

# sliding-verify-wgmma-split-combine-on-v3chain

## Change

Board-candidate twin of sliding-verify-wgmma-window-s2-split-combine, built on the best code instead of s1's base. Base: merge-perkvhead-v3chain-sv32-tiled (claim restacks its patch onto master 5d528bf3). Then (1) apply muse_attention_decode/hypotheses/done/neutral/sliding-verify-wgmma-window-s1.patch with git apply -3 inside the worktree (both patches are against master 5d528bf3 and touch disjoint code except kernel_fn and the end-of-import setup calls): in kernel_fn keep v3chain's _C320_FN line and tiled pre-attention branch AND s1's _sw_gate sliding-verify route; at the end of import run s1's _sw_setup() after v3chain's piece B under the same 480 s start rule (s1's _SW_*_BY deadlines measured from v3chain's _IMPORT_T0 clock; v3chain ended its phases at 212 s, s1 started its gate at 250 s on its own base). (2) Make exactly the change specified in sliding-verify-wgmma-window-s2-split-combine (route slidewin_verify_wgmma_kernel<false> plus the one-pass parallel sw_combine_kernel, grid (B * QL, HKV), with the shared seq_s / tile_prefix helper; module renamed _v2; gate >= 1.05x in both final rounds, unchanged checks); if that hypothesis is already implemented on its branch, copy its diff. Nothing else changes: per-kvhead SD1, the c320 chain, SV32 tiled GATE B and every master route stay as on v3chain.

## Expected

- `*layer=sliding load=verify_16 *` faster: as the s2 split-combine: kernel-only 14.9-16.1 us vs master path 24.4 us at B=16; shape 22.84 -> ~20 us at 0.21%/us
- `*layer=sliding load=verify_32 *` faster: band B=17..32, same mechanism; 29.07 us shape at 0.124%/us
- `*layer=full load=decode_1 *` flat: v3chain's c320 chain kept (17.77 us)
- `*layer=sliding load=decode_1 *` flat: per-kvhead kept (10.81 us)

## Risk

If the sw route wins, this is the run to submit: ~1.5360 x (1 + 0.6-1.0%) clears the 1.5347 board. Risk medium: a three-way merge plus the s2 change; the import clock (v3chain 212 s + s1's build and gate) must stay under the 480 s start rule, so watch the [sw] phase lines; any overrun or failed check leaves sliding verify on FA3 with a PATH CHOSEN line (the run then reproduces v3chain).
