# Rank ladder v3: LOG

## Now (2026-10-30T14:50Z)

- **Mode:** H100 first (user 14:35Z): tokens only on advancing H100 trials; gpunode06 any GPU (0/2 held: overheat at idle).
  Judge checks only for unions that held on the H100.
- **Boards:** attention 1.805 (525/1130), bf16 1.3425 (466/1471), decode 1.699 (469/1039), moe 2.100 (340/1186),
  prefill 1.572 (274/691). Overall 472/1690.
- **Quota 14:29Z:** checks left 66/100, scored 26/50.
- **Lanes:** see LANES.tsv (GPU 1 MO3-S2, 3 VT-D16S, 4 D-ATTN, 5 BF-VT; queue on 6/7 runs rl-d-moe, rl-d-decode).
- **Runner:** hopper-queue restarted 14:45Z (poll 120 s); queue/reserved `gpunode06 0,1,2,3,4,5`.
- **User approvals 10-30:** H100 on gpunode06 freely; D (public files on gpunode06, measurement only); VT rule (public values
  seed sweep ranges only; we keep our own measured optimum).
- **Resume:** PLAN.md "Resume" section.
