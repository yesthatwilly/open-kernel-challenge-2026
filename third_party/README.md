# Third-party sources

Code and published findings from these repositories are used here under their licenses. Each
directory keeps the upstream LICENSE, NOTICE and the exact commit (SOURCE). Every kernel file that
copies or adapts upstream code keeps the upstream notices in its header.

| Directory | Upstream | Owner | License | How we use it |
|---|---|---|---|---|
| [okc-muse-spark-kernels/](okc-muse-spark-kernels/) | [alygg77/okc-muse-spark-kernels](https://github.com/alygg77/okc-muse-spark-kernels) @ 6cae49a | alygg77 (with vjeux, Nathen Fernandes, Matsvei) | Apache-2.0 (+ MIT for Nathen Fernandes' routes) | Foundation of every `submission.py` since 2026-10-31 (user ruling) |
| [okc-2026-muse-kernels/](okc-2026-muse-kernels/) | [trxphxle/okc-2026-muse-kernels](https://github.com/trxphxle/okc-2026-muse-kernels) @ 72db638 | trxphxle (with vjeux, alygg77, Nathen Fernandes, Matsvei) | Apache-2.0 (+ MIT for Nathen Fernandes' routes) | Studied from 2026-11-02 (campaign `2026-11-02-trx-study`): kernels, measurements, negative results, ceilings |

Upstreams of those: [vjeux/okc-2026](https://github.com/vjeux/okc-2026) (Apache-2.0),
[NathenFernandes/muse-bf16-proj](https://github.com/NathenFernandes/muse-bf16-proj) (MIT), SGLang
(Apache-2.0), NVIDIA CUTLASS (BSD-3-Clause).
