# Hangang configs (32 and 64 nodes)

Krai's v6.0 configs cover 1, 2, 4, 8 and 16 nodes and are used unchanged from
`third_party/krai60/...`. For 32 and 64 nodes Krai published no config; these files follow the
same format. `config_gh200.sh` and `config_common.sh` are copied here from Krai's n16
directory by `scripts/00_fetch_sources.sh`.

`GBS = MINIBS * (GPUs / (TP*PP)) / CP`

| File | Nodes | TP | CP | CP comm | GBS | LR / max steps | Status |
|---|---:|---:|---:|---|---:|---|---|
| `config_hangang_n32_tp4pp1cp2.sh` | 32 | 4 | 2 | a2a | 16 | 0.00055 / 800 (Krai) | **official rehearsal** (3.921 min) |
| `config_hangang_n64_tp4pp1cp2_lr3e-4.sh` | 64 | 4 | 2 | a2a | 32 | 0.0003 / 1024 | **official rehearsal** (3.434 min) |
| `config_hangang_n32_tp4pp1cp4.sh` | 32 | 4 | 4 | p2p | 8 | 0.00055 / 800 | tested: 4.319 min (slower convergence) |
| `config_hangang_n64_tp4pp1cp4.sh` | 64 | 4 | 4 | p2p | 16 | 0.00055 / 800 | tested once: 22.6 samples/s, not used |

Why not GBS 32 with Krai's LR at 64 nodes: it converged in 4992 samples, below the RCP lower
limit of 5802 for GBS 32 (the package checker would reject it). LR 0.0003 with 1024 max steps
converges at ~6100-6900 samples and passes.

Why CP4 needs `p2p`: Llama2-70B has 8 KV heads, i.e. 2 per GPU at TP=4; the default `a2a`
context parallelism requires the head count to be divisible by CP.
