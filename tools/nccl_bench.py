"""Minimal NCCL all-reduce bandwidth test (torch.distributed), launched by torchrun."""
import os
import time

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
tag = os.environ.get("BENCH_TAG", "?")

for mb in [1, 16, 128, 512]:
    x = torch.ones(mb * 2**20 // 4, device="cuda")
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    t0, n = time.time(), 20
    for _ in range(n):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    busbw = (mb * 2**20 / dt) * 2 * (world - 1) / world / 1e9
    if rank == 0:
        print(f"BENCH {tag} ranks={world} size={mb}MB time={dt*1e3:.2f}ms busbw={busbw:.2f}GB/s", flush=True)

dist.destroy_process_group()
