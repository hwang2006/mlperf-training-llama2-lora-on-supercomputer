"""Per-GPU sanity check: FP32/BF16 GEMMs against an FP64 CPU reference (run inside the image).
Prints one 'GPUCHECK <host> gpu<N> OK|FAIL ...' line per GPU."""
import socket
import torch

torch.manual_seed(0)
a = torch.randn(8192, 8192); b = torch.randn(8192, 8192)
ref = (a.double() @ b.double()).float()
for d in range(torch.cuda.device_count()):
    x, y = a.to(f"cuda:{d}"), b.to(f"cuda:{d}")
    bad = []
    for dt in (torch.float32, torch.bfloat16):
        for _ in range(20):
            r = (x.to(dt) @ y.to(dt)).float().cpu()
            if not torch.isfinite(r).all():
                bad.append(f"{dt} non-finite"); break
            err = ((r - ref).abs().max() / ref.abs().max()).item()
            if err > (1e-3 if dt == torch.float32 else 3e-2):
                bad.append(f"{dt} err={err:.2e}"); break
    print(f"GPUCHECK {socket.gethostname()} gpu{d} {'OK' if not bad else 'FAIL ' + ';'.join(bad)}", flush=True)
