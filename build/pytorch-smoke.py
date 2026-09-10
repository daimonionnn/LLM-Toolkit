#!/usr/bin/env python3
"""Does PyTorch actually compute on a Vega 8 (gfx90c presented as gfx900)?

Each check exercises one of the three ROCm libraries whose gfx900 support was in
question, and each verifies a NUMBER rather than the absence of an exception --
a kernel that silently returns zeros would pass a "did it crash" test.
"""
import sys, time
import torch

ok = True
def check(name, passed, detail=""):
    global ok
    ok &= passed
    print(f"  {'PASS' if passed else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))

print(f"torch {torch.__version__}, HIP {torch.version.hip}")
if not torch.cuda.is_available():
    print("  FAIL  no GPU visible to torch")
    sys.exit(1)

props = torch.cuda.get_device_properties(0)
print(f"device: {torch.cuda.get_device_name(0)}  arch={props.gcnArchName}  "
      f"CUs={props.multi_processor_count}  mem={props.total_memory/2**30:.1f} GiB")
dev = torch.device("cuda:0")

# ── rocBLAS: matmul, checked against a CPU reference ──────────────────────────
torch.manual_seed(0)
a, b = torch.randn(1024, 1024), torch.randn(1024, 1024)
ref = a @ b
got = (a.to(dev) @ b.to(dev)).cpu()
err = (got - ref).abs().max().item()
check("rocBLAS  fp32 matmul 1024^3", torch.allclose(got, ref, atol=1e-3), f"max abs err {err:.2e}")

# fp16 too: this is the path llama.cpp's attention would use
ah, bh = a.half().to(dev), b.half().to(dev)
errh = ((ah @ bh).float().cpu() - ref).abs().max().item()
check("rocBLAS  fp16 matmul 1024^3", errh < 5.0, f"max abs err {errh:.2e} (fp16 accum)")

# ── rocRAND: GPU-side RNG, the library expected to be the blocker ─────────────
try:
    g = torch.Generator(device=dev).manual_seed(1234)
    r = torch.rand(1 << 20, device=dev, generator=g)
    m, s = r.mean().item(), r.std().item()
    sane = 0.45 < m < 0.55 and 0.27 < s < 0.31 and r.min() >= 0 and r.max() < 1
    check("rocRAND  uniform 2^20", sane, f"mean={m:.4f} std={s:.4f} (expect 0.5 / 0.289)")
except Exception as e:
    check("rocRAND  uniform 2^20", False, f"{type(e).__name__}: {e}")

# ── MIOpen: convolution, checked against a CPU reference ──────────────────────
try:
    x, w = torch.randn(8, 16, 64, 64), torch.randn(32, 16, 3, 3)
    cref = torch.nn.functional.conv2d(x, w, padding=1)
    cgot = torch.nn.functional.conv2d(x.to(dev), w.to(dev), padding=1).cpu()
    cerr = (cgot - cref).abs().max().item()
    check("MIOpen   conv2d 8x16x64x64", torch.allclose(cgot, cref, atol=1e-2), f"max abs err {cerr:.2e}")
except Exception as e:
    check("MIOpen   conv2d 8x16x64x64", False, f"{type(e).__name__}: {e}")

# ── Throughput, for scale ────────────────────────────────────────────────────
n = 2048
x = torch.randn(n, n, device=dev)
for _ in range(3):
    x @ x
torch.cuda.synchronize()
t0 = time.perf_counter()
iters = 20
for _ in range(iters):
    x @ x
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / iters
print(f"\n  fp32 sgemm {n}^3: {dt*1e3:.1f} ms/iter = {2*n**3/dt/1e12:.2f} TFLOP/s")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
