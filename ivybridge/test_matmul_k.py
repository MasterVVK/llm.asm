#!/usr/bin/env python3
"""Exact-reference test of the Q8_0 x Q8_0 matmul at model sizes (K = 1024..3072).

The reference quantizes x the same way the assembly does (float32 max-abs
scale, round-half-even), takes exact int32 block dot products and applies the
per-block scales in float64. The only expected deviation is float32 rounding
of the scale multiply and accumulation, so the error budget is ~1e-6 relative.
"""

import subprocess
import sys

import numpy as np


def quantize_rows(w, fp16):
    m, k = w.shape
    blocks = w.reshape(m, k // 32, 32)
    amax = np.max(np.abs(blocks), axis=2)
    scale = np.where(amax > 0, amax / np.float32(127.0), np.float32(0.0)).astype(np.float32)
    if fp16:
        scale = scale.astype(np.float16)
    eff = scale.astype(np.float32)
    inv = np.where(eff > 0, np.float32(1.0) / eff, np.float32(0.0)).astype(np.float32)
    q = np.clip(np.rint(blocks * inv[:, :, None]), -127, 127).astype(np.int8)
    scale_bytes = scale.astype("<f2" if fp16 else "<f4").view(np.uint8).reshape(m, -1)
    packed = np.concatenate([q.reshape(m, k).view(np.uint8), scale_bytes], axis=1)
    return packed.tobytes(), q, eff


def reference(x, q, eff):
    xb = x.reshape(-1, 32)
    amax = np.max(np.abs(xb), axis=1)
    xs = (amax * np.float32(1.0 / 127.0)).astype(np.float32)
    with np.errstate(divide="ignore"):
        inv = np.where(amax > 0, np.float32(127.0) / amax, np.float32(0.0)).astype(np.float32)
    xq = np.clip(np.rint(xb * inv[:, None]), -127, 127).astype(np.int32)
    dots = np.einsum("mbk,bk->mb", q.astype(np.int32), xq)
    exact = np.sum(dots.astype(np.float64) * (eff.astype(np.float64) * xs.astype(np.float64)), axis=1)
    fp32_truth = np.einsum("mbk,bk->m", q.astype(np.float64) * eff[:, :, None], xb.astype(np.float64))
    return exact, fp32_truth


def main():
    binary = sys.argv[1]
    rng = np.random.default_rng(20261127)
    worst = 0.0
    for k in (1024, 2048, 3072):
        for fp16 in (0, 1):
            m = 256
            x = rng.normal(0.0, 1.5, size=k).astype(np.float32)
            x[rng.integers(0, k, size=k // 16)] *= 8.0      # activation outliers
            x[:32] = 0.0                                     # an all-zero block
            w = rng.normal(0.0, 0.05, size=(m, k)).astype(np.float32)
            w[rng.random((m, k)) < 0.001] *= 20.0            # weight outliers
            packed, q, eff = quantize_rows(w, fp16)
            payload = np.array([k, m, fp16], dtype="<u8").tobytes() + x.tobytes() + packed
            out = np.frombuffer(subprocess.run([binary], input=payload, capture_output=True, check=True).stdout, dtype="<f4")
            exact, truth = reference(x, q, eff)
            rel = float(np.max(np.abs(out - exact)) / np.max(np.abs(exact)))
            q_noise = float(np.max(np.abs(exact - truth)) / np.max(np.abs(truth)))
            worst = max(worst, rel)
            print(f"K={k:4d} M={m} scales={'fp16' if fp16 else 'fp32'}: max rel err vs exact Q8_0xQ8_0 = {rel:.2e}"
                  f"  (activation int8 noise vs FP32 = {q_noise:.2e})")
    assert worst < 1e-5, f"matmul error too high: {worst}"
    print("OK")


if __name__ == "__main__":
    main()
