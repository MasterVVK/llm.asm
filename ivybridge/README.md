# llm.asm on Ivy Bridge (AVX + F16C, no AVX2/FMA)

**English** · [Русский](README.ru.md) · [简体中文](README.zh-CN.md)

A port of [alesha-pro/llm.asm](https://github.com/alesha-pro/llm.asm) (upstream commit `48feccb`) to x86-64 CPUs without AVX2 and FMA. It needs only AVX and F16C, so it runs on Intel Ivy Bridge (2012) and newer, including the Xeon E5 v2 servers that are still common in homelabs and Proxmox clusters. Upstream requires AVX2 + FMA (Haswell or newer) and on these machines stops with `Illegal instruction` (Linux) or `[FATAL] CPU lacks AVX2+FMA` (bare-metal OS).

Both upstream programs are ported: the Linux inference engine (`build/asm-llm`, 15,344 bytes) and the bootable bare-metal OS (`build/asm-llm-os.elf`, 24,888 bytes).

## What changed

- `shared/matmul.asm`: the INT8×INT8 kernel (`FAST_Q8=1`) runs on 128-bit integer SIMD (`vpsignb`, `vpmaddubsw`, `vpmaddwd`). Each 32-byte block is processed as two 16-byte halves whose int32 sums are added exactly before a single conversion to float. |x| of the activation vector is computed once per matmul instead of once per weight row. FMA is replaced by `vmulps` + `vaddps`. The int32→int8 packing and the embedding dequantization work on 128-bit halves.
- `shared/math.asm`: FMA is replaced by multiply + add (`MADD213SD` / `MADD213PS` macros), 2^k in `expf` is built on 128-bit halves, `argmax` keeps indices as floats, and `vbroadcastss` from a register is replaced by the `BCAST_SS` macro.
- `02-baremetal-os/boot.asm`, `kernel.asm`: the CPUID check requires XSAVE + AVX + F16C; banner and error text are updated.
- `FAST_Q8=0` (FP32 activations) is not ported; building it stops with `%error`.
- Checked with a disassembler: neither binary contains a single AVX2 or FMA instruction, and there are no legacy (non-VEX) SSE instructions, so there are no AVX–SSE transition penalties.

## Correctness

- The upstream AVX2 build, run under `qemu-x86_64 -cpu max` emulation, and this port produce **the same 64 of 64 greedy tokens** and byte-identical text on the upstream benchmark prompt. The cosine similarity of the prefill logits is 0.99970: rounding without FMA flips a few activation quantizations. Upstream itself is at 0.99965 against its FP32 reference. See `results/parity-*`.
- Upstream unit tests (`python3 scripts/test_math.py build/test_math`) pass, with every error at or below the upstream README values.
- `ivybridge/test_matmul_k.py` covers the main 4-block loop at model sizes; upstream tests use K = 64 and only reach the tail. For K = 1024 / 2048 / 3072 with FP32 and FP16 scales, the maximum relative error against an exact Q8_0×Q8_0 reference is 4e-8…1.4e-7. Activation quantization itself contributes about 1e-2.
- Output is bit-identical between 1 and 4 threads.

## Results

Hardware: a Proxmox VE 8.4 host with 2× Xeon E5-2696 v2 (Ivy Bridge-EP, 12 cores each) and DDR3-1600. Linux numbers come from a 32-vCPU Ubuntu 22.04 KVM guest on that host. Model: Qwen3-0.6B Q8_0 (`.asmbin` v2 for asm-llm, the official `Qwen3-0.6B-Q8_0.gguf` for llama.cpp). Prompt of 34 tokens and 64 generated tokens, as in the upstream table. llama.cpp: official release b12810 (`ca123ba76`, CPU backend `ivybridge`), `llama-bench -ngl 0`, flash attention `auto` (on; it was faster than off). Ranges cover two sweeps of 3 runs each (`results/linux-vm-run*.csv`), in tokens per second:

| Threads | Decode: this port | Decode: llama.cpp | Gain | Prefill: this port | Prefill: llama.cpp |
|---:|---:|---:|---:|---:|---:|
| 1 | 10.9 | 7.6–8.1 | +36…44% | 14.5 | 20.9–21.9 |
| 4 | 33.7–33.9 | 25.7–26.0 | +31% | 44.3–44.6 | 72–73 |
| 8 | 47.7–49.9 | 39.1–40.0 | +22…25% | 61.5–62.1 | 143–146 |
| 16 | 51.4–54.4 | 39.9–40.9 | +29…33% | 65–69 | 200–223 |
| 24 | 49.5 | 32.2 | +54% | 63.1 | 249 |

Bare-metal OS in a diskless Proxmox VM on the same host (`results/proxmox-os-vm250.txt`): the kernel boots in 170–180 ms and the prompt appears 8 s after the VM starts. On the same prompt it decodes faster than the Linux build by 24% with 1 core, 12% with 4, 4% with 8 and 5% with 12 cores, reaching 55.4 tok/s at 12 cores.

## Limitations

- Prefill runs token by token, so it is 1.5–4× slower than llama.cpp. With long prompts llama.cpp wins on total time.
- Qwen3-0.6B only: the architecture is hard-coded, the context is limited to 2048 tokens, and there is no GPU support or network API.
- **The bare-metal OS keeps all of its vCPUs at 100% even when idle**: worker cores wait in a `pause` loop and there is no `hlt`. On a shared hypervisor, run it only while you use it.
- Under Linux, do not set `-t` above the number of free cores. The default is 32 threads, and on a busy machine throughput drops by more than an order of magnitude (upstream issues #9 and #10).
- F16C is required for the FP16 block scales, so Sandy Bridge (Xeon E5 v1) is not supported.

## Reproduce

```bash
make inference os test-math          # needs nasm and binutils
python3 scripts/test_math.py build/test_math
nasm -f elf64 -DFAST_Q8=1 -Ishared/ ivybridge/test_matmul_k.asm -o build/test_matmul_k.o
ld --gc-sections -s -z noexecstack build/test_matmul_k.o build/syscalls.o build/matmul.o -o build/test_matmul_k
python3 ivybridge/test_matmul_k.py build/test_matmul_k
# model export: see 01-qwen3-inference/README.md
ivybridge/bench.sh build/asm-llm models/qwen3-0.6b-q8_0.asmbin /path/to/llama-bench /path/to/Qwen3-0.6B-Q8_0.gguf "1 4 8 16" 3 out.csv
```

Bootable ISO: `make os-iso MODEL=models/qwen3-0.6b-q8_0.asmbin` (needs `grub-pc-bin grub-efi-amd64-bin mtools xorriso`). A Proxmox VE VM without disk or network:

```bash
qm create 250 --name asm-llm-os --machine pc --bios seabios --cpu host --cores 8 \
  --memory 2048 --balloon 0 --vga std --serial0 socket \
  --sata0 local:iso/asm-llm-os.iso,media=cdrom --boot order=sata0 --ostype other --onboot 0
qm start 250 && python3 ivybridge/serial_bench.py 250 3; qm stop 250
```

Attach the ISO as `sata0`. With `ide2`, as in the upstream instructions, GRUB needs about 64 s to read the 614 MiB model; with `sata0` it needs about 8 s. `serial_bench.py` runs on the Proxmox host: it waits for the `asm-llm>` prompt on the VM serial socket and runs `/bench`.

## License

MIT, as upstream. Original work © 2026 Alexey Fateev ([alesha-pro/llm.asm](https://github.com/alesha-pro/llm.asm)).
