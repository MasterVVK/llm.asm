# llm.asm on AArch64 (Cortex-A76 / Cortex-A55)

**English** · [Русский](README.ru.md) · [简体中文](README.zh-CN.md)

A port of the [llm.asm](https://github.com/alesha-pro/llm.asm) Linux inference engine (Qwen3-0.6B Q8_0) to AArch64 assembly, written from scratch with GNU `as` and `ld`: no C, no libc, a 12 KB static ELF (`build/asm-llm-arm64`). It uses the same model file (`.asmbin` v2), chat template, tokenizer and output format as the x86-64 build.

## What differs from the x86-64 code

- `sdot` (ARMv8.2 dot product) multiplies signed bytes directly, so the x86 sign trick is not needed.
- At load time the weights are repacked with 4 rows interleaved per 4-byte group, the idea behind llama.cpp's `q8_0_4x4`. One `sdot` by element then computes 4 rows, and x is loaded once per 4 rows. The repacked copy keeps the file offsets. `--norepack` switches to the plain row kernel.
- There are two kernel variants: with software prefetch for Cortex-A76 and without it for Cortex-A55, where prefetch costs the in-order core about 35%. Every thread is pinned to one CPU of the `taskset` mask and reads `MIDR_EL1` to pick its variant. The main thread, which does all sequential work, is always placed on a big core.
- Work is distributed dynamically: threads claim 16-row chunks with an LSE `ldadd`, so fast cores take more. The result does not depend on which thread computes which row.
- Threads wait with `ldaxr` + `wfe`.
- The repacked weights are placed on 2 MiB transparent huge pages (`madvise`) by default; `--nohugepages` turns this off.
- Not ported: interactive chat, sampling (greedy only), the bare-metal OS and `FAST_Q8=0`.

## Correctness

- Logits and text are bit-identical on 1×A76, 4×A76, 1×A55, 4×A55 and all 8 cores.
- Against the x86-64 builds, the cosine similarity of the prefill logits is 0.99971. The two x86-64 builds differ from each other at the same level (0.99970). The first 15 of the 64 greedy tokens match the x86-64 output; then a near-tie flips: at that step four candidates lie within 0.23 of each other.

## Results

Board: FriendlyElec CM3588 Plus (RK3588: 4× Cortex-A76 at 2.3 GHz + 4× Cortex-A55 at 1.8 GHz, 32 GB LPDDR4X), Debian 12, kernel 6.1. Qwen3-0.6B Q8_0, prompt of 34 tokens, 64 generated tokens, 3 runs. llama.cpp b12810 was built natively (`-mcpu=cortex-a76.cortex-a55`, dotprod, OpenMP) and measured with `llama-bench`. Cores were selected with `taskset`: cpu0–3 are A55, cpu4–7 are A76.

Decode in tokens per second, with the board's default DDR governor (`dmc_ondemand`) and with DDR fixed at 2400 MHz (`results/*.csv`):

| Cores | This port, default | llama.cpp, default | This port, DDR fixed | llama.cpp, DDR fixed |
|---|---:|---:|---:|---:|
| 1×A76 | 19.7 | 23.0 | 19.6 | 22.7 |
| 2×A76 | 28.0 | 31.5 | 28.1 | 31.6 |
| 4×A76 | **34.3** | 30.5 | **32.7** | 30.7 |
| 1×A55 | 3.0 | 4.8 | 4.5 | 5.7 |
| 2×A55 | 3.5 | 9.7 | 6.2 | 10.0 |
| 4×A55 | 10.3 | 16.0 | 11.0 | 16.1 |
| all 8 | 28.6 | 19.6 | 28.8 | 22.1 |

Prefill with DDR fixed, tokens per second: this port 26.8 / 37.8 / 44.6 on 1 / 2 / 4 A76, 6.1 / 8.5 / 14.8 on 1 / 2 / 4 A55, 35.5 on all 8; llama.cpp 64.4 / 125.8 / 227.7, 14.3 / 28.5 / 56.0 and 156.8.

## Findings

- **Big cores.** llama.cpp is 12–16% faster on 1–2 A76; this port is 7–12% faster on 4 A76, where memory bandwidth is the limit and lower synchronization overhead pays off. 34 tok/s on 4 A76 is the best result on the board.
- **Little cores.** llama.cpp is 1.3–1.6× faster on A55 with DDR fixed.
- **DDR frequency.** With the default governor, work on A55 cores alone keeps the memory at 534 MHz, while any A76 at work raises it to 2400 MHz. A55-only workloads are therefore memory-starved by default: on one A55 this port gains 49% when DDR is fixed.
- **Mixing.** Adding A55 cores to the four A76 slows both engines down, because memory bandwidth is already saturated. Use the 4 big cores: with all cores allowed, `-t 4` picks the A76 automatically.
- **Prefill** is 2.3–5× slower than llama.cpp, because this engine processes the prompt token by token.

## Ollama and two OSes at once

Ollama 0.32.5 got the same Q8_0 GGUF in a private instance; its speed is taken from its own `eval_count` / `eval_duration`. On the host, decode in tokens per second, with the SoC cooled below 70 °C before every measurement (`results/three-engines-host.csv`):

| Cores | This port | llama.cpp | Ollama |
|---|---:|---:|---:|
| 1×A76 | 19.5 | 23.0 | 22.0 |
| 4×A76 | **34.2** | 29.8 | 29.0 |
| 1×A55 | 3.0 | 4.9 | 4.4 |
| 4×A55 | 10.3 | 15.8 | 13.3 |
| all 8 | 28.0 | 22.9 | 22.2 |

Two KVM guests at once (QEMU in Docker, a minimal Linux initramfs in each), one pinned to the four A76 and one to the four A55 (`vm/`, `results/vm/`). Median decode, tokens per second:

| | A76 guest | A55 guest | Sum |
|---|---:|---:|---:|
| This port, each guest alone | 34.2 | 10.0 | — |
| This port, both at once | 21.7 | 6.0 | 27.7 |
| Ollama, each guest alone | 28.0 | 12.7 | — |
| Ollama, both at once | 18.8 | 7.4 | 26.2 |

Both guests together deliver less than the A76 guest alone. Within seconds the A76 guest drops from 30 to 22 tok/s while still at full clock, because the clusters share DDR and the L3. Within a minute the SoC reaches 85 °C and throttles the A76 to 1.8–2.0 GHz and the A55 to 1.2 GHz. KVM itself costs almost nothing: 34.2 tok/s in the guest, 34.3 on the host.

## Automatic threads and Qwen3-1.7B

Without `-t` the engine starts one thread per big core of the `taskset` mask, or one per allowed CPU when the mask has only A55 cores. A plain run on the whole board therefore lands on the four A76: 34.4 tok/s instead of 28.6 with all 8 cores. Work chunks are 16 rows when any big core takes part and 256 rows on A55-only sets: 12.8 instead of 10.3 tok/s on 4×A55. `--chunk-shift n` sets 2^n rows for experiments.

Qwen3-1.7B Q8_0, decode in tokens per second, same method as above:

| Cores | This port | llama.cpp | Ollama |
|---|---:|---:|---:|
| 1×A76 | 7.1 | 8.4 | 8.6 |
| 4×A76 | **11.7** | 10.8 | 11.1 |
| 4×A55 | 5.3 | 6.4 | 6.0 |

## Build and run

```bash
aarch64/build.sh                     # on an AArch64 Linux machine (GNU as + ld)
taskset -c 4-7 build/asm-llm-arm64 -m models/qwen3-0.6b-q8_0.asmbin -p "..." -n 64 -t 4
aarch64/bench_cores.sh build/asm-llm-arm64 <model.asmbin> <llama-bench> <model.gguf> 3 out.csv
```

The model file is the same `.asmbin` v2 as for x86-64 (`scripts/export_weights.py --scale-dtype fp16`).

## License

MIT, as upstream. Original work © 2026 Alexey Fateev ([alesha-pro/llm.asm](https://github.com/alesha-pro/llm.asm)).
