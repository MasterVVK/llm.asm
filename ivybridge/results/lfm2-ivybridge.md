# LFM2-700M on Ivy Bridge: four versus eight physical cores

The same 64-token answer decoded at **37.56 tok/s on four cores** and **39.56 tok/s on eight** (+5.3%). All twelve benchmark runs, including warm-ups, generated the same token IDs. The primary contribution is support for this checkpoint on AVX1, not a claim that doubling the core count doubles throughput.

## Fixed-prompt benchmark

One Dell PowerEdge R720, Intel Xeon E5-2667 v2 (Ivy Bridge-EP, 3.30 GHz nominal). Bare-metal assembly guest under hardware KVM, host CPU passthrough, 2 GiB guest RAM, CPU only. Each vCPU was individually pinned to a distinct physical core on one socket; RAM was bound to its local NUMA node. No SMT siblings in either set. One shared host, not an isolated fleet or a CPU power study.

The guest’s built-in `/bench` uses “Explain how a CPU cache works in 3 concise sentences.” Each invocation starts at position zero. Greedy decoding, 22 prompt tokens, 64 generated tokens. One warm-up then five measured runs per core count; rates are medians of guest-reported tokens/s, not host-side estimates. Wall time is from sending the serial command to seeing the next prompt, excluding boot.

| Physical cores | Prefill (tok/s) | Decode (tok/s) | Decode min–max | Wall (s) |
|---:|---:|---:|---:|---:|
| 4 | 44.26 | 37.56 | 37.11–38.51 | 2.24 |
| 8 | 46.32 | 39.56 | 39.06–40.27 | 2.14 |

[Per-run CSV, including warm-ups](lfm2-bench.csv). Output hashes in both CSVs are SHA-256 over the comma-separated decimal generated token IDs, without a trailing newline.

## Short factual prompts (not a throughput benchmark)

Eight fixed record-reading questions, once per configuration; history cleared before each. Responses were only 1–5 tokens, so their decode rates are much more sensitive to timing overhead. Answers and output IDs matched across core counts for every question, including the four wrong answers. For example, the model supplied a number when the record contained no power measurement. This is a check on repeatability, **not** an accuracy estimate for other tasks.

| Physical cores | Median prefill (tok/s) | Median decode (tok/s) | Median wall (s) | Exact |
|---:|---:|---:|---:|---:|
| 4 | 44.14 | 26.08 | 3.43 | 4/8 |
| 8 | 45.56 | 26.85 | 3.35 | 4/8 |

[Exact prompts, expected/actual answers, timings and token hashes](lfm2-short-tasks.csv). Exact means case-sensitive equality after stripping surrounding whitespace. The table embedded in some prompts is input text, not a measurement of LFM2. The guest reserved 4,096 context slots; these runs reached position 190, not full context. Neither suite compares model families.

## Correctness and backwards compatibility

| Check | Result |
|---|---|
| Build and boot | Linux CLI and Multiboot ELF built; the guest recognized the exported weights as LFM2-700M and reported 16 layers at both core counts. |
| Export and kernel | All 16 descriptors checked; all ten transposed tap arrays match the source checkpoint. At eight widths (including zero, SIMD tails and 1,536), 12 steps × two fresh sequences matched full causal convolution and its state exactly. |
| Numerical reference | Native greedy IDs `1098, 5706, 803` match top-1 from the independent NumPy forward pass using the original BF16 weights. Full-vocabulary logit correlations: 0.999385, 0.999700, 0.999563. This is one prompt and its next two positions, not universal numerical parity. |
| Chat state | Cached second-turn output matches replay of the full token history; `/clear` reproduces both the first prompt IDs and its output. |
| Qwen regression | Built the base and candidate Linux CLIs, then ran the same Qwen3-0.6B ASML weights at two threads. The 34 prompt IDs, all 64 greedy output IDs and answer bytes matched. The inherited math suite and six matrix-kernel cases (K = 1,024, 2,048, 3,072; both scale formats) also passed. |

## Reproduce

1. Follow the [build and export instructions](../LFM2.md). Input checkpoint revision `86f49fc9a3800c3a325b7320bde179c318062583`; exported ALF2 SHA-256 `3b0d5e0ff5872efe12b164bf0ed85cc5e9d6643d6a620d8e64002f0a05bd1605`. No weights are included.
2. Build with `make inference os test-math` (tested with NASM 3.01, GNU binutils 2.45.1). Guest ELF SHA-256 on that build: `873ad3400d9a6325b574bc8ddf1733d06688dcd4ac040667b2db000a49765858`. On a KVM host with AVX + F16C, select one hardware thread per core on the same NUMA node with `lscpu -e=CPU,CORE,SOCKET,NODE`. Use the same four cores as a subset of the eight; pin vCPU threads one-to-one and bind guest memory to that node for a placement comparable to this test.
3. Boot with `make os-run MODEL=models/lfm2-q8.bin RAM=2G SMP=4 CMDLINE='ctx=4096 max_tokens=64 print_tokens=1 raw=0 think=0 temp=0 topp=0.9 seed=42 single_turn=0'`. At the prompt enter `/bench` six times; discard the first. Repeat at `SMP=8` after shutting down the first guest. The make target sets vCPU count, **not** host pinning. Stop the guest when finished: its worker cores spin while idle.
4. For the short suite, use the same command with `max_tokens=24`. Send `/clear` then the CSV’s `prompt` for each case, as a single line. Compare the response with `expected`. Run the [component, reference and replay tests](../LFM2.md#reproduce-the-checks) separately on Linux.
5. For the Qwen comparison, build both `26e5e40` and this branch, and run each CLI on the same Qwen3-0.6B Q8 ASML with `-t 2 -c 256 -n 64 --print-tokens -p "Explain the difference between L1, L2, and L3 CPU caches and why cache locality matters."`. Compare stdout and the `PROMPT_TOKENS` / `GEN_TOKENS` lines, not the timing lines on stderr.
