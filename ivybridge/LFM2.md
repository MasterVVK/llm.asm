# LFM2-700M on Ivy Bridge

This branch runs the original LiquidAI/LFM2-700M checkpoint in the static Linux executable and the Multiboot guest. The Qwen ASML reader remains available. The new ALF2 reader is specific to this model: 16 blocks (six grouped-query attention, ten gated depthwise convolutions), 1,536 hidden dimensions and tied embeddings. Each convolution retains two steps of state per channel. This is not generic support for the LFM family.

## Build and run

Start with the original BF16 model.safetensors, tokenizer.json and config.json from [the tested checkpoint](https://huggingface.co/LiquidAI/LFM2-700M/tree/86f49fc9a3800c3a325b7320bde179c318062583). Check the publisher’s license and hashes; no weights are included here. The converter requires Python 3 and NumPy. On Linux x86-64, with NASM and GNU binutils:

```sh
python3 scripts/export_lfm2.py --model-dir /path/to/checkpoint --out models/lfm2-q8.bin
make inference os
build/asm-llm -m models/lfm2-q8.bin --chat -c 4096 -t 4 -n 32
# On a KVM host with AVX and F16C:
make os-run MODEL=models/lfm2-q8.bin RAM=2G SMP=4 CMDLINE='ctx=4096 max_tokens=32'
```

The default export is 798,229,440 bytes for the pinned model. Matrices are row-split Q8 with FP16 scales; norms and the convolution taps are FP32. The export supports 1–4,096 context slots. That is an allocation limit, **not** a measured useful retrieval length. Chat supports user/assistant turns and reset. It does not expose the publisher’s full tool or system-message API. The existing byte/BPE path does not implement all upstream regex pre-tokenization boundaries; arbitrary text is not guaranteed to produce identical IDs.

## Reproduce the checks

```sh
nasm -f elf64 -Ishared/ shared/lfm2_shortconv.asm -o build/lfm2-shortconv-test.o
ld -shared build/lfm2-shortconv-test.o -o build/lfm2-shortconv-test.so
python3 tests/lfm2_shortconv.py ./build/lfm2-shortconv-test.so
python3 tests/lfm2_export.py /path/to/checkpoint/model.safetensors models/lfm2-q8.bin
python3 tests/lfm2_engine.py build/asm-llm models/lfm2-q8.bin
python3 tests/lfm2_chat.py build/asm-llm models/lfm2-q8.bin
OPENBLAS_NUM_THREADS=1 python3 tests/lfm2_reference.py build/asm-llm models/lfm2-q8.bin /path/to/checkpoint/model.safetensors
```

The independent full-sequence NumPy reference reads the original BF16 weights. For one 16-token input and two subsequent greedy steps, the native and reference top-1 IDs matched at all three positions (full-vocabulary logit correlations 0.999385, 0.999700, 0.999563). The chat check compares a cached continuation with a fresh replay of the same whole token history; clearing the session must reproduce the first turn’s IDs.

See the [four- and eight-core test report](results/lfm2-ivybridge.md) for hardware placement, per-run data, Qwen regression checks and reproduction instructions.
