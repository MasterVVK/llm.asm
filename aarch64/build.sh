#!/bin/sh
# Build the AArch64 port natively (GNU as + ld, no C, no libc).
set -e
cd "$(dirname "$0")"
OUT=../build/aarch64
mkdir -p "$OUT"
for f in sys math matmul threads tokenizer transformer main; do
    as -o "$OUT/$f.o" "$f.s"
done
ld -static --gc-sections -s -z noexecstack -o ../build/asm-llm-arm64 "$OUT"/*.o
ls -l ../build/asm-llm-arm64
