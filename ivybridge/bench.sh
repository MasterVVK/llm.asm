#!/usr/bin/env bash
# Decode/prefill comparison of asm-llm (this port) and llama.cpp on one host.
# Same model (Qwen3-0.6B Q8_0), same prompt length (34 tokens) and 64 generated
# tokens as the upstream README table; median of RUNS runs per thread count.
#
# usage: bench.sh <asm-llm> <model.asmbin> <llama-bench> <model.gguf> "<threads...>" [runs] [out.csv]
set -euo pipefail

ASM_BIN=$1
ASM_MODEL=$2
LLAMA_BENCH=$3
GGUF=$4
THREADS=$5
RUNS=${6:-3}
OUT=${7:-}
TMP=$(mktemp)

PROMPT="Explain the difference between L1, L2, and L3 CPU caches and why cache locality matters."

median() { sort -n | awk '{a[NR]=$1} END{print (NR%2)? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2}'; }

echo "engine,threads,prefill_tok_s,decode_tok_s,runs" > "$TMP"
for t in $THREADS; do
    pf=(); dc=()
    for _ in $(seq "$RUNS"); do
        stats=$("$ASM_BIN" -m "$ASM_MODEL" -p "$PROMPT" -n 64 -t "$t" 2>&1 >/dev/null)
        pf+=("$(sed -n 's/.*prefill: .* (\([0-9.]*\) tok\/s).*/\1/p' <<<"$stats")")
        dc+=("$(sed -n 's/.*decode : .* (\([0-9.]*\) tok\/s, threads.*/\1/p' <<<"$stats")")
    done
    echo "asm-llm-ivb,$t,$(printf '%s\n' "${pf[@]}" | median),$(printf '%s\n' "${dc[@]}" | median),$RUNS" >> "$TMP"

    # llama-bench reports the mean of -r runs for pp34 and tg64 separately
    read -r pp tg < <("$LLAMA_BENCH" -m "$GGUF" -p 34 -n 64 -r "$RUNS" -t "$t" -ngl 0 -o json 2>/dev/null |
        python3 -I -c 'import json,sys; r=json.load(sys.stdin); pp=[x["avg_ts"] for x in r if x["n_prompt"]>0]; tg=[x["avg_ts"] for x in r if x["n_gen"]>0]; print(f"{pp[0]:.2f} {tg[0]:.2f}")')
    echo "llama.cpp,$t,$pp,$tg,$RUNS" >> "$TMP"
    tail -n 2 "$TMP" >&2
done
if [[ -z "$OUT" ]]; then cat "$TMP"; rm -f "$TMP"; else mv "$TMP" "$OUT"; fi
