#!/usr/bin/env bash
# Compare asm-llm (AArch64) and llama.cpp on big.LITTLE core sets.
# usage: bench_cores.sh <asm-llm-arm64> <model.asmbin> <llama-bench> <model.gguf> [runs] [out.csv]
set -euo pipefail
ASM=$1; ASMBIN=$2; LB=$3; GGUF=$4; RUNS=${5:-3}; OUT=${6:-}
PROMPT="Explain the difference between L1, L2, and L3 CPU caches and why cache locality matters."
# label:cpu-list:threads
CONFIGS="A76x1:4:1 A76x2:4-5:2 A76x4:4-7:4 A55x1:0:1 A55x2:0-1:2 A55x4:0-3:4 all8:0-7:8"
median() { sort -n | awk '{a[NR]=$1} END{print (NR%2)? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2}'; }
TMP=$(mktemp)
echo "config,cpus,threads,engine,prefill_tok_s,decode_tok_s" > "$TMP"
for c in $CONFIGS; do
    IFS=: read -r label cpus t <<<"$c"
    pf=(); dc=()
    for _ in $(seq "$RUNS"); do
        s=$(taskset -c "$cpus" "$ASM" -m "$ASMBIN" -p "$PROMPT" -n 64 -t "$t" 2>&1 >/dev/null)
        pf+=("$(sed -n 's/.*prefill: .* (\([0-9.]*\) tok\/s).*/\1/p' <<<"$s")")
        dc+=("$(sed -n 's/.*decode : .* (\([0-9.]*\) tok\/s, threads.*/\1/p' <<<"$s")")
    done
    echo "$label,$cpus,$t,asm-llm,$(printf '%s\n' "${pf[@]}" | median),$(printf '%s\n' "${dc[@]}" | median)" >> "$TMP"
    read -r pp tg < <(taskset -c "$cpus" "$LB" -m "$GGUF" -p 34 -n 64 -r "$RUNS" -t "$t" -o json 2>/dev/null |
        python3 -I -c 'import json,sys; r=json.load(sys.stdin); print(" ".join("%.2f" % [x["avg_ts"] for x in r if (x["n_prompt"]>0) == k][0] for k in (True, False)))')
    echo "$label,$cpus,$t,llama.cpp,$pp,$tg" >> "$TMP"
    tail -n 2 "$TMP" >&2
done
if [[ -z "$OUT" ]]; then cat "$TMP"; rm -f "$TMP"; else mv "$TMP" "$OUT"; fi
