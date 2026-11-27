#!/usr/bin/env bash
# llm-asm vs llama.cpp vs Ollama on big.LITTLE core sets, same Q8_0 weights.
# Waits for the SoC to cool below COOL_MC before every measurement (the board
# throttles at 75 C). A private Ollama instance (own port and model dir) is
# restarted under each taskset mask; the system Ollama service is not touched.
# usage: bench_all.sh <asm-llm> <model.asmbin> <llama-bench> <model.gguf> <ollama-models-dir> [runs] [out.csv]
set -euo pipefail
ASM=$1; ASMBIN=$2; LB=$3; GGUF=$4; OMODELS=$5; RUNS=${6:-3}; OUT=${7:-}
PORT=11500; OMODEL=${OMODEL:-qwen3-06b-q8}; COOL_MC=${COOL_MC:-70000}
CONFIGS=${CONFIGS:-"A76x1:4:1 A76x4:4-7:4 A55x1:0:1 A55x4:0-3:4 all8:0-7:8"}
PROMPT="Explain the difference between L1, L2, and L3 CPU caches and why cache locality matters."
TEMP=/sys/class/thermal/thermal_zone0/temp
median() { sort -n | awk '{a[NR]=$1} END{print (NR%2)? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2}'; }
wait_cool() { while [ "$(cat $TEMP)" -gt "$COOL_MC" ]; do sleep 10; done; }
ollama_stop() {
    local pid
    pid=$(ss -ltnpH "sport = :$PORT" | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1)
    [ -n "$pid" ] && kill -TERM -- "-$(ps -o pgid= -p "$pid" | tr -d ' ')" 2>/dev/null || true
    sleep 2
}
ollama_start() {  # cpus
    ollama_stop
    OLLAMA_HOST=127.0.0.1:$PORT OLLAMA_MODELS=$OMODELS setsid taskset -c "$1" /usr/local/bin/ollama serve > /tmp/ollama-bench.log 2>&1 < /dev/null &
    for _ in $(seq 30); do curl -s -m 1 "http://127.0.0.1:$PORT/api/version" >/dev/null && return; sleep 1; done
}
ollama_gen() {  # threads -> "prefill decode"
    curl -s -m 600 "http://127.0.0.1:$PORT/api/generate" -d "{\"model\":\"$OMODEL\",\"prompt\":\"$PROMPT\",\"stream\":false,\"think\":false,\"keep_alive\":\"10m\",\"options\":{\"num_predict\":64,\"temperature\":0,\"seed\":1,\"num_thread\":$1}}" |
        python3 -I -c 'import json,sys; r=json.load(sys.stdin); print("%.2f %.2f" % (r["prompt_eval_count"]*1e9/r["prompt_eval_duration"], r["eval_count"]*1e9/r["eval_duration"]))'
}
TMP=$(mktemp)
echo "config,cpus,threads,engine,prefill_tok_s,decode_tok_s,max_temp_c" > "$TMP"
for c in $CONFIGS; do
    IFS=: read -r label cpus t <<<"$c"
    # llm-asm
    wait_cool; pf=(); dc=(); hot=0
    for _ in $(seq "$RUNS"); do
        s=$(taskset -c "$cpus" "$ASM" -m "$ASMBIN" -p "$PROMPT" -n 64 -t "$t" 2>&1 >/dev/null)
        pf+=("$(sed -n 's/.*prefill: .* (\([0-9.]*\) tok\/s).*/\1/p' <<<"$s")")
        dc+=("$(sed -n 's/.*decode : .* (\([0-9.]*\) tok\/s, threads.*/\1/p' <<<"$s")")
        hot=$(( $(cat $TEMP) > hot ? $(cat $TEMP) : hot ))
    done
    echo "$label,$cpus,$t,llm-asm,$(printf '%s\n' "${pf[@]}" | median),$(printf '%s\n' "${dc[@]}" | median),$((hot / 1000))" >> "$TMP"
    # llama.cpp
    wait_cool
    read -r pp tg < <(taskset -c "$cpus" "$LB" -m "$GGUF" -p 34 -n 64 -r "$RUNS" -t "$t" -o json 2>/dev/null |
        python3 -I -c 'import json,sys; r=json.load(sys.stdin); print(" ".join("%.2f" % [x["avg_ts"] for x in r if (x["n_prompt"]>0) == k][0] for k in (True, False)))')
    echo "$label,$cpus,$t,llama.cpp,$pp,$tg,$(( $(cat $TEMP) / 1000 ))" >> "$TMP"
    # Ollama
    ollama_start "$cpus"; ollama_gen "$t" > /dev/null     # warm-up: load + repack
    wait_cool; pf=(); dc=()
    for _ in $(seq "$RUNS"); do read -r p d < <(ollama_gen "$t"); pf+=("$p"); dc+=("$d"); done
    echo "$label,$cpus,$t,ollama,$(printf '%s\n' "${pf[@]}" | median),$(printf '%s\n' "${dc[@]}" | median),$(( $(cat $TEMP) / 1000 ))" >> "$TMP"
    tail -n 3 "$TMP" >&2
done
ollama_stop
if [[ -z "$OUT" ]]; then cat "$TMP"; rm -f "$TMP"; else mv "$TMP" "$OUT"; fi
