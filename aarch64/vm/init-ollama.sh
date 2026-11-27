#!/bin/busybox sh
# Minimal guest with Ollama: warm up once, then generate in a loop until uptime reaches llmuntil.
/bin/busybox mount -t proc proc /proc
/bin/busybox mount -t sysfs sys /sys
/bin/busybox mount -t devtmpfs dev /dev
/bin/busybox --install -s /bin
/bin/busybox ifconfig lo 127.0.0.1 up
export PATH=/bin:/usr/local/bin HOME=/root OLLAMA_MODELS=/models OLLAMA_HOST=127.0.0.1:11434
ROLE=$(sed -n "s/.*llmrole=\([^ ]*\).*/\1/p" /proc/cmdline)
UNTIL=$(sed -n "s/.*llmuntil=\([0-9]*\).*/\1/p" /proc/cmdline)
PART=$(grep -m1 "CPU part" /proc/cpuinfo | sed "s/.*: //")
echo "[$ROLE] guest up at $(cut -d" " -f1 /proc/uptime)s: $(nproc) vCPU, CPU part $PART"
ollama serve > /tmp/serve.log 2>&1 &
for i in $(seq 60); do wget -q -O /dev/null http://127.0.0.1:11434/api/version && break; sleep 1; done
P="Explain the difference between L1, L2, and L3 CPU caches and why cache locality matters."
J="{\"model\":\"qwen3-06b-q8\",\"prompt\":\"$P\",\"stream\":false,\"think\":false,\"keep_alive\":\"30m\",\"options\":{\"num_predict\":64,\"temperature\":0,\"seed\":1,\"num_thread\":4}}"
gen() { wget -q -O - --header "Content-Type: application/json" --post-data "$J" http://127.0.0.1:11434/api/generate; }
gen > /dev/null
echo "[$ROLE] ollama ready at $(cut -d" " -f1 /proc/uptime)s"
i=0
while :; do
    up=$(cut -d. -f1 /proc/uptime)
    [ "$up" -ge "$UNTIL" ] && break
    i=$((i + 1))
    R=$(gen)
    ec=$(echo "$R" | sed -n "s/.*\"eval_count\":\([0-9]*\).*/\1/p")
    ed=$(echo "$R" | sed -n "s/.*\"eval_duration\":\([0-9]*\).*/\1/p")
    echo "[$ROLE] run $i at ${up}s: eval_count=$ec eval_duration_ns=$ed"
done
echo "[$ROLE] done at $(cut -d" " -f1 /proc/uptime)s"
poweroff -f
