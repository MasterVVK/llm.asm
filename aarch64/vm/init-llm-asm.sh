#!/bin/busybox sh
# Minimal guest: runs llm-asm in a loop until uptime reaches llmuntil, then powers off.
/bin/busybox mount -t proc proc /proc
/bin/busybox mount -t sysfs sys /sys
/bin/busybox mount -t devtmpfs dev /dev
/bin/busybox --install -s /bin
export PATH=/bin
ROLE=$(sed -n "s/.*llmrole=\([^ ]*\).*/\1/p" /proc/cmdline)
UNTIL=$(sed -n "s/.*llmuntil=\([0-9]*\).*/\1/p" /proc/cmdline)
PART=$(grep -m1 "CPU part" /proc/cpuinfo | sed "s/.*: //")
echo "[$ROLE] guest up at $(cut -d" " -f1 /proc/uptime)s: $(nproc) vCPU, CPU part $PART, $(grep MemTotal /proc/meminfo | tr -s " ")"
P="Explain the difference between L1, L2, and L3 CPU caches and why cache locality matters."
i=0
while :; do
    up=$(cut -d. -f1 /proc/uptime)
    [ "$up" -ge "$UNTIL" ] && break
    i=$((i + 1))
    /opt/asm-llm-arm64 -m /opt/qwen3-0.6b-q8_0.asmbin -p "$P" -n 64 -t 4 2>&1 >/dev/null | grep decode | sed "s/^/[$ROLE] run $i at ${up}s: /"
done
echo "[$ROLE] done at $(cut -d" " -f1 /proc/uptime)s"
poweroff -f
