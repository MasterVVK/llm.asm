#!/bin/sh
# usage: run_vms2.sh <asm|ollama> <a76|a55|both> <until_seconds> <log_prefix>
cd /root/llm-asm-arm/vm
ENGINE=$1; MODE=$2; UNTIL=$3; LOG=$4
if [ "$ENGINE" = ollama ]; then INITRD=/vm/initrd-ollama.cpio; MEM=4096; else INITRD=/vm/initrd.cpio; MEM=3072; fi
T=/sys/class/thermal/thermal_zone0/temp
while [ "$(cat $T)" -gt 68000 ]; do sleep 10; done
vm() {  # name cpus role
    docker run --rm --name llmvm-$1 --device /dev/kvm --cpuset-cpus $2 -v /root/llm-asm-arm/vm:/vm:ro llmasm-qemu:local \
        qemu-system-aarch64 -M virt,gic-version=3 -cpu host -enable-kvm -smp 4 -m $MEM -nic none \
        -kernel /vm/vmlinuz -initrd $INITRD \
        -append "console=ttyAMA0 quiet llmrole=$3 llmuntil=$UNTIL" -nographic -no-reboot 2>&1 |
        tr -d "\r" | grep "^\[A" > $LOG-$3.log
}
( while :; do echo "$(date +%s) temp=$(cat $T) a55=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) a76=$(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq) ddr=$(cat /sys/class/devfreq/dmc/cur_freq)"; sleep 5; done ) > $LOG-sensors.log &
S=$!
case $MODE in
    a76) vm a76 4-7 A76 ;;
    a55) vm a55 0-3 A55 ;;
    both) vm a76 4-7 A76 & P1=$!; vm a55 0-3 A55 & P2=$!; wait $P1 $P2 ;;
esac
kill $S
