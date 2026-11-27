#!/usr/bin/env python3
"""Drive the asm-llm OS REPL over a Proxmox VM serial socket and run /bench.

Runs on the Proxmox host:  python3 serial_bench.py <vmid> [runs]
Prints the boot banner and every /bench result line.
"""

import socket
import sys
import time

vmid = sys.argv[1]
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(f"/var/run/qemu-server/{vmid}.serial0")
sock.settimeout(1.0)
buf = b""


def read_until(marker, timeout, poke=None):
    global buf
    deadline = time.time() + timeout
    next_poke = time.time() + 150
    while time.time() < deadline:
        if marker in buf:
            head, _, buf = buf.partition(marker)
            return head
        if poke and time.time() > next_poke:
            sock.sendall(poke)
            next_poke = time.time() + 150
        try:
            chunk = sock.recv(65536)
        except socket.timeout:
            continue
        if not chunk:
            break
        buf += chunk
    raise SystemExit(f"timeout waiting for {marker!r}; got: {buf[-2000:]!r}")


t0 = time.time()
boot = read_until(b"asm-llm> ", 180, poke=b"\r")
print(f"[host] prompt after {time.time() - t0:.1f} s since connect")
print(boot.decode("utf-8", "replace"))
time.sleep(1.0)
try:
    while sock.recv(65536):
        pass
except socket.timeout:
    pass
buf = b""
for i in range(runs):
    sock.sendall(b"/bench\r")
    out = read_until(b"[asm-llm-os] decode", 120)
    out += b"[asm-llm-os] decode" + read_until(b"\n", 10)
    read_until(b"asm-llm> ", 30)
    lines = [l.strip() for l in out.decode("utf-8", "replace").splitlines() if "[asm-llm-os]" in l]
    print(f"[bench {i + 1}]", " | ".join(lines))
