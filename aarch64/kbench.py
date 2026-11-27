"""Run aarch64/kbench at a few shapes and print cycles per (row, token, 32-value block).

usage: taskset -c 4 python3 aarch64/kbench.py build/kbench 1    # A76, prefetching kernels
       taskset -c 0 python3 aarch64/kbench.py build/kbench 0    # A55, no prefetch
Build (from aarch64/ after build.sh):
       as -o ../build/kbench.o kbench.s      (not in build/aarch64: build.sh links every .o there)
       ld -static -o ../build/kbench ../build/kbench.o ../build/aarch64/matmul.o
Assemble with --defsym NEWK=1 to time candidate kernels named
matmul_q8_0_4x4t_{pf,nopf} against the current ones and check the outputs.
"""
import struct, subprocess, sys

exe, pf = sys.argv[1], int(sys.argv[2])
for K, M, B, R in [(1024, 16, 8, 4000), (1024, 16, 16, 2000), (3072, 8, 16, 1000), (1024, 128, 16, 250)]:
    p = subprocess.run([exe], input=struct.pack('<5Q', K, M, B, R, pf), capture_output=True, check=True)
    cal, bns, gns, nns = struct.unpack_from('<4Q', p.stdout)
    ghz = 20e6 * 10 / cal                          # 2e7 x 10 dependent adds, 1 cycle each
    blocks = R * M * B * K / 32
    out = p.stdout[32:32 + 4 * M * B]
    new = p.stdout[32 + 4 * M * B:]
    line = f'K={K} M={M} B={B}: {ghz:.2f} GHz | batch {bns * ghz / blocks:5.2f} | gemv {gns * ghz / blocks:5.2f}'
    if nns:
        line += f' | new {nns * ghz / blocks:5.2f} identical={out == new}'
    print(line)
