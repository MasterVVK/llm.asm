"""Run ivybridge/kbench at a few shapes and print cycles per (row, token, 32-value block).

usage: python3 ivybridge/kbench.py build/kbench
Build: nasm -f elf64 -DFAST_Q8=1 -Ishared/ ivybridge/kbench.asm -o build/kbench.o
       ld --gc-sections -z noexecstack build/kbench.o build/matmul.o -o build/kbench
Add -DNEWK=<symbol> to time a candidate kernel with the matmul_q8_0_batch_range
signature against the current one and check that the outputs are bit-identical.
"""
import struct, subprocess, sys

exe = sys.argv[1]
for K, M, B, R in [(1024, 8, 8, 20000), (1024, 8, 4, 40000), (3072, 4, 8, 20000), (1024, 64, 8, 2000), (3072, 64, 8, 700)]:
    p = subprocess.run([exe], input=struct.pack('<4Q', K, M, B, R), capture_output=True, check=True)
    cal, bns, gns, nns = struct.unpack_from('<4Q', p.stdout)
    ghz = 30e6 * 10 * 3 / cal                      # 3e7 x 10 dependent imul, 3 cycles each
    blocks = R * M * B * K / 32
    out = p.stdout[32:32 + 4 * M * B]
    new = p.stdout[32 + 4 * M * B:]
    line = f'K={K} M={M} B={B}: {ghz:.2f} GHz | batch {bns * ghz / blocks:5.2f} | gemv {gns * ghz / blocks:5.2f}'
    if nns:
        line += f' | new {nns * ghz / blocks:5.2f} identical={out == new}'
    print(line)
