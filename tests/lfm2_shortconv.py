"""Run against a NASM-assembled ELF shared object. Independent full-sequence oracle."""
import ctypes as C
import random
import sys

rng = random.Random(700)
lib = C.CDLL(sys.argv[1])
f = lib.lfm2_shortconv_step
f.argtypes = [C.POINTER(C.c_float)]*4 + [C.c_size_t]
f.restype = None
def arr(xs):
    return (C.c_float*len(xs))(*xs)

for n in (0,1,7,8,9,31,64,1536):
    # Independent dyadic inputs allow exact reference comparison at every step.
    rand = lambda: rng.randint(-7,7)/8
    kernel = [[rand() for _ in range(3)] for _ in range(n)]
    w = arr([kernel[c][i] for i in range(3) for c in range(n)])
    for _ in range(2):  # Fresh conversation/cache must not leak prior history.
        history=[]; state=arr([0.]*(2*n)); out=arr([0.]*n)
        for t in range(12):
            b,g,x=([rand() for _ in range(n)] for _ in range(3))
            history.append([u*v for u,v in zip(b,x)])
            f(out,arr(b+g+x),w,state,n)
            ref=[g[c]*sum(kernel[c][k]*history[t-2+k][c] for k in range(3) if t-2+k>=0) for c in range(n)]
            assert list(out)==ref, (n,t)
            assert list(state)==(history[t-1] if t else [0.]*n)+history[t],(n,t,'state')
print('PASS: 8 widths, 2 fresh sequences each, 12 decode steps; outputs and cache match full convolution exactly')
