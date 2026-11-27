"""Check a produced file independently against the original safetensors."""
import sys, json, struct
from pathlib import Path
from math import prod
w,p=map(Path,sys.argv[1:])
with p.open('rb') as f, w.open('rb') as ref:
    h=struct.unpack('<4s9I2f4I6Q16s',f.read(128))
    assert h[:9]==(b'ALF2',1,1536,6912,16,24,8,64,65536)
    tok,emb,table,stride,norm,size=h[-7:-1]; assert stride==128 and size==p.stat().st_size
    assert 128<=tok<emb<table<norm<size
    n=struct.unpack('<Q',ref.read(8))[0];md=json.loads(ref.read(n));base=8+n
    for i in range(16):
        f.seek(table+i*stride);a=struct.unpack('<16Q',f.read(128))
        att=i in (2,5,8,10,12,14);assert a[0]==att
        for pos in a[1:13]:
            assert pos==0 or table+16*stride<=pos<norm and pos%64==0
        assert bool(a[9])!=att
        if att: assert a[5]<a[6]<a[7]<a[8] and a[3] and a[4]
        else:
            assert not any(a[k] for k in (3,4,6,7))
            m=md[f'model.layers.{i}.conv.conv.weight'];ref.seek(base+m['data_offsets'][0])
            data=struct.unpack('<4608H',ref.read(9216))
            f.seek(a[9]);got=struct.unpack('<4608I',f.read(18432))
            assert got==tuple(data[c*3+t]<<16 for t in range(3) for c in range(1536))
        assert 0<a[10]<a[11]<a[12]<norm
    print('PASS: full file length, architecture, 16 descriptors, bounds/alignment, all ten transposed kernels exactly match original BF16')
