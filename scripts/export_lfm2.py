"""Export the bias-free tied LFM2-700M checkpoint. No inference or remote code.

ALF2 v1 uses the 128-byte ASMBIN header layout, except off_layers is a table
of 128-byte descriptors and layer_stride is the *descriptor* stride.
All table offsets are absolute, 64-byte aligned. Zero means not applicable.
The tag is deliberately distinct: an unmodified Qwen loader must reject it.
"""
import argparse
import json
import os
import struct
import tempfile
from pathlib import Path
import numpy as np
from export_weights import build_tokenizer_blob, HT_MASK, pad64, load_safetensors_header, quantize_matrix_q8_0_row_split, read_tensor_f32

DESCRIPTOR = struct.Struct('<16Q')
HEADER = struct.Struct('<4s9I2f4I6Q16s')
assert HEADER.size == DESCRIPTOR.size == 128

def export(src, filename, ctx):
    c=json.loads((src/'config.json').read_text())
    expected={'model_type':'lfm2','hidden_size':1536,'num_hidden_layers':16,'num_attention_heads':24,'num_key_value_heads':8,
              'vocab_size':65536,'conv_L_cache':3,'conv_bias':False,'full_attn_idxs':[2,5,8,10,12,14],'bos_token_id':1,'eos_token_id':7}
    if any(c.get(k)!=v for k,v in expected.items()) or not c.get('tie_word_embeddings',True) or not 1<=ctx<=4096:
        raise ValueError('unsupported architecture or context')
    d,n,h,v,hd,kv=1536,16,6912,65536,64,512
    weights=src/'model.safetensors'
    hdr,base=load_safetensors_header(weights)
    if 'lm_head.weight' in hdr: raise ValueError('explicit output head not supported')
    filename.parent.mkdir(exist_ok=True,parents=True)
    tmp=tempfile.NamedTemporaryFile(mode='wb',dir=filename.parent,delete=False)
    try:
        with tmp as f, weights.open('rb') as sf:
            def put(name,shape,mode):
                entry=hdr[name]
                if entry['dtype']!='BF16' or entry['shape']!=list(shape):
                    raise ValueError('unexpected input tensor: '+name)
                begin,end=entry['data_offsets']
                if begin<0 or end-begin!=2*int(np.prod(shape)) or base+end>weights.stat().st_size:raise ValueError('bad tensor bounds: '+name)
                pos=pad64(f)
                if mode=='q8':
                    # Row-split layout; chunk the large embedding, no multi-GB temporaries.
                    sf.seek(base+begin)
                    for first in range(0,shape[0],128):
                        rows=min(128,shape[0]-first); count=rows*shape[1]
                        raw=sf.read(2*count)
                        if len(raw)!=2*count: raise ValueError('short tensor')
                        mat=(np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16).view(np.float32).reshape(rows,shape[1])
                        f.write(quantize_matrix_q8_0_row_split(mat,'fp16'))
                else:
                    t=read_tensor_f32(sf,hdr,base,name)
                    if mode=='taps':t=t[:,0,:].T
                    f.write(t.astype('<f4').tobytes(order='C'))
                return pos
            f.write(bytes(HEADER.size))
            tok=pad64(f);f.write(build_tokenizer_blob(str(src/'tokenizer.json'),v))
            emb=put('model.embed_tokens.weight',(v,d),'q8')
            table=pad64(f);f.write(bytes(n*DESCRIPTOR.size))
            descriptors=[]
            for i in range(n):
                prefix='model.layers.'+str(i)+'.'
                def one(name,shape,mode): return put(prefix+name+'.weight',shape,mode)
                att=i in expected['full_attn_idxs']
                fields=[int(att),one('operator_norm',(d,),'fp32'),one('ffn_norm',(d,),'fp32')]
                if att:
                    qn=one('self_attn.q_layernorm',(hd,),'fp32');kn=one('self_attn.k_layernorm',(hd,),'fp32')
                    qp=one('self_attn.q_proj',(d,d),'q8');kp=one('self_attn.k_proj',(kv,d),'q8');vp=one('self_attn.v_proj',(kv,d),'q8')
                    op=one('self_attn.out_proj',(d,d),'q8');taps=0
                else:
                    qn=kn=kp=vp=0
                    qp=one('conv.in_proj',(3*d,d),'q8');op=one('conv.out_proj',(d,d),'q8');taps=one('conv.conv',(d,1,3),'taps')
                gate=one('feed_forward.w1',(h,d),'q8');up=one('feed_forward.w3',(h,d),'q8');down=one('feed_forward.w2',(d,h),'q8')
                descriptors.append(DESCRIPTOR.pack(*fields,qn,kn,qp,kp,vp,op,taps,gate,up,down,0,0,0))
            norm=put('model.embedding_norm.weight',(d,),'fp32');size=pad64(f)
            f.seek(table);f.writelines(descriptors)
            f.seek(0);f.write(HEADER.pack(b'ALF2',1,d,h,n,24,8,hd,v,ctx,c['norm_eps'],c['rope_theta'],7,2,HT_MASK,7,tok,emb,table,DESCRIPTOR.size,norm,size,bytes(16)))
        os.replace(tmp.name,filename)
    finally:
        if os.path.exists(tmp.name):os.unlink(tmp.name)
    return size

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model-dir',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);ap.add_argument('--context',type=int,default=4096)
    a=ap.parse_args();print('written',export(a.model_dir,a.out,a.context),'bytes')
