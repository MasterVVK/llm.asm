"""Independent full-sequence F32 oracle using the original BF16 safetensors.

Tests the native engine on identical IDs (no tokenizer dependency), including
two generated positions. Report centered logit correlation, RMSE and top IDs;
quantized weights/activations need not reproduce BF16 bit-for-bit.
Usage: OPENBLAS_NUM_THREADS=1 python tests/lfm2_reference.py ELF ALF2 SAFETENSORS
"""
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

import numpy as np

PROMPT = [1, 6, 6423, 708, 3493, 856, 779, 5706, 803, 4481, 540, 7, 708, 6, 64015, 708]


class Weights:
    def __init__(self, path):
        with open(path, 'rb') as f:
            size, = struct.unpack('<Q', f.read(8))
            self.meta = json.loads(f.read(size))
        self.path, self.base = path, size + 8

    def __call__(self, name, rows=None):
        m = self.meta[name + '.weight']
        assert m['dtype'] == 'BF16'
        a = np.memmap(self.path, dtype='<u2', mode='r', shape=tuple(m['shape']),
                      offset=self.base + m['data_offsets'][0])
        data = a if rows is None else a[rows]
        return (data.astype(np.uint32) << 16).view(np.float32)


def norm(x, w):
    return x * np.reciprocal(np.sqrt((x*x).mean(axis=-1, keepdims=True) + 1e-5)) * w


def rope(x):
    t, _, dim = x.shape
    freq = np.float32(1000000.) ** (-np.arange(0, dim, 2, dtype=np.float32) / dim)
    angle = np.arange(t, dtype=np.float32)[:, None, None] * freq
    c, s = np.cos(angle), np.sin(angle)
    a, b = np.split(x, 2, axis=-1)
    return np.concatenate((a*c-b*s, a*s+b*c), axis=-1)


def reference(w, ids):
    x = w('model.embed_tokens', ids)
    t = len(ids)
    for i in range(16):
        def weight(name):
            return w(f'model.layers.{i}.' + name)
        z = norm(x, weight('operator_norm'))
        if i in (2, 5, 8, 10, 12, 14):
            def proj(which, heads):
                return (z @ weight('self_attn.'+which+'_proj').T).reshape(t, heads, 64)
            q = rope(norm(proj('q', 24), weight('self_attn.q_layernorm')))
            k = rope(norm(proj('k', 8), weight('self_attn.k_layernorm')))
            v = proj('v', 8)
            logits = np.einsum('thd,shd->hts', q, np.repeat(k, 3, axis=1))/8
            logits[:, np.triu_indices(t, 1)[0], np.triu_indices(t, 1)[1]] = -np.inf
            att = np.exp(logits - logits.max(axis=-1, keepdims=True))
            att /= att.sum(axis=-1, keepdims=True)
            y = np.einsum('hts,shd->thd', att, np.repeat(v, 3, axis=1)).reshape(t, -1)
            x += y @ weight('self_attn.out_proj').T
        else:
            b, c, a = np.split(z @ weight('conv.in_proj').T, 3, axis=-1)
            history = np.pad(b*a, ((2, 0), (0, 0)))
            taps = weight('conv.conv')[:, 0, :]
            y = sum(history[j:j+t] * taps[:, j] for j in range(3))
            x += (y*c) @ weight('conv.out_proj').T
        z = norm(x, weight('ffn_norm'))
        gate = z @ weight('feed_forward.w1').T
        up = z @ weight('feed_forward.w3').T
        x += (gate / (1 + np.exp(-np.clip(gate, -85, 85))) * up) @ weight('feed_forward.w2').T
    z = norm(x, w('model.embedding_norm'))
    return np.concatenate([z @ w('model.embed_tokens', slice(r, r+1024)).T
                           for r in range(0, 65536, 1024)], axis=-1)


def native(exe, model, ids, out):
    run = subprocess.run([exe, '-m', model, '--tokens', ','.join(map(str, ids)),
                          '--print-tokens', '--dump-logits', str(out),
                          '-n', '3', '-t', '2', '-c', '64'],
                         check=True, capture_output=True, timeout=120)
    generated = re.search(rb'GEN_TOKENS:([\d,]+)', run.stderr)[1]
    scores = np.fromfile(out, dtype='<f4')
    assert scores.shape == (65536,) and np.isfinite(scores).all()
    return list(map(int, generated.split(b','))), scores


if __name__ == '__main__':
    exe, model, original = sys.argv[1:]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'logits'
        generated, first = native(exe, model, PROMPT, path)
        assert len(generated) == 3
        ids = PROMPT + generated[:2]
        expected = reference(Weights(original), ids)
        for step in range(3):
            scores = first if not step else native(exe, model, ids[:len(PROMPT)+step], path)[1]
            ref = expected[len(PROMPT)-1+step]
            corr = float(np.corrcoef(scores, ref)[0, 1])
            rmse = float(np.sqrt(np.mean((ref-scores)**2)))
            top1 = int(np.argmax(scores))
            ref_top = np.argsort(ref)[-5:][::-1].tolist()
            print(f'step={step} correlation={corr:.6f} RMSE={rmse:.5f} native={top1} BF16_top5={ref_top}', flush=True)
            assert top1 == generated[step] == ref_top[0], (step, top1, generated[step], ref_top)
            assert corr > .99, (step, corr)
        print("PASS: native greedy IDs match BF16 top-1 at all three positions")
