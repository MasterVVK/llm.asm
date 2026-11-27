"""Native smoke test on the pinned LiquidAI/LFM2-700M weights (no network)."""
import argparse
import re
import subprocess

def run(exe, model, prompt):
    p = subprocess.run([exe, '-m', model, '-p', prompt, '-n', '8', '-t', '2',
                        '-c', '128', '--print-tokens'], capture_output=True,
                       check=True, timeout=120)
    meta = p.stderr.decode()
    print(meta, flush=True)
    ids = list(map(int, re.search(r'PROMPT_TOKENS:\s*([\d,]+)', meta)[1].split(',')))
    assert ids[:4] == [1, 6, 6423, 708] and ids[-5:] == [7, 708, 6, 64015, 708], ids
    return p.stdout.decode().strip()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('exe')
    ap.add_argument('model')
    args = ap.parse_args()
    answer = run(args.exe, args.model, 'What is the capital of France?')
    print('answer:', answer)
    assert 'paris' in answer.lower(), answer
