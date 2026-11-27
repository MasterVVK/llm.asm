"""Check native greedy chat state against complete token replay and /clear."""
import argparse
import re
import subprocess


def sequence(meta, label):
    return [[int(x) for x in match.split(',') if x]
            for match in re.findall(rf'{label}: *([0-9,]*)', meta)]


def check(exe, model):
    argv = [exe, '-m', model, '-t', '2', '-c', '256', '-n', '4', '--print-tokens']
    first = 'What is the capital of France?'
    second = 'Name the river in that city.'
    chat = subprocess.run(argv + ['--chat'], input='\n'.join([first, second, '/clear', first, '/quit', '']),
                          text=True, capture_output=True, timeout=120, check=True)
    p = sequence(chat.stderr, 'PROMPT_TOKENS')
    g = sequence(chat.stderr, 'GEN_TOKENS')
    assert len(p) == len(g) == 3 and all(g), (p, g)
    assert p[0][:4] == [1, 6, 6423, 708] and p[1][:3] == [6, 6423, 708]
    assert p[2] == p[0] and g[2] == g[0], 'reset did not restore fresh output'
    ids = p[0] + g[0] + [7, 708] + p[1]
    fresh = subprocess.run(argv + ['--tokens', ','.join(map(str, ids))], capture_output=True,
                           text=True, timeout=120, check=True)
    assert sequence(fresh.stderr, 'PROMPT_TOKENS') == [ids]
    assert sequence(fresh.stderr, 'GEN_TOKENS') == [g[1]], 'cached turn differs from full replay'
    print('PASS: continuation matches full replay; /clear restores fresh prompt and output IDs')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('exe')
    ap.add_argument('model')
    a = ap.parse_args()
    check(a.exe, a.model)
