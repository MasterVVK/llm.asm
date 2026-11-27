#!/usr/bin/env python3
"""Summarize guest benchmark logs and host sensor logs written by run_vms2.sh.

usage: parse_vm_logs.py <log_prefix> [<log_prefix> ...]
For each prefix reads <prefix>-A76.log / <prefix>-A55.log (guest output) and
<prefix>-sensors.log (host: temperature, A55/A76 and DDR frequency every 5 s).
"""

import re
import statistics
import sys


def runs(path):
    out = []
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except FileNotFoundError:
        return out
    for line in lines:
        m = re.search(r"run \d+ at (\d+)s: .*\(([\d.]+) tok/s", line)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
            continue
        m = re.search(r"run \d+ at (\d+)s: eval_count=(\d+) eval_duration_ns=(\d+)", line)
        if m and int(m.group(3)) > 0:
            out.append((int(m.group(1)), int(m.group(2)) * 1e9 / int(m.group(3))))
    return out


def sensors(path):
    rows = []
    try:
        lines = open(path).read().splitlines()
    except FileNotFoundError:
        return rows
    for line in lines:
        kv = dict(p.split("=") for p in line.split()[1:])
        rows.append({k: int(v) for k, v in kv.items()})
    return rows


for prefix in sys.argv[1:]:
    print(f"== {prefix}")
    for role in ("A76", "A55"):
        r = runs(f"{prefix}-{role}.log")
        if not r:
            continue
        v = [x for _, x in r]
        half = len(v) // 2 or 1
        print(f"  {role}: {len(v)} runs, median {statistics.median(v):.2f} tok/s, "
              f"min {min(v):.2f}, max {max(v):.2f}, first half {statistics.median(v[:half]):.2f}, "
              f"second half {statistics.median(v[half:]):.2f}")
    s = sensors(f"{prefix}-sensors.log")
    if s:
        print(f"  host: temp {min(x['temp'] for x in s) / 1000:.1f}..{max(x['temp'] for x in s) / 1000:.1f} C, "
              f"A76 freq min {min(x['a76'] for x in s) // 1000} MHz, A55 freq min {min(x['a55'] for x in s) // 1000} MHz, "
              f"DDR {sorted(set(x['ddr'] // 1000000 for x in s))} MHz")
