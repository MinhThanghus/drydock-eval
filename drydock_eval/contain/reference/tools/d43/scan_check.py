#!/usr/bin/env python3
# Copyright 2026 The drydock-eval Authors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""D4.3 scanner check (stdlib only; runs locally and on the host with the serving Python).
usage: python -I scan_check.py <lib dir with the NEW g4s_d4.py> [--quick] [--out result.json]
  (a) linear time: scan_json_shape() of the new library on the D4.2 escape verifier's attack forms (its fixcheck2.py five
      forms and redos_scan.py three forms) and further adversarial forms, sizes 64 KiB .. 1 MiB (the body cap), best of 3;
      the per-doubling time ratio and the absolute time at 1 MiB; tracemalloc peak at 1 MiB;
  (b) exactness: for valid JSON bodies (random generators incl. the verifier's, escapes, unicode, deep nesting) the
      structural bytes outside strings equal those of the D4.2 regex (on these the regex is fast) and the scanner's container
      count, closing count and depth equal those of the parsed object;
  (c) safety on invalid bodies (random byte mutations of valid bodies): whenever json.loads accepts a mutated body, (b)
      holds for it; otherwise json.loads refuses it anyway;
  (d) the D4.2 regex on the same attack forms at small sizes (quadratic reference; never run above 32k pairs)."""
import json, os, random, re, sys, time, tracemalloc

sys.path.insert(0, sys.argv[1])
import g4s_d4 as D  # noqa: E402

NS = bytes(b for b in range(256) if b not in b'[]{},:')

QUICK = '--quick' in sys.argv
OUTF = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else None
OLD = re.compile(rb'"[^"\\]*(?:\\.[^"\\]*)*"')


def scan_d42(body, max_depth=64, max_containers=20_000, max_values=50_000):
    """lib/g4s_d4.py 81cd2d0e (D4.2) scan_json_shape, verbatim apart from the name (the decision reference)."""
    rest = OLD.sub(b'0', body)
    struct = rest.translate(None, NS)
    n_open = struct.count(b'[') + struct.count(b'{')
    if n_open > max_containers:
        return f'body has over {max_containers} JSON containers'
    if struct.count(b',') + struct.count(b':') > max_values:
        return f'body has over {max_values} JSON values'
    d = mx = 0
    for b in struct:
        if b in (91, 123):
            d += 1
            if d > mx:
                mx = d
                if mx > max_depth:
                    return f'body nesting deeper than {max_depth}'
        elif b in (93, 125):
            d -= 1
    return None


def loads_all(b):
    """json.loads without key de-duplication (a duplicate key's earlier value is still built by json.loads)."""
    return json.loads(b, object_pairs_hook=lambda p: dict(enumerate(v for _, v in p)))
CAP = 1 << 20
R = {'python': sys.version.split()[0], 'lib_sha256': D.sha256_file(os.path.join(sys.argv[1], 'g4s_d4.py'))}


def fill(unit, size, head=b'', tail=b''):
    n = max(0, (size - len(head) - len(tail)) // len(unit))
    return head + unit * n + tail


FORMS = {   # name -> builder(size) giving a body of at most `size` bytes
    'esc_quotes (vd42e K)': lambda s: fill(b'\\"', s, b'"'),
    'esc_quotes_trailing_bs': lambda s: fill(b'\\"', s, b'"', b'\\'),
    'esc_quotes_bs_newline': lambda s: fill(b'\\"', s, b'"', b'\\\n'),
    'bs_newline_each': lambda s: fill(b'"\\\n', s),
    'quote_bs_newline_quote': lambda s: fill(b'"\\\n"', s),
    'esc_any': lambda s: fill(b'\\a', s, b'"'),
    'quote_esc_mix': lambda s: fill(b'"\\"', s),
    'esc_quotes_in_object': lambda s: fill(b'\\"', s, b'{"model":"m","messages":[{"role":"user","content":"'),
    'all_quotes': lambda s: fill(b'"', s),
    'two_byte_strings': lambda s: fill(b'"ab', s),
    'backslash_run_then_quote': lambda s: fill(b'\\', s - 2, b'"') + b'"',
    'escaped_backslash_pairs': lambda s: fill(b'\\\\', s, b'"'),
    'closing_brackets': lambda s: fill(b']', s),
    'opening_brackets': lambda s: fill(b'[', s),
    'commas': lambda s: fill(b',', s),
    'string_list_valid': lambda s: fill(b'"x",', s, b'[', b'"x"]'),
    'u_escapes_valid': lambda s: fill(b'\\u0022', s, b'"', b'"'),
    'huge_valid_string': lambda s: json.dumps({'model': 'm', 'messages': [{'role': 'user', 'content': 'a' * (s - 80)}]}).encode(),
    # the most pieces the string cap lets through (70,001 strings of 13 bytes): cut into pieces, then refused by the value cap
    'strings_at_string_cap': lambda s: fill(b'"abcdefghijklm"', min(s, 70001 * 15)),
    'strings_at_string_cap_no_commas': lambda s: fill(b'"abcdefghijklm" ', min(s, 70001 * 16), b'[', b']'),
    # the D4.2 escape verifier's within-cap bodies (its _within(): vals, dicts, floats, valid), scaled to the size
    'within_vals': lambda s: fill(b'"' + b'v' * 18 + b'",', s, b'{"model":"m","messages":[{"role":"user","content":"x"}],"max_tokens":8,"stop":[', b'"v"]}'),
    'within_dicts': lambda s: fill(b'{"k":"' + b'w' * 40 + b'"},', s, b'{"model":"m","messages":[{"role":"user","content":"x"}],"max_tokens":8,"stop":[', b'{}]}'),
    'within_floats': lambda s: fill(b'0.' + b'1' * 18 + b',', s, b'{"model":"m","messages":[{"role":"user","content":"x"}],"max_tokens":8,"stop":[', b'0]}'),
}
SIZES = [64 << 10, 128 << 10, 256 << 10, 512 << 10, CAP]


def best(f, x, n=3):
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        f(x)
        t.append(time.perf_counter() - t0)
    return min(t)


# (a) linear time and memory
lin = {}
for name, mk in FORMS.items():
    ts = {}
    for s in SIZES:
        b = mk(s)
        assert len(b) <= CAP, (name, len(b))
        ts[len(b)] = best(D.scan_json_shape, b)
    sz = sorted(ts)
    ratios = [ts[sz[i + 1]] / max(ts[sz[i]], 1e-9) for i in range(len(sz) - 1)]
    b = mk(CAP)
    tracemalloc.start()
    D.scan_json_shape(b)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    lin[name] = {'bytes_at_cap': len(b), 'seconds_at_cap': round(ts[sz[-1]], 5), 'seconds_by_size': {k: round(v, 5) for k, v in ts.items()},
                 'doubling_ratios': [round(r, 2) for r in ratios], 'tracemalloc_peak_MB_at_cap': round(peak / 1e6, 2),
                 'decision_at_cap': D.scan_json_shape(b)}
R['linear'] = lin
R['max_seconds_at_cap'] = max(v['seconds_at_cap'] for v in lin.values())
R['max_peak_MB_at_cap'] = max(v['tracemalloc_peak_MB_at_cap'] for v in lin.values())
# a per-byte cost bound: the slowest form at the cap, in ns per byte
R['max_ns_per_byte_at_cap'] = round(max(v['seconds_at_cap'] / v['bytes_at_cap'] for v in lin.values()) * 1e9, 2)

# (b) exactness on valid JSON
random.seed(43)
ALPH = 'ab"\\\n\t/é\U0001F600 {}[],:\x01'


def rstr():
    return ''.join(random.choice(ALPH) for _ in range(random.randint(0, 24)))


def rnd(d=0, maxd=6):
    c = random.random()
    if d >= maxd or c < 0.3:
        return random.choice([1, -2.5e-3, None, True, False, 12345678901234567890, rstr(), rstr()])
    if c < 0.65:
        return [rnd(d + 1, maxd) for _ in range(random.randint(0, 5))]
    return {rstr(): rnd(d + 1, maxd) for _ in range(random.randint(0, 5))}


def rnd_v(d=0):     # the verifier's fixcheck2 generator
    c = random.random()
    if d > 4 or c < 0.3:
        return random.choice([1, 2.5, None, True, ''.join(random.choice('ab"\\\né\U0001F600 /') for _ in range(random.randint(0, 20)))])
    if c < 0.65:
        return [rnd_v(d + 1) for _ in range(random.randint(0, 5))]
    return {''.join(random.choice('k"\\ ') for _ in range(random.randint(1, 5))): rnd_v(d + 1) for _ in range(random.randint(0, 5))}


def obj_counts(o):
    n, mx, st = 0, 0, [(o, 1)]
    if not isinstance(o, (dict, list)):
        return 0, 0
    while st:
        x, d = st.pop()
        n += 1
        mx = max(mx, d)
        for v in (x.values() if isinstance(x, dict) else x):
            if isinstance(v, (dict, list)):
                st.append((v, d + 1))
    return n, mx


def scan_counts(b):
    s = D.json_outside_strings(b).translate(None, NS)
    d = mx = 0
    for c in s:
        if c in (91, 123):
            d += 1
            mx = max(mx, d)
        elif c in (93, 125):
            d -= 1
    return s.count(b'[') + s.count(b'{'), s.count(b']') + s.count(b'}'), mx


def bodies(n):
    for i in range(n):
        g = i % 4
        o = rnd_v() if g == 0 else rnd(maxd=random.choice((3, 6, 12)))
        if g == 3:
            o = {'model': 'm', 'messages': [{'role': 'user', 'content': rstr() * 3}, {'role': 'assistant', 'content': None,
                                                                                      'tool_calls': [{'id': rstr(), 'function': {'name': 'f', 'arguments': json.dumps(rnd())}}]}]}
        yield json.dumps(o, ensure_ascii=random.random() < 0.5, indent=random.choice((None, None, 1))).encode()


N = 4000 if QUICK else 20000


def near_cap():
    """valid bodies at and just over the D4.2 count caps (strings as values, keys, nested one-string arrays)"""
    yield json.dumps(['s'] * 50001).encode()                        # 50,000 commas: within
    yield json.dumps(['s'] * 50002).encode()                        # 50,001 commas: over (both versions)
    yield json.dumps({f'k{i}': 'v' for i in range(25000)}).encode()  # 25,000 ':' + 24,999 ',': within; 50,000 strings
    yield json.dumps({f'k{i}': 'v' for i in range(25001)}).encode()  # over
    yield json.dumps([['s']] * 19999).encode()                      # 20,000 containers: within
    yield json.dumps([['s']] * 20000).encode()                      # 20,001: over
    yield json.dumps([['s', 's']] * 16000).encode()                 # 16,001 containers, 31,999 commas, 32,000 strings
    yield json.dumps([{'k': 's', 'k2': 's'}] * 12000).encode()      # 12,001 containers, 47,999 values, 48,000 strings
    yield json.dumps([{'k': 's', 'k2': 's'}] * 12500).encode()      # 50,999 values: over
    yield json.dumps({'model': 'm', 'messages': [{'role': 'user', 'content': 'a"\\' * 60000}]}).encode()
    yield (b'[' + b'[' * 63 + b'"x"' + b']' * 64)                  # depth 64: within
    yield (b'[' + b'[' * 64 + b'"x"' + b']' * 65)                  # depth 65: over


ex = {'n': 0, 'old_regex_structural_bytes_equal': 0, 'containers_closings_depth_equal_to_parsed': 0, 'decision_none_equal_to_d42': 0,
      'mismatch_examples': []}
import itertools  # noqa: E402
for b in itertools.chain(near_cap(), bodies(N)):
    ex['n'] += 1
    old = OLD.sub(b'0', b).translate(None, NS)
    new = D.json_outside_strings(b).translate(None, NS)
    ex['old_regex_structural_bytes_equal'] += old == new
    ex['decision_none_equal_to_d42'] += (scan_d42(b) is None) == (D.scan_json_shape(b) is None)
    o = loads_all(b)
    nc, md = obj_counts(o)
    so, sc, sd = scan_counts(b)
    ok = so == nc and sc == nc and sd == md
    ex['containers_closings_depth_equal_to_parsed'] += ok
    if (old != new or not ok) and len(ex['mismatch_examples']) < 5:
        ex['mismatch_examples'].append(b[:200].decode('utf-8', 'replace'))
R['exact_valid'] = ex

# (c) mutated bodies
random.seed(4343)
mut = {'n': 0, 'accepted_by_json': 0, 'accepted_and_exact': 0, 'mismatch_examples': []}
PIECES = [b'"', b'\\', b'\\"', b'\\\\', b'[', b']', b'{', b'}', b',', b':', b'\n', b'\\u0022']
for b in bodies(N // 2):
    for _ in range(4):
        x = bytearray(b)
        for _ in range(random.randint(1, 4)):
            p = random.randrange(len(x) + 1)
            op = random.random()
            if op < 0.4:
                x[p:p] = random.choice(PIECES)
            elif op < 0.7 and len(x) > p:
                del x[p:p + random.randint(1, 3)]
            elif len(x) > p:
                x[p] = random.choice(b'"\\[]{},:a')
        x = bytes(x)
        mut['n'] += 1
        try:
            o = loads_all(x)
        except (ValueError, RecursionError):
            continue
        mut['accepted_by_json'] += 1
        nc, md = obj_counts(o)
        so, sc, sd = scan_counts(x)
        if so == nc and sc == nc and sd == md:
            mut['accepted_and_exact'] += 1
        elif len(mut['mismatch_examples']) < 5:
            mut['mismatch_examples'].append(x[:200].decode('utf-8', 'replace'))
R['mutated'] = mut

# (d) the D4.2 regex (reference, small sizes only)
oldt = {}
for name in ('esc_quotes (vd42e K)', 'esc_quotes_trailing_bs', 'esc_quotes_bs_newline', 'bs_newline_each', 'quote_bs_newline_quote'):
    oldt[name] = {}
    for k in ((2000, 4000) if QUICK else (4000, 8000, 16000)):
        b = FORMS[name](2 * k + 1)
        oldt[name][len(b)] = round(best(lambda x: OLD.sub(b'0', x), b, 1), 4)
R['old_regex_seconds'] = oldt

R['pass'] = (R['max_seconds_at_cap'] < 0.1 and all(max(v['doubling_ratios']) < 3.0 or v['seconds_at_cap'] < 0.005 for v in lin.values())
             and ex['old_regex_structural_bytes_equal'] == ex['n'] and ex['containers_closings_depth_equal_to_parsed'] == ex['n']
             and ex['decision_none_equal_to_d42'] == ex['n']
             and mut['accepted_and_exact'] == mut['accepted_by_json'])
print(json.dumps({k: R[k] for k in ('python', 'pass', 'max_seconds_at_cap', 'max_ns_per_byte_at_cap', 'max_peak_MB_at_cap')}))
print(json.dumps({k: (v['seconds_at_cap'], v['doubling_ratios'], v['tracemalloc_peak_MB_at_cap'], v['decision_at_cap']) for k, v in lin.items()}, indent=0))
print(json.dumps({'exact_valid': {k: v for k, v in ex.items() if k != 'mismatch_examples'}, 'mutated': {k: v for k, v in mut.items() if k != 'mismatch_examples'},
                  'old_regex_seconds': oldt, 'mismatches': ex['mismatch_examples'][:2] + mut['mismatch_examples'][:2]}))
if OUTF:
    with open(OUTF, 'w') as f:
        json.dump(R, f, indent=1)
