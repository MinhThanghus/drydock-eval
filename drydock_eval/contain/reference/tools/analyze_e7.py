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
"""E7 analysis: the frozen decision rules of PROTOCOL.md ("Decision rules") with PROTOCOL_AMENDMENTS.md A0-A9, and the secondary metrics of the pre-registration.

Rules: --rules v2.2 (default) applies PROTOCOL_AMENDMENTS.md A0-A9 (gate_v22, screening_v22, rerun_jobs_v22, records_v22,
parity_ratio_v22; see the "Rules v2.2" block); --rules v2.1 reproduces the A0-A8 implementation exactly (gate_v21,
screening_v21, rerun_jobs_v21, parity --rules v2.1); --rules v2 reproduces the A0-A7 implementation exactly (gate_v2,
screening_v2); --rules v1 reproduces the first implementation of PROTOCOL.md exactly (gate, screening; described below).

Subcommands (each writes <out>.json and <out>.md; v2.1/v2.2 also: reruns, parity, error-table --rules v2|v2.1|v2.2;
v2.2 also: records, the blind record check of A9):
  screen     screening(results, panel_S): gain per arm vs bundle0, edit-call success, mean agent time, finalist.
  gate       gate(results, panel, arm_a, arm_b): task-weighted gain, null-imposed wild repository-cluster
             bootstrap-t (Rademacher, 9,999 draws, one-sided), repository-equal and leave-one-repository-out
             estimates, positive repositories, agent-time ratio, harness errors, PASS/FAIL; plus secondary metrics.
  secondary  secondary(results, traces): descriptive metrics with paired task differences and repository-cluster
             intervals.

A run directory is either the directory that holds results.jsonl (rows written by worker_berth.py) or its parent
(holding out/results.jsonl). Traces are read from <out>/<arm>/s<seed>/traces/trace_<task>.json.

Frozen choices made here (documented in ANALYSIS_VALIDATION.md, written before any E7 outcome existed):
  * Solve = the row's `resolved`. A planned (arm, task, repeat) with no row is a MISSING run: it is excluded from
    the task's mean (repeats are averaged over completed runs), listed, and counted as a harness failure for
    criterion (4). A row that ended in a harness error (worker exception, sandbox/evaluation/infrastructure error,
    unknown error text) is a completed, unsolved run and is counted as a harness error.
  * A task without any completed run in one of the two arms is excluded from every paired statistic and listed,
    with best-case/worst-case imputations reported (never silently dropped).
  * Duplicate rows for one planned run: the first row in load order that is not a harness error is used; all
    duplicates are listed.
  * Null-imposed bootstrap (WCR): per-task differences D_i are regressed on a constant; the null H0: mean = 0 is
    imposed by taking the restricted residuals D_i - 0 (the differences centred at the null value 0, not at their
    mean); D*_i = v_g * D_i with one weight per repository; t* = mean(D*) / CR1 se(D*); one-sided
    p = (1 + #{t* >= t}) / (B + 1). The unrestricted variant (differences recentred at their own mean) and Webb
    six-point weights are reported as sensitivities only.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import sys
import time
from fractions import Fraction

import numpy as np

# ----------------------------------------------------------------------------------------------------------------
# Frozen constants (PROTOCOL.md "Decision rules")
# ----------------------------------------------------------------------------------------------------------------
BASELINE = 'bundle0'
SCREEN_ARMS = ['bundle0', 'bundle0_variant_1', 'bundle0_variant_3', 'bundle0_variant_2', 'bundle0_variant_4', 'bundle0_variant_5']
GAIN_MIN = Fraction(2, 58)          # criterion (1): task-weighted mean gain >= 2/58
ALPHA = Fraction(1, 40)             # criterion (2): one-sided p <= 0.025
N_BOOT = 9999                       # bootstrap draws
BOOT_SEED = 20261109                # frozen seed of the decision draws (reported in every output)
MIN_POS_REPOS = 3                   # criterion (3)
TIME_RATIO_MAX = Fraction(6, 5)     # criterion (4): mean agent time <= 1.2 x baseline
DEFAULT_REPEATS = {'S': 2, 'C': 4, 'D': 4}
TOOL_BUDGET = 28
ANALYSIS_ID = 'analyze_e7 v1 (2026-11-09)'

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(os.path.dirname(HERE))
DEFAULT_TASKS = os.path.join(RESEARCH, 'data', 'berth_ds', 'tasks.jsonl')

# ----------------------------------------------------------------------------------------------------------------
# Row status: harness errors versus agent outcomes
# ----------------------------------------------------------------------------------------------------------------
AGENT_OUTCOMES = (
    ('budget_tool_calls', 'Agent exceeded tool call budget'),
    ('budget_turns', 'Agent exceeded turns budget'),
    ('budget_turns', 'Agent exceeded maximum allowed LLM turns'),
    ('timeout', 'Agent exceeded session timeout'),
    ('no_submit_no_patch', 'Agent completed execution without calling submit_patch'),
)
GRADING_REJECTIONS = ('Required test node did not pass', 'Malformed JUnit XML report',
                      'Pytest stdout summary indicates', 'Missing JUnit XML report')
INFRA_PREFIXES = ('Sandbox execution error', 'Evaluation error', 'Snapshot file not found',
                  'Unexpected evaluation worker error', 'Missing test specification', 'Failed to apply test_patch',
                  'Failed to apply agent patch', 'Agent patch file missing', 'Test patch file missing')


def row_status(row):
    """Return (category, detail). category is 'ok', an agent outcome ('budget_tool_calls', 'budget_turns',
    'timeout', 'no_submit_no_patch'), 'grading_rejection', or 'harness_error'."""
    if row.get('exc') or row.get('tb'):
        return 'harness_error', 'worker exception: ' + str(row.get('exc') or row.get('tb'))[:160]
    if row.get('resolved') is None:
        return 'harness_error', 'row has no resolved field'
    if row.get('resolved') is True:
        return 'ok', ''
    err = str(row.get('error') or '').strip()
    if not err:
        return 'ok', ''
    for cat, prefix in AGENT_OUTCOMES:
        if err.startswith(prefix):
            return cat, err[:160]
    for prefix in GRADING_REJECTIONS:
        if err.startswith(prefix):
            return 'grading_rejection', err[:160]
    if err.startswith(INFRA_PREFIXES):
        return 'harness_error', err[:160]
    return 'harness_error', 'unknown error text: ' + err[:160]


def is_harness_error(row):
    return row_status(row)[0] == 'harness_error'


def solved(row):
    return 1 if row.get('resolved') is True else 0


# ----------------------------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------------------------
def load_task_repos(path=DEFAULT_TASKS, needed=None):
    """instance_id -> repository, from tasks.jsonl (read as data, line by line)."""
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            iid = r.get('instance_id')
            if needed is None or iid in needed:
                out[iid] = r.get('repo')
    return out


def load_panel(path):
    """A panel file: {'panel': name, 'tasks': [...], 'per_repo': {...}} or a plain list of task ids."""
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    if isinstance(d, list):
        d = {'tasks': list(d)}
    elif 'tasks' not in d and 'tasks_frozen_order' in d:
        d = dict(d, tasks=list(d['tasks_frozen_order']))
    d = dict(d)
    d.setdefault('panel', os.path.splitext(os.path.basename(path))[0].replace('panel_', ''))
    d['path'] = os.path.abspath(path)
    with open(path, 'rb') as f:
        d['sha256'] = hashlib.sha256(f.read()).hexdigest()
    if len(set(d['tasks'])) != len(d['tasks']):
        raise ValueError(f'panel {path} lists a task twice')
    return d


def check_panel_repos(panel, task_repo):
    """Every panel task must have a repository; a frozen per_repo table must match tasks.jsonl."""
    missing = [t for t in panel['tasks'] if not task_repo.get(t)]
    if missing:
        raise ValueError(f'tasks without repository in tasks.jsonl: {missing[:10]}')
    rc = collections.Counter(task_repo[t] for t in panel['tasks'])
    if panel.get('per_repo') is not None and dict(rc) != dict(panel['per_repo']):
        raise ValueError('panel per_repo does not match tasks.jsonl repositories')
    return dict(rc)


def find_out_dir(d):
    if os.path.isfile(os.path.join(d, 'results.jsonl')):
        return d
    if os.path.isfile(os.path.join(d, 'out', 'results.jsonl')):
        return os.path.join(d, 'out')
    raise FileNotFoundError(f'no results.jsonl in {d} or {d}/out')


def _dir_label(d, od):
    base = os.path.normpath(od)
    if os.path.basename(base) == 'out':
        base = os.path.dirname(base)
    return os.path.basename(base) or os.path.basename(os.path.normpath(d))


def load_runs(dirs, arm_from_repeat=False, arm_tag_from_dir=False):
    """Read result rows from run directories. Returns (runs, problems).

    Each run: arm, task, repeat, row, source (out dir), label, order (dir index, line), trace_path.
    arm_from_repeat / arm_tag_from_dir relabel arms for smoke tests only (repeat k of arm X becomes arm 'X#rk' with
    repeat 1; arm X of directory L becomes 'X@L')."""
    runs, problems = [], []
    for di, d in enumerate(dirs):
        od = find_out_dir(d)
        label = _dir_label(d, od)
        with open(os.path.join(od, 'results.jsonl'), encoding='utf-8') as f:
            for li, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    problems.append(dict(source=od, line=li + 1, problem=f'unparsable row: {e}'))
                    continue
                arm, task = row.get('arm'), row.get('task')
                rep = row.get('repeat_id', row.get('seed'))
                seed = row.get('seed', rep)
                try:
                    rep = int(rep)
                except (TypeError, ValueError):
                    problems.append(dict(source=od, line=li + 1, problem='row without integer repeat_id/seed'))
                    continue
                if not arm or not task:
                    problems.append(dict(source=od, line=li + 1, problem='row without arm or task'))
                    continue
                trace = os.path.join(od, arm, f's{seed}', 'traces', 'trace_' + task.replace('/', '__') + '.json')
                a, r = arm, rep
                if arm_from_repeat:
                    a, r = f'{arm}#r{rep}', 1
                if arm_tag_from_dir:
                    a = f'{a}@{label}'
                runs.append(dict(arm=a, orig_arm=arm, task=task, repeat=r, orig_repeat=rep, row=row, source=od,
                                 label=label, order=(di, li), trace_path=trace))
    return runs, problems


def assemble(runs, tasks, arms, repeats):
    """Map planned runs (arm x task x repeat 1..repeats) to rows; list missing runs, duplicates and ignored rows."""
    task_set, arm_set = set(tasks), set(arms)
    by_key = collections.defaultdict(list)
    ignored = collections.Counter()
    for r in runs:
        if r['arm'] not in arm_set:
            ignored['arm not analysed'] += 1
        elif r['task'] not in task_set:
            ignored['task not in panel'] += 1
        elif not 1 <= r['repeat'] <= repeats:
            ignored['repeat id outside 1..%d' % repeats] += 1
        else:
            by_key[(r['arm'], r['task'], r['repeat'])].append(r)
    chosen, missing, duplicates = {}, [], []
    for a in arms:
        for t in tasks:
            for k in range(1, repeats + 1):
                lst = sorted(by_key.get((a, t, k), []), key=lambda x: x['order'])
                if not lst:
                    missing.append(dict(arm=a, task=t, repeat=k))
                    continue
                pick = next((x for x in lst if not is_harness_error(x['row'])), lst[0])
                if len(lst) > 1:
                    duplicates.append(dict(arm=a, task=t, repeat=k, n_rows=len(lst),
                                           used=f"{pick['label']}:{pick['order'][1] + 1}",
                                           all=[f"{x['label']}:{x['order'][1] + 1}" for x in lst],
                                           same_directory=len({x['source'] for x in lst}) < len(lst)))
                chosen[(a, t, k)] = pick
    cells = {(a, t): [chosen[(a, t, k)] for k in range(1, repeats + 1) if (a, t, k) in chosen]
             for a in arms for t in tasks}
    return dict(tasks=list(tasks), arms=list(arms), repeats=repeats, chosen=chosen, cells=cells, missing=missing,
                duplicates=duplicates, ignored=dict(ignored), n_planned=len(arms) * len(tasks) * repeats)


# ----------------------------------------------------------------------------------------------------------------
# Shell / Python source-write classification (for run_command calls)
# ----------------------------------------------------------------------------------------------------------------
_SCRATCH = ('/tmp', '/dev/', '/proc/', '/sys/', '/var/tmp')
_OPS = ('&>>', '<<<', '<<-', '&&', '||', '>>', '>|', '&>', '>&', '<&', '<<', ';', '|', '&', '>', '<', '(', ')')
_HEREDOC = re.compile(r"<<(-?)[ \t]*(?:'([^'\n]+)'|\"([^\"\n]+)\"|\\?([A-Za-z_][A-Za-z0-9_]*))")
_PYTHON = re.compile(r'^python(\d+(\.\d+)?)?$')


def _scratch(path):
    p = (path or '').strip()
    if p in ('', '-', '/dev/null') or p.startswith('&'):
        return True
    return p.startswith(_SCRATCH)


def _split_heredocs(cmd):
    """Remove here-document bodies from a shell command; return (shell text, [bodies in order])."""
    lines = cmd.split('\n')
    out, bodies, i = [], [], 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        pending = []
        for m in _HEREDOC.finditer(line):
            if m.start() > 0 and line[m.start() - 1] == '<':
                continue  # here-string <<<
            pending.append((m.group(2) or m.group(3) or m.group(4), m.group(1) == '-'))
        for delim, strip_tabs in pending:
            body = []
            while i < len(lines):
                l = lines[i]
                i += 1
                if (l.lstrip('\t') if strip_tabs else l).rstrip() == delim:
                    break
                body.append(l)
            bodies.append('\n'.join(body))
    return '\n'.join(out), bodies


def _tokenize_shell(text):
    """Quote-aware tokenizer: [(kind, value)] with kind 'w' (word, quotes removed) or 'op'. Newlines -> ';'."""
    toks, buf, inword, i, n = [], [], False, 0, len(text)

    def flush():
        nonlocal buf, inword
        if inword:
            toks.append(('w', ''.join(buf)))
        buf, inword = [], False

    while i < n:
        ch = text[i]
        if ch == '\\' and i + 1 < n:
            if text[i + 1] != '\n':
                buf.append(text[i + 1])
                inword = True
            i += 2
            continue
        if ch == "'":
            j = text.find("'", i + 1)
            j = n if j < 0 else j
            buf.append(text[i + 1:j])
            inword, i = True, j + 1
            continue
        if ch == '"':
            j, part = i + 1, []
            while j < n and text[j] != '"':
                if text[j] == '\\' and j + 1 < n and text[j + 1] in '"\\$`\n':
                    part.append(text[j + 1])
                    j += 2
                    continue
                part.append(text[j])
                j += 1
            buf.append(''.join(part))
            inword, i = True, j + 1
            continue
        if ch == '#' and not inword and (i == 0 or text[i - 1] in ' \t\n;&|()'):
            j = text.find('\n', i)
            i = n if j < 0 else j
            continue
        if ch in ' \t\r':
            flush()
            i += 1
            continue
        if ch == '\n':
            flush()
            toks.append(('op', ';'))
            i += 1
            continue
        op = next((o for o in _OPS if text.startswith(o, i)), None)
        if op:
            if op[0] in '<>' and inword and ''.join(buf).isdigit():
                buf, inword = [], False  # file-descriptor prefix such as 2>
            else:
                flush()
            toks.append(('op', op))
            i += len(op)
            continue
        buf.append(ch)
        inword = True
        i += 1
    flush()
    return toks


def _simple_commands(toks):
    cmds, cur, hd = [], dict(words=[], redirs=[], heredocs=[]), 0
    i = 0
    while i < len(toks):
        kind, val = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if kind == 'op' and val in (';', '&&', '||', '|', '&', '(', ')'):
            if cur['words'] or cur['redirs'] or cur['heredocs']:
                cmds.append(cur)
            cur = dict(words=[], redirs=[], heredocs=[])
            i += 1
        elif kind == 'op' and val in ('<<', '<<-'):
            cur['heredocs'].append(hd)
            hd += 1
            i += 2 if nxt and nxt[0] == 'w' else 1
        elif kind == 'op':
            target = nxt[1] if nxt and nxt[0] == 'w' else ''
            cur['redirs'].append((val, target))
            i += 2 if nxt and nxt[0] == 'w' else 1
        else:
            cur['words'].append(val)
            i += 1
    if cur['words'] or cur['redirs'] or cur['heredocs']:
        cmds.append(cur)
    return cmds


def _command_word(words):
    j = 0
    while j < len(words):
        w = words[j]
        if re.match(r'^[A-Za-z_][A-Za-z0-9_]*=', w) or w in ('sudo', 'env', 'time', 'command', 'nohup', 'exec'):
            j += 1
            continue
        break
    return j, (os.path.basename(words[j]) if j < len(words) else '')


def _positional_after_script(args, script_flag_letter='e', file_flag='f'):
    """Non-option arguments after the script of sed/perl (the script is given by -e/-f or is the first positional)."""
    positional, explicit_script, skip = [], False, False
    for a in args:
        if skip:
            skip = False
            continue
        if a.startswith('--'):
            if a.startswith(('--expression', '--file')):
                explicit_script = True
                skip = '=' not in a
            continue
        if a.startswith('-') and len(a) > 1:
            letters = a[1:]
            if letters.endswith(script_flag_letter) or letters.endswith(file_flag):
                explicit_script = True
                skip = True
            continue
        positional.append(a)
    return positional if explicit_script else positional[1:]


def _balanced_args(code, open_idx):
    """Top-level comma-separated arguments of the call whose '(' is at open_idx (approximate, quote aware)."""
    depth, i, start, args, q = 0, open_idx, open_idx + 1, [], None
    while i < len(code):
        c = code[i]
        if q:
            if c == '\\':
                i += 2
                continue
            if code.startswith(q, i):
                i += len(q)
                q = None
                continue
        elif c in '\'"':
            q = code[i:i + 3] if code[i:i + 3] in ('"""', "'''") else c
            i += len(q)
            continue
        elif c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
            if depth == 0:
                args.append(code[start:i])
                return [a.strip() for a in args if a.strip()]
        elif c == ',' and depth == 1:
            args.append(code[start:i])
            start = i + 1
        i += 1
    args.append(code[start:])
    return [a.strip() for a in args if a.strip()]


_STR_LIT = re.compile(r"""^[rRbBuU]?(['"])(.*)\1$""", re.S)
_PATH_LIT = re.compile(r"""^(?:pathlib\.)?Path\(\s*[rR]?(['"])([^'"]*)\1\s*\)$""")


def _resolve_path_expr(expr, code):
    expr = expr.strip()
    m = _STR_LIT.match(expr) or _PATH_LIT.match(expr)
    if m:
        return [m.group(2)]
    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', expr):
        vals = []
        for am in re.finditer(r'(?m)^[ \t]*' + re.escape(expr) + r'[ \t]*=[ \t]*([^\n=][^\n]*)$', code):
            rhs = am.group(1).strip()
            mm = re.match(r"""^(?:(?:pathlib\.)?Path\(\s*)?[rR]?(['"])([^'"]*)\1""", rhs)
            vals.append(mm.group(2) if mm else '?')
        return vals or ['?']
    return ['?']


def python_write_targets(code):
    """File targets written by a Python snippet ('?' when the target cannot be resolved)."""
    targets = []
    for m in re.finditer(r'(?<![A-Za-z0-9_])open\s*\(', code):
        args = _balanced_args(code, m.end() - 1)
        if not args:
            continue
        mode = None
        if len(args) > 1 and '=' not in args[1].split('(')[0]:
            mode = args[1]
        for a in args[1:]:
            km = re.match(r'^mode\s*=\s*(.+)$', a, re.S)
            if km:
                mode = km.group(1)
        lit = _STR_LIT.match(mode.strip()) if mode else None
        if lit and set(lit.group(2)) & set('wax+'):
            targets += _resolve_path_expr(args[0], code)
    for m in re.finditer(r'\.\s*write_(?:text|bytes)\s*\(', code):
        prefix = code[:m.start()].rstrip()
        lm = re.search(r"""(?:pathlib\.)?Path\(\s*[rR]?(['"])([^'"]*)\1\s*\)$""", prefix)
        if lm:
            targets.append(lm.group(2))
            continue
        im = re.search(r'([A-Za-z_][A-Za-z0-9_\.]*)$', prefix)
        if im and '.' not in im.group(1):
            targets += _resolve_path_expr(im.group(1), code)
        else:
            targets.append('?')
    for m in re.finditer(r'fileinput\.(?:input|FileInput)\s*\(', code):
        args = _balanced_args(code, m.end() - 1)
        if any(re.match(r'^inplace\s*=\s*(True|1)$', a) for a in args):
            targets += _resolve_path_expr(args[0], code) if args else ['?']
    for m in re.finditer(r'\b(?:shutil\.(?:copy|copy2|copyfile|move)|os\.(?:replace|rename))\s*\(', code):
        args = _balanced_args(code, m.end() - 1)
        targets += _resolve_path_expr(args[1], code) if len(args) > 1 else ['?']
    return targets


def classify_shell_write(cmd):
    """Is a run_command a source write? Returns {'write': bool, 'writes': [(kind, target)]}.

    Writes: shell redirections (> >> &> >|) into a non-scratch path, sed -i / perl -i, tee into a file, patch /
    git apply, and Python snippets (python -c CODE or a here-document fed to python) that open a file for writing,
    call write_text/write_bytes, fileinput inplace, or shutil/os moves. Targets under /tmp, /dev, /proc, /sys,
    /var/tmp are scratch and do not count; an unresolvable target counts."""
    if not isinstance(cmd, str) or not cmd.strip():
        return dict(write=False, writes=[])
    shell, bodies = _split_heredocs(cmd)
    writes = []
    for c in _simple_commands(_tokenize_shell(shell)):
        for op, tgt in c['redirs']:
            if op in ('>', '>>', '>|', '&>', '&>>') or (op == '>&' and tgt and not tgt.isdigit() and tgt != '-'):
                if not _scratch(tgt):
                    writes.append(('redirect', tgt))
        j, name = _command_word(c['words'])
        args = c['words'][j + 1:]
        if name in ('sed', 'gsed'):
            if any(re.match(r'^-[a-zA-Z]*i', a) or a.startswith('--in-place') for a in args):
                tg = _positional_after_script(args)
                writes += [('sed', t) for t in tg if not _scratch(t)] if tg else [('sed', '?')]
        elif name == 'perl':
            if any(re.match(r'^-[a-zA-Z0-9]*i', a) for a in args):
                tg = _positional_after_script(args)
                writes += [('perl', t) for t in tg if not _scratch(t)] if tg else [('perl', '?')]
        elif name == 'tee':
            writes += [('tee', a) for a in args if not a.startswith('-') and not _scratch(a)]
        elif name == 'patch' or (name == 'git' and args[:1] == ['apply']):
            writes.append(('patch', '?'))
        elif _PYTHON.match(name):
            code = None
            if '-c' in args:
                k = args.index('-c')
                code = args[k + 1] if k + 1 < len(args) else ''
            elif c['heredocs'] and all(a == '-' or a.startswith('-') for a in args):
                code = '\n'.join(bodies[h] for h in c['heredocs'] if h < len(bodies))
            if code:
                writes += [('python', t) for t in python_write_targets(code) if t == '?' or not _scratch(t)]
    return dict(write=bool(writes), writes=writes)


# ----------------------------------------------------------------------------------------------------------------
# Trace parsing (ATIF-v1.7 traces written by the harness)
# ----------------------------------------------------------------------------------------------------------------
def call_outcome(obs):
    """(status, detail) of one tool call from its observation: status 'ok', 'error', 'malformed' (the call was
    rejected for missing mandatory parameters or similar), 'budget' (rejected because the budget was exhausted), or
    'unknown' (no or unparsable observation)."""
    if not isinstance(obs, dict):
        return 'unknown', 'no observation'
    content = obs.get('content')
    if content is None or content == '':
        return 'unknown', 'no observation'
    try:
        j = json.loads(content) if isinstance(content, str) else content
    except (TypeError, ValueError):
        return 'unknown', 'non-json observation'
    if not isinstance(j, dict):
        return 'unknown', 'non-dict observation'
    st = j.get('status')
    if st == 'ok':
        return 'ok', ''
    if st == 'error':
        et = j.get('error_type') or ''
        return ('budget', et) if et == 'BudgetExceeded' else ('error', et)
    if 'error' in j:
        return 'malformed', str(j.get('error'))[:120]
    return 'unknown', 'observation without status'


def _new_response():
    return dict(thought=False, thought_chars=0, tools=[])


def parse_trace(path):
    """Per-run trace metrics. Model responses are reconstructed from the steps (validated against raw request dumps,
    see ANALYSIS_VALIDATION.md): a step with usage metrics starts a response; agent steps without metrics continue
    it; a step that carries only tool calls joins the open response if that response has no tool call yet (and, if
    the step carries metrics, those belong to a further, empty response)."""
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        return dict(error=f'unreadable trace: {e}')
    steps = d.get('steps') if isinstance(d, dict) else None
    if not isinstance(steps, list):
        return dict(error='trace without steps')
    responses, calls, cur = [], [], None
    for s in steps:
        if not isinstance(s, dict) or s.get('source') != 'agent':
            cur = None
            continue
        tcs = [tc for tc in (s.get('tool_calls') or []) if isinstance(tc, dict)]
        msg = s.get('message')
        has_msg = isinstance(msg, str) and msg.strip() != ''
        has_metrics = bool(s.get('metrics'))
        event = (s.get('extra') or {}).get('event_type')
        if tcs and not has_msg and cur is not None and not cur['tools']:
            cur['tools'] += [tc.get('function_name') for tc in tcs]
            if has_metrics:
                cur = _new_response()
                responses.append(cur)
        else:
            if has_metrics or cur is None:
                cur = _new_response()
                responses.append(cur)
            if event == 'thinking' and has_msg:
                cur['thought'] = True
                cur['thought_chars'] += len(msg)
            cur['tools'] += [tc.get('function_name') for tc in tcs]
        obs = s.get('observation')
        for k, tc in enumerate(tcs):
            if len(tcs) == 1:
                status, detail = call_outcome(obs)
            else:  # several calls in one step share one observation: outcome unknown except for the last call
                status, detail = call_outcome(obs) if k == len(tcs) - 1 else ('unknown', 'shared observation')
            calls.append(dict(name=tc.get('function_name'), args=tc.get('arguments'), status=status, detail=detail))
    first_submit = next((i for i, r in enumerate(responses) if 'submit_patch' in r['tools']), None)
    nonterminal = responses if first_submit is None else responses[:first_submit]
    m = dict(error=None, n_responses=len(responses), n_thought=sum(r['thought'] for r in responses),
             nonterm_responses=len(nonterminal), nonterm_thought=sum(r['thought'] for r in nonterminal),
             thought_chars=sum(r['thought_chars'] for r in responses), n_calls=len(calls),
             budget_rejected_calls=0, unknown_outcome_calls=0, malformed_calls=0,
             submitted=any(c['name'] == 'submit_patch' and c['status'] == 'ok' for c in calls),
             write_attempts=0, ok_writes=0, failed_writes=0, unknown_writes=0, budget_rejected_writes=0,
             repeat_identical_failed_writes=0, repeat_identical_failed_calls=0, shell_write_kinds={})
    for ch in ('edit_file', 'write_file', 'shell'):
        m[f'{ch}_attempts'] = 0
        m[f'{ch}_failed'] = 0
    seen_failed = set()
    kinds = collections.Counter()
    for c in calls:
        name, status = c['name'], c['status']
        channel = None
        if name in ('edit_file', 'write_file'):
            channel = name
        elif name == 'run_command':
            cmd = (c['args'] or {}).get('command') if isinstance(c['args'], dict) else None
            cls = classify_shell_write(cmd)
            if cls['write']:
                channel = 'shell'
                for kind, _ in cls['writes']:
                    kinds[kind] += 1
        if status == 'budget':
            m['budget_rejected_calls'] += 1
            m['budget_rejected_writes'] += channel is not None
            continue
        if status == 'unknown':
            m['unknown_outcome_calls'] += 1
            m['unknown_writes'] += channel is not None
            continue
        m['malformed_calls'] += status == 'malformed'
        failed = status in ('error', 'malformed')
        if channel:
            m['write_attempts'] += 1
            m[f'{channel}_attempts'] += 1
            if failed:
                m['failed_writes'] += 1
                m[f'{channel}_failed'] += 1
            else:
                m['ok_writes'] += 1
        if failed:
            try:
                key = (name, json.dumps(c['args'], sort_keys=True, ensure_ascii=False))
            except (TypeError, ValueError):
                key = (name, repr(c['args']))
            if key in seen_failed:
                m['repeat_identical_failed_calls'] += 1
                m['repeat_identical_failed_writes'] += channel is not None
            seen_failed.add(key)
    m['shell_write_kinds'] = dict(kinds)
    return m


class TraceStore:
    """Lazy cache: trace path -> parsed metrics (None when the file does not exist)."""

    def __init__(self):
        self._c = {}

    def get(self, path):
        if path not in self._c:
            self._c[path] = parse_trace(path)
        return self._c[path]


# ----------------------------------------------------------------------------------------------------------------
# Small statistics helpers (no scipy)
# ----------------------------------------------------------------------------------------------------------------
def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-300 else 1e-300
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def t_cdf(x, df):
    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + x * x))
    return 1.0 - tail if x >= 0 else tail


def t_ppf(p, df):
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def percentile(values, q):
    """Linear-interpolation percentile (numpy default, Hyndman-Fan type 7); q in [0, 100]."""
    v = sorted(values)
    if not v:
        return None
    h = (len(v) - 1) * q / 100.0
    lo = math.floor(h)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (h - lo) * (v[hi] - v[lo])


def cluster_mean_ci(values, clusters, level=0.95):
    """Mean of task-level values with a CR1 repository-cluster standard error and a t(G-1) interval."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return dict(n=0, G=0, mean=None, se=None, lo=None, hi=None)
    names = sorted(set(clusters))
    G = len(names)
    mean = sum(vals) / n
    if G < 2:
        return dict(n=n, G=G, mean=mean, se=None, lo=None, hi=None)
    idx = {c: i for i, c in enumerate(names)}
    s = [0.0] * G
    for v, c in zip(vals, clusters):
        s[idx[c]] += v - mean
    se = math.sqrt(G / (G - 1) * sum(x * x for x in s)) / n
    q = t_ppf(0.5 + level / 2, G - 1)
    return dict(n=n, G=G, mean=mean, se=se, lo=mean - q * se, hi=mean + q * se)


# ----------------------------------------------------------------------------------------------------------------
# Wild cluster bootstrap-t for the mean of task differences (regression on a constant)
# ----------------------------------------------------------------------------------------------------------------
WEBB = np.array([-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)])


def _t_values(beta, se):
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(se > 0, beta / np.where(se > 0, se, 1.0), np.sign(beta) * np.inf)
    return np.where(np.isnan(t), 0.0, t)


def _count_ge(tstar, t):
    if math.isfinite(t):
        return int(np.count_nonzero(tstar >= t - 1e-9 * max(1.0, abs(t))))
    return int(np.count_nonzero(tstar >= t))


def cluster_t(D, idx, G):
    """OLS of D on a constant with a CR1 cluster-robust standard error: (beta, se, t)."""
    D = np.asarray(D, dtype=float)
    N = D.size
    n_g = np.bincount(idx, minlength=G).astype(float)
    T_g = np.bincount(idx, weights=D, minlength=G)
    beta = T_g.sum() / N
    S_g = T_g - n_g * beta
    se = math.sqrt(G / (G - 1) * float(S_g @ S_g)) / N
    t = float(_t_values(np.array(beta), np.array(se)))
    return beta, se, t


def wild_cluster_bootstrap_t(D, idx, G, n_boot=N_BOOT, seed=BOOT_SEED, weights='rademacher', null='restricted',
                             rng=None):
    """One-sided (H1: mean > 0) wild cluster bootstrap-t. null='restricted' imposes H0 (residuals D_i - 0);
    null='unrestricted' uses residuals D_i - mean(D) (sensitivity). Returns t, se, p, count, n_boot."""
    D = np.asarray(D, dtype=float)
    idx = np.asarray(idx)
    N = D.size
    n_g = np.bincount(idx, minlength=G).astype(float)
    T_g = np.bincount(idx, weights=D, minlength=G)
    beta, se, t = cluster_t(D, idx, G)
    R_g = T_g if null == 'restricted' else T_g - n_g * beta
    rng = rng if rng is not None else np.random.default_rng(seed)
    if weights == 'rademacher':
        V = rng.integers(0, 2, size=(n_boot, G)).astype(np.float64) * 2.0 - 1.0
    elif weights == 'webb':
        V = WEBB[rng.integers(0, 6, size=(n_boot, G))]
    else:
        raise ValueError(weights)
    VR = V * R_g
    bstar = VR.sum(axis=1) / N
    Sstar = VR - np.outer(bstar, n_g)
    sestar = np.sqrt(G / (G - 1) * np.einsum('ij,ij->i', Sstar, Sstar)) / N
    tstar = _t_values(bstar, sestar)
    count = _count_ge(tstar, t)
    return dict(t=t, se=se, beta=beta, count=count, n_boot=n_boot, p=(1 + count) / (n_boot + 1),
                weights=weights, null=null, n_clusters=G,
                n_clusters_nonzero=int(np.count_nonzero(np.abs(R_g) > 1e-12)))


def exact_rademacher_p(D, idx, G, null='restricted', max_nonzero=22):
    """Exact Rademacher wild-bootstrap p-value by enumerating all sign patterns of the clusters whose residual total
    is nonzero (other clusters do not change t*). Returns (p, n_nonzero) or (None, n_nonzero)."""
    D = np.asarray(D, dtype=float)
    idx = np.asarray(idx)
    N = D.size
    n_g = np.bincount(idx, minlength=G).astype(float)
    T_g = np.bincount(idx, weights=D, minlength=G)
    beta, se, t = cluster_t(D, idx, G)
    R_g = T_g if null == 'restricted' else T_g - n_g * beta
    nz = np.flatnonzero(np.abs(R_g) > 1e-12)
    k = len(nz)
    if k > max_nonzero:
        return None, k
    total, count, chunk = 1 << k, 0, 1 << 16
    for start in range(0, total, chunk):
        codes = np.arange(start, min(total, start + chunk), dtype=np.int64)
        V = np.ones((codes.size, G))
        if k:
            V[:, nz] = ((codes[:, None] >> np.arange(k)) & 1) * 2.0 - 1.0
        VR = V * R_g
        bstar = VR.sum(axis=1) / N
        Sstar = VR - np.outer(bstar, n_g)
        sestar = np.sqrt(G / (G - 1) * np.einsum('ij,ij->i', Sstar, Sstar)) / N
        count += _count_ge(_t_values(bstar, sestar), t)
    return count / total, k


def _fmt_frac(x):
    return None if x is None else f'{x.numerator}/{x.denominator}'


def gate_core(K, L, repos, n_boot=N_BOOT, seed=BOOT_SEED, rng=None, sensitivities=True, exact=True):
    """Solve-based gate statistics from task differences D_i = K_i / L (K integer numerators, common denominator L).

    Returns the task-weighted gain, repository table, leave-one-repository-out estimates, positive repositories,
    the decision bootstrap (Rademacher, restricted) and sensitivities, and criteria (1)-(3)."""
    K = [int(k) for k in K]
    N = len(K)
    names = sorted(set(repos))
    G = len(names)
    pos = {r: i for i, r in enumerate(names)}
    idx = np.array([pos[r] for r in repos], dtype=np.int64)
    out = dict(n_tasks=N, n_repos=G)
    if N == 0:
        out.update(d=None, criteria=dict(gain=False, p=False, pos_repos=False, loo=False), notes=['no tasks'])
        return out
    sumK = sum(K)
    d = Fraction(sumK, N * L)
    per = {r: dict(n=0, sumK=0) for r in names}
    for k, r in zip(K, repos):
        per[r]['n'] += 1
        per[r]['sumK'] += k
    repo_table, loo = {}, {}
    for r in names:
        n_r, s_r = per[r]['n'], per[r]['sumK']
        rest = N - n_r
        loo[r] = Fraction(sumK - s_r, rest * L) if rest > 0 else None
        repo_table[r] = dict(n_tasks=n_r, sum_gain=float(Fraction(s_r, L)), sum_gain_exact=_fmt_frac(Fraction(s_r, L)),
                             mean_gain=float(Fraction(s_r, n_r * L)), loo=None if loo[r] is None else float(loo[r]),
                             loo_exact=_fmt_frac(loo[r]))
    repo_equal = sum(Fraction(per[r]['sumK'], per[r]['n'] * L) for r in names) / G
    n_pos = sum(1 for r in names if per[r]['sumK'] > 0)
    loo_vals = [v for v in loo.values()]
    loo_ok = G >= 2 and all(v is not None and v > 0 for v in loo_vals)
    D = np.array(K, dtype=float) / L
    notes = []
    if G >= 2:
        # decision draws: default_rng(seed); sensitivities: default_rng(seed + 1), default_rng(seed + 2).
        # A caller-supplied rng (simulation) is used for all three in sequence.
        rngs = [rng, rng, rng] if rng is not None else [np.random.default_rng(seed + j) for j in range(3)]
        boot = wild_cluster_bootstrap_t(D, idx, G, n_boot=n_boot, rng=rngs[0], weights='rademacher', null='restricted')
        boot['seed'] = None if rng is not None else seed
        p_ok = Fraction(1 + boot['count'], n_boot + 1) <= ALPHA
        sens = {}
        if sensitivities:
            sens['webb_restricted'] = wild_cluster_bootstrap_t(D, idx, G, n_boot=n_boot, weights='webb',
                                                               null='restricted', rng=rngs[1])
            sens['rademacher_unrestricted'] = wild_cluster_bootstrap_t(D, idx, G, n_boot=n_boot, weights='rademacher',
                                                                       null='unrestricted', rng=rngs[2])
        if exact:
            pe, k = exact_rademacher_p(D, idx, G, null='restricted')
            boot['p_exact_enumeration'] = pe
        boot['min_attainable_p_exact'] = 2.0 ** (-boot['n_clusters_nonzero'])
    else:
        boot, sens, p_ok = None, {}, False
        notes.append('fewer than 2 repositories: cluster-robust inference impossible')
    crit = dict(gain=d >= GAIN_MIN, p=bool(p_ok), pos_repos=n_pos >= MIN_POS_REPOS, loo=bool(loo_ok))
    out.update(d=float(d), d_exact=_fmt_frac(d), sum_gain=float(Fraction(sumK, L)), repo_equal=float(repo_equal),
               repo_equal_exact=_fmt_frac(repo_equal), n_repos_positive=n_pos,
               n_repos_negative=sum(1 for r in names if per[r]['sumK'] < 0),
               loo_min=min((float(v) for v in loo_vals if v is not None), default=None),
               repos=repo_table, bootstrap=boot, sensitivity=sens, criteria=crit, notes=notes)
    return out


def _common_denominator(fracs):
    L = 1
    for f in fracs:
        L = L * f.denominator // math.gcd(L, f.denominator)
    return [int(f * L) for f in fracs], L


# ----------------------------------------------------------------------------------------------------------------
# Screening
# ----------------------------------------------------------------------------------------------------------------
def _arm_run_summary(asm, arm, store):
    runs = [r for t in asm['tasks'] for r in asm['cells'][(arm, t)]]
    statuses = collections.Counter(row_status(r['row'])[0] for r in runs)
    times = [float(r['row']['t_agent']) for r in runs if isinstance(r['row'].get('t_agent'), (int, float))]
    tr = [store.get(r['trace_path']) for r in runs]
    ok_tr = [m for m in tr if m and not m.get('error')]
    att = sum(m['write_attempts'] for m in ok_tr)
    okw = sum(m['ok_writes'] for m in ok_tr)
    planned = len(asm['tasks']) * asm['repeats']
    missing = [m for m in asm['missing'] if m['arm'] == arm]
    return dict(planned=planned, completed=len(runs), missing=len(missing), solves=sum(solved(r['row']) for r in runs),
                status_counts=dict(statuses), harness_errors=statuses.get('harness_error', 0),
                harness_failures=statuses.get('harness_error', 0) + len(missing),
                mean_agent_time=(sum(times) / len(times)) if times else None, n_agent_time=len(times),
                edit_attempts=att, edit_ok=okw, edit_success=(Fraction(okw, att) if att else None),
                traces_found=len(ok_tr), traces_missing=sum(1 for m in tr if m is None),
                traces_unreadable=sum(1 for m in tr if m and m.get('error')),
                edit_channels={ch: [sum(m[f'{ch}_attempts'] for m in ok_tr), sum(m[f'{ch}_failed'] for m in ok_tr)]
                               for ch in ('edit_file', 'write_file', 'shell')})


def screening(results, panel_S, task_repo=None, baseline=BASELINE, arms=None, repeats=None, traces=None):
    """PROTOCOL.md screening: per arm, gain = sum over tasks of (mean solves of arm - mean solves of baseline);
    finalist = largest gain if > 0, ties broken by higher edit-call success rate, then lower mean agent time
    (a remaining exact tie, which the protocol does not cover, falls back to the PROTOCOL arm order and is flagged)."""
    tasks = list(panel_S['tasks'])
    repeats = repeats or DEFAULT_REPEATS.get(panel_S.get('panel'), 2)
    if arms is None:
        present = {r['arm'] for r in results}
        arms = [a for a in SCREEN_ARMS if a in present and a != baseline] + sorted(
            a for a in present if a not in SCREEN_ARMS and a != baseline)
    arms = [a for a in arms if a != baseline]
    store = traces if traces is not None else TraceStore()
    asm = assemble(results, tasks, [baseline] + arms, repeats)
    summ = {a: _arm_run_summary(asm, a, store) for a in [baseline] + arms}
    rows = []
    for a in arms:
        gain, used, excluded = Fraction(0), [], []
        per_repo = collections.defaultdict(Fraction)
        for t in tasks:
            ra, rb = asm['cells'][(baseline, t)], asm['cells'][(a, t)]
            if not ra or not rb:
                excluded.append(dict(task=t, n_baseline=len(ra), n_arm=len(rb)))
                continue
            diff = Fraction(sum(solved(r['row']) for r in rb), len(rb)) - Fraction(sum(solved(r['row']) for r in ra), len(ra))
            gain += diff
            used.append(t)
            if task_repo:
                per_repo[task_repo.get(t)] += diff
        s = summ[a]
        rows.append(dict(arm=a, gain=gain, gain_float=float(gain), gain_exact=_fmt_frac(gain), tasks_used=len(used),
                         tasks_excluded=excluded, mean_gain_per_task=float(gain / len(used)) if used else None,
                         repos_positive=sum(1 for v in per_repo.values() if v > 0),
                         repos_negative=sum(1 for v in per_repo.values() if v < 0),
                         per_repo_gain={k: float(v) for k, v in sorted(per_repo.items())},
                         edit_success=s['edit_success'], mean_agent_time=s['mean_agent_time']))
    order = {a: i for i, a in enumerate(SCREEN_ARMS)}

    def key(r):
        es = r['edit_success']
        mt = r['mean_agent_time']
        return (-r['gain'], -(es if es is not None else Fraction(-1)), mt if mt is not None else float('inf'),
                order.get(r['arm'], len(order)), r['arm'])

    ranked = sorted(rows, key=key)
    finalist, tie_log = None, []
    if ranked and ranked[0]['gain'] > 0:
        finalist = ranked[0]['arm']
        top = [r for r in ranked if r['gain'] == ranked[0]['gain']]
        if len(top) > 1:
            tie_log.append(f"gain tie among {[r['arm'] for r in top]}; broken by edit-call success rate")
            es_top = [r for r in top if r['edit_success'] == top[0]['edit_success']]
            if len(es_top) > 1:
                tie_log.append(f"edit-success tie among {[r['arm'] for r in es_top]}; broken by mean agent time")
                mt_top = [r for r in es_top if r['mean_agent_time'] == es_top[0]['mean_agent_time']]
                if len(mt_top) > 1:
                    tie_log.append('UNRESOLVED by the protocol: fell back to PROTOCOL arm order (flag to owner)')
    for r in rows:
        r['edit_success_float'] = None if r['edit_success'] is None else float(r['edit_success'])
        r['edit_success'] = _fmt_frac(r['edit_success']) if r['edit_success'] is not None else None
        r['gain'] = r['gain_float']
    for a, s in summ.items():
        s['edit_success_float'] = None if s['edit_success'] is None else float(s['edit_success'])
        s['edit_success'] = _fmt_frac(s['edit_success']) if s['edit_success'] is not None else None
    return dict(kind='screening', analysis=ANALYSIS_ID, panel=panel_S.get('panel'), panel_sha256=panel_S.get('sha256'),
                baseline=baseline, arms=arms, repeats=repeats, n_tasks=len(tasks),
                ranking=[r['arm'] for r in ranked], finalist=finalist,
                decision=(f'finalist {finalist}' if finalist else 'no finalist (largest gain <= 0): bundle0 stays'),
                tie_log=tie_log, arms_table=rows, run_summary=summ,
                data=dict(n_planned=asm['n_planned'], n_missing=len(asm['missing']), missing=asm['missing'],
                          duplicates=asm['duplicates'], ignored_rows=asm['ignored']))


# ----------------------------------------------------------------------------------------------------------------
# Gate (confirmation on C, dev on D)
# ----------------------------------------------------------------------------------------------------------------
def _time_ratio(runs_a, runs_b):
    ta = [Fraction(str(r['row']['t_agent'])) for r in runs_a if isinstance(r['row'].get('t_agent'), (int, float))]
    tb = [Fraction(str(r['row']['t_agent'])) for r in runs_b if isinstance(r['row'].get('t_agent'), (int, float))]
    if not ta or not tb or sum(ta) == 0:
        return None, ta, tb
    return (sum(tb) / len(tb)) / (sum(ta) / len(ta)), ta, tb


def gate(results, panel, task_repo, arm_a=BASELINE, arm_b=None, repeats=None, n_boot=N_BOOT, seed=BOOT_SEED,
         traces=None, with_secondary=True, exact=True):
    """PROTOCOL.md confirmation/dev gate of arm_b (candidate) against arm_a (bundle0). See module docstring."""
    tasks = list(panel['tasks'])
    repeats = repeats or DEFAULT_REPEATS.get(panel.get('panel'), 4)
    check_panel_repos(panel, task_repo)
    asm = assemble(results, tasks, [arm_a, arm_b], repeats)
    diffs, reps, used, excluded = [], [], [], []
    for t in tasks:
        ra, rb = asm['cells'][(arm_a, t)], asm['cells'][(arm_b, t)]
        if not ra or not rb:
            excluded.append(dict(task=t, repo=task_repo[t], completed_a=len(ra), completed_b=len(rb),
                                 mean_a=(sum(solved(r['row']) for r in ra) / len(ra)) if ra else None,
                                 mean_b=(sum(solved(r['row']) for r in rb) / len(rb)) if rb else None))
            continue
        diffs.append(Fraction(sum(solved(r['row']) for r in rb), len(rb)) - Fraction(sum(solved(r['row']) for r in ra), len(ra)))
        reps.append(task_repo[t])
        used.append(t)
    K, L = _common_denominator(diffs) if diffs else ([], 1)
    core = gate_core(K, L, reps, n_boot=n_boot, seed=seed, exact=exact)
    runs_a = [r for t in used for r in asm['cells'][(arm_a, t)]]
    runs_b = [r for t in used for r in asm['cells'][(arm_b, t)]]
    ratio, ta, tb = _time_ratio(runs_a, runs_b)
    he = {}
    for a in (arm_a, arm_b):
        rs = [r for t in tasks for r in asm['cells'][(a, t)]]
        miss = sum(1 for m in asm['missing'] if m['arm'] == a)
        cats = collections.Counter(row_status(r['row'])[0] for r in rs)
        details = collections.Counter(row_status(r['row'])[1][:80] for r in rs if row_status(r['row'])[0] == 'harness_error')
        he[a] = dict(planned=len(tasks) * repeats, completed=len(rs), missing=miss, harness_error_rows=cats.get('harness_error', 0),
                     harness_failures=cats.get('harness_error', 0) + miss, status_counts=dict(cats),
                     harness_error_details=dict(details))
    fa, fb = he[arm_a], he[arm_b]
    harness_ok = fb['harness_failures'] * fa['planned'] <= fa['harness_failures'] * fb['planned']
    time_ok = ratio is not None and ratio <= TIME_RATIO_MAX
    crit = dict(core['criteria'], time=bool(time_ok), harness=bool(harness_ok))
    decision = 'PASS' if all(crit.values()) else 'FAIL'
    # explicit handling of excluded tasks: best and worst case for the candidate
    sens_missing = None
    if excluded:
        n_all = len(used) + len(excluded)
        s = sum(diffs, Fraction(0))
        worst = (s - len(excluded)) / n_all
        best = (s + len(excluded)) / n_all
        sens_missing = dict(n_excluded=len(excluded), d_if_excluded_tasks_were_minus1=float(worst),
                            d_if_excluded_tasks_were_plus1=float(best), d_if_excluded_tasks_were_0=float(s / n_all))
    complete = not asm['missing'] and not excluded
    notes = list(core.get('notes', []))
    if not complete:
        notes.append(f"DATA INCOMPLETE: {len(asm['missing'])} planned runs missing, {len(excluded)} tasks excluded; "
                     'missing runs count as harness failures in criterion (4)')
    if seed != BOOT_SEED or n_boot != N_BOOT:
        notes.append(f'NON-FROZEN bootstrap settings (seed {seed}, draws {n_boot}); the frozen ones are seed {BOOT_SEED}, draws {N_BOOT}')
    out = dict(kind='gate', analysis=ANALYSIS_ID, panel=panel.get('panel'), panel_sha256=panel.get('sha256'),
               arm_a=arm_a, arm_b=arm_b, repeats=repeats, n_tasks_planned=len(tasks), n_tasks_used=len(used),
               excluded_tasks=excluded, excluded_sensitivity=sens_missing, decision=decision, criteria=crit,
               data_complete=complete, notes=notes,
               thresholds=dict(gain_min=_fmt_frac(GAIN_MIN), alpha=float(ALPHA), min_pos_repos=MIN_POS_REPOS,
                               time_ratio_max=float(TIME_RATIO_MAX), n_boot=N_BOOT, seed=BOOT_SEED),
               gain=core, time=dict(ratio=None if ratio is None else float(ratio),
                                    mean_a=float(sum(ta) / len(ta)) if ta else None,
                                    mean_b=float(sum(tb) / len(tb)) if tb else None, n_a=len(ta), n_b=len(tb)),
               harness=dict(by_arm=he, increase=not harness_ok),
               data=dict(n_planned=asm['n_planned'], n_missing=len(asm['missing']), missing=asm['missing'],
                         duplicates=asm['duplicates'], ignored_rows=asm['ignored']))
    if with_secondary:
        out['secondary'] = secondary(results, traces if traces is not None else TraceStore(), panel, task_repo,
                                     arm_a, arm_b, repeats, asm=asm)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Secondary metrics (descriptive; cannot rescue a failed gate)
# ----------------------------------------------------------------------------------------------------------------
def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def run_metrics(run, store, status=row_status):
    """Per-run values of the secondary metrics (row-based and trace-based). `status` classifies the row (v1:
    row_status; v2: the A1 table)."""
    row = run['row']
    cat = status(row)[0]
    m = dict(solved=solved(row), t_agent=_num(row.get('t_agent')), t_setup=_num(row.get('t_setup')),
             t_verify=_num(row.get('t_verify')), tok_in=_num(row.get('tok_in')), tok_out=_num(row.get('tok_out')),
             tool_calls=_num(row.get('tool_calls')), llm_calls=_num(row.get('llm_calls')),
             timeout=int(cat == 'timeout'), budget_exhausted=int(cat in ('budget_tool_calls', 'budget_turns')),
             budget_tool_calls=int(cat == 'budget_tool_calls'), harness_error=int(cat == 'harness_error'),
             no_submit_no_patch=int(cat == 'no_submit_no_patch'), grading_rejection=int(cat == 'grading_rejection'),
             nonempty_patch=int((_num(row.get('patch_chars')) or 0) > 0))
    tm = store.get(run['trace_path'])
    m['_trace'] = 'missing' if tm is None else ('unreadable' if tm.get('error') else 'ok')
    if tm and not tm.get('error'):
        for k in ('n_responses', 'n_thought', 'nonterm_responses', 'nonterm_thought', 'thought_chars', 'write_attempts',
                  'ok_writes', 'failed_writes', 'repeat_identical_failed_writes', 'repeat_identical_failed_calls',
                  'budget_rejected_calls', 'malformed_calls', 'edit_file_attempts', 'edit_file_failed',
                  'write_file_attempts', 'write_file_failed', 'shell_attempts', 'shell_failed'):
            m[k] = tm[k]
        m['submitted'] = int(tm['submitted'])
    return m


PER_RUN = ('solved', 't_agent', 't_setup', 't_verify', 'tok_in', 'tok_out', 'tool_calls', 'llm_calls', 'timeout',
           'budget_exhausted', 'budget_tool_calls', 'harness_error', 'no_submit_no_patch', 'grading_rejection',
           'nonempty_patch', 'submitted', 'write_attempts', 'failed_writes', 'repeat_identical_failed_writes',
           'repeat_identical_failed_calls', 'malformed_calls', 'budget_rejected_calls', 'thought_chars')
RATIOS = (('thought_rate_all', 'n_thought', 'n_responses'),
          ('thought_rate_nonterminal', 'nonterm_thought', 'nonterm_responses'),
          ('failed_writes_per_attempted_write', 'failed_writes', 'write_attempts'),
          ('edit_file_failure_rate', 'edit_file_failed', 'edit_file_attempts'),
          ('write_file_failure_rate', 'write_file_failed', 'write_file_attempts'),
          ('shell_write_failure_rate', 'shell_failed', 'shell_attempts'))


def _describe(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return dict(n=0, mean=None, median=None, p95=None)
    return dict(n=len(vals), mean=sum(vals) / len(vals), median=percentile(vals, 50), p95=percentile(vals, 95))


def secondary(results, traces, panel, task_repo, arm_a=BASELINE, arm_b=None, repeats=None, asm=None, status=row_status):
    """Descriptive secondary metrics per arm, and paired task differences (arm_b - arm_a; repeats averaged or pooled
    within task) with CR1 repository-cluster 95% intervals (t with G-1 degrees of freedom)."""
    tasks = list(panel['tasks'])
    repeats = repeats or DEFAULT_REPEATS.get(panel.get('panel'), 4)
    store = traces if traces is not None else TraceStore()
    asm = asm or assemble(results, tasks, [arm_a, arm_b], repeats)
    per = {}
    for a in (arm_a, arm_b):
        per[a] = {t: [run_metrics(r, store, status) for r in asm['cells'][(a, t)]] for t in tasks}
    arms_out = {}
    for a in (arm_a, arm_b):
        runs = [m for t in tasks for m in per[a][t]]
        d = dict(n_runs=len(runs), planned=len(tasks) * repeats,
                 completion_rate=(sum(1 - m['harness_error'] for m in runs) / (len(tasks) * repeats)) if tasks else None,
                 traces=collections.Counter(m['_trace'] for m in runs))
        for k in PER_RUN:
            d[k] = _describe([m.get(k) for m in runs])
        for name, num, den in RATIOS:
            nn = sum(m[num] for m in runs if num in m)
            dd = sum(m[den] for m in runs if den in m)
            d[name] = dict(num=nn, den=dd, rate=(nn / dd) if dd else None)
        d['traces'] = dict(d['traces'])
        arms_out[a] = d
    paired = {}
    for k in PER_RUN:
        vals, cl = [], []
        for t in tasks:
            xa = [m[k] for m in per[arm_a][t] if m.get(k) is not None]
            xb = [m[k] for m in per[arm_b][t] if m.get(k) is not None]
            if xa and xb:
                vals.append(sum(xb) / len(xb) - sum(xa) / len(xa))
                cl.append(task_repo[t])
        paired[k] = cluster_mean_ci(vals, cl)
    for name, num, den in RATIOS:
        vals, cl = [], []
        for t in tasks:
            na = sum(m[num] for m in per[arm_a][t] if num in m)
            da = sum(m[den] for m in per[arm_a][t] if den in m)
            nb = sum(m[num] for m in per[arm_b][t] if num in m)
            db = sum(m[den] for m in per[arm_b][t] if den in m)
            if da and db:
                vals.append(nb / db - na / da)
                cl.append(task_repo[t])
        paired[name] = cluster_mean_ci(vals, cl)
    return dict(kind='secondary', arm_a=arm_a, arm_b=arm_b, panel=panel.get('panel'), repeats=repeats, arms=arms_out,
                paired_differences=paired,
                note='descriptive only (P1_PREREG.md); paired = arm_b minus arm_a per task, CR1 repository-cluster 95% CI')


# ================================================================================================================
# Rules v2: PROTOCOL_AMENDMENTS.md A0-A7 (written before any run of the experiment).
# Everything above this block is rules v1 and is unchanged; `--rules v1` reproduces the earlier outputs exactly
# (ANALYSIS_VALIDATION.md). The CLI default is v2.2 (blocks "Rules v2.1", A8, and "Rules v2.2", A9, below). Python
# callers: gate() and screening() are v1; gate_v2() and screening_v2() are v2.
# ================================================================================================================
DEFAULT_RULES = 'v2.2'
ANALYSIS_ID_V2 = 'analyze_e7 v2 (2026-11-09; PROTOCOL_AMENDMENTS.md A0-A7)'
MAX_ATTEMPTS = 3                    # A1: the first run of a planned slot plus at most two re-runs
ALPHA_4B = Fraction(1, 20)          # A1 (4b): fails at one-sided exact binomial p <= 0.05
DROP_MAX = Fraction(1, 10)          # A0: more than 10% of a panel's tasks dropped -> stop
GUARD_H_LIMIT = Fraction(-2)        # A4: guard H fails when a repository's total difference is <= -2 task-equivalents
GUARD_H_ADOPTED = True              # A4 adoption result (ANALYSIS_VALIDATION.md V.9): at a true +3/58 even gain,
#                                     rho 0.7, H removed 0 of 18,130 D passes and 0 of 11,961 C passes in 20,000 panels each
#                                     (loss 0.00 pp; 95% upper bound 0.018 pp <= 2 pp on both) -> criterion (5)
ADAPTER_ARMS = ('bundle0_adapter_A', 'bundle0_adapter_B')   # A7: labels of the adapter arms
PROV_BLOCK_FIELDS = ('docker_image_id', 'harness_sha256', 'vllm_version', 'served_model_name')  # A3, one value per block
PROV_ARM_FIELDS = ('bundle_sha256', 'adapter_sha256')      # A3, arm-specific: one value per arm (adapter: adapter arms only)
PROV_USED_ARM_FIELDS = ('sampling_sent',)                  # PROTOCOL "Runs": one value per arm over the rows used


class AnalysisStop(Exception):
    """The analysis stops with an error and no decision (A0 > 10% dropped; A2 duplicate rows; malformed attempts)."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = list(details or [])


# ----------------------------------------------------------------------------------------------------------------
# A1 error-class table. Built from the harness that the reference host runs: swegemma 0.2.11 (<env>/lib/python3.12/
# site-packages/swegemma; all 34 .py files match the wheel RECORD) and the reference Berth worker and its Docker layer
# (<runner>/lib/{worker_berth.py,g4s_docker.py,sandbox_berth.py}). The worker
# writes one row per job: error = TaskResult.error[:200] (worker_berth.py:195) or, on any exception out of
# evaluate_task or the row's side files, exc = repr(e)[:300] and tb (worker_berth.py:203-204). The runner
# (<runner>/run_berth.py) writes no rows. Paths below are relative to those two roots. Frozen with this file.
# ----------------------------------------------------------------------------------------------------------------
ErrorEntry = collections.namedtuple('ErrorEntry', 'id klass category kind pattern source')
ERROR_TABLE_V2 = (
    # agent outcomes: the row is a completed run; only A00 is a solve
    ErrorEntry('A00', 'agent', 'solved', 'row', 'resolved true, error empty',
               'swegemma/harness/verification.py:646-683 (pytest exit 0 and the JUnit checks pass), 697 (error None)'),
    ErrorEntry('A01', 'agent', 'unsolved', 'row', 'resolved false, error empty',
               'swegemma/harness/verification.py:646, 697 (tests fail and the agent loop ended without an error)'),
    ErrorEntry('A02', 'agent', 'budget_tool_calls', 'prefix', 'Agent exceeded tool call budget',
               'swegemma/harness/agent_runner.py:932-939'),
    ErrorEntry('A03', 'agent', 'budget_turns', 'prefix', 'Agent exceeded turns budget',
               'swegemma/harness/agent_runner.py:789-795, 947-953, 1032-1037'),
    ErrorEntry('A04', 'agent', 'budget_turns', 'prefix', 'Agent exceeded maximum allowed LLM turns',
               'swegemma/harness/agent_runner.py:794, 952, 1036 (turn budget not configured)'),
    ErrorEntry('A05', 'agent', 'timeout', 'prefix', 'Agent exceeded session timeout',
               'swegemma/harness/agent_runner.py:866-870, 925-930, 1026-1031'),
    ErrorEntry('A06', 'agent', 'no_submit_no_patch', 'prefix', 'Agent completed execution without calling submit_patch',
               'swegemma/harness/agent_runner.py:1079-1083, 1088-1097'),
    ErrorEntry('A07', 'agent', 'agent_patch_apply', 'prefix', 'Failed to apply agent patch',
               'swegemma/harness/verification.py:463-488'),
    ErrorEntry('A08', 'agent', 'agent_patch_missing', 'exc',
               r"""^ContainerSetupError\(['"]Agent patch file missing in container on task """,
               'swegemma/harness/verification.py:465-468 raises ContainerSetupError (re-raised at 701-702); it reaches '
               'the row only as exc/tb (lib/worker_berth.py:203-204). A1 lists it as an agent outcome'),
    ErrorEntry('A09', 'agent', 'grading_rejection', 'prefix', 'Required test node did not pass',
               'swegemma/harness/verification.py:339-340 via 675-677'),
    ErrorEntry('A10', 'agent', 'grading_rejection', 'prefix', 'Missing JUnit XML report',
               'swegemma/harness/verification.py:681-683'),
    ErrorEntry('A11', 'agent', 'grading_rejection', 'prefix', 'Missing or empty JUnit XML report',
               'swegemma/harness/verification.py:275-276 (not reachable: 668 validates only a non-empty report)'),
    ErrorEntry('A12', 'agent', 'grading_rejection', 'prefix', 'Malformed JUnit XML report',
               'swegemma/harness/verification.py:277-280 via 675-677'),
    ErrorEntry('A13', 'agent', 'grading_rejection', 'prefix', 'No <testsuite> elements found in JUnit XML',
               'swegemma/harness/verification.py:282-284 via 675-677'),
    ErrorEntry('A14', 'agent', 'grading_rejection', 'prefix', 'No passing tests recorded in JUnit XML',
               'swegemma/harness/verification.py:292-296 via 675-677'),
    ErrorEntry('A15', 'agent', 'grading_rejection', 'prefix', 'Test failures/errors recorded in JUnit XML',
               'swegemma/harness/verification.py:297-301 via 675-677'),
    ErrorEntry('A16', 'agent', 'grading_rejection', 'prefix', 'Pytest stdout summary indicates zero or no passing tests',
               'swegemma/harness/verification.py:678-680'),
    # infrastructure errors: the planned slot is missing (not unsolved) and is re-run (A1)
    ErrorEntry('I01', 'infra', 'llm_connection_http', 'prefix', 'Sandbox execution error: litellm.',
               'swegemma/harness/agent_runner.py:1039-1043 (an LLM request raised a litellm exception: connection, '
               'timeout, HTTP status including context-length 400s)'),
    ErrorEntry('I02', 'infra', 'sandbox_setup_or_agent_loop', 'prefix', 'Sandbox execution error',
               'swegemma/harness/agent_runner.py:1039-1043, 1107-1109 (sandbox setup and any other exception in the '
               'agent loop; host-side setup failures arrive here: lib/g4s_docker.py:421, 460-461, lib/sandbox_berth.py:57)'),
    ErrorEntry('I03', 'infra', 'evaluation_infrastructure', 'prefix', 'Evaluation error',
               'swegemma/harness/verification.py:703-717 (any exception in the evaluation container except '
               'ContainerSetupError; reference runner: lib/g4s_docker.py:421, 460-461, lib/sandbox_berth.py:57)'),
    ErrorEntry('I04', 'infra', 'snapshot', 'prefix', 'Snapshot file not found',
               'swegemma/evaluate.py:280-295'),
    ErrorEntry('I05', 'infra', 'test_spec', 'prefix', 'Missing test specification',
               'swegemma/harness/verification.py:363-381'),
    ErrorEntry('I06', 'infra', 'test_patch', 'prefix', 'Failed to apply test_patch',
               'swegemma/harness/verification.py:573-597'),
    ErrorEntry('I07', 'infra', 'evaluation_infrastructure', 'prefix', 'Unexpected evaluation worker error',
               'swegemma/evaluate.py:539 (Evaluator.run only; not reachable from lib/worker_berth.py, which calls '
               'evaluate_task)'),
    ErrorEntry('I08', 'infra', 'worker_exception', 'exc', r'(?s).',
               'lib/worker_berth.py:193-204: any other exception (exc/tb). Includes ContainerSetupError from '
               'swegemma/harness/container_setup.py:449-451 (snapshot extraction), 464-466 (incremental patch), '
               '542-549 and 559-563 (test-dependency setup; in the reference runner lib/sandbox_berth.py:95-99 replaces that function, '
               'so they cannot raise there), '
               'verification.py:575-578 (test patch file missing); sandbox start failures outside the try blocks '
               '(agent_runner.py:493, verification.py:384; reference runner: lib/g4s_docker.py:509, 514); a task id missing from '
               'tasks.jsonl (worker_berth.py:194); a failure writing the row side files (worker_berth.py:199-202)'),
    ErrorEntry('I09', 'infra', 'job_died', 'synthetic', 'started.jsonl entry without a results row',
               'lib/worker_berth.py:188-189 writes started.jsonl before the job, 214-215 the row after it (worker '
               'killed, runner deadline, crash)'),
    ErrorEntry('I10', 'infra', 'malformed_row', 'row', 'resolved missing or not a boolean',
               'lib/worker_berth.py:195, 204 always write a boolean'),
    ErrorEntry('I11', 'infra', 'contradictory_row', 'row', 'resolved true with a non-empty error',
               'swegemma/harness/verification.py:697 (error is None when resolved)'),
    ErrorEntry('I12', 'infra', 'unknown_error_text', 'row', 'any other error text',
               'none of the sources above writes it; every such row is listed in the report'),
)
ERROR_TABLE_SHA256 = hashlib.sha256(repr(ERROR_TABLE_V2).encode()).hexdigest()
_ERR_BY_ID = {e.id: e for e in ERROR_TABLE_V2}
_PREFIX_ENTRIES = tuple(e for e in ERROR_TABLE_V2 if e.kind == 'prefix')   # table order: I01 before I02
_EXC_A08 = re.compile(_ERR_BY_ID['A08'].pattern)


def _cls(eid, detail=''):
    e = _ERR_BY_ID[eid]
    return dict(klass=e.klass, category=e.category, id=eid, detail=detail)


def classify_row_v2(row):
    """A1 class of one result row: dict(klass 'agent' | 'infra', category, id (ERROR_TABLE_V2), detail)."""
    if row.get('_synthetic') == 'started_without_result':
        return _cls('I09', 'job in started.jsonl without a results row')
    exc, tb = row.get('exc'), row.get('tb')
    if exc or tb:
        text = str(exc) if exc else ''
        if _EXC_A08.match(text):
            return _cls('A08', text[:160])
        return _cls('I08', 'worker exception: ' + str(exc or tb)[:160])
    res = row.get('resolved')
    if not isinstance(res, bool):
        return _cls('I10', f'resolved = {res!r}'[:160])
    err = row.get('error')
    err = '' if err is None else str(err).strip()
    if res:
        return _cls('I11', err[:160]) if err else _cls('A00')
    if not err:
        return _cls('A01')
    for e in _PREFIX_ENTRIES:
        if err.startswith(e.pattern):
            return _cls(e.id, err[:160])
    return _cls('I12', err[:160])


def is_infra_v2(row):
    return classify_row_v2(row)['klass'] == 'infra'


_V2_METRIC_STATUS = {'solved': 'ok', 'unsolved': 'ok', 'agent_patch_apply': 'agent_patch',
                     'agent_patch_missing': 'agent_patch'}


def _status_v2_metric(row):
    """row_status-like (category, detail) under the v2 table, for the secondary metrics (run_metrics)."""
    c = classify_row_v2(row)
    if c['klass'] == 'infra':
        return 'harness_error', c['detail']
    return _V2_METRIC_STATUS.get(c['category'], c['category']), c['detail']


def error_table_markdown():
    L = ['| id | class | category | match | pattern | source |', '|---|---|---|---|---|---|']
    for e in ERROR_TABLE_V2:
        pat = e.pattern.replace('|', '\\|')
        L.append(f'| {e.id} | {e.klass} | {e.category} | {e.kind} | `{pat}` | {e.source} |')
    return '\n'.join(L) + '\n'


# ----------------------------------------------------------------------------------------------------------------
# v2 loading: attempts, job ids, blocks, started-without-result records
# ----------------------------------------------------------------------------------------------------------------
def _as_int(x):
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str) and x.strip().isdigit():
        return int(x.strip())
    return None


def job_id_v2(block, arm, task, repeat, attempt):
    """Job id format shared with make_jobs.py: '<block>:<arm>:<task>:r<repeat>:a<attempt>'."""
    return f'{block}:{arm}:{task}:r{repeat}:a{attempt}'


def arm_order_key(task, repeat, arm):
    """Arm order within one task and repeat of an interleaved block (make_jobs.py, A3)."""
    return hashlib.sha256(f'g4s-armorder:{task}:{repeat}:{arm}'.encode()).hexdigest()


def interleaved_slots(tasks, arms, repeats):
    """Planned slots (arm, task, repeat) in block job order: by task, then repeat, then arm by arm_order_key."""
    return [(a, t, r) for t in tasks for r in range(1, repeats + 1)
            for a in sorted(arms, key=lambda a: arm_order_key(t, r, a))]


def _slot_key_of(rec):
    rep = _as_int(rec.get('repeat_id', rec.get('seed')))
    att = rec.get('attempt')
    return (rec.get('arm'), rec.get('task'), rep, 1 if att is None else _as_int(att))


def _v2_fields(r):
    row = r['row']
    att = row.get('attempt')
    r['attempt_recorded'] = att is not None
    r['attempt'] = 1 if att is None else _as_int(att)
    if r['attempt'] is not None and r['attempt'] < 1:
        r['attempt'] = None
    r['attempt_raw'] = att
    r['job_id'] = row.get('job_id')
    r['synthetic'] = row.get('_synthetic')
    return r


def load_runs_v2(dirs, arm_from_repeat=False, arm_tag_from_dir=False):
    """load_runs() plus the v2 fields (attempt, job_id) and, for every out dir with started.jsonl, one synthetic
    infrastructure-error record (I09) per started job without a results row (matched on arm, task, repeat, attempt).
    The same out dir given twice is loaded once. Returns (runs, problems, notes)."""
    uniq, seen, notes = [], set(), []
    for d in dirs:
        od = os.path.realpath(find_out_dir(d))
        if od in seen:
            notes.append(f'run directory {d} resolves to an out dir already given; loaded once')
            continue
        seen.add(od)
        uniq.append(d)
    runs, problems = load_runs(uniq, arm_from_repeat=arm_from_repeat, arm_tag_from_dir=arm_tag_from_dir)
    for r in runs:
        _v2_fields(r)
    for di, d in enumerate(uniq):
        od = find_out_dir(d)
        sp = os.path.join(od, 'started.jsonl')
        if not os.path.isfile(sp):
            continue
        label = _dir_label(d, od)
        have = collections.Counter(_slot_key_of(r['row']) for r in runs if r['source'] == od and not r['synthetic'])
        with open(sp, encoding='utf-8') as f:
            for li, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    problems.append(dict(source=sp, line=li + 1, problem=f'unparsable started record: {e}'))
                    continue
                key = _slot_key_of(rec)
                if have[key] > 0:
                    have[key] -= 1
                    continue
                arm, task, rep = rec.get('arm'), rec.get('task'), key[2]
                if not arm or not task or rep is None:
                    problems.append(dict(source=sp, line=li + 1, problem='started record without arm, task or repeat'))
                    continue
                row = dict(rec)
                row.update(_synthetic='started_without_result', resolved=False, seed=rec.get('seed', rep), repeat_id=rep)
                a, rr = arm, rep
                if arm_from_repeat:
                    a, rr = f'{arm}#r{rep}', 1
                if arm_tag_from_dir:
                    a = f'{a}@{label}'
                trace = os.path.join(od, arm, f"s{row['seed']}", 'traces', 'trace_' + task.replace('/', '__') + '.json')
                runs.append(_v2_fields(dict(arm=a, orig_arm=arm, task=task, repeat=rr, orig_repeat=rep, row=row, source=od,
                                            label=label, order=(di, 10 ** 9 + li), trace_path=trace)))
    return runs, problems, notes


def load_drop_file(path):
    """A0 drop list: a JSON list of task ids, a JSON object with 'dropped' (task ids or objects with 'task' or
    'instance_id'), or text with one task id per line ('#' starts a comment). Returns (set of ids, sha256)."""
    with open(path, 'rb') as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    text = raw.decode('utf-8')
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        ids = [ln.split('#', 1)[0].strip() for ln in text.splitlines()]
        return {i for i in ids if i}, sha
    items = d.get('dropped') if isinstance(d, dict) else d
    if not isinstance(items, list):
        raise ValueError(f'drop file {path}: expected a list or an object with key "dropped"')
    out = set()
    for it in items:
        tid = it if isinstance(it, str) else (it.get('task') or it.get('instance_id') if isinstance(it, dict) else None)
        if not tid:
            raise ValueError(f'drop file {path}: entry without a task id: {it!r}')
        out.add(tid)
    return out, sha


def apply_drops(panel, dropped=None):
    """A0: remove dropped tasks from the panel (not replaced); stop when more than 10% of its tasks are dropped."""
    tasks = list(panel['tasks'])
    dropped = set(dropped or ())
    hit = [t for t in tasks if t in dropped]
    info = dict(n_panel=len(tasks), n_dropped=len(hit), n_used=len(tasks) - len(hit), dropped_in_panel=hit,
                fraction=float(Fraction(len(hit), len(tasks))) if tasks else 0.0, limit='more than 10% stops')
    if tasks and Fraction(len(hit), len(tasks)) > DROP_MAX:
        raise AnalysisStop(f"A0: {len(hit)} of {len(tasks)} tasks of panel {panel.get('panel')} are dropped (more than "
                           '10%): no E7 run starts until the cause is understood', details=hit)
    return [t for t in tasks if t not in dropped], info


def _ref(r):
    return f"{r['label']}:{'started' if r.get('synthetic') else 'line'} {r['order'][1] % 10 ** 9 + 1}"


def assemble_v2(runs, tasks, arms, repeats, block=None, dropped=(), max_attempts=MAX_ATTEMPTS):
    """A1/A2 slot assembly. A planned slot is (arm, task, repeat); its rows are attempts. Infrastructure-error rows
    are missing; the slot's used row is its one row that is not an infrastructure error, at attempt <= 3. Stops
    (AnalysisStop) on two non-infrastructure rows in one slot, two rows with one attempt index, or an attempt index
    that is not a positive integer. Rows outside the plan are ignored and listed; the order of the run directories
    never decides a row."""
    task_set, arm_set, dropped = set(tasks), set(arms), set(dropped or ())
    by_slot = collections.defaultdict(list)
    ignored = []
    for r in runs:
        rb = r['row'].get('block')
        reason = None
        if r['arm'] not in arm_set:
            reason = 'arm not analysed'
        elif r['task'] not in task_set:
            reason = 'task dropped (A0)' if r['task'] in dropped else 'task not in panel'
        elif not 1 <= r['repeat'] <= repeats:
            reason = f'repeat id outside 1..{repeats}'
        elif block is not None and rb is not None and rb != block:
            reason = f'row from another block ({rb})'
        if reason:
            ignored.append(dict(reason=reason, arm=r['arm'], task=r['task'], repeat=r['repeat'], attempt=r['attempt'],
                                ref=_ref(r)))
            continue
        by_slot[(r['arm'], r['task'], r['repeat'])].append(r)
    fatal, chosen, lacking, deviations, slots, mismatch = [], {}, [], [], {}, []
    infra = {a: [] for a in arms}
    attempts_hist = collections.Counter()
    for a in arms:
        for t in tasks:
            for k in range(1, repeats + 1):
                key = (a, t, k)
                lst = sorted(by_slot.get(key, []), key=lambda x: (x['attempt'] or 0, x['source'], x['order'][1]))
                slots[key] = lst
                if any(x['attempt'] is None for x in lst):
                    fatal.append(dict(problem='attempt index is not a positive integer', slot=key,
                                      rows=[(_ref(x), x['attempt_raw']) for x in lst]))
                    continue
                cnt = collections.Counter(x['attempt'] for x in lst)
                if any(c > 1 for c in cnt.values()):
                    fatal.append(dict(problem='two rows with the same attempt index (A2)', slot=key,
                                      rows=[(_ref(x), x['attempt'], x['job_id']) for x in lst]))
                    continue
                cl = [(x, classify_row_v2(x['row'])) for x in lst]
                for x, c in cl:   # RUN_RECORD_SPEC.md: the runner records the class; the analysis's own one decides
                    rec = x['row'].get('error_class_id')
                    if rec is not None and rec != c['id']:
                        mismatch.append(dict(slot=key, attempt=x['attempt'], recorded=rec, computed=c['id'], ref=_ref(x)))
                good = [x for x, c in cl if c['klass'] == 'agent']
                if len(good) > 1:
                    fatal.append(dict(problem='more than one row that is not an infrastructure error (A2)', slot=key,
                                      rows=[(_ref(x), x['attempt'], x['job_id']) for x in good]))
                    continue
                for x, c in cl:
                    if c['klass'] == 'infra':
                        infra[a].append(dict(slot=key, attempt=x['attempt'], id=c['id'], category=c['category'],
                                             detail=c['detail'][:100], ref=_ref(x), job_id=x['job_id']))
                attempts_hist[len(lst)] += 1
                atts = sorted(cnt)
                if atts and atts != list(range(1, atts[-1] + 1)):
                    deviations.append(dict(problem='attempt indices with a gap (an attempt has no row and no started '
                                                   'record; its infrastructure error may be uncounted in (4b))', slot=key,
                                           rows=[(_ref(x), x['attempt']) for x in lst]))
                over = [x for x in lst if x['attempt'] > max_attempts]
                if over:
                    deviations.append(dict(problem=f'attempt index above {max_attempts} (more than two re-runs)', slot=key,
                                           rows=[(_ref(x), x['attempt']) for x in over]))
                if good and good[0]['attempt'] <= max_attempts:
                    chosen[key] = good[0]
                    late = [x for x in lst if x['attempt'] > good[0]['attempt']]
                    if late:
                        deviations.append(dict(problem='re-run of a slot that already had a valid row', slot=key,
                                               rows=[(_ref(x), x['attempt']) for x in late]))
                else:
                    lacking.append(dict(arm=a, task=t, repeat=k, attempts=[x['attempt'] for x in lst],
                                        classes=[c['id'] for _, c in cl],
                                        reason=('no row' if not lst else ('valid row only above attempt 3' if good
                                                                          else 'only infrastructure-error rows'))))
    if fatal:
        raise AnalysisStop(f'{len(fatal)} planned slot(s) with conflicting rows: the analysis stops without a decision '
                           '(A2: the lead resolves them from job ids and logs the deviation)', details=fatal)
    cells = {(a, t): [chosen[(a, t, k)] for k in range(1, repeats + 1) if (a, t, k) in chosen] for a in arms for t in tasks}
    n_rows = sum(len(v) for v in slots.values())
    return dict(tasks=list(tasks), arms=list(arms), repeats=repeats, chosen=chosen, cells=cells, slots=slots,
                lacking=lacking, deviations=deviations, infra=infra, ignored=ignored, class_mismatches=mismatch,
                n_planned=len(arms) * len(tasks) * repeats, n_rows_in_plan=n_rows,
                attempts_hist={str(k): v for k, v in sorted(attempts_hist.items())},
                rows_without_attempt=sum(1 for v in slots.values() for x in v if not x['attempt_recorded']),
                rows_without_job_id=sum(1 for v in slots.values() for x in v if not x['job_id']))


# ----------------------------------------------------------------------------------------------------------------
# v2 criteria: (4b) exact binomial, guard H, provenance (A3), decision
# ----------------------------------------------------------------------------------------------------------------
def binom_upper_tail(k, n, pi):
    """Exact P(X >= k) for X ~ Binomial(n, pi) (pi a Fraction); a Fraction."""
    k, n, pi = int(k), int(n), Fraction(pi)
    if k <= 0:
        return Fraction(1)
    if k > n:
        return Fraction(0)
    q = 1 - pi
    return sum(math.comb(n, j) * pi ** j * q ** (n - j) for j in range(k, n + 1))


def harness_4b(k_a, k_b, planned_a, planned_b):
    """A1 (4b): one-sided exact binomial test of the candidate's share of all infrastructure-error rows (all
    attempts, both arms) against its planned-slot share; fails at p <= 0.05; no errors in either arm passes."""
    n = k_a + k_b
    share = Fraction(planned_b, planned_a + planned_b) if planned_a + planned_b else Fraction(1, 2)
    out = dict(k_baseline=k_a, k_candidate=k_b, n=n, planned_baseline=planned_a, planned_candidate=planned_b,
               share_candidate=_fmt_frac(share), alpha=float(ALPHA_4B))
    if n == 0:
        out.update(p=None, p_exact=None, fails=False, note='no infrastructure-error rows in either arm: (4b) passes')
        return out
    p = binom_upper_tail(k_b, n, share)
    out.update(p=float(p), p_exact=_fmt_frac(p) if p.denominator < 10 ** 30 else None, fails=bool(p <= ALPHA_4B))
    return out


def guard_h(K, L, repos, limit=GUARD_H_LIMIT):
    """A4 guard H: a repository's total difference = sum over its tasks of (candidate mean - baseline mean), from
    task differences D_i = K_i / L. H fails when any total is <= limit (-2 task-equivalents). Exact."""
    totals, counts = collections.defaultdict(Fraction), collections.Counter()
    for k, r in zip(K, repos):
        totals[r] += Fraction(int(k), L)
        counts[r] += 1
    if not totals:
        return dict(limit=float(limit), fails=False, min_total=None, min_total_exact=None, min_repo=None,
                    repos_at_or_below=[], table=[])
    table = sorted(totals, key=lambda r: (totals[r], r))
    low = [r for r in table if totals[r] <= limit]
    rows = [dict(repo=r, n_tasks=counts[r], total=float(totals[r]), total_exact=_fmt_frac(totals[r]),
                 mean=float(totals[r] / counts[r]), at_or_below_limit=totals[r] <= limit) for r in table]
    return dict(limit=float(limit), fails=bool(low), min_total=float(totals[table[0]]),
                min_total_exact=_fmt_frac(totals[table[0]]), min_repo=table[0], repos_at_or_below=low, table=rows)


def _canon(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)


def _empty(v):
    return v is None or v == '' or v == {} or v == []


def provenance_check_v2(planned_rows, used_rows, arms, adapter_arms=ADAPTER_ARMS):
    """A3: every row of the block's planned slots (all attempts) must agree on the block-wide fields and, within its
    arm, on the arm-specific ones; the rows used must agree within arm on the sampling actually sent. A missing value
    counts as a problem (the adapter sha256 is required only for adapter arms). Returns dict(ok, missing,
    disagreements, values)."""
    missing, examples, disagreements, values = collections.Counter(), {}, [], {}

    def miss(f, a, r):
        missing[(f, a)] += 1
        examples.setdefault((f, a), _ref(r))

    for f in ('block',) + PROV_BLOCK_FIELDS:
        c = collections.Counter()
        for a in arms:
            for r in planned_rows.get(a, []):
                v = r['row'].get(f)
                if _empty(v):
                    miss(f, a, r)
                else:
                    c[_canon(v)] += 1
        values[f] = dict(c)
        if len(c) > 1:
            disagreements.append(dict(field=f, scope='block (all arms)', values=dict(c)))
    for f in PROV_ARM_FIELDS:
        for a in arms:
            c = collections.Counter()
            need = f != 'adapter_sha256' or a in adapter_arms
            for r in planned_rows.get(a, []):
                v = r['row'].get(f)
                if _empty(v):
                    if need:
                        miss(f, a, r)
                else:
                    c[_canon(v)] += 1
            values[f'{f}[{a}]'] = dict(c)
            if len(c) > 1:
                disagreements.append(dict(field=f, scope=f'arm {a}', values=dict(c)))
    for f in PROV_USED_ARM_FIELDS:
        for a in arms:
            c = collections.Counter()
            for r in used_rows.get(a, []):
                v = r['row'].get(f)
                if _empty(v):
                    miss(f, a, r)
                else:
                    c[_canon(v)] += 1
            values[f'{f}[{a}]'] = dict(c)
            if len(c) > 1:
                disagreements.append(dict(field=f, scope=f'arm {a} (rows used)', values=dict(c)))
    miss_list = [dict(field=f, arm=a, rows=n, example=examples[(f, a)]) for (f, a), n in sorted(missing.items())]
    return dict(ok=not miss_list and not disagreements, missing=miss_list, disagreements=disagreements, values=values)


def decision_v2(criteria, inconclusive_reasons):
    """INCONCLUSIVE while any reason exists (A1 slots, A3 provenance, unrecorded agent time); else PASS iff all
    criteria hold (criterion (5), guard H, is in `criteria` only when adopted)."""
    if inconclusive_reasons:
        return 'INCONCLUSIVE'
    return 'PASS' if all(bool(v) for v in criteria.values()) else 'FAIL'


def _frozen_notes(panel, repeats, n_boot, seed):
    notes, frozen = [], True
    dflt = DEFAULT_REPEATS.get(panel.get('panel'))
    if dflt is None or repeats != dflt:
        frozen = False
        notes.append(f"NON-FROZEN repeat count {repeats} for panel {panel.get('panel')} (frozen: S 2, C 4, D 4)")
    if seed != BOOT_SEED or n_boot != N_BOOT:
        frozen = False
        notes.append(f'NON-FROZEN bootstrap settings (seed {seed}, draws {n_boot}); the frozen ones are seed {BOOT_SEED}, '
                     f'draws {N_BOOT}')
    return frozen, notes


def _infra_summary(asm, arm):
    lst = asm['infra'][arm]
    return dict(rows=len(lst), by_id=dict(collections.Counter(x['id'] for x in lst)),
                by_attempt=dict(collections.Counter(str(x['attempt']) for x in lst)),
                details=dict(collections.Counter(x['detail'][:80] for x in lst).most_common(20)))


def _silent_verification_suspects(runs):
    """Diagnostic only: used unsolved rows with no error text whose pytest exit code is not one pytest or a timeout
    produces (0-5, 124); e.g. 125 (docker could not start the verification container, lib/g4s_docker.py:321-334)."""
    out = []
    for r in runs:
        row = r['row']
        if row.get('resolved') is False and not str(row.get('error') or '').strip():
            te = row.get('test_exit')
            if te is not None and te not in (0, 1, 2, 3, 4, 5, 124):
                out.append(dict(task=r['task'], repeat=r['repeat'], test_exit=te, ref=_ref(r)))
    return out


# ----------------------------------------------------------------------------------------------------------------
# Gate under rules v2
# ----------------------------------------------------------------------------------------------------------------
def gate_v2(results, panel, task_repo, arm_a=BASELINE, arm_b=None, repeats=None, n_boot=N_BOOT, seed=BOOT_SEED,
            traces=None, with_secondary=True, exact=True, dropped=None, drop_source=None, block=None,
            guard_adopted=None, load_problems=None):
    """Confirmation/dev gate under PROTOCOL_AMENDMENTS.md A0-A7. `results` are runs from load_runs_v2();
    `load_problems` are its unreadable or unplaceable records (any makes the gate INCONCLUSIVE)."""
    adopted = GUARD_H_ADOPTED if guard_adopted is None else bool(guard_adopted)
    repeats = repeats or DEFAULT_REPEATS.get(panel.get('panel'), 4)
    check_panel_repos(panel, task_repo)
    tasks, a0 = apply_drops(panel, dropped)
    if drop_source:
        a0.update(drop_source)
    block = block or panel.get('panel')
    asm = assemble_v2(results, tasks, [arm_a, arm_b], repeats, block=block, dropped=dropped)
    diffs, reps, used, excluded = [], [], [], []
    for t in tasks:
        ra, rb = asm['cells'][(arm_a, t)], asm['cells'][(arm_b, t)]
        if not ra or not rb:
            excluded.append(dict(task=t, repo=task_repo[t], used_a=len(ra), used_b=len(rb)))
            continue
        diffs.append(Fraction(sum(solved(r['row']) for r in rb), len(rb)) - Fraction(sum(solved(r['row']) for r in ra), len(ra)))
        reps.append(task_repo[t])
        used.append(t)
    K, L = _common_denominator(diffs) if diffs else ([], 1)
    core = gate_core(K, L, reps, n_boot=n_boot, seed=seed, exact=exact)
    guard = guard_h(K, L, reps)
    for row in guard['table']:
        lo = (core.get('repos') or {}).get(row['repo'], {})
        row['loo'] = lo.get('loo')
    runs_a = [r for t in used for r in asm['cells'][(arm_a, t)]]
    runs_b = [r for t in used for r in asm['cells'][(arm_b, t)]]
    ratio, ta, tb = _time_ratio(runs_a, runs_b)
    no_time = [dict(arm=r['arm'], task=r['task'], repeat=r['repeat'], ref=_ref(r)) for r in runs_a + runs_b
               if not isinstance(r['row'].get('t_agent'), (int, float)) or isinstance(r['row'].get('t_agent'), bool)]
    planned = len(tasks) * repeats
    h4b = harness_4b(len(asm['infra'][arm_a]), len(asm['infra'][arm_b]), planned, planned)
    planned_rows = {a: [x for (aa, t, k), lst in asm['slots'].items() if aa == a for x in lst] for a in (arm_a, arm_b)}
    used_rows = {a: [r for t in tasks for r in asm['cells'][(a, t)]] for a in (arm_a, arm_b)}
    prov = provenance_check_v2(planned_rows, used_rows, [arm_a, arm_b])
    reasons = []
    if asm['lacking']:
        reasons.append(f"{len(asm['lacking'])} planned slot(s) without a row that is not an infrastructure error (A1)")
    if not prov['ok']:
        reasons.append(f"provenance: {len(prov['missing'])} missing field group(s), {len(prov['disagreements'])} "
                       'disagreement(s) (A3)')
    if no_time:
        reasons.append(f'{len(no_time)} used row(s) without a recorded agent time ((4a) cannot be computed on all used rows)')
    if load_problems:
        reasons.append(f'{len(load_problems)} result or started record(s) could not be read or placed (load_problems); '
                       'one may be an uncounted infrastructure error')
    time_ok = ratio is not None and ratio <= TIME_RATIO_MAX
    crit = dict(core['criteria'], time=bool(time_ok), harness=not h4b['fails'])
    if adopted:
        crit['guard_h'] = not guard['fails']
    decision = decision_v2(crit, reasons)
    frozen, notes = _frozen_notes(panel, repeats, n_boot, seed)
    notes = list(core.get('notes', [])) + notes
    if arm_b not in ADAPTER_ARMS and panel.get('panel') == 'D':
        notes.append(f'arm {arm_b} on panel D: A7 labels the adapter arm bundle0_adapter_A or bundle0_adapter_B')
    if asm['deviations']:
        notes.append(f"{len(asm['deviations'])} protocol deviation(s) in attempts (listed under slots)")
    if asm['rows_without_attempt']:
        notes.append(f"{asm['rows_without_attempt']} row(s) without an attempt index were taken as attempt 1")
    if asm['class_mismatches']:
        notes.append(f"{len(asm['class_mismatches'])} row(s) whose recorded error_class_id differs from the A1 table "
                     '(the table decides; check the runner)')
    unknown = [x for a in (arm_a, arm_b) for x in asm['infra'][a] if x['id'] == 'I12']
    if unknown:
        notes.append(f'{len(unknown)} row(s) with error text outside the frozen A1 table were counted as infrastructure '
                     'errors (I12); the lead should check them')
    sens_missing = None
    if excluded:
        n_all = len(used) + len(excluded)
        s = sum(diffs, Fraction(0))
        sens_missing = dict(n_excluded=len(excluded), d_if_excluded_tasks_were_minus1=float((s - len(excluded)) / n_all),
                            d_if_excluded_tasks_were_plus1=float((s + len(excluded)) / n_all),
                            d_if_excluded_tasks_were_0=float(s / n_all))
    out = dict(kind='gate', rules='v2', analysis=ANALYSIS_ID_V2, error_table_sha256=ERROR_TABLE_SHA256,
               panel=panel.get('panel'), panel_sha256=panel.get('sha256'), block=block, arm_a=arm_a, arm_b=arm_b,
               repeats=repeats, frozen_settings=frozen, decision=decision, inconclusive_reasons=reasons,
               criteria=crit, criteria_note=('provisional values on the rows available; not a decision'
                                             if reasons else 'decision criteria'),
               guard_h=dict(guard, adopted=adopted, role='criterion (5)' if adopted else 'diagnostic only (A4)'),
               n_tasks_planned=len(panel['tasks']), n_tasks_after_a0=len(tasks), n_tasks_used=len(used),
               excluded_tasks=excluded, excluded_sensitivity=sens_missing, a0=a0, notes=notes,
               thresholds=dict(gain_min=_fmt_frac(GAIN_MIN), alpha=float(ALPHA), min_pos_repos=MIN_POS_REPOS,
                               time_ratio_max=float(TIME_RATIO_MAX), alpha_4b=float(ALPHA_4B),
                               guard_h_limit=float(GUARD_H_LIMIT), max_attempts=MAX_ATTEMPTS, n_boot=N_BOOT,
                               seed=BOOT_SEED),
               gain=core,
               time=dict(ratio=None if ratio is None else float(ratio), mean_a=float(sum(ta) / len(ta)) if ta else None,
                         mean_b=float(sum(tb) / len(tb)) if tb else None, n_a=len(ta), n_b=len(tb),
                         rows_without_t_agent=no_time, note='(4a) on the rows used in the task means only (A1)'),
               harness=dict(test_4b=h4b, by_arm={a: _infra_summary(asm, a) for a in (arm_a, arm_b)},
                            rows=[dict(arm=a, **x) for a in (arm_a, arm_b) for x in asm['infra'][a]][:400],
                            silent_verification_suspects={a: _silent_verification_suspects(used_rows[a])
                                                          for a in (arm_a, arm_b)}),
               provenance_check=prov,
               slots=dict(n_planned=asm['n_planned'], n_with_used_row=len(asm['chosen']), lacking=asm['lacking'],
                          deviations=asm['deviations'], attempts_hist=asm['attempts_hist'],
                          rows_in_plan=asm['n_rows_in_plan'], rows_without_attempt=asm['rows_without_attempt'],
                          rows_without_job_id=asm['rows_without_job_id'], class_mismatches=asm['class_mismatches']),
               data=dict(ignored_rows=dict(collections.Counter(x['reason'] for x in asm['ignored'])),
                         ignored=asm['ignored'][:300]))
    if with_secondary:
        out['secondary'] = secondary(results, traces if traces is not None else TraceStore(),
                                     dict(panel=panel.get('panel'), tasks=tasks), task_repo, arm_a, arm_b, repeats,
                                     asm=asm, status=_status_v2_metric)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Screening under rules v2 (exploratory; A1 slots and A2 duplicates apply; A0 drops apply)
# ----------------------------------------------------------------------------------------------------------------
def _arm_used_summary_v2(asm, arm, store):
    runs = [r for t in asm['tasks'] for r in asm['cells'][(arm, t)]]
    cats = collections.Counter(classify_row_v2(r['row'])['category'] for r in runs)
    times = [float(r['row']['t_agent']) for r in runs if isinstance(r['row'].get('t_agent'), (int, float))]
    tr = [store.get(r['trace_path']) for r in runs]
    ok_tr = [m for m in tr if m and not m.get('error')]
    att = sum(m['write_attempts'] for m in ok_tr)
    okw = sum(m['ok_writes'] for m in ok_tr)
    planned = len(asm['tasks']) * asm['repeats']
    return dict(planned=planned, used=len(runs), lacking=sum(1 for x in asm['lacking'] if x['arm'] == arm),
                infra_rows=len(asm['infra'][arm]), infra_by_id=dict(collections.Counter(x['id'] for x in asm['infra'][arm])),
                solves=sum(solved(r['row']) for r in runs), outcome_counts=dict(cats),
                mean_agent_time=(sum(times) / len(times)) if times else None, n_agent_time=len(times),
                edit_attempts=att, edit_ok=okw, edit_success=(Fraction(okw, att) if att else None),
                traces_found=len(ok_tr), traces_missing=sum(1 for m in tr if m is None),
                traces_unreadable=sum(1 for m in tr if m and m.get('error')),
                edit_channels={ch: [sum(m[f'{ch}_attempts'] for m in ok_tr), sum(m[f'{ch}_failed'] for m in ok_tr)]
                               for ch in ('edit_file', 'write_file', 'shell')})


def screening_v2(results, panel_S, task_repo=None, baseline=BASELINE, arms=None, repeats=None, traces=None,
                 dropped=None, drop_source=None, block=None, load_problems=None):
    """PROTOCOL.md screening with the A1 slot rule (infrastructure-error rows are missing, not unsolved), A2
    duplicate stop and A0 drops. Gain, edit-call success (A6/A7) and the finalist rule are unchanged."""
    repeats = repeats or DEFAULT_REPEATS.get(panel_S.get('panel'), 2)
    tasks, a0 = apply_drops(panel_S, dropped)
    if drop_source:
        a0.update(drop_source)
    block = block or panel_S.get('panel')
    if arms is None:
        present = {r['arm'] for r in results}
        arms = [a for a in SCREEN_ARMS if a in present and a != baseline] + sorted(
            a for a in present if a not in SCREEN_ARMS and a != baseline)
    arms = [a for a in arms if a != baseline]
    store = traces if traces is not None else TraceStore()
    asm = assemble_v2(results, tasks, [baseline] + arms, repeats, block=block, dropped=dropped)
    summ = {a: _arm_used_summary_v2(asm, a, store) for a in [baseline] + arms}
    rows = []
    for a in arms:
        gain, used, excluded = Fraction(0), [], []
        per_repo = collections.defaultdict(Fraction)
        for t in tasks:
            ra, rb = asm['cells'][(baseline, t)], asm['cells'][(a, t)]
            if not ra or not rb:
                excluded.append(dict(task=t, n_baseline=len(ra), n_arm=len(rb)))
                continue
            diff = Fraction(sum(solved(r['row']) for r in rb), len(rb)) - Fraction(sum(solved(r['row']) for r in ra), len(ra))
            gain += diff
            used.append(t)
            if task_repo:
                per_repo[task_repo.get(t)] += diff
        s = summ[a]
        rows.append(dict(arm=a, gain=gain, gain_float=float(gain), gain_exact=_fmt_frac(gain), tasks_used=len(used),
                         tasks_excluded=excluded, mean_gain_per_task=float(gain / len(used)) if used else None,
                         repos_positive=sum(1 for v in per_repo.values() if v > 0),
                         repos_negative=sum(1 for v in per_repo.values() if v < 0),
                         per_repo_gain={k: float(v) for k, v in sorted(per_repo.items())},
                         edit_success=s['edit_success'], mean_agent_time=s['mean_agent_time']))
    order = {a: i for i, a in enumerate(SCREEN_ARMS)}

    def key(r):
        es, mt = r['edit_success'], r['mean_agent_time']
        return (-r['gain'], -(es if es is not None else Fraction(-1)), mt if mt is not None else float('inf'),
                order.get(r['arm'], len(order)), r['arm'])

    ranked = sorted(rows, key=key)
    finalist, tie_log = None, []
    if ranked and ranked[0]['gain'] > 0:
        finalist = ranked[0]['arm']
        top = [r for r in ranked if r['gain'] == ranked[0]['gain']]
        if len(top) > 1:
            tie_log.append(f"gain tie among {[r['arm'] for r in top]}; broken by edit-call success rate")
            es_top = [r for r in top if r['edit_success'] == top[0]['edit_success']]
            if len(es_top) > 1:
                tie_log.append(f"edit-success tie among {[r['arm'] for r in es_top]}; broken by mean agent time")
                mt_top = [r for r in es_top if r['mean_agent_time'] == es_top[0]['mean_agent_time']]
                if len(mt_top) > 1:
                    tie_log.append('UNRESOLVED by the protocol: fell back to PROTOCOL arm order (flag to owner)')
    for r in rows:
        r['edit_success_float'] = None if r['edit_success'] is None else float(r['edit_success'])
        r['edit_success'] = _fmt_frac(r['edit_success']) if r['edit_success'] is not None else None
        r['gain'] = r['gain_float']
    for a, s in summ.items():
        s['edit_success_float'] = None if s['edit_success'] is None else float(s['edit_success'])
        s['edit_success'] = _fmt_frac(s['edit_success']) if s['edit_success'] is not None else None
    all_arms = [baseline] + arms
    planned_rows = {a: [x for (aa, t, k), lst in asm['slots'].items() if aa == a for x in lst] for a in all_arms}
    used_rows = {a: [r for t in tasks for r in asm['cells'][(a, t)]] for a in all_arms}
    prov = provenance_check_v2(planned_rows, used_rows, all_arms)
    provisional = []
    if asm['lacking']:
        provisional.append(f"{len(asm['lacking'])} planned slot(s) without a valid row: re-run (A1) before selecting")
    if not prov['ok']:
        provisional.append('provenance problems (PROTOCOL stop rule; A3)')
    if load_problems:
        provisional.append(f'{len(load_problems)} unreadable or unplaceable record(s)')
    decision = f'finalist {finalist}' if finalist else 'no finalist (largest gain <= 0): bundle0 stays'
    if provisional:
        decision = f"{decision} (PROVISIONAL: {'; '.join(provisional)})"
    frozen, notes = _frozen_notes(panel_S, repeats, N_BOOT, BOOT_SEED)
    return dict(kind='screening', rules='v2', analysis=ANALYSIS_ID_V2, error_table_sha256=ERROR_TABLE_SHA256,
                panel=panel_S.get('panel'), panel_sha256=panel_S.get('sha256'), block=block, baseline=baseline, arms=arms,
                repeats=repeats, frozen_settings=frozen, notes=notes, n_tasks=len(tasks), n_tasks_planned=len(panel_S['tasks']),
                a0=a0, ranking=[r['arm'] for r in ranked], finalist=finalist, decision=decision, provisional=provisional,
                tie_log=tie_log, arms_table=rows, run_summary=summ, provenance_check=prov,
                slots=dict(n_planned=asm['n_planned'], n_with_used_row=len(asm['chosen']), lacking=asm['lacking'],
                           deviations=asm['deviations'], attempts_hist=asm['attempts_hist'],
                           rows_without_attempt=asm['rows_without_attempt']),
                data=dict(ignored_rows=dict(collections.Counter(x['reason'] for x in asm['ignored'])),
                          ignored=asm['ignored'][:300]))


def rerun_jobs_v2(results, panel, arms, repeats=None, block=None, dropped=None):
    """A1 re-run list: every planned slot without a valid row, in block job order (interleaved_slots), with the same
    arm, task and repeat and the next attempt index; slots that already used attempt 3 are listed as exhausted."""
    repeats = repeats or DEFAULT_REPEATS.get(panel.get('panel'), 4)
    tasks, a0 = apply_drops(panel, dropped)
    block = block or panel.get('panel')
    asm = assemble_v2(results, tasks, list(arms), repeats, block=block, dropped=dropped)
    jobs, exhausted = [], []
    for a, t, k in interleaved_slots(tasks, list(arms), repeats):
        if (a, t, k) in asm['chosen']:
            continue
        lst = asm['slots'].get((a, t, k), [])
        nxt = max((x['attempt'] for x in lst), default=0) + 1
        if nxt > MAX_ATTEMPTS:
            exhausted.append(dict(arm=a, task=t, repeat=k, attempts=[x['attempt'] for x in lst]))
            continue
        jobs.append(dict(arm=a, task=t, seed=k, repeat_id=k, attempt=nxt, block=block,
                         job_id=job_id_v2(block, a, t, k, nxt)))
    return dict(kind='reruns', rules='v2', block=block, arms=list(arms), repeats=repeats, a0=a0, jobs=jobs,
                exhausted=exhausted, n_planned=asm['n_planned'], n_with_used_row=len(asm['chosen']))


# ================================================================================================================
# Rules v2.1: PROTOCOL_AMENDMENTS.md A0-A8 (A8, written before any run of the experiment). This is
# the CLI default. Everything above this block (rules v1 and v2) is unchanged: `--rules v2` and `--rules v1` reproduce
# their earlier outputs exactly (ANALYSIS_VALIDATION.md W.11). Python callers: gate_v21(), screening_v21(),
# rerun_jobs_v21() on the output of load_runs_v21().
# What v2.1 adds to v2 (A8):
#   * A0 at analysis time: the drop file's sha256 must equal the one every row and started record carries (copied by
#     the runner from the job file that make_jobs.py --drop-file wrote); a row of a dropped task in the block, a hash
#     mismatch, or any frozen panel (S, C, D) losing more than 10% STOPS the analysis.
#   * Reclassification (table ERROR_TABLE_V21): 'Failed to apply test_patch' (v2 I06) and context-length HTTP 400s (v2
#     part of I01) are agent outcomes (A17, A18); 'Failed to apply agent patch' with a Docker/g4s error text is
#     infrastructure (I13); 'Agent patch file missing' is infrastructure (I14, v2 A08).
#   * Binding candidate-penalised sensitivity d_pen: every candidate slot with an infrastructure-error row of a class in
#     D_PEN_IDS (any attempt) is scored unsolved, the baseline unchanged; criteria (1), (2), (3) and (5) must also hold
#     with d_pen in place of d.
#   * Criterion (4b) stratified by job batch (one batch = one run directory = one run tag): within batch b the
#     candidate's infrastructure rows are Binomial(n_b, s_b) under the null, s_b = the candidate's share of the batch's
#     jobs (its jobs.json); exact one-sided p of the total; fails at p <= 0.05.
#   * Started records are matched to rows by job_id. A block manifest (written by the runner) lists every run tag of the
#     block; a missing tag, an unlisted run directory or an attempt gap makes the gate INCONCLUSIVE. Every field that
#     RUN_RECORD_SPEC.md marks required is enforced (missing -> INCONCLUSIVE).
#   * INCONCLUSIVE and STOP reports are blind (slots, provenance, error counts and load problems only). Pre-registered
#     next step: re-run batches up to attempt 3 (A1), then one blind extra batch of at most two more attempts (4 and 5)
#     for every slot still without a valid row; a gate still INCONCLUSIVE after it is a FAIL (bundle0 stays).
#   * Block job order with counterbalanced arm order (interleaved_slots_v21; make_jobs.py writes the job files).
#   * The screen defaults to the six frozen arms; an arm without rows makes it PROVISIONAL.
# ================================================================================================================
ANALYSIS_ID_V21 = 'analyze_e7 v2.1 (2026-11-09; PROTOCOL_AMENDMENTS.md A0-A8)'
RERUN_MAX_ATTEMPT = 3               # A1: the first run plus at most two re-runs (re-run batches: attempts 2 and 3)
MAX_ATTEMPTS_V21 = 5                # A8: at most two more attempts (4 and 5) in one blind extra batch
FROZEN_PANEL_FILES = ('panel_S.json', 'panel_C.json', 'panel_D.json')   # A8: A0 is checked on every panel
D_BLOCK_MIN_DEADLINE_S = 12 * 3600  # A8: run deadlines above the block's expected duration; D block at least 12 h
HOST_RUNS_PER_HOUR = 200             # deadline note only: about 200 runs/h measured with 8 workers on the reference host
PARITY_RATIO_LIMIT = Fraction(23, 20)   # A8: agent-time ratio of a second server configuration above 1.15 -> investigate (no rule)

CONTEXT_LENGTH_PATTERN = (r"(?is)^Sandbox execution error: litellm\.(?:ContextWindowExceededError\b|BadRequestError\b"
                          r"(?=.*?(?:context[ _-]?window|context[ _-]?length|maximum model length|max_model_len)))")
AGENT_PATCH_INFRA_PATTERN = (r"(?s)^Failed to apply agent patch:\s*(?:docker:|g4s[: ]|.*?(?:Error response from daemon|"
                             r"Cannot connect to the Docker daemon|/usr/bin/docker))")

_V2E = {e.id: e for e in ERROR_TABLE_V2}
ERROR_TABLE_V21 = (
    _V2E['A00'], _V2E['A01'], _V2E['A02'], _V2E['A03'], _V2E['A04'], _V2E['A05'], _V2E['A06'],
    ErrorEntry('A07', 'agent', 'agent_patch_apply', 'prefix', 'Failed to apply agent patch',
               'swegemma/harness/verification.py:463-488 (git apply / patch rejected the agent patch); with a Docker/g4s '
               'error text it is I13 (A8)'),
    _V2E['A09'], _V2E['A10'], _V2E['A11'], _V2E['A12'], _V2E['A13'], _V2E['A14'], _V2E['A15'], _V2E['A16'],
    ErrorEntry('A17', 'agent', 'test_patch_conflict', 'prefix', 'Failed to apply test_patch',
               'swegemma/harness/verification.py:573-597 (v2: I06). A8: an agent outcome (unsolved), as on Kaggle: the '
               'gold and empty controls show that the test patch applies to the baseline'),
    ErrorEntry('A18', 'agent', 'context_length', 'regex', CONTEXT_LENGTH_PATTERN,
               'swegemma/harness/agent_runner.py:1039-1043 with a litellm context-window or HTTP 400 context-length '
               'error (litellm 1.89.7 litellm_core_utils/exception_mapping_utils.py:69-95, 408-416; exceptions.py:376-406) '
               '(v2: part of I01). A8: an agent outcome (unsolved), as on Kaggle'),
    ErrorEntry('I01', 'infra', 'llm_connection_http', 'prefix', 'Sandbox execution error: litellm.',
               'swegemma/harness/agent_runner.py:1039-1043 (an LLM request raised a litellm exception: connection, '
               'timeout, HTTP status; context-length 400s are A18)'),
    _V2E['I02'], _V2E['I03'], _V2E['I04'], _V2E['I05'], _V2E['I07'], _V2E['I08'], _V2E['I09'], _V2E['I10'],
    _V2E['I11'], _V2E['I12'],
    ErrorEntry('I13', 'infra', 'evaluation_infrastructure', 'regex', AGENT_PATCH_INFRA_PATTERN,
               'swegemma/harness/verification.py:463-474 with the Docker/g4s error that lib/g4s_docker.py returns as the '
               "apply command's stderr (297-299 refused, 309-312 popen_error, 320-328 docker rc 125 after 3 tries, "
               '350-354 exec_error) (v2: part of A07). A8: infrastructure'),
    ErrorEntry('I14', 'infra', 'agent_patch_missing', 'exc',
               r"""^ContainerSetupError\(['"]Agent patch file missing in container on task """,
               'swegemma/harness/verification.py:465-468 raises ContainerSetupError (re-raised at 701-702); it reaches '
               'the row only as exc/tb (lib/worker_berth.py:203-204) (v2: A08). A8: infrastructure, covered by d_pen'),
)
ERROR_TABLE_SHA256_V21 = hashlib.sha256(repr(ERROR_TABLE_V21).encode()).hexdigest()
_ERR_BY_ID_V21 = {e.id: e for e in ERROR_TABLE_V21}
_REGEX_ENTRIES_V21 = tuple((e, re.compile(e.pattern)) for e in ERROR_TABLE_V21 if e.kind == 'regex')
_PREFIX_ENTRIES_V21 = tuple(e for e in ERROR_TABLE_V21 if e.kind == 'prefix')   # table order: I01 before I02
_EXC_I14 = re.compile(_ERR_BY_ID_V21['I14'].pattern)
ERROR_ID_MAP_V2_TO_V21 = {'I06': 'A17', 'A08': 'I14', 'I01': 'I01 or A18 (context length)', 'A07': 'A07 or I13 (Docker/g4s text)'}
# A8: d_pen scores a candidate slot unsolved when any of its rows (any attempt) has one of these classes. A8 names I01,
# I02, I03, I09, I12 and says 'Agent patch file missing' (I14) is covered; I13 (a Docker/g4s failure to apply the agent
# patch, an evaluation-infrastructure failure that v2 filed under A07) is added as the conservative reading (section 4
# of ANALYSIS_VALIDATION.md, now W.4; A9 ratified it).
D_PEN_IDS = ('I01', 'I02', 'I03', 'I09', 'I12', 'I13', 'I14')


def _cls21(eid, detail=''):
    e = _ERR_BY_ID_V21[eid]
    return dict(klass=e.klass, category=e.category, id=eid, detail=detail)


def classify_row_v21(row):
    """A1/A8 class of one result row under ERROR_TABLE_V21: dict(klass 'agent' | 'infra', category, id, detail).
    Order: started job without a row (I09); exc/tb (I14, else I08); resolved (I10, I11, A00); empty error (A01); the
    regex entries (A18, I13); the prefix entries in table order; anything else I12."""
    if row.get('_synthetic') == 'started_without_result':
        return _cls21('I09', 'job in started.jsonl without a results row')
    exc, tb = row.get('exc'), row.get('tb')
    if exc or tb:
        text = str(exc) if exc else ''
        if _EXC_I14.match(text):
            return _cls21('I14', text[:160])
        return _cls21('I08', 'worker exception: ' + str(exc or tb)[:160])
    res = row.get('resolved')
    if not isinstance(res, bool):
        return _cls21('I10', f'resolved = {res!r}'[:160])
    err = row.get('error')
    err = '' if err is None else str(err).strip()
    if res:
        return _cls21('I11', err[:160]) if err else _cls21('A00')
    if not err:
        return _cls21('A01')
    for e, rx in _REGEX_ENTRIES_V21:
        if rx.match(err):
            return _cls21(e.id, err[:160])
    for e in _PREFIX_ENTRIES_V21:
        if err.startswith(e.pattern):
            return _cls21(e.id, err[:160])
    return _cls21('I12', err[:160])


_V21_METRIC_STATUS = {'solved': 'ok', 'unsolved': 'ok', 'agent_patch_apply': 'agent_patch',
                      'test_patch_conflict': 'test_patch', 'context_length': 'context_length'}


def _status_v21_metric(row):
    """row_status-like (category, detail) under the v2.1 table, for the secondary metrics (run_metrics)."""
    c = classify_row_v21(row)
    if c['klass'] == 'infra':
        return 'harness_error', c['detail']
    return _V21_METRIC_STATUS.get(c['category'], c['category']), c['detail']


def error_table_markdown_v21():
    L = ['| id | class | category | match | pattern | source |', '|---|---|---|---|---|---|']
    for e in ERROR_TABLE_V21:
        pat = e.pattern.replace('|', '\\|')
        L.append(f'| {e.id} | {e.klass} | {e.category} | {e.kind} | `{pat}` | {e.source} |')
    return '\n'.join(L) + '\n'


# ----------------------------------------------------------------------------------------------------------------
# Required fields (RUN_RECORD_SPEC.md section 1 and 3; A8: every field marked required is enforced)
# ----------------------------------------------------------------------------------------------------------------
REQ_ROW_ALL_V21 = ('job_id', 'block', 'arm', 'task', 'seed', 'repeat_id', 'attempt', 'worker', 't_start_utc',
                   't_end_utc', 'resolved', 'error_class', 'error_class_id', 'error_table_sha256', 'docker_image_id',
                   'harness_sha256', 'vllm_version', 'served_model_name', 'request_model', 'bundle_sha256',
                   'a0_drop_sha256')                       # non-empty on every result row
REQ_ROW_KEYS_V21 = ('t_agent', 't_setup', 't_verify', 't_phase1', 'sampling_sent')   # key present on every row
REQ_ROW_NOEXC_KEYS_V21 = ('error', 'test_exit', 'test_summary')                       # key present on rows without exc
REQ_ROW_AGENT_V21 = ('t_agent', 'sampling_sent')            # non-empty on every agent-outcome row
REQ_ADAPTER_V21 = ('adapter_sha256', 'adapter_name')        # non-empty on adapter-arm rows
REQ_STARTED_V21 = ('job_id', 'block', 'arm', 'task', 'seed', 'repeat_id', 'attempt', 'worker', 't_start_utc',
                   'docker_image_id', 'harness_sha256', 'vllm_version', 'served_model_name', 'bundle_sha256',
                   'a0_drop_sha256')                       # non-empty on every started record
REQ_STARTED_ADAPTER_V21 = ('adapter_sha256',)
REQ_MANIFEST_V21 = ('block', 'runs')
REQ_MANIFEST_RUN_V21 = ('run_tag', 'jobs_sha256', 'deadline_s')
PROV_ARM_FIELDS_V21 = ('request_model', 'adapter_name')     # v2.1 addition to A3: one value per arm (adapter_name: adapter arms only)


def required_fields_v21(planned_rows, started_records, adapter_arms=ADAPTER_ARMS, classify=None):
    """Every RUN_RECORD_SPEC.md required field on the rows of the planned slots (all attempts) and the started records
    of the block's plan. Returns dict(ok, missing=[dict(field, scope, arm, rows, example)], malformed=[...]).
    'Missing' = absent, None or '' for non-empty fields; absent for key-only fields. `classify`: the row classifier (default
    classify_row_v21; rules v2.2 pass classify_row_v22)."""
    classify = classify or classify_row_v21
    missing, examples, malformed = collections.Counter(), {}, []

    def miss(f, scope, arm, ref):
        missing[(f, scope, arm)] += 1
        examples.setdefault((f, scope, arm), ref)

    for arm, rows in planned_rows.items():
        for r in rows:
            row = r['row']
            if r.get('synthetic'):
                continue                                    # a started job without a row: checked as a started record
            ref = _ref(r)
            for f in REQ_ROW_ALL_V21:
                if _empty(row.get(f)):
                    miss(f, 'result row', arm, ref)
            for f in REQ_ROW_KEYS_V21:
                if f not in row:
                    miss(f, 'result row (key)', arm, ref)
            if not (row.get('exc') or row.get('tb')):
                for f in REQ_ROW_NOEXC_KEYS_V21:
                    if f not in row:
                        miss(f, 'result row without exc (key)', arm, ref)
            if classify(row)['klass'] == 'agent':
                for f in REQ_ROW_AGENT_V21:
                    v = row.get(f)
                    if _empty(v) or (f == 't_agent' and (isinstance(v, bool) or not isinstance(v, (int, float)))):
                        miss(f, 'agent-outcome row', arm, ref)
            if row.get('arm') in adapter_arms:
                for f in REQ_ADAPTER_V21:
                    if _empty(row.get(f)):
                        miss(f, 'adapter-arm row', arm, ref)
            jid = row.get('job_id')
            if not _empty(jid) and not _empty(row.get('block')):
                want = job_id_v2(row.get('block'), row.get('arm'), row.get('task'), r.get('orig_repeat', r['repeat']),
                                 r['attempt'])
                if r['attempt'] is not None and jid != want:
                    malformed.append(dict(problem='job_id does not match block/arm/task/repeat/attempt', ref=ref,
                                          job_id=jid, expected=want))
    for s in started_records:
        rec, arm = s['rec'], s['rec'].get('arm')
        for f in REQ_STARTED_V21:
            if _empty(rec.get(f)):
                miss(f, 'started record', arm, s['ref'])
        if arm in adapter_arms:
            for f in REQ_STARTED_ADAPTER_V21:
                if _empty(rec.get(f)):
                    miss(f, 'started record (adapter arm)', arm, s['ref'])
    miss_list = [dict(field=f, scope=sc, arm=a, rows=n, example=examples[(f, sc, a)])
                 for (f, sc, a), n in sorted(missing.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], str(kv[0][2])))]
    return dict(ok=not miss_list and not malformed, missing=miss_list, malformed=malformed[:100],
                n_malformed=len(malformed))


# ----------------------------------------------------------------------------------------------------------------
# v2.1 loading: rows, started records matched by job_id (A8), batches (run directories and their jobs.json)
# ----------------------------------------------------------------------------------------------------------------
def _jobs_json_path(d, od):
    cands = [os.path.join(d, 'jobs.json')]
    if os.path.basename(os.path.normpath(od)) == 'out':
        cands.append(os.path.join(os.path.dirname(os.path.normpath(od)), 'jobs.json'))
    for p in cands:
        if os.path.isfile(p):
            return p
    return None


def load_runs_v21(dirs, arm_from_repeat=False, arm_tag_from_dir=False, dedupe_started=False):
    """load_runs() plus the v2 fields, each row's batch (its run directory's run tag), the started records matched to
    rows by job_id (A8; a record without job_id falls back to arm, task, repeat, attempt and is listed), one synthetic
    infrastructure-error row (I09) per started job without a row, and each batch's job list (<run dir>/jobs.json, which
    run_berth.py writes). The same out dir given twice is loaded once. dedupe_started (rules v2.2, A9): a started record
    that repeats a job_id already seen in its run directory is listed in flags['duplicate_started_job_ids'] (a record
    problem) and makes no synthetic row; rules v2.1 (default False) keep the earlier behaviour.
    Returns dict(runs, problems, notes, batches, started, flags)."""
    uniq, seen, notes = [], set(), []
    for d in dirs:
        od = os.path.realpath(find_out_dir(d))
        if od in seen:
            notes.append(f'run directory {d} resolves to an out dir already given; loaded once')
            continue
        seen.add(od)
        uniq.append(d)
    runs, problems = load_runs(uniq, arm_from_repeat=arm_from_repeat, arm_tag_from_dir=arm_tag_from_dir)
    for r in runs:
        _v2_fields(r)
    batches, started_all = {}, []
    flags = dict(no_started_file=[], started_without_job_id=[], rows_without_started=[], duplicate_run_tags=[])
    if dedupe_started:
        flags['duplicate_started_job_ids'] = []
    key_of_source = {}
    for di, d in enumerate(uniq):
        od = find_out_dir(d)
        label = _dir_label(d, od)
        key = label
        if key in batches:
            flags['duplicate_run_tags'].append(label)
            key = f'{label}#{di}'
        key_of_source[od] = key
        jp = _jobs_json_path(d, od)
        info = dict(run_tag=label, key=key, dir=os.path.abspath(d), out_dir=os.path.abspath(od), jobs_path=jp,
                    jobs_sha256=None, jobs=None, jobs_error=None if jp else 'no jobs.json', has_started=False,
                    n_started=0, n_rows=0)
        if jp:
            with open(jp, 'rb') as f:
                raw = f.read()
            info['jobs_sha256'] = hashlib.sha256(raw).hexdigest()
            try:
                jl = json.loads(raw.decode('utf-8'))
                if not isinstance(jl, list) or not all(isinstance(j, dict) for j in jl):
                    raise ValueError('not a list of job objects')
                info['jobs'] = jl
            except (ValueError, UnicodeDecodeError) as e:
                info['jobs_error'] = f'unreadable jobs.json: {e}'[:200]
        batches[key] = info
        dir_rows = [r for r in runs if r['source'] == od]
        info['n_rows'] = len(dir_rows)
        for r in dir_rows:
            r['batch'] = key
        sp = os.path.join(od, 'started.jsonl')
        if not os.path.isfile(sp):
            flags['no_started_file'].append(key)
            continue
        info['has_started'] = True
        by_jid = collections.defaultdict(list)
        by_key = collections.defaultdict(list)
        for r in dir_rows:
            if r['job_id']:
                by_jid[r['job_id']].append(r)
            by_key[_slot_key_of(r['row'])].append(r)
        matched, seen_jids = set(), set()
        with open(sp, encoding='utf-8') as f:
            for li, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    problems.append(dict(source=sp, line=li + 1, problem=f'unparsable started record: {e}'))
                    continue
                if not isinstance(rec, dict):
                    problems.append(dict(source=sp, line=li + 1, problem='started record is not an object'))
                    continue
                info['n_started'] += 1
                ref = f"{label}:started {li + 1}"
                srec = dict(rec=rec, batch=key, ref=ref, matched=False)
                started_all.append(srec)
                jid = rec.get('job_id')
                if dedupe_started and jid:
                    if jid in seen_jids:
                        flags['duplicate_started_job_ids'].append(ref)
                        continue
                    seen_jids.add(jid)
                cand = by_jid.get(jid, []) if jid else by_key.get(_slot_key_of(rec), [])
                if not jid:
                    flags['started_without_job_id'].append(ref)
                hit = next((r for r in cand if id(r) not in matched), None)
                if hit is None and jid:          # a row without job_id (listed as a missing field): match by slot key
                    hit = next((r for r in by_key.get(_slot_key_of(rec), []) if not r['job_id'] and id(r) not in matched),
                               None)
                if hit is not None:
                    matched.add(id(hit))
                    srec['matched'] = True
                    continue
                arm, task, rep = rec.get('arm'), rec.get('task'), _as_int(rec.get('repeat_id', rec.get('seed')))
                if not arm or not task or rep is None:
                    problems.append(dict(source=sp, line=li + 1, problem='started record without arm, task or repeat'))
                    continue
                row = dict(rec)
                row.update(_synthetic='started_without_result', resolved=False, seed=rec.get('seed', rep), repeat_id=rep)
                a, rr = arm, rep
                if arm_from_repeat:
                    a, rr = f'{arm}#r{rep}', 1
                if arm_tag_from_dir:
                    a = f'{a}@{label}'
                trace = os.path.join(od, arm, f"s{row['seed']}", 'traces', 'trace_' + task.replace('/', '__') + '.json')
                syn = _v2_fields(dict(arm=a, orig_arm=arm, task=task, repeat=rr, orig_repeat=rep, row=row, source=od,
                                      label=label, order=(di, 10 ** 9 + li), trace_path=trace))
                syn['batch'] = key
                runs.append(syn)
        for r in dir_rows:
            if id(r) not in matched:
                flags['rows_without_started'].append(_ref(r))
    return dict(runs=runs, problems=problems, notes=notes, batches=batches, started=started_all, flags=flags)


def load_manifest_v21(path):
    """A8 block manifest (RUN_RECORD_SPEC.md section 3): JSON {"block": ..., "runs": [{"run_tag", "jobs_sha256",
    "deadline_s", ...}, ...]}. Returns dict(path, sha256, data or error)."""
    with open(path, 'rb') as f:
        raw = f.read()
    out = dict(path=os.path.abspath(path), sha256=hashlib.sha256(raw).hexdigest())
    try:
        out['data'] = json.loads(raw.decode('utf-8'))
    except (ValueError, UnicodeDecodeError) as e:
        out['error'] = f'unreadable manifest: {e}'[:200]
    return out


def check_manifest_v21(manifest, batches, block):
    """A8: the manifest lists every run tag of the block. 'clearable' problems are analysis-input problems (no
    manifest, a listed tag whose run directory was not given, a run directory the manifest does not list, a manifest of
    another block, two directories with one run tag): fix the inputs; they never turn into a FAIL. 'data' problems are
    record problems (a malformed manifest entry, a batch without a readable jobs.json or whose sha256 differs from the
    manifest). Notes: deadlines below A8's minimum or below the batch's expected duration."""
    clear, data, notes, runs_out = [], [], [], []
    if manifest is None:
        return dict(ok=False, clearable=['no block manifest given (A8)'], data=[], notes=[], runs=[], tags=[])
    if manifest.get('error'):
        return dict(ok=False, clearable=[f"block manifest {manifest['path']}: {manifest['error']}"], data=[], notes=[],
                    runs=[], tags=[])
    m = manifest.get('data')
    if not isinstance(m, dict):
        return dict(ok=False, clearable=['block manifest is not a JSON object'], data=[], notes=[], runs=[], tags=[])
    for f in REQ_MANIFEST_V21:
        if _empty(m.get(f)):
            data.append(f'manifest without required field {f!r}')
    if not _empty(m.get('block')) and m.get('block') != block:
        clear.append(f"manifest is for block {m.get('block')!r}, not {block!r}")
    entries = m.get('runs') if isinstance(m.get('runs'), list) else []
    tags = []
    by_tag = {}
    for i, e in enumerate(entries):
        if not isinstance(e, dict) or _empty(e.get('run_tag')):
            data.append(f'manifest run entry {i} without run_tag')
            continue
        for f in REQ_MANIFEST_RUN_V21:
            if f not in ('run_tag',) and (f not in e or e.get(f) is None or e.get(f) == ''):
                data.append(f"manifest run {e['run_tag']}: required field {f!r} missing")
        if e['run_tag'] in by_tag:
            data.append(f"manifest lists run tag {e['run_tag']} twice")
        by_tag[e['run_tag']] = e
        tags.append(e['run_tag'])
    for dup in sorted({b['run_tag'] for b in batches.values() if b['key'] != b['run_tag']}):
        clear.append(f'two run directories carry the run tag {dup}')
    given = {b['run_tag'] for b in batches.values()}
    for t in tags:
        if t not in given:
            clear.append(f'run tag {t} is in the block manifest but its run directory was not given (--runs)')
    for g in sorted(given):
        if g not in by_tag:
            clear.append(f'run directory {g} is not in the block manifest')
    for i, t in enumerate(tags):
        e = by_tag[t]
        b = next((x for x in batches.values() if x['run_tag'] == t and x['key'] == t), None)
        info = dict(run_tag=t, deadline_s=e.get('deadline_s'), jobs_sha256_manifest=e.get('jobs_sha256'), given=b is not None)
        if b is not None:
            info.update(jobs_sha256=b['jobs_sha256'], n_jobs=len(b['jobs']) if b['jobs'] is not None else None)
            if b['jobs'] is None:
                data.append(f"run {t}: {b['jobs_error']}")
            elif e.get('jobs_sha256') not in (None, '') and e.get('jobs_sha256') != b['jobs_sha256']:
                data.append(f'run {t}: sha256 of jobs.json differs from the block manifest')
            dl = e.get('deadline_s')
            if isinstance(dl, (int, float)) and not isinstance(dl, bool) and b['jobs'] is not None:
                expected = len(b['jobs']) / HOST_RUNS_PER_HOUR * 3600
                if dl < expected:
                    notes.append(f'run {t}: deadline {dl} s is below the expected duration of its {len(b["jobs"])} jobs '
                                 f'({expected:.0f} s at {HOST_RUNS_PER_HOUR} runs/h) (A8)')
                if block == 'D' and i == 0 and dl < D_BLOCK_MIN_DEADLINE_S:
                    notes.append(f'run {t}: the D block deadline {dl} s is below 12 h (A8)')
        runs_out.append(info)
    return dict(ok=not clear and not data, clearable=clear, data=data, notes=notes, runs=runs_out, tags=tags)


# ----------------------------------------------------------------------------------------------------------------
# v2.1 slot assembly, A0 checks, (4b) stratified, d_pen
# ----------------------------------------------------------------------------------------------------------------
def interleaved_slots_v21(tasks, arms, repeats):
    """A8 (amends A3): block job order with counterbalanced arm order. For each task (panel order) and repeat r, the
    arms in the order of sha256('g4s-armorder:{task}:{r'}:{arm}') with r' the odd repeat of the pair (r' = r for r = 1,
    3, ...); for the even repeat (2, 4, ...) the reverse of that order. With two arms and an even repeat count each arm
    runs first in exactly half of the pairs of every task. Repeats 1 and 3 keep the A3 (v2) order."""
    out = []
    for t in tasks:
        for r in range(1, repeats + 1):
            r_odd = r if r % 2 == 1 else r - 1
            order = sorted(arms, key=lambda a: arm_order_key(t, r_odd, a))
            if r % 2 == 0:
                order = order[::-1]
            out += [(a, t, r) for a in order]
    return out


def check_drops_all_panels_v21(dropped, panels):
    """A8: A0 is checked on every panel, not only the analysed one: STOP if any loses more than 10% (apply_drops)."""
    info = []
    for p in panels:
        _, a0 = apply_drops(p, dropped)
        info.append(dict(panel=p.get('panel'), n_panel=a0['n_panel'], n_dropped=a0['n_dropped']))
    return info


def a0_row_stops_v21(results, block, dropped, drop_sha256):
    """A8: STOP if any row or started record of the analysed block (rows without a block label count as the block's)
    belongs to a dropped task, or carries an a0_drop_sha256 that differs from the drop file's sha256."""
    dropped = set(dropped or ())
    in_block = [r for r in results if r['row'].get('block') in (block, None)]
    hit = [r for r in in_block if r['task'] in dropped]
    if hit:
        raise AnalysisStop(f'A0/A8: {len(hit)} row(s) or started record(s) of block {block} belong to dropped tasks: the '
                           'drop list is not the one frozen before the block, or a dropped task was run',
                           details=[dict(arm=r['arm'], task=r['task'], repeat=r['repeat'], attempt=r['attempt'],
                                         ref=_ref(r)) for r in hit])
    if drop_sha256 is not None:
        bad = [r for r in in_block if not _empty(r['row'].get('a0_drop_sha256'))
               and r['row'].get('a0_drop_sha256') != drop_sha256]
        if bad:
            raise AnalysisStop(f'A0/A8: {len(bad)} row(s) or started record(s) of block {block} carry a drop-file sha256 '
                               f'that differs from the drop file given ({drop_sha256[:16]}...)',
                               details=[dict(arm=r['arm'], task=r['task'], repeat=r['repeat'], attempt=r['attempt'],
                                             a0_drop_sha256=r['row'].get('a0_drop_sha256'), ref=_ref(r)) for r in bad])


def assemble_v21(runs, tasks, arms, repeats, block=None, max_attempts=MAX_ATTEMPTS_V21, classify=None):
    """A1/A2/A8 slot assembly. A planned slot is (arm, task, repeat); its rows are attempts. Infrastructure-error rows
    (ERROR_TABLE_V21) are missing; the slot's used row is its one row that is not an infrastructure error, at attempt
    <= 5. Stops (AnalysisStop) on two non-infrastructure rows in one slot, two rows with one attempt index, or an attempt
    index that is not a positive integer. Attempt gaps are returned (an INCONCLUSIVE reason in v2.1). Rows outside the
    plan are ignored and listed; the order of the run directories never decides a row. `classify`: the row classifier
    (default classify_row_v21; rules v2.2 pass classify_row_v22)."""
    classify = classify or classify_row_v21
    task_set, arm_set = set(tasks), set(arms)
    by_slot = collections.defaultdict(list)
    ignored = []
    for r in runs:
        rb = r['row'].get('block')
        reason = None
        if r['arm'] not in arm_set:
            reason = 'arm not analysed'
        elif r['task'] not in task_set:
            reason = 'task not in panel'
        elif not 1 <= r['repeat'] <= repeats:
            reason = f'repeat id outside 1..{repeats}'
        elif block is not None and rb is not None and rb != block:
            reason = f'row from another block ({rb})'
        if reason:
            ignored.append(dict(reason=reason, arm=r['arm'], task=r['task'], repeat=r['repeat'], attempt=r['attempt'],
                                ref=_ref(r)))
            continue
        by_slot[(r['arm'], r['task'], r['repeat'])].append(r)
    fatal, chosen, lacking, deviations, gaps, slots, mismatch = [], {}, [], [], [], {}, []
    infra = {a: [] for a in arms}
    attempts_hist = collections.Counter()
    for a in arms:
        for t in tasks:
            for k in range(1, repeats + 1):
                key = (a, t, k)
                lst = sorted(by_slot.get(key, []), key=lambda x: (x['attempt'] or 0, x['source'], x['order'][1]))
                slots[key] = lst
                if any(x['attempt'] is None for x in lst):
                    fatal.append(dict(problem='attempt index is not a positive integer', slot=key,
                                      rows=[(_ref(x), x['attempt_raw']) for x in lst]))
                    continue
                cnt = collections.Counter(x['attempt'] for x in lst)
                if any(c > 1 for c in cnt.values()):
                    fatal.append(dict(problem='two rows with the same attempt index (A2/A8)', slot=key,
                                      rows=[(_ref(x), x['attempt'], x['job_id']) for x in lst]))
                    continue
                cl = [(x, classify(x['row'])) for x in lst]
                for x, c in cl:
                    rec = x['row'].get('error_class_id')
                    if rec is not None and rec != c['id']:
                        mismatch.append(dict(slot=key, attempt=x['attempt'], recorded=rec, computed=c['id'], ref=_ref(x)))
                good = [x for x, c in cl if c['klass'] == 'agent']
                if len(good) > 1:
                    fatal.append(dict(problem='more than one row that is not an infrastructure error (A2)', slot=key,
                                      rows=[(_ref(x), x['attempt'], x['job_id']) for x in good]))
                    continue
                for x, c in cl:
                    if c['klass'] == 'infra':
                        infra[a].append(dict(slot=key, attempt=x['attempt'], id=c['id'], category=c['category'],
                                             detail=c['detail'][:100], ref=_ref(x), job_id=x['job_id'],
                                             batch=x.get('batch')))
                attempts_hist[len(lst)] += 1
                atts = sorted(cnt)
                if atts and atts != list(range(1, atts[-1] + 1)):
                    gaps.append(dict(slot=key, attempts=atts, rows=[(_ref(x), x['attempt']) for x in lst]))
                over = [x for x in lst if x['attempt'] > max_attempts]
                if over:
                    deviations.append(dict(problem=f'attempt index above {max_attempts}', slot=key,
                                           rows=[(_ref(x), x['attempt']) for x in over]))
                if good and good[0]['attempt'] <= max_attempts:
                    chosen[key] = good[0]
                    late = [x for x in lst if x['attempt'] > good[0]['attempt']]
                    if late:
                        deviations.append(dict(problem='re-run of a slot that already had a valid row', slot=key,
                                               rows=[(_ref(x), x['attempt']) for x in late]))
                else:
                    lacking.append(dict(arm=a, task=t, repeat=k, attempts=[x['attempt'] for x in lst],
                                        max_attempt=max((x['attempt'] for x in lst), default=0),
                                        classes=[c['id'] if c['klass'] == 'infra' else 'agent' for _, c in cl],
                                        reason=('no row' if not lst else (f'valid row only above attempt {max_attempts}'
                                                                          if good else 'only infrastructure-error rows'))))
    if fatal:
        raise AnalysisStop(f'{len(fatal)} planned slot(s) with conflicting rows: the analysis stops without a decision '
                           '(A2: the lead resolves them from job ids and logs the deviation)', details=fatal)
    cells = {(a, t): [chosen[(a, t, k)] for k in range(1, repeats + 1) if (a, t, k) in chosen] for a in arms for t in tasks}
    n_rows = sum(len(v) for v in slots.values())
    return dict(tasks=list(tasks), arms=list(arms), repeats=repeats, chosen=chosen, cells=cells, slots=slots,
                lacking=lacking, deviations=deviations, gaps=gaps, infra=infra, ignored=ignored, class_mismatches=mismatch,
                n_planned=len(arms) * len(tasks) * repeats, n_rows_in_plan=n_rows,
                attempts_hist={str(k): v for k, v in sorted(attempts_hist.items())},
                rows_without_attempt=sum(1 for v in slots.values() for x in v if not x['attempt_recorded']),
                rows_without_job_id=sum(1 for v in slots.values() for x in v if not x['job_id']))


def _binom_weights(n, c, m):
    """Integer weights w[j] = C(n, j) c^j (m - c)^(n - j), j = 0..n: Binomial(n, c/m) has P(X = j) = w[j] / m^n."""
    return [math.comb(n, j) * c ** j * (m - c) ** (n - j) for j in range(n + 1)]


def stratified_upper_tail(strata, t_obs):
    """Exact P(sum_b X_b >= t_obs) for independent X_b ~ Binomial(n_b, c_b / m_b); strata = [(n_b, c_b, m_b)]. A
    Fraction (integer convolution, no rounding)."""
    W, den = [1], 1
    for n, c, m in strata:
        if n == 0:
            continue
        w = _binom_weights(n, c, m)
        new = [0] * (len(W) + n)
        for i, a in enumerate(W):
            if a:
                for j, b in enumerate(w):
                    if b:
                        new[i + j] += a * b
        W, den = new, den * m ** n
    t_obs = int(t_obs)
    if t_obs <= 0:
        return Fraction(1)
    if t_obs >= len(W):
        return Fraction(0)
    return Fraction(sum(W[t_obs:]), den)


def harness_4b_stratified(strata, alpha=ALPHA_4B):
    """A8 (4b): one-sided exact test of the candidate's infrastructure-error rows (all attempts; started jobs without a
    row included), stratified by job batch. Under the null, in batch b the candidate's count is Binomial(n_b, s_b), n_b
    = the batch's infrastructure rows of both arms, s_b = the candidate's share of the batch's jobs. p = P(total >=
    observed); fails at p <= 0.05; no errors in either arm passes.
    strata: list of dict(batch, jobs_baseline, jobs_candidate, k_baseline, k_candidate)."""
    rows, tri, bad = [], [], []
    for s in strata:
        n = s['k_baseline'] + s['k_candidate']
        m = s['jobs_baseline'] + s['jobs_candidate']
        share = Fraction(s['jobs_candidate'], m) if m else None
        rows.append(dict(s, n=n, share_candidate=_fmt_frac(share) if share is not None else None))
        if n and (m == 0 or (s['k_candidate'] and s['jobs_candidate'] == 0) or (s['k_baseline'] and s['jobs_baseline'] == 0)):
            bad.append(s['batch'])
        if n:
            tri.append((n, s['jobs_candidate'], m))
    k_a = sum(s['k_baseline'] for s in strata)
    k_b = sum(s['k_candidate'] for s in strata)
    out = dict(k_baseline=k_a, k_candidate=k_b, n=k_a + k_b, alpha=float(alpha), strata=rows, inconsistent_batches=bad,
               method='exact, stratified by job batch (A8)')
    if bad:
        out.update(p=None, p_exact=None, fails=None, note='a batch has infrastructure rows of an arm it has no jobs of')
        return out
    if k_a + k_b == 0:
        out.update(p=None, p_exact=None, fails=False, note='no infrastructure-error rows in either arm: (4b) passes')
        return out
    p = stratified_upper_tail(tri, k_b)
    out.update(p=float(p), p_exact=_fmt_frac(p) if p.denominator < 10 ** 30 else None, fails=bool(p <= alpha))
    return out


def _batch_job_counts(batch_info, arms, tasks, repeats, block):
    """Jobs of the analysed arms and plan in one batch's job list: Counter(arm) and the set of job ids."""
    cnt, ids = collections.Counter(), set()
    task_set = set(tasks)
    for j in batch_info.get('jobs') or []:
        rep = _as_int(j.get('repeat_id', j.get('seed')))
        if j.get('arm') in arms and j.get('task') in task_set and rep is not None and 1 <= rep <= repeats \
                and j.get('block', block) == block:
            cnt[j['arm']] += 1
            if j.get('job_id'):
                ids.add(j['job_id'])
    return cnt, ids


def batch_checks_v21(asm, batches, arms, tasks, repeats, block):
    """Every in-plan row (all attempts, started jobs without a row included) must come from a batch with a readable job
    list that contains its job id. Returns (strata per batch for (4b) with job counts, problems)."""
    problems, used_batches = [], collections.OrderedDict()
    for key in sorted({x.get('batch') for lst in asm['slots'].values() for x in lst}, key=lambda k: str(k)):
        b = batches.get(key)
        if b is None:
            problems.append(f'rows from an unknown batch {key}')
            continue
        if b['jobs'] is None:
            problems.append(f"batch {b['run_tag']}: {b['jobs_error']}")
            continue
        used_batches[key] = _batch_job_counts(b, arms, tasks, repeats, block)
    n_out = 0
    for lst in asm['slots'].values():
        for x in lst:
            bc = used_batches.get(x.get('batch'))
            if bc is not None and (not x['job_id'] or x['job_id'] not in bc[1]):
                n_out += 1
                if n_out <= 20:
                    problems.append(f"row {_ref(x)} (job_id {x['job_id']}) is not in its batch's job list")
    if n_out > 20:
        problems.append(f"... {n_out} rows in all are not in their batch's job list")
    return used_batches, problems


def d_pen_task_diffs(asm, tasks, arm_a, arm_b, pen_ids=D_PEN_IDS, classify=None):
    """A8 d_pen: per task, mean over the candidate's used rows with every slot that has a penalised infrastructure row
    (any attempt) scored unsolved, minus the baseline's mean (unchanged). Returns (diffs as Fractions, penalised slots)."""
    classify = classify or classify_row_v21
    diffs, pen = [], []
    for t in tasks:
        ra, rb = asm['cells'][(arm_a, t)], asm['cells'][(arm_b, t)]
        yb = []
        for r in rb:
            ids = [classify(x['row'])['id'] for x in asm['slots'][(arm_b, t, r['repeat'])]]
            hit = [i for i in ids if i in pen_ids]
            if hit:
                pen.append(dict(task=t, repeat=r['repeat'], ids=hit, used_row_solved=bool(solved(r['row']))))
                yb.append(0)
            else:
                yb.append(solved(r['row']))
        diffs.append(Fraction(sum(yb), len(rb)) - Fraction(sum(solved(r['row']) for r in ra), len(ra)))
    return diffs, pen


def provenance_check_v21(planned_rows, used_rows, arms, adapter_arms=ADAPTER_ARMS):
    """provenance_check_v2 (A3, unchanged) plus one value per arm of request_model and (adapter arms) adapter_name."""
    prov = provenance_check_v2(planned_rows, used_rows, arms, adapter_arms=adapter_arms)
    extra = []
    for f in PROV_ARM_FIELDS_V21:
        for a in arms:
            if f == 'adapter_name' and a not in adapter_arms:
                continue
            c = collections.Counter(_canon(r['row'].get(f)) for r in planned_rows.get(a, [])
                                    if not r.get('synthetic') and not _empty(r['row'].get(f)))
            prov['values'][f'{f}[{a}]'] = dict(c)
            if len(c) > 1:
                extra.append(dict(field=f, scope=f'arm {a}', values=dict(c)))
    prov['disagreements'] = prov['disagreements'] + extra
    prov['ok'] = prov['ok'] and not extra
    return prov


def _reason(kind, text, clearable=False):
    return dict(kind=kind, text=text, clearable=clearable)


def next_step_v21(reasons, lacking):
    """A8's pre-registered next step. Returns (decision, decision_basis, next_step)."""
    if not reasons:
        return None, 'criteria', None
    if any(r['clearable'] for r in reasons):
        return ('INCONCLUSIVE', 'analysis inputs incomplete',
                'fix the analysis inputs listed as clearable (run directories, block manifest, drop file) and run the '
                'analysis again; no runs are needed for them')
    open_slots = [x for x in lacking if x['max_attempt'] < MAX_ATTEMPTS_V21 and not x['reason'].startswith('valid row only')]
    if open_slots:
        if any(x['max_attempt'] < RERUN_MAX_ATTEMPT for x in open_slots):
            return ('INCONCLUSIVE', 'slots open for re-runs (A1)',
                    f"`analyze_e7.py reruns`: a re-run batch (attempts <= 3) for "
                    f"{sum(1 for x in open_slots if x['max_attempt'] < RERUN_MAX_ATTEMPT)} slot(s); run it under a new run "
                    'tag, add the tag to the block manifest, then run the gate again')
        return ('INCONCLUSIVE', 'slots open for the extra batch (A8)',
                f'`analyze_e7.py reruns`: the blind extra batch (attempt {min(x["max_attempt"] for x in open_slots) + 1}; '
                f'at most attempts 4 and 5) for {len(open_slots)} slot(s) of both arms; then run the gate again. A gate '
                'still INCONCLUSIVE after the extra batch is a FAIL')
    return ('FAIL', 'INCONCLUSIVE after the extra batch (A8)',
            'none: the gate is a FAIL (A8: still INCONCLUSIVE after the extra batch; bundle0 stays)')


def _blind_infra(asm, arms):
    out = {}
    for a in arms:
        lst = asm['infra'][a]
        out[a] = dict(rows=len(lst), by_id=dict(collections.Counter(x['id'] for x in lst)),
                      by_attempt=dict(collections.Counter(str(x['attempt']) for x in lst)),
                      by_batch=dict(collections.Counter(str(x['batch']) for x in lst)),
                      details=dict(collections.Counter(x['detail'][:80] for x in lst).most_common(20)))
    return out


def _collect_reasons_v21(asm, arms, tasks, repeats, block, load, manifest, adapter_arms=ADAPTER_ARMS,
                         drop_given=True, kind='gate', classify=None):
    """INCONCLUSIVE (gate) / PROVISIONAL (screen) reasons that need no outcome: slots, attempt gaps, manifest, batch job
    lists, required fields, started records, provenance, load problems. Returns (reasons, parts)."""
    reasons = []
    batches = load.get('batches', {}) if load else {}
    flags = load.get('flags', {}) if load else {}
    if not drop_given:
        reasons.append(_reason('drop_file', 'no A0 drop file given: the drop-file sha256 in the rows cannot be checked',
                               clearable=True))
    if asm['lacking']:
        reasons.append(_reason('slots', f"{len(asm['lacking'])} planned slot(s) without a row that is not an "
                                        'infrastructure error (A1/A8)'))
    if asm['gaps']:
        reasons.append(_reason('attempt_gaps', f"{len(asm['gaps'])} slot(s) with a gap in their attempt indices (A8)"))
    man = check_manifest_v21(manifest, batches, block)
    for t in man['clearable']:
        reasons.append(_reason('manifest', t, clearable=True))
    for t in man['data']:
        reasons.append(_reason('manifest_record', t))
    used_batches, bprob = batch_checks_v21(asm, batches, arms, tasks, repeats, block)
    for t in bprob:
        reasons.append(_reason('batch_jobs', t))
    if batches:
        no_jobs = [a for a in arms if not any(cnt.get(a) for cnt, _ in used_batches.values())]
        if no_jobs:
            reasons.append(_reason('arms', f'arm(s) {no_jobs} have no job of block {block} in the job lists of the run '
                                           'directories given: check --arm-b / --arms / --block', clearable=True))
    planned_rows = {a: [x for (aa, t, k), lst in asm['slots'].items() if aa == a for x in lst] for a in arms}
    used_rows = {a: [r for t in tasks for r in asm['cells'][(a, t)]] for a in arms}
    task_set = set(tasks)
    started_in_plan = [s for s in (load.get('started', []) if load else [])
                       if s['rec'].get('arm') in arms and s['rec'].get('task') in task_set
                       and _as_int(s['rec'].get('repeat_id', s['rec'].get('seed'))) in range(1, repeats + 1)
                       and s['rec'].get('block', block) == block]
    req = required_fields_v21(planned_rows, started_in_plan, adapter_arms=adapter_arms, classify=classify)
    if req['missing']:
        reasons.append(_reason('required_fields', f"{len(req['missing'])} required field group(s) missing "
                                                  '(RUN_RECORD_SPEC.md; A8)'))
    if req['n_malformed']:
        reasons.append(_reason('required_fields', f"{req['n_malformed']} row(s) whose job_id does not match their "
                                                  'block, arm, task, repeat and attempt'))
    no_started = list(flags.get('no_started_file', []))      # every run directory of the block needs started.jsonl
    if no_started:
        reasons.append(_reason('started_records', f'{len(no_started)} run directory(ies) without started.jsonl: '
                                                  f'{no_started[:5]}'))
    in_plan_refs = {_ref(x) for lst in asm['slots'].values() for x in lst}
    rws = [r for r in flags.get('rows_without_started', []) if r in in_plan_refs]
    if rws:
        reasons.append(_reason('started_records', f'{len(rws)} row(s) without a started record (matched by job_id; A8)'))
    prov = provenance_check_v21(planned_rows, used_rows, arms, adapter_arms=adapter_arms)
    if not prov['ok']:
        reasons.append(_reason('provenance', f"provenance: {len(prov['missing'])} missing field group(s), "
                                             f"{len(prov['disagreements'])} disagreement(s) (A3)"))
    probs = load.get('problems', []) if load else []
    if probs:
        reasons.append(_reason('load_problems', f'{len(probs)} result or started record(s) could not be read or placed; '
                                                'one may be an uncounted infrastructure error'))
    parts = dict(manifest=man, used_batches=used_batches, batch_problems=bprob, required_fields=req,
                 started=dict(no_started_file=no_started, rows_without_started=rws[:50], n_rows_without_started=len(rws),
                              started_without_job_id=flags.get('started_without_job_id', [])[:50],
                              n_started_without_job_id=len(flags.get('started_without_job_id', []))),
                 provenance=prov, planned_rows=planned_rows, used_rows=used_rows)
    return reasons, parts


def _blind_slots(asm):
    return dict(n_planned=asm['n_planned'], n_with_used_row=len(asm['chosen']),
                lacking=[dict(arm=x['arm'], task=x['task'], repeat=x['repeat'], attempts=x['attempts'],
                              classes=x['classes'], reason=x['reason']) for x in asm['lacking']],
                gaps=asm['gaps'], deviations=asm['deviations'], attempts_hist=asm['attempts_hist'],
                rows_in_plan=asm['n_rows_in_plan'], rows_without_attempt=asm['rows_without_attempt'],
                rows_without_job_id=asm['rows_without_job_id'], n_class_mismatches=len(asm['class_mismatches']))


def gate_v21(results, panel, task_repo, arm_a=BASELINE, arm_b=None, repeats=None, n_boot=N_BOOT, seed=BOOT_SEED,
             traces=None, with_secondary=True, exact=True, dropped=None, drop_sha256=None, drop_source=None, block=None,
             guard_adopted=None, load=None, manifest=None, all_panels=None):
    """Confirmation/dev gate under PROTOCOL_AMENDMENTS.md A0-A8. `results` = load_runs_v21()['runs'], `load` = that
    dict (batches, started records, flags, problems), `manifest` = load_manifest_v21(). `dropped`, `drop_sha256`: the
    A0 drop file (load_drop_file). `all_panels`: the frozen panels A0 is checked on (default: panel_S/C/D.json).
    Raises AnalysisStop (no decision) on A0/A8 and A2 violations. INCONCLUSIVE results are blind."""
    adopted = GUARD_H_ADOPTED if guard_adopted is None else bool(guard_adopted)
    repeats = repeats or DEFAULT_REPEATS.get(panel.get('panel'), 4)
    check_panel_repos(panel, task_repo)
    if all_panels is None:
        all_panels = [load_panel(os.path.join(HERE, f)) for f in FROZEN_PANEL_FILES]
    a0_panels = check_drops_all_panels_v21(dropped, list(all_panels) + [panel])
    tasks, a0 = apply_drops(panel, dropped)
    a0.update(drop_source or {}, drop_file_sha256=drop_sha256, all_panels=a0_panels)
    block = block or panel.get('panel')
    a0_row_stops_v21(results, block, dropped, drop_sha256)
    arms = [arm_a, arm_b]
    asm = assemble_v21(results, tasks, arms, repeats, block=block)
    reasons, parts = _collect_reasons_v21(asm, arms, tasks, repeats, block, load, manifest,
                                          drop_given=drop_sha256 is not None)
    frozen, fnotes = _frozen_notes(panel, repeats, n_boot, seed)
    notes = list(fnotes) + list(parts['manifest']['notes'])
    if arm_b not in ADAPTER_ARMS and panel.get('panel') == 'D':
        notes.append(f'arm {arm_b} on panel D: A7 labels the adapter arm bundle0_adapter_A or bundle0_adapter_B')
    if asm['deviations']:
        notes.append(f"{len(asm['deviations'])} protocol deviation(s) in attempts (listed under slots)")
    if asm['class_mismatches']:
        notes.append(f"{len(asm['class_mismatches'])} row(s) whose recorded error_class_id differs from the v2.1 table "
                     '(the table decides; check the runner)')
    tabs = collections.Counter(_canon(x['row'].get('error_table_sha256')) for lst in asm['slots'].values() for x in lst
                               if not x.get('synthetic') and not _empty(x['row'].get('error_table_sha256')))
    if tabs and set(tabs) != {_canon(ERROR_TABLE_SHA256_V21)}:
        notes.append(f'rows carry error_table_sha256 values other than the v2.1 table ({ERROR_TABLE_SHA256_V21[:16]}...): '
                     f'{dict(tabs)}; the analysis classifies every row itself')
    unknown = [x for a in arms for x in asm['infra'][a] if x['id'] == 'I12']
    if unknown:
        notes.append(f'{len(unknown)} row(s) with error text outside the frozen table were counted as infrastructure '
                     'errors (I12; penalised in d_pen); the lead should check them')
    common = dict(kind='gate', rules='v2.1', analysis=ANALYSIS_ID_V21, error_table_sha256=ERROR_TABLE_SHA256_V21,
                  panel=panel.get('panel'), panel_sha256=panel.get('sha256'), block=block, arm_a=arm_a, arm_b=arm_b,
                  repeats=repeats, frozen_settings=frozen, a0=a0, n_tasks_planned=len(panel['tasks']),
                  n_tasks_after_a0=len(tasks), d_pen_ids=list(D_PEN_IDS),
                  thresholds=dict(gain_min=_fmt_frac(GAIN_MIN), alpha=float(ALPHA), min_pos_repos=MIN_POS_REPOS,
                                  time_ratio_max=float(TIME_RATIO_MAX), alpha_4b=float(ALPHA_4B),
                                  guard_h_limit=float(GUARD_H_LIMIT), rerun_max_attempt=RERUN_MAX_ATTEMPT,
                                  max_attempts=MAX_ATTEMPTS_V21, n_boot=N_BOOT, seed=BOOT_SEED),
                  manifest=dict(path=(manifest or {}).get('path'), sha256=(manifest or {}).get('sha256'),
                                **{k: v for k, v in parts['manifest'].items() if k != 'ok'}),
                  required_fields=parts['required_fields'], started=parts['started'],
                  provenance_check=parts['provenance'], load_problems=(load or {}).get('problems', []),
                  load_notes=(load or {}).get('notes', []),
                  data=dict(ignored_rows=dict(collections.Counter(x['reason'] for x in asm['ignored'])),
                            ignored=asm['ignored'][:300]))
    if reasons:
        decision, basis, step = next_step_v21(reasons, asm['lacking'])
        return dict(common, decision=decision, decision_basis=basis, blind=True,
                    inconclusive_reasons=[r['text'] for r in reasons], reasons=reasons, next_step=step, notes=notes,
                    blind_note=('A8: the report of an INCONCLUSIVE or STOP gate is blind (slots, provenance, error counts '
                                'and load problems only) until the reasons are cleared'),
                    slots=_blind_slots(asm), harness=dict(by_arm=_blind_infra(asm, arms),
                                                          rows=[dict(arm=a, **x) for a in arms for x in asm['infra'][a]][:400]))
    # ------------------------------------------------------------ no reason left: the decision statistics
    diffs, reps = [], []
    for t in tasks:
        ra, rb = asm['cells'][(arm_a, t)], asm['cells'][(arm_b, t)]
        diffs.append(Fraction(sum(solved(r['row']) for r in rb), len(rb)) - Fraction(sum(solved(r['row']) for r in ra), len(ra)))
        reps.append(task_repo[t])
    K, L = _common_denominator(diffs)
    core = gate_core(K, L, reps, n_boot=n_boot, seed=seed, exact=exact)
    core.update(notes=list(core.get('notes', [])))
    guard = guard_h(K, L, reps)
    for row in guard['table']:
        row['loo'] = (core.get('repos') or {}).get(row['repo'], {}).get('loo')
    pdiffs, pen_slots = d_pen_task_diffs(asm, tasks, arm_a, arm_b)
    Kp, Lp = _common_denominator(pdiffs)
    core_pen = gate_core(Kp, Lp, reps, n_boot=n_boot, seed=seed, sensitivities=False, exact=exact)
    guard_pen = guard_h(Kp, Lp, reps)
    runs_a = [r for t in tasks for r in asm['cells'][(arm_a, t)]]
    runs_b = [r for t in tasks for r in asm['cells'][(arm_b, t)]]
    ratio, ta, tb = _time_ratio(runs_a, runs_b)
    strata = []
    for key, (cnt, _) in parts['used_batches'].items():
        strata.append(dict(batch=key, run_tag=(load or {}).get('batches', {}).get(key, {}).get('run_tag', key),
                           jobs_baseline=cnt.get(arm_a, 0), jobs_candidate=cnt.get(arm_b, 0),
                           k_baseline=sum(1 for x in asm['infra'][arm_a] if x['batch'] == key),
                           k_candidate=sum(1 for x in asm['infra'][arm_b] if x['batch'] == key)))
    h4b = harness_4b_stratified(strata)
    time_ok = ratio is not None and ratio <= TIME_RATIO_MAX
    crit = dict(core['criteria'], time=bool(time_ok), harness=h4b['fails'] is False)
    if adopted:
        crit['guard_h'] = not guard['fails']
    cp = core_pen['criteria']
    crit.update(gain_pen=bool(cp['gain']), p_pen=bool(cp['p']), pos_repos_pen=bool(cp['pos_repos']), loo_pen=bool(cp['loo']))
    if adopted:
        crit['guard_h_pen'] = not guard_pen['fails']
    decision = 'PASS' if all(bool(v) for v in crit.values()) else 'FAIL'
    notes = list(core.get('notes', [])) + notes
    out = dict(common, decision=decision, decision_basis='criteria', blind=False, inconclusive_reasons=[], reasons=[],
               next_step=None, notes=notes, criteria=crit,
               guard_h=dict(guard, adopted=adopted, role='criterion (5)' if adopted else 'diagnostic only (A4)'),
               n_tasks_used=len(tasks), gain=core,
               d_pen=dict(gain=core_pen, guard_h=guard_pen, penalised_slots=pen_slots, n_penalised=len(pen_slots),
                          n_penalised_solved=sum(1 for x in pen_slots if x['used_row_solved']),
                          note='candidate slots with a row of class ' + '/'.join(D_PEN_IDS) +
                               ' (any attempt) scored unsolved; baseline unchanged; criteria (1), (2), (3), (5) must '
                               'also hold (A8)'),
               time=dict(ratio=None if ratio is None else float(ratio), mean_a=float(sum(ta) / len(ta)) if ta else None,
                         mean_b=float(sum(tb) / len(tb)) if tb else None, n_a=len(ta), n_b=len(tb),
                         note='(4a) on the rows used in the task means only (A1)'),
               harness=dict(test_4b=h4b, by_arm=_blind_infra(asm, arms),
                            rows=[dict(arm=a, **x) for a in arms for x in asm['infra'][a]][:400],
                            silent_verification_suspects={a: _silent_verification_suspects(parts['used_rows'][a])
                                                          for a in arms}),
               slots=dict(_blind_slots(asm), class_mismatches=asm['class_mismatches']))
    if with_secondary:
        out['secondary'] = secondary(results, traces if traces is not None else TraceStore(),
                                     dict(panel=panel.get('panel'), tasks=tasks), task_repo, arm_a, arm_b, repeats,
                                     asm=asm, status=_status_v21_metric)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Screening and re-runs under rules v2.1
# ----------------------------------------------------------------------------------------------------------------
def _arm_used_summary_v21(asm, arm, store, classify=None):
    classify = classify or classify_row_v21
    runs = [r for t in asm['tasks'] for r in asm['cells'][(arm, t)]]
    cats = collections.Counter(classify(r['row'])['category'] for r in runs)
    times = [float(r['row']['t_agent']) for r in runs if isinstance(r['row'].get('t_agent'), (int, float))]
    tr = [store.get(r['trace_path']) for r in runs]
    ok_tr = [m for m in tr if m and not m.get('error')]
    att = sum(m['write_attempts'] for m in ok_tr)
    okw = sum(m['ok_writes'] for m in ok_tr)
    planned = len(asm['tasks']) * asm['repeats']
    n_rows = sum(len(asm['slots'][(arm, t, k)]) for t in asm['tasks'] for k in range(1, asm['repeats'] + 1))
    return dict(planned=planned, used=len(runs), rows_in_plan=n_rows,
                lacking=sum(1 for x in asm['lacking'] if x['arm'] == arm),
                infra_rows=len(asm['infra'][arm]), infra_by_id=dict(collections.Counter(x['id'] for x in asm['infra'][arm])),
                solves=sum(solved(r['row']) for r in runs), outcome_counts=dict(cats),
                mean_agent_time=(sum(times) / len(times)) if times else None, n_agent_time=len(times),
                edit_attempts=att, edit_ok=okw, edit_success=(Fraction(okw, att) if att else None),
                traces_found=len(ok_tr), traces_missing=sum(1 for m in tr if m is None),
                traces_unreadable=sum(1 for m in tr if m and m.get('error')),
                edit_channels={ch: [sum(m[f'{ch}_attempts'] for m in ok_tr), sum(m[f'{ch}_failed'] for m in ok_tr)]
                               for ch in ('edit_file', 'write_file', 'shell')})


def screening_v21(results, panel_S, task_repo=None, baseline=BASELINE, arms=None, repeats=None, traces=None,
                  dropped=None, drop_sha256=None, drop_source=None, block=None, load=None, manifest=None,
                  all_panels=None):
    """PROTOCOL.md screening under A0-A8: the six frozen arms by default (A8), the v2.1 table, A1 slots (attempts <= 5),
    A2 stops, A0/A8 stops. PROVISIONAL while a slot lacks a valid row, an arm has no rows, or a manifest, record or
    provenance problem exists. Gain, edit-call success (A6/A7) and the finalist rule are unchanged."""
    repeats = repeats or DEFAULT_REPEATS.get(panel_S.get('panel'), 2)
    if all_panels is None:
        all_panels = [load_panel(os.path.join(HERE, f)) for f in FROZEN_PANEL_FILES]
    a0_panels = check_drops_all_panels_v21(dropped, list(all_panels) + [panel_S])
    tasks, a0 = apply_drops(panel_S, dropped)
    a0.update(drop_source or {}, drop_file_sha256=drop_sha256, all_panels=a0_panels)
    block = block or panel_S.get('panel')
    a0_row_stops_v21(results, block, dropped, drop_sha256)
    arms = [a for a in (SCREEN_ARMS if arms is None else arms) if a != baseline]
    store = traces if traces is not None else TraceStore()
    all_arms = [baseline] + arms
    asm = assemble_v21(results, tasks, all_arms, repeats, block=block)
    summ = {a: _arm_used_summary_v21(asm, a, store) for a in all_arms}
    rows = []
    for a in arms:
        gain, used, excluded = Fraction(0), [], []
        per_repo = collections.defaultdict(Fraction)
        for t in tasks:
            ra, rb = asm['cells'][(baseline, t)], asm['cells'][(a, t)]
            if not ra or not rb:
                excluded.append(dict(task=t, n_baseline=len(ra), n_arm=len(rb)))
                continue
            diff = Fraction(sum(solved(r['row']) for r in rb), len(rb)) - Fraction(sum(solved(r['row']) for r in ra), len(ra))
            gain += diff
            used.append(t)
            if task_repo:
                per_repo[task_repo.get(t)] += diff
        s = summ[a]
        rows.append(dict(arm=a, gain=gain, gain_float=float(gain), gain_exact=_fmt_frac(gain), tasks_used=len(used),
                         tasks_excluded=excluded, mean_gain_per_task=float(gain / len(used)) if used else None,
                         repos_positive=sum(1 for v in per_repo.values() if v > 0),
                         repos_negative=sum(1 for v in per_repo.values() if v < 0),
                         per_repo_gain={k: float(v) for k, v in sorted(per_repo.items())},
                         edit_success=s['edit_success'], mean_agent_time=s['mean_agent_time']))
    order = {a: i for i, a in enumerate(SCREEN_ARMS)}

    def key(r):
        es, mt = r['edit_success'], r['mean_agent_time']
        return (-r['gain'], -(es if es is not None else Fraction(-1)), mt if mt is not None else float('inf'),
                order.get(r['arm'], len(order)), r['arm'])

    ranked = sorted(rows, key=key)
    finalist, tie_log = None, []
    if ranked and ranked[0]['gain'] > 0:
        finalist = ranked[0]['arm']
        top = [r for r in ranked if r['gain'] == ranked[0]['gain']]
        if len(top) > 1:
            tie_log.append(f"gain tie among {[r['arm'] for r in top]}; broken by edit-call success rate")
            es_top = [r for r in top if r['edit_success'] == top[0]['edit_success']]
            if len(es_top) > 1:
                tie_log.append(f"edit-success tie among {[r['arm'] for r in es_top]}; broken by mean agent time")
                mt_top = [r for r in es_top if r['mean_agent_time'] == es_top[0]['mean_agent_time']]
                if len(mt_top) > 1:
                    tie_log.append('UNRESOLVED by the protocol: fell back to PROTOCOL arm order (flag to owner)')
    for r in rows:
        r['edit_success_float'] = None if r['edit_success'] is None else float(r['edit_success'])
        r['edit_success'] = _fmt_frac(r['edit_success']) if r['edit_success'] is not None else None
        r['gain'] = r['gain_float']
    for a, s in summ.items():
        s['edit_success_float'] = None if s['edit_success'] is None else float(s['edit_success'])
        s['edit_success'] = _fmt_frac(s['edit_success']) if s['edit_success'] is not None else None
    reasons, parts = _collect_reasons_v21(asm, all_arms, tasks, repeats, block, load, manifest,
                                          drop_given=drop_sha256 is not None, kind='screen')
    no_rows = [a for a in all_arms if summ[a]['rows_in_plan'] == 0]
    if no_rows:
        reasons.insert(0, _reason('arm_without_rows', f'arm(s) without any row: {no_rows} (A8)'))
    provisional = [r['text'] for r in reasons]
    _, basis, step = next_step_v21(reasons, asm['lacking'])
    procedure_complete = bool(reasons) and basis == 'INCONCLUSIVE after the extra batch (A8)'
    decision = f'finalist {finalist}' if finalist else 'no finalist (largest gain <= 0): bundle0 stays'
    if provisional:
        decision = f"{decision} (PROVISIONAL: {'; '.join(provisional[:6])}{'; ...' if len(provisional) > 6 else ''})"
    frozen, notes = _frozen_notes(panel_S, repeats, N_BOOT, BOOT_SEED)
    notes = list(notes) + list(parts['manifest']['notes'])
    if panel_S.get('panel') == 'S' and (baseline != BASELINE or sorted(arms) != sorted(SCREEN_ARMS[1:])):
        frozen = False
        notes.append(f'NON-FROZEN arm set {[baseline] + arms} (A8: the six frozen arms {SCREEN_ARMS})')
    if procedure_complete:
        notes.append('every slot has used its five attempts and the screen is still PROVISIONAL: A8 sets no screening '
                     'rule for this case (lead)')
    return dict(kind='screening', rules='v2.1', analysis=ANALYSIS_ID_V21, error_table_sha256=ERROR_TABLE_SHA256_V21,
                panel=panel_S.get('panel'), panel_sha256=panel_S.get('sha256'), block=block, baseline=baseline, arms=arms,
                repeats=repeats, frozen_settings=frozen, notes=notes, n_tasks=len(tasks),
                n_tasks_planned=len(panel_S['tasks']), a0=a0, ranking=[r['arm'] for r in ranked], finalist=finalist,
                decision=decision, provisional=provisional, reasons=reasons,
                next_step=(step if reasons and not procedure_complete else None), procedure_complete=procedure_complete,
                tie_log=tie_log, arms_table=rows, run_summary=summ, provenance_check=parts['provenance'],
                manifest=dict(path=(manifest or {}).get('path'), sha256=(manifest or {}).get('sha256'),
                              **{k: v for k, v in parts['manifest'].items() if k != 'ok'}),
                required_fields=parts['required_fields'], started=parts['started'],
                slots=dict(_blind_slots(asm)), load_problems=(load or {}).get('problems', []),
                load_notes=(load or {}).get('notes', []),
                data=dict(ignored_rows=dict(collections.Counter(x['reason'] for x in asm['ignored'])),
                          ignored=asm['ignored'][:300]))


def rerun_jobs_v21(results, panel, arms, repeats=None, block=None, dropped=None, drop_sha256=None, load=None,
                   manifest=None, all_panels=None):
    """A1/A8 next batch, blind (slots only). Every planned slot without a valid row whose highest attempt is below 3
    gets its next attempt (a re-run batch). When no such slot is left, every slot still without a valid row gets its
    next attempt up to 5 (the extra batch: attempts 4 and 5, both arms). Jobs in block job order
    (interleaved_slots_v21), with the same arm, task, repeat, the next attempt, a new job_id and the drop-file sha256.
    Refuses (AnalysisStop) on A0/A8/A2 stops and while the block manifest has an input problem."""
    repeats = repeats or DEFAULT_REPEATS.get(panel.get('panel'), 4)
    if all_panels is None:
        all_panels = [load_panel(os.path.join(HERE, f)) for f in FROZEN_PANEL_FILES]
    check_drops_all_panels_v21(dropped, list(all_panels) + [panel])
    tasks, a0 = apply_drops(panel, dropped)
    block = block or panel.get('panel')
    a0_row_stops_v21(results, block, dropped, drop_sha256)
    asm = assemble_v21(results, tasks, list(arms), repeats, block=block)
    man = check_manifest_v21(manifest, (load or {}).get('batches', {}), block)
    if man['clearable']:
        raise AnalysisStop('reruns refused: the analysis inputs are incomplete (A8 manifest); a re-run list made from '
                           'them could repeat or skip attempts', details=man['clearable'])
    if drop_sha256 is None:
        raise AnalysisStop('reruns refused: no A0 drop file (its sha256 goes into every job)')
    batches = (load or {}).get('batches', {})
    used_batches, _ = batch_checks_v21(asm, batches, list(arms), tasks, repeats, block)
    no_jobs = [a for a in arms if not any(cnt.get(a) for cnt, _ in used_batches.values())]
    if batches and no_jobs:
        raise AnalysisStop(f'reruns refused: arm(s) {no_jobs} have no job of block {block} in the job lists of the run '
                           'directories given (check --arms and --block)', details=no_jobs)
    lack = {(x['arm'], x['task'], x['repeat']): x for x in asm['lacking']}
    open_ = {k: x for k, x in lack.items() if x['max_attempt'] < MAX_ATTEMPTS_V21
             and not x['reason'].startswith('valid row only')}
    rerun_phase = any(x['max_attempt'] < RERUN_MAX_ATTEMPT for x in open_.values())
    kind = 'rerun' if rerun_phase else 'extra'
    jobs, waiting, exhausted = [], [], []
    for a, t, k in interleaved_slots_v21(tasks, list(arms), repeats):
        x = lack.get((a, t, k))
        if x is None:
            continue
        if (a, t, k) not in open_:
            exhausted.append(dict(arm=a, task=t, repeat=k, attempts=x['attempts'], reason=x['reason']))
            continue
        nxt = x['max_attempt'] + 1
        if rerun_phase and nxt > RERUN_MAX_ATTEMPT:
            waiting.append(dict(arm=a, task=t, repeat=k, attempts=x['attempts']))
            continue
        jobs.append(dict(arm=a, task=t, seed=k, repeat_id=k, attempt=nxt, block=block,
                         job_id=job_id_v2(block, a, t, k, nxt), a0_drop_sha256=drop_sha256, batch_kind=kind))
    return dict(kind='reruns', rules='v2.1', block=block, arms=list(arms), repeats=repeats, a0=a0, batch_kind=kind,
                jobs=jobs, waiting_for_extra_batch=waiting, exhausted=exhausted, n_planned=asm['n_planned'],
                n_with_used_row=len(asm['chosen']), n_lacking=len(asm['lacking']), manifest_data_problems=man['data'],
                note=('blind: slots only (A8). A re-run batch gives attempts 2-3 (A1); the extra batch, made only when no '
                      'slot is below attempt 3, gives at most attempts 4 and 5 to every slot of both arms still without a '
                      'valid row (A8). Run it under a new run tag and add the tag to the block manifest.'))


def parity_ratio_v21(results_plain, results_lora, arm=BASELINE):
    """A8: mean agent time of `arm` on the server with adapter serving enabled over the plain server (agent-outcome
    rows only, v2.1 table). Above 1.15 -> investigate (no rule change)."""
    def times(rs):
        return [Fraction(str(r['row']['t_agent'])) for r in rs if r['arm'] == arm and not r.get('synthetic')
                and classify_row_v21(r['row'])['klass'] == 'agent' and isinstance(r['row'].get('t_agent'), (int, float))
                and not isinstance(r['row'].get('t_agent'), bool)]
    tp, tl = times(results_plain), times(results_lora)
    ratio = (sum(tl) / len(tl)) / (sum(tp) / len(tp)) if tp and tl and sum(tp) else None
    return dict(kind='parity', rules='v2.1', arm=arm, n_plain=len(tp), n_lora=len(tl),
                mean_plain=float(sum(tp) / len(tp)) if tp else None, mean_lora=float(sum(tl) / len(tl)) if tl else None,
                ratio=None if ratio is None else float(ratio), limit=float(PARITY_RATIO_LIMIT),
                investigate=None if ratio is None else bool(ratio > PARITY_RATIO_LIMIT),
                note='A8: above 1.15 means investigate before the D block; it changes no rule')


# ================================================================================================================
# Rules v2.2: PROTOCOL_AMENDMENTS.md A0-A9 (A9, written before any run of the experiment). This is
# the CLI default. Everything above this block (rules v1, v2, v2.1) keeps its behaviour: `--rules v2.1`, `--rules v2`
# and `--rules v1` reproduce their earlier outputs exactly (ANALYSIS_VALIDATION.md section 7). Python callers:
# gate_v22(), screening_v22(), rerun_jobs_v22(), records_v22(), parity_ratio_v22() on the output of
# load_runs_v21(..., dedupe_started=True).
# What v2.2 adds to v2.1 (A9):
#   * A0 job-file hash: every job entry of every batch's jobs.json (entries of the analysed block or without a block)
#     must carry the drop file's sha256, and every started record that carries an a0_drop_sha256 must carry that one:
#     STOP otherwise (a job entry without it is the signature of a template job file; the runner refuses such jobs,
#     job_refusal_v22). Rows keep the v2.1 checks (a different value STOPs; a missing one is a record problem).
#   * Record problems void the block: the record checks run first and blind (no outcome is computed); a block whose
#     analysis inputs are complete but whose records have a problem is VOID (not FAIL): its rows are archived unopened
#     and the block is re-run under a new block id <panel>_v<n> (make_jobs.py --new-block-id), whose manifest names the
#     voided block and the sha256 of its VOID report (`replaces`; the analysis checks it with --void-report). A FAIL by
#     INCONCLUSIVE (A8) is left only for slots still without a valid row after attempt 5. A started record repeating a
#     job_id of its run directory is a record problem (v2.1: it became a synthetic row and a STOP).
#   * Error table v2.2: I13 ratified as a d_pen class and anchored to the Docker/g4s message forms at the start of the
#     patch detail; the per-job container runner's row for a job that died without a results row
#     ('g4s-d<N> infra: ...') is I09 (penalised), as on the host runner.
#   * Blind reports show only recorded-vs-computed class mismatches that involve an infrastructure id, agent ids masked.
#   * Provenance: max_model_len and the adapter-serving settings (enable_lora, max_lora_rank, served_adapter_names) are
#     required in every run entry of the block manifest and must agree across the block; an adapter arm needs a server
#     with adapter serving enabled that serves its adapter name. Missing or inconsistent -> record problem -> VOID.
#   * Parity: the agent-time ratio of the adapter-serving server to the plain one is paired by task (common tasks, task means), with a warning when the
#     task sets differ.
#   * `records`: the blind record check alone (the smoke block's check, mandatory for each runner version).
# ================================================================================================================
ANALYSIS_ID_V22 = 'analyze_e7 v2.2 (2026-11-09; PROTOCOL_AMENDMENTS.md A0-A9)'
RUNNER_NO_ROW_PATTERN = r'^g4s-d[0-9]+ infra: '
AGENT_PATCH_INFRA_PATTERN_V22 = (r'(?s)^Failed to apply agent patch:\s*(?:docker:\s|g4s:\s|g4s run\.py:\s|\[Errno [0-9]+\]\s|'
                                 r'Error response from daemon|Cannot connect to the Docker daemon)')
BLOCK_ID_PATTERN = r'^(?P<panel>[A-Za-z0-9_]+?)_v(?P<n>[2-9]|[1-9][0-9]+)$'   # A9: block id of a re-run after a void
SERVER_FIELDS_V22 = ('max_model_len', 'enable_lora', 'max_lora_rank', 'served_adapter_names')
REQ_MANIFEST_RUN_V22 = REQ_MANIFEST_RUN_V21 + SERVER_FIELDS_V22
KAGGLE_DEFAULT_MAX_MODEL_LEN = 32768   # note only: adk_submission VllmConfig default (adk_submission/server.py:152)
HEX64 = re.compile(r'^[0-9a-f]{64}$')

_V21E = {e.id: e for e in ERROR_TABLE_V21}
_V22_REPLACED = {
    'I08': ErrorEntry('I08', 'infra', 'worker_exception', 'exc', r'(?s).',
                      _V21E['I08'].source + "; not the per-job container runner's 'g4s-d<N> infra: ...' rows (I09, A9)"),
    'I09': ErrorEntry('I09', 'infra', 'job_died', 'synthetic or exc', RUNNER_NO_ROW_PATTERN,
                      'a started.jsonl entry without a results row (lib/worker_berth.py:188-189 writes started.jsonl before '
                      'the job, 214-215 the row after it: worker killed, runner deadline, crash); A9: also the exc row that '
                      'the per-job container runner writes for a job that died without a results row '
                      "(<runner>/run_berth.py:290-294: 'g4s-d3 infra: job deadline', 'g4s-d3 infra: no results row' "
                      "or 'g4s-d3 infra: <frame error>'; the D4 runner's 'g4s-d4 infra: ...'), as on the host runner"),
    'I13': ErrorEntry('I13', 'infra', 'evaluation_infrastructure', 'regex', AGENT_PATCH_INFRA_PATTERN_V22,
                      'swegemma/harness/verification.py:463-474 (detail = stderr or stdout of the apply command) when the '
                      'detail STARTS with a Docker/g4s form that lib/g4s_docker.py returns as stderr: docker: (docker run '
                      'exit 125 after 3 tries, 320-328), g4s: (refused, 297-299; the g4s ValueError/RuntimeError/OSError '
                      'texts 131-514), g4s run.py: (docker/run.py:59), [Errno N] (popen_error 309-312, e.g. /usr/bin/docker '
                      'missing; exec_error 350-354), or the daemon\'s "Error response from daemon" / "Cannot connect to the '
                      'Docker daemon". A9: anchored at the start of the detail (patch-tool output that only contains these '
                      'strings stays A07) and ratified as a d_pen class'),
}
ERROR_TABLE_V22 = tuple(_V22_REPLACED.get(e.id, e) for e in ERROR_TABLE_V21)
ERROR_TABLE_SHA256_V22 = hashlib.sha256(repr(ERROR_TABLE_V22).encode()).hexdigest()
_ERR_BY_ID_V22 = {e.id: e for e in ERROR_TABLE_V22}
_REGEX_ENTRIES_V22 = tuple((e, re.compile(e.pattern)) for e in ERROR_TABLE_V22 if e.kind == 'regex')
_PREFIX_ENTRIES_V22 = tuple(e for e in ERROR_TABLE_V22 if e.kind == 'prefix')   # table order: I01 before I02
_RUNNER_NO_ROW = re.compile(RUNNER_NO_ROW_PATTERN)
_INFRA_IDS_V22 = frozenset(e.id for e in ERROR_TABLE_V22 if e.klass == 'infra')
_AGENT_IDS_V22 = frozenset(e.id for e in ERROR_TABLE_V22 if e.klass == 'agent')
ERROR_ID_MAP_V21_TO_V22 = {'I08': "I08 or I09 (exc 'g4s-d<N> infra: ...')", 'I13': 'I13 or A07 (Docker/g4s text not at the start)'}
# A9 ratifies I13; the set is the v2.1 one (I09 now also covers the per-job container runner's rows).
D_PEN_IDS_V22 = ('I01', 'I02', 'I03', 'I09', 'I12', 'I13', 'I14')
# A9 reason categories: input problems are cleared by running the analysis with the right inputs (INCONCLUSIVE, never
# a FAIL); record problems void the block (VOID); slot problems follow A1/A8 (re-runs, the extra batch, then FAIL).
REASON_CATEGORY_V22 = dict(drop_file='input', manifest='input', arms='input', void_chain='input',
                           slots='slots', arm_without_rows='slots',
                           attempt_gaps='record', manifest_record='record', batch_jobs='record', required_fields='record',
                           started_records='record', provenance='record', load_problems='record',
                           server_settings='record')
VOID_STEP_V22 = ('A9: the block is VOID (record problems). Archive its run directories unopened (no outcome is computed '
                 'or shown), log the deviation, fix the runner (the smoke block is mandatory for each runner version), '
                 'then re-run the block under a new block id: `make_jobs.py --drop-file <the same A0 list> --new-block-id '
                 '<panel>_v<n>`, a new block manifest whose `replaces` names this block and the sha256 of this report, '
                 'and pass this report to the analysis of the new block as --void-report')


def _cls22(eid, detail=''):
    e = _ERR_BY_ID_V22[eid]
    return dict(klass=e.klass, category=e.category, id=eid, detail=detail)


def classify_row_v22(row):
    """A1/A8/A9 class of one result row under ERROR_TABLE_V22: dict(klass 'agent' | 'infra', category, id, detail).
    Order: started job without a row (I09); exc/tb: the per-job container runner's 'g4s-d<N> infra: ' row (I09), I14,
    else I08; resolved (I10, I11, A00); empty error (A01); the regex entries (A18, I13 anchored); the prefix entries in
    table order; anything else I12."""
    if row.get('_synthetic') == 'started_without_result':
        return _cls22('I09', 'job in started.jsonl without a results row')
    exc, tb = row.get('exc'), row.get('tb')
    if exc or tb:
        text = str(exc) if exc else ''
        if _RUNNER_NO_ROW.match(text):
            return _cls22('I09', text[:160])
        if _EXC_I14.match(text):
            return _cls22('I14', text[:160])
        return _cls22('I08', 'worker exception: ' + str(exc or tb)[:160])
    res = row.get('resolved')
    if not isinstance(res, bool):
        return _cls22('I10', f'resolved = {res!r}'[:160])
    err = row.get('error')
    err = '' if err is None else str(err).strip()
    if res:
        return _cls22('I11', err[:160]) if err else _cls22('A00')
    if not err:
        return _cls22('A01')
    for e, rx in _REGEX_ENTRIES_V22:
        if rx.match(err):
            return _cls22(e.id, err[:160])
    for e in _PREFIX_ENTRIES_V22:
        if err.startswith(e.pattern):
            return _cls22(e.id, err[:160])
    return _cls22('I12', err[:160])


def _status_v22_metric(row):
    """row_status-like (category, detail) under the v2.2 table, for the secondary metrics (run_metrics)."""
    c = classify_row_v22(row)
    if c['klass'] == 'infra':
        return 'harness_error', c['detail']
    return _V21_METRIC_STATUS.get(c['category'], c['category']), c['detail']


def error_table_markdown_v22():
    L = ['| id | class | category | match | pattern | source |', '|---|---|---|---|---|---|']
    for e in ERROR_TABLE_V22:
        pat = e.pattern.replace('|', '\\|')
        L.append(f'| {e.id} | {e.klass} | {e.category} | {e.kind} | `{pat}` | {e.source} |')
    return '\n'.join(L) + '\n'


def job_refusal_v22(job):
    """A9: the runner refuses a job without a0_drop_sha256. Returns the reason to refuse (a string) or None. A runner
    that copies analyze_e7.py next to the worker can call this on every job entry before it starts the job."""
    if not isinstance(job, dict):
        return 'job entry is not an object'
    v = job.get('a0_drop_sha256')
    if not isinstance(v, str) or not HEX64.match(v):
        return (f'job {job.get("job_id") or (job.get("arm"), job.get("task"), job.get("seed"))} has no a0_drop_sha256 '
                f'(a 64-character hex sha256): run only the .A0.json job files of make_jobs.py --drop-file (A8/A9)')
    return None


def _mask_class_id(i):
    if isinstance(i, str) and i in _INFRA_IDS_V22:
        return i
    if isinstance(i, str) and i in _AGENT_IDS_V22:
        return 'agent'
    return 'other'


def blind_class_mismatches_v22(mismatches):
    """A9: a blind report shows only recorded-vs-computed mismatches that involve an infrastructure id, with every agent
    id shown as 'agent' (A00 and A01 never appear, so no count reveals a solve)."""
    c = collections.Counter()
    for m in mismatches:
        rec, comp = _mask_class_id(m.get('recorded')), _mask_class_id(m.get('computed'))
        if rec in _INFRA_IDS_V22 or comp in _INFRA_IDS_V22:
            c[f'{rec} -> {comp}'] += 1
    return dict(n=sum(c.values()), by_kind=dict(sorted(c.items())),
                note='A9: only mismatches that involve an infrastructure id; agent ids masked')


def _blind_slots_v22(asm):
    s = _blind_slots(asm)
    s.pop('n_class_mismatches', None)
    s['infra_class_mismatches'] = blind_class_mismatches_v22(asm['class_mismatches'])
    return s


def _pos_int(x):
    return isinstance(x, int) and not isinstance(x, bool) and x > 0


def check_server_settings_v22(manifest, arms, planned_rows, adapter_arms=ADAPTER_ARMS):
    """A9: max_model_len and the adapter-serving settings (enable_lora, max_lora_rank, served_adapter_names) are required in
    every run entry of the block manifest (the runner writes the values it started the server with). Record problems:
    a missing or malformed value, a value that differs between the runs of the block (A3: one server configuration per
    block), or an adapter arm analysed on a server without adapter serving or without its adapter name. Notes: max_model_len other
    than the adk_submission default 32768 (A18 counts context-length errors as agent outcomes 'as on Kaggle')."""
    probs, notes, values = [], [], {}
    m = (manifest or {}).get('data')
    if not isinstance(m, dict) or not isinstance(m.get('runs'), list):
        return dict(ok=True, checked=False, problems=[], notes=[], values={})   # reported by check_manifest_v21
    seen = collections.defaultdict(collections.Counter)
    served = set()
    for e in m['runs']:
        if not isinstance(e, dict) or _empty(e.get('run_tag')):
            continue                                                            # reported by check_manifest_v21
        tag, bad = e['run_tag'], []
        for f in SERVER_FIELDS_V22:
            if f not in e:
                bad.append(f'{f} missing')
        mml, el, rank, names = (e.get(f) for f in SERVER_FIELDS_V22)
        if 'max_model_len' in e and not _pos_int(mml):
            bad.append(f'max_model_len {mml!r} is not a positive integer')
        if 'enable_lora' in e and not isinstance(el, bool):
            bad.append(f'enable_lora {el!r} is not true or false')
        if 'max_lora_rank' in e:
            if el is True and not _pos_int(rank):
                bad.append(f'max_lora_rank {rank!r} is not a positive integer (LoRA enabled)')
            elif el is not True and rank is not None and not _pos_int(rank):
                bad.append(f'max_lora_rank {rank!r} is neither null nor a positive integer')
        if 'served_adapter_names' in e:
            if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
                bad.append(f'served_adapter_names {names!r} is not a list of names')
            elif el is True and not names:
                bad.append('LoRA enabled but no served adapter name')
            elif el is False and names:
                bad.append(f'served adapter names {names} on a server without LoRA')
            else:
                served |= set(names)
        for b in bad:
            probs.append(f'manifest run {tag}: {b} (A9 server settings)')
        for f in SERVER_FIELDS_V22:
            if f in e:
                seen[f][_canon(e.get(f))] += 1
    for f in SERVER_FIELDS_V22:
        values[f] = dict(seen.get(f, {}))
        if len(seen.get(f, {})) > 1:
            probs.append(f'server setting {f} differs between the runs of the block: {dict(seen[f])} (A3/A9)')
    for a in [x for x in arms if x in adapter_arms]:
        if seen.get('enable_lora') and set(seen['enable_lora']) != {_canon(True)}:
            probs.append(f'adapter arm {a}: LoRA is not enabled in every run of the block (A3/A9)')
        used = sorted({r['row'].get('adapter_name') for r in planned_rows.get(a, []) if not r.get('synthetic')
                       and isinstance(r['row'].get('adapter_name'), str) and r['row'].get('adapter_name')})
        unserved = [n for n in used if n not in served]
        if unserved and seen.get('served_adapter_names'):          # a missing field is reported above
            probs.append(f'adapter arm {a}: adapter name(s) {unserved} not among the served adapter names {sorted(served)} (A9)')
    mmls = [json.loads(k) for k in seen.get('max_model_len', {})]
    if mmls and all(_pos_int(x) for x in mmls) and any(x != KAGGLE_DEFAULT_MAX_MODEL_LEN for x in mmls):
        notes.append(f'max_model_len {sorted(mmls)} differs from the adk_submission default {KAGGLE_DEFAULT_MAX_MODEL_LEN} '
                     '(A18 counts context-length errors as agent outcomes "as on Kaggle"; check the scorer value)')
    return dict(ok=not probs, checked=True, problems=probs, notes=notes, values=values)


def a0_jobfile_stops_v22(load, block, drop_sha256):
    """A9: STOP when a job entry of any batch's jobs.json (entries of the analysed block or without a block label)
    carries no a0_drop_sha256 or another one than the drop file's, or when a started record of the block carries an
    a0_drop_sha256 other than the drop file's. (A started record without the value is a missing required field: a record
    problem, VOID.) Returns a summary of what was checked."""
    if drop_sha256 is None:
        return dict(checked=False, note='no drop file: not checked')
    bad_jobs, bad_started, n_jobs, n_started = collections.Counter(), [], 0, 0
    examples = []
    for b in (load or {}).get('batches', {}).values():
        for i, j in enumerate(b.get('jobs') or []):
            if j.get('block') not in (block, None):
                continue
            n_jobs += 1
            v = j.get('a0_drop_sha256')
            if v != drop_sha256:
                bad_jobs[(b['run_tag'], 'none' if _empty(v) else str(v)[:16])] += 1
                if len(examples) < 20:
                    examples.append(dict(run_tag=b['run_tag'], job_index=i, job_id=j.get('job_id'),
                                         a0_drop_sha256=None if _empty(v) else str(v)[:64]))
    for s in (load or {}).get('started', []):
        rec = s['rec']
        if rec.get('block') not in (block, None):
            continue
        n_started += 1
        v = rec.get('a0_drop_sha256')
        if not _empty(v) and v != drop_sha256:
            bad_started.append(dict(ref=s['ref'], job_id=rec.get('job_id'), a0_drop_sha256=str(v)[:64]))
    if bad_jobs or bad_started:
        parts = []
        if bad_jobs:
            parts.append(f'{sum(bad_jobs.values())} job entr(ies) in the batch job lists carry no a0_drop_sha256 or another '
                         f'one ({ {f"{k[0]} ({k[1]})": n for k, n in sorted(bad_jobs.items())} }; "none" = a template job '
                         'file was run)')
        if bad_started:
            parts.append(f'{len(bad_started)} started record(s) carry another a0_drop_sha256')
        raise AnalysisStop(f'A0/A9: {"; ".join(parts)}: the drop file given ({drop_sha256[:16]}...) is not the one frozen '
                           f'into the job files of block {block}', details=examples + bad_started[:50])
    return dict(checked=True, n_job_entries=n_jobs, n_started_records=n_started, ok=True)


def load_void_report_v22(path):
    """The VOID report of the block that a re-run block replaces (A9; --void-report): its sha256 and JSON."""
    with open(path, 'rb') as f:
        raw = f.read()
    out = dict(path=os.path.abspath(path), sha256=hashlib.sha256(raw).hexdigest())
    try:
        out['data'] = json.loads(raw.decode('utf-8'))
        if not isinstance(out['data'], dict):
            out['error'] = 'not a JSON object'
    except (ValueError, UnicodeDecodeError) as e:
        out['error'] = f'unreadable report: {e}'[:200]
    return out


def check_void_chain_v22(block, panel_name, manifest, void_report, drop_sha256, arms):
    """A9: a block re-run after a void carries a new block id <panel>_v<n> (n >= 2); its manifest names the voided block
    and the sha256 of its VOID report (`replaces`), and the analysis is given that report (--void-report). Returns
    (input problems, notes, info). STOPS (A0) when the voided block was analysed with another drop file."""
    clear, notes = [], []
    m = manifest.get('data') if manifest and not manifest.get('error') and isinstance(manifest.get('data'), dict) else None
    rep = m.get('replaces') if m is not None else None
    info = dict(block=block, panel=panel_name, replaces=rep, void_report=None)
    if block == panel_name:
        if rep is not None:
            clear.append(f'the block manifest names a block it replaces ({rep!r}) but the block id is the panel name '
                         f'{panel_name!r}: a re-run after a void needs a new block id {panel_name}_v<n> (A9)')
        if void_report is not None:
            notes.append('--void-report ignored: the block id is the panel name (a first block)')
        return clear, notes, info
    mt = re.match(BLOCK_ID_PATTERN, str(block))
    if not mt or mt.group('panel') != panel_name:
        clear.append(f'block id {block!r} is neither the panel name {panel_name!r} nor {panel_name}_v<n> with n >= 2 (A9)')
        return clear, notes, info
    if m is None:
        return clear, notes, info                                   # the missing manifest is an input problem already
    if not isinstance(rep, dict) or _empty(rep.get('block')) or _empty(rep.get('void_report_sha256')):
        clear.append(f'block {block} is a re-run after a void: its manifest must carry "replaces": {{"block": <voided block '
                     'id>, "void_report_sha256": <sha256 of its VOID report>}} (A9)')
        return clear, notes, info
    rb = str(rep['block'])
    mp = re.match(BLOCK_ID_PATTERN, rb)
    if not (rb == panel_name or (mp and mp.group('panel') == panel_name and int(mp.group('n')) < int(mt.group('n')))):
        clear.append(f'manifest replaces block {rb!r}, which is not an earlier block of panel {panel_name} (A9)')
    if void_report is None:
        clear.append(f'no --void-report given: block {block} needs the VOID report of block {rb} (A9)')
        return clear, notes, info
    info['void_report'] = dict(path=void_report.get('path'), sha256=void_report.get('sha256'))
    if void_report.get('error'):
        clear.append(f"--void-report {void_report.get('path')}: {void_report['error']}")
        return clear, notes, info
    d = void_report['data']
    if void_report['sha256'] != rep.get('void_report_sha256'):
        clear.append('the sha256 of --void-report differs from replaces.void_report_sha256 in the block manifest (A9)')
    if d.get('rules') != 'v2.2' or d.get('decision') != 'VOID':
        clear.append(f"--void-report is not a v2.2 VOID report (rules {d.get('rules')!r}, decision {d.get('decision')!r})")
    if d.get('block') != rb or d.get('panel') != panel_name:
        clear.append(f"--void-report is for block {d.get('block')!r} of panel {d.get('panel')!r}, not block {rb!r} of "
                     f'panel {panel_name!r}')
    if d.get('arms_analysed') is not None and set(d.get('arms_analysed') or []) != set(arms):
        clear.append(f"--void-report analysed the arms {d.get('arms_analysed')}, not {list(arms)}")
    info['void_report'].update(block=d.get('block'), decision=d.get('decision'))
    vs = (d.get('a0') or {}).get('drop_file_sha256')
    if not clear and drop_sha256 is not None and vs != drop_sha256:
        raise AnalysisStop(f'A0/A9: the voided block {rb} was analysed with drop file {str(vs)[:16]}..., this block with '
                           f'{drop_sha256[:16]}...: the A0 drop list changed between the blocks',
                           details=[dict(voided_block=rb, voided_drop_sha256=vs, drop_sha256=drop_sha256)])
    return clear, notes, info


def _stops_and_records_v22(results, panel, arms, repeats, dropped, drop_sha256, drop_source, block, load, manifest,
                           all_panels, void_report, kind):
    """The STOP checks (A0, A8, A9, A2) and the blind record checks of A9, before any outcome is computed. Returns
    dict(tasks, a0, block, asm, reasons (each with its category), parts, notes)."""
    if all_panels is None:
        all_panels = [load_panel(os.path.join(HERE, f)) for f in FROZEN_PANEL_FILES]
    a0_panels = check_drops_all_panels_v21(dropped, list(all_panels) + [panel])
    tasks, a0 = apply_drops(panel, dropped)
    a0.update(drop_source or {}, drop_file_sha256=drop_sha256, all_panels=a0_panels)
    block = block or panel.get('panel')
    a0_row_stops_v21(results, block, dropped, drop_sha256)
    a0['job_files'] = a0_jobfile_stops_v22(load, block, drop_sha256)
    vclear, vnotes, vinfo = check_void_chain_v22(block, panel.get('panel'), manifest, void_report, drop_sha256, arms)
    asm = assemble_v21(results, tasks, arms, repeats, block=block, classify=classify_row_v22)
    reasons, parts = _collect_reasons_v21(asm, arms, tasks, repeats, block, load, manifest,
                                          drop_given=drop_sha256 is not None, kind=kind, classify=classify_row_v22)
    server = check_server_settings_v22(manifest, arms, parts['planned_rows'])
    for t in server['problems']:
        reasons.append(_reason('server_settings', t))
    dup = (load or {}).get('flags', {}).get('duplicate_started_job_ids', [])
    if dup:
        reasons.append(_reason('started_records', f'{len(dup)} started record(s) repeat a job_id of their run directory '
                                                  f'(A9: a record problem): {dup[:5]}'))
    for t in vclear:
        reasons.append(_reason('void_chain', t, clearable=True))
    for r in reasons:
        r['category'] = REASON_CATEGORY_V22[r['kind']]
    notes = list(vnotes) + list(server['notes'])
    m = (manifest or {}).get('data')
    if isinstance(m, dict) and _empty(m.get('smoke_block')):
        notes.append('A9: the RUN_RECORD_SPEC smoke block is mandatory before the first E7 job of each runner version; '
                     'the block manifest names none (optional field smoke_block)')
    parts.update(server=server, void_chain=dict(vinfo, problems=vclear, notes=vnotes),
                 duplicate_started=dict(n=len(dup), refs=dup[:50]))
    return dict(tasks=tasks, a0=a0, block=block, asm=asm, reasons=reasons, parts=parts, notes=notes)


def next_step_v22(reasons, lacking):
    """A9 (on A8): input problems -> INCONCLUSIVE (fix the inputs); else record problems -> VOID; else the A1/A8 slot
    steps (re-run batches, the extra batch, then FAIL). Returns (decision, decision_basis, next_step)."""
    if not reasons:
        return None, 'criteria', None
    if any(r['category'] == 'input' for r in reasons):
        return next_step_v21([r for r in reasons if r['category'] == 'input'], lacking)
    if any(r['category'] == 'record' for r in reasons):
        return 'VOID', 'record problems (A9)', VOID_STEP_V22
    return next_step_v21(reasons, lacking)


def _common_v22(pre, panel, repeats, arms, manifest, load):
    parts = pre['parts']
    return dict(rules='v2.2', analysis=ANALYSIS_ID_V22, error_table_sha256=ERROR_TABLE_SHA256_V22, panel=panel.get('panel'),
                panel_sha256=panel.get('sha256'), block=pre['block'], arms_analysed=list(arms), repeats=repeats,
                a0=pre['a0'], n_tasks_planned=len(panel['tasks']), n_tasks_after_a0=len(pre['tasks']),
                d_pen_ids=list(D_PEN_IDS_V22),
                manifest=dict(path=(manifest or {}).get('path'), sha256=(manifest or {}).get('sha256'),
                              **{k: v for k, v in parts['manifest'].items() if k != 'ok'}),
                server_settings=parts['server'], void_chain=parts['void_chain'],
                required_fields=parts['required_fields'],
                started=dict(parts['started'], duplicate_job_ids=parts['duplicate_started']['refs'],
                             n_duplicate_job_ids=parts['duplicate_started']['n']),
                provenance_check=parts['provenance'], load_problems=(load or {}).get('problems', []),
                load_notes=(load or {}).get('notes', []),
                data=dict(ignored_rows=dict(collections.Counter(x['reason'] for x in pre['asm']['ignored'])),
                          ignored=pre['asm']['ignored'][:300]))


def _record_notes_v22(asm, arms, blind):
    """Notes that need no outcome; the class-mismatch note is masked in blind reports (A9)."""
    notes = []
    if asm['deviations']:
        notes.append(f"{len(asm['deviations'])} protocol deviation(s) in attempts (listed under slots)")
    if blind:
        bm = blind_class_mismatches_v22(asm['class_mismatches'])
        if bm['n']:
            notes.append(f"{bm['n']} row(s) whose recorded error_class_id differs from the v2.2 table in an infrastructure "
                         f"id ({bm['by_kind']}; agent ids masked, A9); the table decides; check the runner")
    elif asm['class_mismatches']:
        notes.append(f"{len(asm['class_mismatches'])} row(s) whose recorded error_class_id differs from the v2.2 table "
                     '(the table decides; check the runner)')
    tabs = collections.Counter(_canon(x['row'].get('error_table_sha256')) for lst in asm['slots'].values() for x in lst
                               if not x.get('synthetic') and not _empty(x['row'].get('error_table_sha256')))
    if tabs and set(tabs) != {_canon(ERROR_TABLE_SHA256_V22)}:
        notes.append(f'rows carry error_table_sha256 values other than the v2.2 table ({ERROR_TABLE_SHA256_V22[:16]}...): '
                     f'{dict(tabs)}; the analysis classifies every row itself')
    unknown = [x for a in arms for x in asm['infra'][a] if x['id'] == 'I12']
    if unknown:
        notes.append(f'{len(unknown)} row(s) with error text outside the frozen table were counted as infrastructure '
                     'errors (I12; penalised in d_pen); the lead should check them')
    return notes


def gate_v22(results, panel, task_repo, arm_a=BASELINE, arm_b=None, repeats=None, n_boot=N_BOOT, seed=BOOT_SEED,
             traces=None, with_secondary=True, exact=True, dropped=None, drop_sha256=None, drop_source=None, block=None,
             guard_adopted=None, load=None, manifest=None, all_panels=None, void_report=None):
    """Confirmation/dev gate under PROTOCOL_AMENDMENTS.md A0-A9. `results` = load_runs_v21(..., dedupe_started=True)
    ['runs'], `load` = that dict, `manifest` = load_manifest_v21(), `void_report` = load_void_report_v22() (re-run
    blocks only). Raises AnalysisStop (no decision) on A0/A8/A9 and A2 violations. The record checks run first and
    blind; INCONCLUSIVE, VOID and FAIL-by-INCONCLUSIVE results are blind."""
    adopted = GUARD_H_ADOPTED if guard_adopted is None else bool(guard_adopted)
    repeats = repeats or DEFAULT_REPEATS.get(panel.get('panel'), 4)
    check_panel_repos(panel, task_repo)
    arms = [arm_a, arm_b]
    pre = _stops_and_records_v22(results, panel, arms, repeats, dropped, drop_sha256, drop_source, block, load, manifest,
                                 all_panels, void_report, kind='gate')
    tasks, block, asm, reasons, parts = pre['tasks'], pre['block'], pre['asm'], pre['reasons'], pre['parts']
    decision, basis, step = next_step_v22(reasons, asm['lacking'])
    blind = decision is not None
    frozen, fnotes = _frozen_notes(panel, repeats, n_boot, seed)
    notes = list(fnotes) + list(parts['manifest']['notes']) + pre['notes']
    if arm_b not in ADAPTER_ARMS and panel.get('panel') == 'D':
        notes.append(f'arm {arm_b} on panel D: A7 labels the adapter arm bundle0_adapter_A or bundle0_adapter_B')
    notes += _record_notes_v22(asm, arms, blind)
    common = dict(_common_v22(pre, panel, repeats, arms, manifest, load), kind='gate', arm_a=arm_a, arm_b=arm_b,
                  frozen_settings=frozen,
                  thresholds=dict(gain_min=_fmt_frac(GAIN_MIN), alpha=float(ALPHA), min_pos_repos=MIN_POS_REPOS,
                                  time_ratio_max=float(TIME_RATIO_MAX), alpha_4b=float(ALPHA_4B),
                                  guard_h_limit=float(GUARD_H_LIMIT), rerun_max_attempt=RERUN_MAX_ATTEMPT,
                                  max_attempts=MAX_ATTEMPTS_V21, n_boot=N_BOOT, seed=BOOT_SEED))
    if blind:
        out = dict(common, decision=decision, decision_basis=basis, blind=True,
                   inconclusive_reasons=[r['text'] for r in reasons], reasons=reasons, next_step=step, notes=notes,
                   blind_note=('A8/A9: the report of an INCONCLUSIVE, VOID or STOP gate is blind (slots, provenance, error '
                               'counts, record and load problems only) until the reasons are cleared'),
                   slots=_blind_slots_v22(asm), harness=dict(by_arm=_blind_infra(asm, arms),
                                                             rows=[dict(arm=a, **x) for a in arms for x in asm['infra'][a]][:400]))
        if decision == 'VOID':
            out['void'] = dict(block=block, record_problems=[r['text'] for r in reasons if r['category'] == 'record'],
                               instructions=VOID_STEP_V22)
        return out
    # ------------------------------------------------------------ no reason left: the decision statistics
    diffs, reps = [], []
    for t in tasks:
        ra, rb = asm['cells'][(arm_a, t)], asm['cells'][(arm_b, t)]
        diffs.append(Fraction(sum(solved(r['row']) for r in rb), len(rb)) - Fraction(sum(solved(r['row']) for r in ra), len(ra)))
        reps.append(task_repo[t])
    K, L = _common_denominator(diffs)
    core = gate_core(K, L, reps, n_boot=n_boot, seed=seed, exact=exact)
    core.update(notes=list(core.get('notes', [])))
    guard = guard_h(K, L, reps)
    for row in guard['table']:
        row['loo'] = (core.get('repos') or {}).get(row['repo'], {}).get('loo')
    pdiffs, pen_slots = d_pen_task_diffs(asm, tasks, arm_a, arm_b, pen_ids=D_PEN_IDS_V22, classify=classify_row_v22)
    Kp, Lp = _common_denominator(pdiffs)
    core_pen = gate_core(Kp, Lp, reps, n_boot=n_boot, seed=seed, sensitivities=False, exact=exact)
    guard_pen = guard_h(Kp, Lp, reps)
    runs_a = [r for t in tasks for r in asm['cells'][(arm_a, t)]]
    runs_b = [r for t in tasks for r in asm['cells'][(arm_b, t)]]
    ratio, ta, tb = _time_ratio(runs_a, runs_b)
    strata = []
    for key, (cnt, _) in parts['used_batches'].items():
        strata.append(dict(batch=key, run_tag=(load or {}).get('batches', {}).get(key, {}).get('run_tag', key),
                           jobs_baseline=cnt.get(arm_a, 0), jobs_candidate=cnt.get(arm_b, 0),
                           k_baseline=sum(1 for x in asm['infra'][arm_a] if x['batch'] == key),
                           k_candidate=sum(1 for x in asm['infra'][arm_b] if x['batch'] == key)))
    h4b = harness_4b_stratified(strata)
    time_ok = ratio is not None and ratio <= TIME_RATIO_MAX
    crit = dict(core['criteria'], time=bool(time_ok), harness=h4b['fails'] is False)
    if adopted:
        crit['guard_h'] = not guard['fails']
    cp = core_pen['criteria']
    crit.update(gain_pen=bool(cp['gain']), p_pen=bool(cp['p']), pos_repos_pen=bool(cp['pos_repos']), loo_pen=bool(cp['loo']))
    if adopted:
        crit['guard_h_pen'] = not guard_pen['fails']
    decision = 'PASS' if all(bool(v) for v in crit.values()) else 'FAIL'
    notes = list(core.get('notes', [])) + notes
    out = dict(common, decision=decision, decision_basis='criteria', blind=False, inconclusive_reasons=[], reasons=[],
               next_step=None, notes=notes, criteria=crit,
               guard_h=dict(guard, adopted=adopted, role='criterion (5)' if adopted else 'diagnostic only (A4)'),
               n_tasks_used=len(tasks), gain=core,
               d_pen=dict(gain=core_pen, guard_h=guard_pen, penalised_slots=pen_slots, n_penalised=len(pen_slots),
                          n_penalised_solved=sum(1 for x in pen_slots if x['used_row_solved']),
                          note='candidate slots with a row of class ' + '/'.join(D_PEN_IDS_V22) +
                               ' (any attempt) scored unsolved; baseline unchanged; criteria (1), (2), (3), (5) must '
                               'also hold (A8; I13 ratified by A9)'),
               time=dict(ratio=None if ratio is None else float(ratio), mean_a=float(sum(ta) / len(ta)) if ta else None,
                         mean_b=float(sum(tb) / len(tb)) if tb else None, n_a=len(ta), n_b=len(tb),
                         note='(4a) on the rows used in the task means only (A1)'),
               harness=dict(test_4b=h4b, by_arm=_blind_infra(asm, arms),
                            rows=[dict(arm=a, **x) for a in arms for x in asm['infra'][a]][:400],
                            silent_verification_suspects={a: _silent_verification_suspects(parts['used_rows'][a])
                                                          for a in arms}),
               slots=dict(_blind_slots(asm), class_mismatches=asm['class_mismatches']))
    if with_secondary:
        out['secondary'] = secondary(results, traces if traces is not None else TraceStore(),
                                     dict(panel=panel.get('panel'), tasks=tasks), task_repo, arm_a, arm_b, repeats,
                                     asm=asm, status=_status_v22_metric)
    return out


def screening_v22(results, panel_S, task_repo=None, baseline=BASELINE, arms=None, repeats=None, traces=None,
                  dropped=None, drop_sha256=None, drop_source=None, block=None, load=None, manifest=None,
                  all_panels=None, void_report=None):
    """PROTOCOL.md screening under A0-A9. The STOP and record checks run first; a screen with an input problem
    (INCONCLUSIVE) or a record problem (VOID) is blind (no gain, ranking or finalist). Otherwise as v2.1: PROVISIONAL
    while a slot lacks a valid row or an arm has no rows; gain, edit-call success (A6/A7) and the finalist rule are
    unchanged."""
    repeats = repeats or DEFAULT_REPEATS.get(panel_S.get('panel'), 2)
    arms = [a for a in (SCREEN_ARMS if arms is None else arms) if a != baseline]
    all_arms = [baseline] + arms
    pre = _stops_and_records_v22(results, panel_S, all_arms, repeats, dropped, drop_sha256, drop_source, block, load,
                                 manifest, all_panels, void_report, kind='screen')
    tasks, block, asm, reasons, parts = pre['tasks'], pre['block'], pre['asm'], pre['reasons'], pre['parts']
    no_rows = [a for a in all_arms if not any(asm['slots'][(a, t, k)] for t in tasks for k in range(1, repeats + 1))]
    if no_rows:
        reasons.insert(0, dict(_reason('arm_without_rows', f'arm(s) without any row: {no_rows} (A8)'), category='slots'))
    decision, basis, step = next_step_v22(reasons, asm['lacking'])
    frozen, notes = _frozen_notes(panel_S, repeats, N_BOOT, BOOT_SEED)
    notes = list(notes) + list(parts['manifest']['notes']) + pre['notes']
    if panel_S.get('panel') == 'S' and (baseline != BASELINE or sorted(arms) != sorted(SCREEN_ARMS[1:])):
        frozen = False
        notes.append(f'NON-FROZEN arm set {[baseline] + arms} (A8: the six frozen arms {SCREEN_ARMS})')
    blind = decision == 'VOID' or basis == 'analysis inputs incomplete'
    notes += _record_notes_v22(asm, all_arms, blind)
    common = dict(_common_v22(pre, panel_S, repeats, all_arms, manifest, load), kind='screening', baseline=baseline,
                  arms=arms, frozen_settings=frozen, n_tasks=len(tasks))
    if blind:
        out = dict(common, decision=decision, decision_basis=basis, blind=True, finalist=None,
                   provisional=[r['text'] for r in reasons], reasons=reasons, next_step=step, notes=notes,
                   blind_note='A9: the record checks run blind before any outcome; this screen shows no gain, ranking or '
                              'finalist until its reasons are cleared',
                   slots=_blind_slots_v22(asm), harness=dict(by_arm=_blind_infra(asm, all_arms)))
        if decision == 'VOID':
            out['void'] = dict(block=block, record_problems=[r['text'] for r in reasons if r['category'] == 'record'],
                               instructions=VOID_STEP_V22)
        return out
    store = traces if traces is not None else TraceStore()
    summ = {a: _arm_used_summary_v21(asm, a, store, classify=classify_row_v22) for a in all_arms}
    rows = []
    for a in arms:
        gain, used, excluded = Fraction(0), [], []
        per_repo = collections.defaultdict(Fraction)
        for t in tasks:
            ra, rb = asm['cells'][(baseline, t)], asm['cells'][(a, t)]
            if not ra or not rb:
                excluded.append(dict(task=t, n_baseline=len(ra), n_arm=len(rb)))
                continue
            diff = Fraction(sum(solved(r['row']) for r in rb), len(rb)) - Fraction(sum(solved(r['row']) for r in ra), len(ra))
            gain += diff
            used.append(t)
            if task_repo:
                per_repo[task_repo.get(t)] += diff
        s = summ[a]
        rows.append(dict(arm=a, gain=gain, gain_float=float(gain), gain_exact=_fmt_frac(gain), tasks_used=len(used),
                         tasks_excluded=excluded, mean_gain_per_task=float(gain / len(used)) if used else None,
                         repos_positive=sum(1 for v in per_repo.values() if v > 0),
                         repos_negative=sum(1 for v in per_repo.values() if v < 0),
                         per_repo_gain={k: float(v) for k, v in sorted(per_repo.items())},
                         edit_success=s['edit_success'], mean_agent_time=s['mean_agent_time']))
    order = {a: i for i, a in enumerate(SCREEN_ARMS)}

    def key(r):
        es, mt = r['edit_success'], r['mean_agent_time']
        return (-r['gain'], -(es if es is not None else Fraction(-1)), mt if mt is not None else float('inf'),
                order.get(r['arm'], len(order)), r['arm'])

    ranked = sorted(rows, key=key)
    finalist, tie_log = None, []
    if ranked and ranked[0]['gain'] > 0:
        finalist = ranked[0]['arm']
        top = [r for r in ranked if r['gain'] == ranked[0]['gain']]
        if len(top) > 1:
            tie_log.append(f"gain tie among {[r['arm'] for r in top]}; broken by edit-call success rate")
            es_top = [r for r in top if r['edit_success'] == top[0]['edit_success']]
            if len(es_top) > 1:
                tie_log.append(f"edit-success tie among {[r['arm'] for r in es_top]}; broken by mean agent time")
                mt_top = [r for r in es_top if r['mean_agent_time'] == es_top[0]['mean_agent_time']]
                if len(mt_top) > 1:
                    tie_log.append('UNRESOLVED by the protocol: fell back to PROTOCOL arm order (flag to owner)')
    for r in rows:
        r['edit_success_float'] = None if r['edit_success'] is None else float(r['edit_success'])
        r['edit_success'] = _fmt_frac(r['edit_success']) if r['edit_success'] is not None else None
        r['gain'] = r['gain_float']
    for a, s in summ.items():
        s['edit_success_float'] = None if s['edit_success'] is None else float(s['edit_success'])
        s['edit_success'] = _fmt_frac(s['edit_success']) if s['edit_success'] is not None else None
    provisional = [r['text'] for r in reasons]
    procedure_complete = bool(reasons) and basis == 'INCONCLUSIVE after the extra batch (A8)'
    text = f'finalist {finalist}' if finalist else 'no finalist (largest gain <= 0): bundle0 stays'
    if provisional:
        text = f"{text} (PROVISIONAL: {'; '.join(provisional[:6])}{'; ...' if len(provisional) > 6 else ''})"
    if procedure_complete:
        notes.append('every slot has used its five attempts and the screen is still PROVISIONAL: A8/A9 set no screening '
                     'rule for this case (lead)')
    return dict(common, decision=text, decision_basis=basis, blind=False, ranking=[r['arm'] for r in ranked],
                finalist=finalist, provisional=provisional, reasons=reasons,
                next_step=(step if reasons and not procedure_complete else None), procedure_complete=procedure_complete,
                notes=notes, tie_log=tie_log, arms_table=rows, run_summary=summ, slots=dict(_blind_slots(asm)))


def rerun_jobs_v22(results, panel, arms, repeats=None, block=None, dropped=None, drop_sha256=None, drop_source=None,
                   load=None, manifest=None, all_panels=None, void_report=None):
    """A1/A8/A9 next batch, blind (slots only). Refuses (AnalysisStop) on A0/A8/A9/A2 stops and while an analysis input
    is incomplete; a block with a record problem is VOID: no batch (A9: re-run the whole block under a new block id).
    Otherwise as v2.1: re-run batches up to attempt 3, then the extra batch (attempts 4 and 5, both arms)."""
    repeats = repeats or DEFAULT_REPEATS.get(panel.get('panel'), 4)
    if drop_sha256 is None:
        raise AnalysisStop('reruns refused: no A0 drop file (its sha256 goes into every job)')
    arms = list(arms)
    pre = _stops_and_records_v22(results, panel, arms, repeats, dropped, drop_sha256, drop_source, block, load, manifest,
                                 all_panels, void_report, kind='reruns')
    tasks, block, asm, reasons = pre['tasks'], pre['block'], pre['asm'], pre['reasons']
    inputs = [r['text'] for r in reasons if r['category'] == 'input']
    if inputs:
        raise AnalysisStop('reruns refused: the analysis inputs are incomplete (A8/A9); a re-run list made from them could '
                           'repeat or skip attempts', details=inputs)
    base = dict(kind='reruns', rules='v2.2', analysis=ANALYSIS_ID_V22, panel=panel.get('panel'), block=block, arms=arms,
                arms_analysed=arms, repeats=repeats, a0=pre['a0'], n_planned=asm['n_planned'],
                n_with_used_row=len(asm['chosen']), n_lacking=len(asm['lacking']),
                manifest_data_problems=pre['parts']['manifest']['data'],
                server_settings=pre['parts']['server'], reasons=reasons, blind=True)
    records = [r['text'] for r in reasons if r['category'] == 'record']
    if records:
        return dict(base, decision='VOID', decision_basis='record problems (A9)', batch_kind=None, jobs=[],
                    waiting_for_extra_batch=[], exhausted=[], next_step=VOID_STEP_V22,
                    void=dict(block=block, record_problems=records, instructions=VOID_STEP_V22),
                    note='blind (A9): the block is VOID; no re-run batch is made for a void block')
    lack = {(x['arm'], x['task'], x['repeat']): x for x in asm['lacking']}
    open_ = {k: x for k, x in lack.items() if x['max_attempt'] < MAX_ATTEMPTS_V21
             and not x['reason'].startswith('valid row only')}
    rerun_phase = any(x['max_attempt'] < RERUN_MAX_ATTEMPT for x in open_.values())
    kind = 'rerun' if rerun_phase else 'extra'
    jobs, waiting, exhausted = [], [], []
    for a, t, k in interleaved_slots_v21(tasks, arms, repeats):
        x = lack.get((a, t, k))
        if x is None:
            continue
        if (a, t, k) not in open_:
            exhausted.append(dict(arm=a, task=t, repeat=k, attempts=x['attempts'], reason=x['reason']))
            continue
        nxt = x['max_attempt'] + 1
        if rerun_phase and nxt > RERUN_MAX_ATTEMPT:
            waiting.append(dict(arm=a, task=t, repeat=k, attempts=x['attempts']))
            continue
        jobs.append(dict(arm=a, task=t, seed=k, repeat_id=k, attempt=nxt, block=block,
                         job_id=job_id_v2(block, a, t, k, nxt), a0_drop_sha256=drop_sha256, batch_kind=kind))
    refused = [job_refusal_v22(j) for j in jobs if job_refusal_v22(j)]
    if refused:
        raise AnalysisStop('reruns refused: the next batch would hold jobs the runner must refuse (A9)', details=refused[:20])
    return dict(base, decision=None, decision_basis='slots', batch_kind=kind, jobs=jobs, waiting_for_extra_batch=waiting,
                exhausted=exhausted,
                note=('blind: slots only (A8). A re-run batch gives attempts 2-3 (A1); the extra batch, made only when no '
                      'slot is below attempt 3, gives at most attempts 4 and 5 to every slot of both arms still without a '
                      'valid row (A8). Run it under a new run tag and add the tag to the block manifest.'))


def records_v22(results, panel, arms, repeats=None, block=None, dropped=None, drop_sha256=None, drop_source=None,
                load=None, manifest=None, all_panels=None, void_report=None):
    """A9: the record checks alone, blind (no outcome is computed): the STOP checks, input problems, record problems and
    the slots without a valid row. Decision 'RECORDS COMPLETE' (no input or record problem; open slots are listed),
    'VOID' (record problems) or 'INCONCLUSIVE' (analysis inputs incomplete). The smoke block (RUN_RECORD_SPEC.md 3.7)
    must give RECORDS COMPLETE."""
    repeats = repeats or DEFAULT_REPEATS.get(panel.get('panel'), 4)
    arms = list(arms)
    pre = _stops_and_records_v22(results, panel, arms, repeats, dropped, drop_sha256, drop_source, block, load, manifest,
                                 all_panels, void_report, kind='records')
    asm, reasons, parts = pre['asm'], pre['reasons'], pre['parts']
    decision, basis, step = next_step_v22(reasons, asm['lacking'])
    if decision == 'VOID':
        verdict = 'VOID'
    elif basis == 'analysis inputs incomplete':
        verdict = 'INCONCLUSIVE'
    else:
        verdict = 'RECORDS COMPLETE'
    frozen, notes = _frozen_notes(panel, repeats, N_BOOT, BOOT_SEED)
    notes = list(notes) + list(parts['manifest']['notes']) + pre['notes'] + _record_notes_v22(asm, arms, True)
    out = dict(_common_v22(pre, panel, repeats, arms, manifest, load), kind='records', decision=verdict,
               decision_basis=basis, blind=True, frozen_settings=frozen, reasons=reasons,
               slot_step=(step if verdict == 'RECORDS COMPLETE' else None),
               next_step=(step if verdict != 'RECORDS COMPLETE' else None), notes=notes,
               slots=_blind_slots_v22(asm), harness=dict(by_arm=_blind_infra(asm, arms)),
               note='A9: record checks only (blind); no outcome is computed')
    if verdict == 'VOID':
        out['void'] = dict(block=pre['block'], record_problems=[r['text'] for r in reasons if r['category'] == 'record'],
                           instructions=VOID_STEP_V22)
    return out


def parity_ratio_v22(results_plain, results_lora, arm=BASELINE):
    """A8/A9: agent-time ratio of `arm` on the adapter-serving server over the plain server, paired by
    task (A9): on the tasks with agent-outcome rows (v2.2 table) on both servers, sum of the task means (repeats
    averaged within task) on the adapter-serving server over the same sum on the plain server. A warning lists tasks that
    only one server has. Above 1.15 -> investigate (no rule change). The unpaired ratio of v2.1 is reported as
    a diagnostic."""
    def by_task(rs):
        d = collections.defaultdict(list)
        for r in rs:
            t = r['row'].get('t_agent')
            if r['arm'] == arm and not r.get('synthetic') and classify_row_v22(r['row'])['klass'] == 'agent' \
                    and isinstance(t, (int, float)) and not isinstance(t, bool):
                d[r['task']].append(Fraction(str(t)))
        return d
    P, L = by_task(results_plain), by_task(results_lora)
    common = sorted(set(P) & set(L))
    only_p, only_l = sorted(set(P) - set(L)), sorted(set(L) - set(P))
    sp = sum((sum(P[t]) / len(P[t]) for t in common), Fraction(0))
    sl = sum((sum(L[t]) / len(L[t]) for t in common), Fraction(0))
    ratio = sl / sp if common and sp else None
    tp = [x for v in P.values() for x in v]
    tl = [x for v in L.values() for x in v]
    unpaired = (sum(tl) / len(tl)) / (sum(tp) / len(tp)) if tp and tl and sum(tp) else None
    warnings = []
    if only_p or only_l:
        warnings.append(f'the task sets differ: {len(only_p)} task(s) only on the plain server {only_p[:20]}, {len(only_l)} '
                        f'only on the LoRA-enabled server {only_l[:20]}; the ratio uses the {len(common)} common task(s) (A9)')
    return dict(kind='parity', rules='v2.2', arm=arm, n_plain=len(tp), n_lora=len(tl), n_tasks_plain=len(P),
                n_tasks_lora=len(L), n_tasks_common=len(common), tasks_only_plain=only_p, tasks_only_lora=only_l,
                mean_plain=float(sp / len(common)) if common else None, mean_lora=float(sl / len(common)) if common else None,
                ratio=None if ratio is None else float(ratio), ratio_exact=None if ratio is None else _fmt_frac(ratio),
                ratio_unpaired=None if unpaired is None else float(unpaired), limit=float(PARITY_RATIO_LIMIT),
                investigate=None if ratio is None else bool(ratio > PARITY_RATIO_LIMIT), warnings=warnings,
                method='paired by task (A9): sum over common tasks of the LoRA-server task mean / the same sum on the plain '
                       'server; agent-outcome rows only',
                note='A8: above 1.15 means investigate before the D block; it changes no rule')


# ----------------------------------------------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------------------------------------------
def _f(x, nd=4):
    if x is None:
        return 'n/a'
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isinf(x):
            return 'inf' if x > 0 else '-inf'
        return f'{x:.{nd}f}'
    return str(x)


def _jsonable(o):
    if isinstance(o, Fraction):
        return float(o)
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    return o


def md_secondary(sec):
    a, b = sec['arm_a'], sec['arm_b']
    L = [f'## Secondary metrics (descriptive; {b} minus {a})', '',
         f"Runs: {a} {sec['arms'][a]['n_runs']}/{sec['arms'][a]['planned']}, {b} {sec['arms'][b]['n_runs']}/{sec['arms'][b]['planned']}; "
         f"traces {a} {sec['arms'][a]['traces']}, {b} {sec['arms'][b]['traces']}.", '',
         f'| metric | {a} mean | {a} median | {a} p95 | {b} mean | {b} median | {b} p95 | paired diff | 95% CI | tasks |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for k in PER_RUN:
        x, y, p = sec['arms'][a][k], sec['arms'][b][k], sec['paired_differences'][k]
        L.append(f"| {k} | {_f(x['mean'])} | {_f(x['median'])} | {_f(x['p95'])} | {_f(y['mean'])} | {_f(y['median'])} | "
                 f"{_f(y['p95'])} | {_f(p['mean'])} | [{_f(p['lo'])}, {_f(p['hi'])}] | {p['n']} |")
    L += ['', f'| rate (pooled over runs) | {a} | {b} | paired task diff | 95% CI | tasks |', '|---|---|---|---|---|---|']
    for name, _, _ in RATIOS:
        x, y, p = sec['arms'][a][name], sec['arms'][b][name], sec['paired_differences'][name]
        L.append(f"| {name} | {_f(x['rate'])} ({x['num']}/{x['den']}) | {_f(y['rate'])} ({y['num']}/{y['den']}) | "
                 f"{_f(p['mean'])} | [{_f(p['lo'])}, {_f(p['hi'])}] | {p['n']} |")
    return '\n'.join(L) + '\n'


def md_gate(g):
    c = g['criteria']
    core = g['gain']
    bt = core.get('bootstrap') or {}
    L = [f"# E7 gate: {g['arm_b']} vs {g['arm_a']} on panel {g['panel']}", '',
         f"**Decision: {g['decision']}** (frozen rule, PROTOCOL.md \"Decision rules\"). Data complete: {_f(g['data_complete'])}.", '']
    for n in g['notes']:
        L.append(f'> {n}')
    L += ['', '| criterion | value | threshold | met |', '|---|---|---|---|',
          f"| (1) task-weighted mean gain | {_f(core.get('d'))} ({core.get('d_exact')}) | >= 2/58 = 0.0345 | {_f(c['gain'])} |",
          f"| (2) bootstrap-t one-sided p (Rademacher, null imposed) | {_f(bt.get('p'))} (t = {_f(bt.get('t'))}) | <= 0.025 | {_f(c['p'])} |",
          f"| (3a) repositories with positive gain | {core.get('n_repos_positive')} of {core.get('n_repos')} | >= 3 | {_f(core.get('n_repos_positive', 0) >= MIN_POS_REPOS)} |",
          f"| (3b) min leave-one-repository-out estimate | {_f(core.get('loo_min'))} | all > 0 | {_f(c['loo'])} |",
          f"| (4a) mean agent time ratio | {_f(g['time']['ratio'])} ({_f(g['time']['mean_b'], 1)} s / {_f(g['time']['mean_a'], 1)} s) | <= 1.2 | {_f(c['time'])} |",
          f"| (4b) harness failures (error rows + missing runs) | {g['arm_b']}: {g['harness']['by_arm'][g['arm_b']]['harness_failures']}, {g['arm_a']}: {g['harness']['by_arm'][g['arm_a']]['harness_failures']} | no increase | {_f(c['harness'])} |",
          '', f"Tasks used {g['n_tasks_used']} of {g['n_tasks_planned']}; repeats planned {g['repeats']} per arm and task.",
          f"Repository-equal estimate: {_f(core.get('repo_equal'))}. Sum of task gains: {_f(core.get('sum_gain'))}.", '']
    if bt:
        L += [f"Bootstrap: B = {bt['n_boot']}, seed {g['thresholds']['seed']}, G = {bt['n_clusters']} repositories, "
              f"{bt['n_clusters_nonzero']} with a nonzero total (smallest attainable exact p {bt['min_attainable_p_exact']:.3g}); "
              f"CR1 se = {_f(bt['se'])}; exact enumeration p = {_f(bt.get('p_exact_enumeration'))} (diagnostic)."]
        for name, s in core.get('sensitivity', {}).items():
            L.append(f"Sensitivity (not the decision) {name}: p = {_f(s['p'])}, t = {_f(s['t'])}.")
        L.append('')
    L += ['| repository | tasks | sum gain | mean gain | leave-one-out estimate |', '|---|---|---|---|---|']
    for r, v in (core.get('repos') or {}).items():
        L.append(f"| {r} | {v['n_tasks']} | {_f(v['sum_gain'])} | {_f(v['mean_gain'])} | {_f(v['loo'])} |")
    d = g['data']
    L += ['', '## Data handling', '',
          f"Planned runs {d['n_planned']}, missing {d['n_missing']}, duplicates {len(d['duplicates'])}, ignored rows {d['ignored_rows']}."]
    for a, h in g['harness']['by_arm'].items():
        L.append(f"- {a}: completed {h['completed']}/{h['planned']}, missing {h['missing']}, harness-error rows "
                 f"{h['harness_error_rows']}, outcomes {h['status_counts']}, details {h['harness_error_details']}")
    if g['excluded_tasks']:
        L.append(f"- Excluded tasks (no completed run in one arm): {[e['task'] for e in g['excluded_tasks']]}; "
                 f"sensitivity {g['excluded_sensitivity']}")
    if d['missing']:
        L.append(f"- Missing runs: {[(m['arm'], m['task'], m['repeat']) for m in d['missing']][:50]}")
    if d['duplicates']:
        L.append(f"- Duplicates: {d['duplicates'][:20]}")
    L.append('')
    if 'secondary' in g:
        L.append(md_secondary(g['secondary']))
    return '\n'.join(L)


def md_screen(s):
    L = [f"# E7 screening on panel {s['panel']} (exploratory)", '',
         f"**{s['decision']}**. Baseline {s['baseline']}; {s['n_tasks']} tasks x {s['repeats']} repeats planned per arm.", '']
    for t in s['tie_log']:
        L.append(f'> {t}')
    L += ['', '| rank | arm | gain (sum over tasks) | tasks used | mean gain/task | repos +/- | edit-call success | mean agent time (s) |',
          '|---|---|---|---|---|---|---|---|']
    rows = {r['arm']: r for r in s['arms_table']}
    for i, a in enumerate(s['ranking'], 1):
        r = rows[a]
        L.append(f"| {i} | {a} | {_f(r['gain'], 2)} ({r['gain_exact']}) | {r['tasks_used']} | {_f(r['mean_gain_per_task'])} | "
                 f"{r['repos_positive']}/{r['repos_negative']} | {_f(r['edit_success_float'])} ({r['edit_success']}) | {_f(r['mean_agent_time'], 1)} |")
    L += ['', '| arm | planned | completed | missing | solves | harness errors | outcomes | traces found/missing | writes (edit_file, write_file, shell) [attempts, failed] |',
          '|---|---|---|---|---|---|---|---|---|']
    for a, v in s['run_summary'].items():
        L.append(f"| {a} | {v['planned']} | {v['completed']} | {v['missing']} | {v['solves']} | {v['harness_errors']} | "
                 f"{v['status_counts']} | {v['traces_found']}/{v['traces_missing']} | {v['edit_channels']} |")
    d = s['data']
    L += ['', f"Planned runs {d['n_planned']}, missing {d['n_missing']}, duplicates {len(d['duplicates'])}, ignored rows {d['ignored_rows']}."]
    if d['missing']:
        L.append(f"Missing runs: {[(m['arm'], m['task'], m['repeat']) for m in d['missing']][:60]}")
    for r in s['arms_table']:
        if r['tasks_excluded']:
            L.append(f"{r['arm']}: tasks excluded from the gain (no completed run in one arm): {r['tasks_excluded']}")
    return '\n'.join(L) + '\n'


def md_gate_v2(g):
    c, core, gh = g['criteria'], g['gain'], g['guard_h']
    bt = core.get('bootstrap') or {}
    h = g['harness']['test_4b']
    L = [f"# E7 gate (rules v2): {g['arm_b']} vs {g['arm_a']} on panel {g['panel']} (block {g['block']})", '',
         '## Per-repository differences (A4; sorted by total difference, most negative first)', '',
         '| repository | tasks | total difference | repository mean | leave-one-out estimate | total <= -2 |',
         '|---|---|---|---|---|---|']
    for r in gh['table']:
        L.append(f"| {r['repo']} | {r['n_tasks']} | {_f(r['total'])} ({r['total_exact']}) | {_f(r['mean'])} | "
                 f"{_f(r.get('loo'))} | {_f(r['at_or_below_limit'])} |")
    L += ['', f"Guard H ({gh['role']}): lowest repository total {_f(gh['min_total'])} ({gh['min_repo']}); "
              f"repositories at or below -2: {gh['repos_at_or_below'] or 'none'}.", '',
          f"## Decision: **{g['decision']}**", '',
          f"Rules v2 (PROTOCOL.md with PROTOCOL_AMENDMENTS.md A0-A7). Frozen settings: {_f(g['frozen_settings'])}. "
          f"Criteria below: {g['criteria_note']}.", '']
    for r in g['inconclusive_reasons']:
        L.append(f'> INCONCLUSIVE: {r}')
    for n in g['notes']:
        L.append(f'> {n}')
    hp = 'n/a (no infrastructure-error rows)' if h['p'] is None else f"{_f(h['p'])}"
    L += ['', '| criterion | value | threshold | met |', '|---|---|---|---|',
          f"| (1) task-weighted mean gain | {_f(core.get('d'))} ({core.get('d_exact')}) | >= 2/58 = 0.0345 | {_f(c['gain'])} |",
          f"| (2) bootstrap-t one-sided p (Rademacher, null imposed) | {_f(bt.get('p'))} (t = {_f(bt.get('t'))}) | <= 0.025 | {_f(c['p'])} |",
          f"| (3a) repositories with positive gain | {core.get('n_repos_positive')} of {core.get('n_repos')} | >= 3 | {_f(core.get('n_repos_positive', 0) >= MIN_POS_REPOS)} |",
          f"| (3b) min leave-one-repository-out estimate | {_f(core.get('loo_min'))} | all > 0 | {_f(c['loo'])} |",
          f"| (4a) mean agent time ratio, rows used | {_f(g['time']['ratio'])} ({_f(g['time']['mean_b'], 1)} s / {_f(g['time']['mean_a'], 1)} s) | <= 1.2 | {_f(c['time'])} |",
          f"| (4b) infrastructure-error rows, all attempts: {g['arm_b']} {h['k_candidate']} of {h['n']} (planned share {h['share_candidate']}) | exact binomial p = {hp} | p > 0.05 | {_f(c['harness'])} |"]
    if 'guard_h' in c:
        L.append(f"| (5) guard H: lowest repository total | {_f(gh['min_total'])} | > -2 | {_f(c['guard_h'])} |")
    L += ['', f"Tasks: {g['n_tasks_planned']} in the panel, {g['n_tasks_after_a0']} after A0, {g['n_tasks_used']} used; "
              f"repeats {g['repeats']} per arm and task.",
          f"Repository-equal estimate: {_f(core.get('repo_equal'))}. Sum of task gains: {_f(core.get('sum_gain'))}.", '']
    if bt:
        L += [f"Bootstrap: B = {bt['n_boot']}, seed {g['thresholds']['seed']}, G = {bt['n_clusters']} repositories, "
              f"{bt['n_clusters_nonzero']} with a nonzero total (smallest attainable exact p {bt['min_attainable_p_exact']:.3g}); "
              f"CR1 se = {_f(bt['se'])}; exact enumeration p = {_f(bt.get('p_exact_enumeration'))} (diagnostic)."]
        for name, s in core.get('sensitivity', {}).items():
            L.append(f"Sensitivity (not the decision) {name}: p = {_f(s['p'])}, t = {_f(s['t'])}.")
        L.append('')
    s = g['slots']
    L += ['## Slots and attempts (A1, A2)', '',
          f"Planned slots {s['n_planned']}; with a used row {s['n_with_used_row']}; without one {len(s['lacking'])}. "
          f"Rows in the plan {s['rows_in_plan']}; rows per slot {s['attempts_hist']}; rows without an attempt index "
          f"{s['rows_without_attempt']}; without a job id {s['rows_without_job_id']}."]
    if s['lacking']:
        L.append(f"- Slots without a valid row (first 40): {[(x['arm'], x['task'], x['repeat'], x['reason']) for x in s['lacking'][:40]]}")
    if s['deviations']:
        L.append(f"- Attempt deviations (first 20): {s['deviations'][:20]}")
    if g['excluded_tasks']:
        L.append(f"- Tasks without a used row in one arm (provisional statistics exclude them): "
                 f"{[e['task'] for e in g['excluded_tasks']][:40]}; sensitivity {g['excluded_sensitivity']}")
    L += ['', '## Infrastructure-error rows by arm (all attempts)', '']
    for a, v in g['harness']['by_arm'].items():
        L.append(f"- {a}: {v['rows']} rows; by table id {v['by_id']}; by attempt {v['by_attempt']}; details {v['details']}")
    sus = g['harness']['silent_verification_suspects']
    if any(sus.values()):
        L.append(f"- Diagnostic: used unsolved rows with an unusual pytest exit code: "
                 f"{ {a: [(x['task'], x['repeat'], x['test_exit']) for x in v][:20] for a, v in sus.items()} }")
    pc = g['provenance_check']
    L += ['', f"## Provenance (A3): {'all rows agree' if pc['ok'] else 'PROBLEMS'}", '']
    for m in pc['missing']:
        L.append(f"- missing {m['field']} in {m['rows']} row(s) of {m['arm']} (e.g. {m['example']})")
    for d in pc['disagreements']:
        L.append(f"- disagreement in {d['field']} ({d['scope']}): {d['values']}")
    for k, v in pc['values'].items():
        L.append(f"- {k}: {v}")
    a0 = g['a0']
    L += ['', '## A0 drops', '',
          f"{a0['n_dropped']} of {a0['n_panel']} panel tasks dropped ({_f(a0['fraction'])}); "
          f"{a0.get('drop_file') or 'no drop file'} (sha256 {a0.get('drop_file_sha256') or 'n/a'}); dropped: {a0['dropped_in_panel']}",
          '', f"## Ignored rows: {g['data']['ignored_rows']}", '']
    if 'secondary' in g:
        L.append(md_secondary(g['secondary']))
    return '\n'.join(L)


def md_screen_v2(s):
    L = [f"# E7 screening on panel {s['panel']} (rules v2; exploratory)", '',
         f"**{s['decision']}**. Baseline {s['baseline']}; {s['n_tasks']} tasks (of {s['n_tasks_planned']}) x "
         f"{s['repeats']} repeats planned per arm. Frozen settings: {_f(s['frozen_settings'])}.", '']
    for t in s['tie_log'] + s['notes']:
        L.append(f'> {t}')
    L += ['', '| rank | arm | gain (sum over tasks) | tasks used | mean gain/task | repos +/- | edit-call success | mean agent time (s) |',
          '|---|---|---|---|---|---|---|---|']
    rows = {r['arm']: r for r in s['arms_table']}
    for i, a in enumerate(s['ranking'], 1):
        r = rows[a]
        L.append(f"| {i} | {a} | {_f(r['gain'], 2)} ({r['gain_exact']}) | {r['tasks_used']} | {_f(r['mean_gain_per_task'])} | "
                 f"{r['repos_positive']}/{r['repos_negative']} | {_f(r['edit_success_float'])} ({r['edit_success']}) | {_f(r['mean_agent_time'], 1)} |")
    L += ['', '| arm | planned | used | lacking | infrastructure-error rows (all attempts) | solves | outcomes | traces found/missing | writes (edit_file, write_file, shell) [attempts, failed] |',
          '|---|---|---|---|---|---|---|---|---|']
    for a, v in s['run_summary'].items():
        L.append(f"| {a} | {v['planned']} | {v['used']} | {v['lacking']} | {v['infra_rows']} {v['infra_by_id']} | {v['solves']} | "
                 f"{v['outcome_counts']} | {v['traces_found']}/{v['traces_missing']} | {v['edit_channels']} |")
    sl = s['slots']
    L += ['', f"Slots: planned {sl['n_planned']}, with a used row {sl['n_with_used_row']}, without {len(sl['lacking'])}; "
              f"rows per slot {sl['attempts_hist']}; ignored rows {s['data']['ignored_rows']}."]
    if sl['lacking']:
        L.append(f"Slots without a valid row (first 40): {[(x['arm'], x['task'], x['repeat'], x['reason']) for x in sl['lacking'][:40]]}")
    pc = s['provenance_check']
    L.append(f"Provenance (A3, diagnostic for screening): {'all rows agree' if pc['ok'] else 'PROBLEMS'}; missing "
             f"{[(m['field'], m['arm'], m['rows']) for m in pc['missing']]}; disagreements {pc['disagreements']}")
    L.append(f"A0: {s['a0']['n_dropped']} of {s['a0']['n_panel']} tasks dropped: {s['a0']['dropped_in_panel']}")
    for r in s['arms_table']:
        if r['tasks_excluded']:
            L.append(f"{r['arm']}: tasks excluded from the gain (no valid row in one arm): {r['tasks_excluded']}")
    return '\n'.join(L) + '\n'


def md_stop(res):
    L = [f"# E7 {res['command']} (rules v2): STOPPED, no decision", '', f"**{res['reason']}**", '']
    for d in res['details'][:60]:
        L.append(f'- {d}')
    return '\n'.join(L) + '\n'


def md_reruns(r):
    L = [f"# E7 re-run list (rules v2), block {r['block']}", '',
         f"{len(r['jobs'])} job(s) to run (same arm, task and repeat; next attempt index); {len(r['exhausted'])} slot(s) "
         f"exhausted after {MAX_ATTEMPTS} attempts (the gate stays INCONCLUSIVE). Planned slots {r['n_planned']}, with a "
         f"used row {r['n_with_used_row']}.", '']
    for j in r['jobs'][:200]:
        L.append(f"- {j['job_id']}")
    if r['exhausted']:
        L += ['', f"Exhausted: {[(x['arm'], x['task'], x['repeat']) for x in r['exhausted']]}"]
    return '\n'.join(L) + '\n'


def _provenance(args):
    p = os.path.join(HERE, 'PROTOCOL.md')
    sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest() if os.path.exists(f) else None
    return dict(analysis=ANALYSIS_ID, analysis_sha256=sha(os.path.abspath(__file__)), protocol_sha256=sha(p),
                numpy=np.__version__, python=sys.version.split()[0], argv=sys.argv, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))


def _write(prefix, obj, md):
    os.makedirs(os.path.dirname(os.path.abspath(prefix)) or '.', exist_ok=True)
    with open(prefix + '.json', 'w', encoding='utf-8') as f:
        json.dump(_jsonable(obj), f, indent=1, sort_keys=False)
    with open(prefix + '.md', 'w', encoding='utf-8') as f:
        f.write(md)


def _provenance_v2(args, extra=None):
    p = _provenance(args)
    p.update(analysis=ANALYSIS_ID_V2, rules='v2', error_table_sha256=ERROR_TABLE_SHA256, guard_h_adopted=GUARD_H_ADOPTED,
             amendments_sha256=(hashlib.sha256(open(os.path.join(HERE, 'PROTOCOL_AMENDMENTS.md'), 'rb').read()).hexdigest()
                                if os.path.exists(os.path.join(HERE, 'PROTOCOL_AMENDMENTS.md')) else None))
    if extra:
        p.update(extra)
    return p


def _md_v21_common_sections(g):
    """Sections shared by the blind and the full v2.1 gate reports (no outcome information)."""
    L = []
    s = g['slots']
    L += ['## Slots and attempts (A1, A2, A8)', '',
          f"Planned slots {s['n_planned']}; with a used row {s['n_with_used_row']}; without one {len(s['lacking'])}. "
          f"Rows in the plan {s['rows_in_plan']}; rows per slot {s['attempts_hist']}; rows without an attempt index "
          f"{s['rows_without_attempt']}; without a job id {s['rows_without_job_id']}."]
    if s['lacking']:
        L.append(f"- Slots without a valid row (first 40): "
                 f"{[(x['arm'], x['task'], x['repeat'], x['attempts'], x['classes']) for x in s['lacking'][:40]]}")
    if s['gaps']:
        L.append(f"- Attempt gaps (A8: INCONCLUSIVE; first 20): {[(x['slot'], x['attempts']) for x in s['gaps'][:20]]}")
    if s['deviations']:
        L.append(f"- Attempt deviations (first 20): {s['deviations'][:20]}")
    L += ['', '## Infrastructure-error rows by arm (all attempts)', '']
    for a, v in g['harness']['by_arm'].items():
        L.append(f"- {a}: {v['rows']} rows; by table id {v['by_id']}; by attempt {v['by_attempt']}; by batch "
                 f"{v['by_batch']}; details {v['details']}")
    pc = g['provenance_check']
    L += ['', f"## Provenance (A3): {'all rows agree' if pc['ok'] else 'PROBLEMS'}", '']
    for m in pc['missing']:
        L.append(f"- missing {m['field']} in {m['rows']} row(s) of {m['arm']} (e.g. {m['example']})")
    for d in pc['disagreements']:
        L.append(f"- disagreement in {d['field']} ({d['scope']}): {d['values']}")
    for k, v in pc['values'].items():
        L.append(f"- {k}: {v}")
    rf = g['required_fields']
    L += ['', f"## Required fields (RUN_RECORD_SPEC.md; A8): {'complete' if rf['ok'] else 'PROBLEMS'}", '']
    for m in rf['missing']:
        L.append(f"- missing {m['field']} ({m['scope']}) in {m['rows']} record(s) of {m['arm']} (e.g. {m['example']})")
    for m in rf['malformed'][:20]:
        L.append(f"- {m['problem']}: {m['ref']} ({m['job_id']}; expected {m['expected']})")
    st = g['started']
    L.append(f"- started records: directories without started.jsonl {st['no_started_file']}; rows without a started "
             f"record {st['n_rows_without_started']}; started records without job_id {st['n_started_without_job_id']}")
    man = g['manifest']
    L += ['', f"## Block manifest (A8): {man.get('path') or 'none'} (sha256 {(man.get('sha256') or 'n/a')[:16]})", '']
    for t in man.get('clearable', []):
        L.append(f'- input problem (clearable): {t}')
    for t in man.get('data', []):
        L.append(f'- record problem: {t}')
    for r in man.get('runs', []):
        L.append(f"- run {r['run_tag']}: given {_f(r['given'])}; jobs {r.get('n_jobs')}; deadline {r.get('deadline_s')} s")
    for t in man.get('notes', []):
        L.append(f'- note: {t}')
    if g.get('load_problems'):
        L += ['', f"## Load problems: {len(g['load_problems'])}", '']
        for p in g['load_problems'][:20]:
            L.append(f'- {p}')
    a0 = g['a0']
    L += ['', '## A0 drops (A8: checked on every panel and against the sha256 the rows carry)', '',
          f"{a0['n_dropped']} of {a0['n_panel']} panel tasks dropped ({_f(a0['fraction'])}); "
          f"{a0.get('drop_file') or 'no drop file'} (sha256 {a0.get('drop_file_sha256') or 'n/a'}); dropped: "
          f"{a0['dropped_in_panel']}; all panels: {a0.get('all_panels')}",
          '', f"## Ignored rows: {g['data']['ignored_rows']}", '']
    return L


def md_gate_v21(g):
    if g.get('blind'):
        return md_gate_v21_blind(g)
    c, core, gh = g['criteria'], g['gain'], g['guard_h']
    dp = g['d_pen']
    cpen, gpen = dp['gain'], dp['guard_h']
    bt, btp = core.get('bootstrap') or {}, cpen.get('bootstrap') or {}
    h = g['harness']['test_4b']
    pen_total = {r['repo']: r for r in gpen['table']}
    L = [f"# E7 gate (rules v2.1): {g['arm_b']} vs {g['arm_a']} on panel {g['panel']} (block {g['block']})", '',
         '## Per-repository differences (A4; sorted by total difference, most negative first)', '',
         '| repository | tasks | total difference | repository mean | leave-one-out estimate | total <= -2 | total under d_pen |',
         '|---|---|---|---|---|---|---|']
    for r in gh['table']:
        L.append(f"| {r['repo']} | {r['n_tasks']} | {_f(r['total'])} ({r['total_exact']}) | {_f(r['mean'])} | "
                 f"{_f(r.get('loo'))} | {_f(r['at_or_below_limit'])} | {_f(pen_total.get(r['repo'], {}).get('total'))} |")
    L += ['', f"Guard H ({gh['role']}): lowest repository total {_f(gh['min_total'])} ({gh['min_repo']}); "
              f"repositories at or below -2: {gh['repos_at_or_below'] or 'none'}. Under d_pen: lowest "
              f"{_f(gpen['min_total'])} ({gpen['min_repo']}).", '',
          f"## Decision: **{g['decision']}**", '',
          f"Rules v2.1 (PROTOCOL.md with PROTOCOL_AMENDMENTS.md A0-A8). Frozen settings: {_f(g['frozen_settings'])}. "
          'PASS needs (1)-(5) on d and (1), (2), (3), (5) on d_pen (A8).', '']
    for n in g['notes']:
        L.append(f'> {n}')
    hp = 'n/a (no infrastructure-error rows)' if h['p'] is None else f"{_f(h['p'])}"
    pc = lambda k: _f(c[k]) if k in c else 'n/a'
    L += ['', f"| criterion | d (rows used) | met | d_pen (candidate slots with a {'/'.join(g['d_pen_ids'])} row scored unsolved) | met |",
          '|---|---|---|---|---|',
          f"| (1) task-weighted mean gain >= 2/58 = 0.0345 | {_f(core.get('d'))} ({core.get('d_exact')}) | {pc('gain')} | "
          f"{_f(cpen.get('d'))} ({cpen.get('d_exact')}) | {pc('gain_pen')} |",
          f"| (2) bootstrap-t one-sided p <= 0.025 (Rademacher, null imposed) | {_f(bt.get('p'))} (t = {_f(bt.get('t'))}) | "
          f"{pc('p')} | {_f(btp.get('p'))} (t = {_f(btp.get('t'))}) | {pc('p_pen')} |",
          f"| (3a) repositories with positive gain >= 3 | {core.get('n_repos_positive')} of {core.get('n_repos')} | "
          f"{pc('pos_repos')} | {cpen.get('n_repos_positive')} of {cpen.get('n_repos')} | {pc('pos_repos_pen')} |",
          f"| (3b) every leave-one-repository-out estimate > 0 | min {_f(core.get('loo_min'))} | {pc('loo')} | "
          f"min {_f(cpen.get('loo_min'))} | {pc('loo_pen')} |",
          f"| (5) guard H: lowest repository total > -2 | {_f(gh['min_total'])} | {pc('guard_h')} | {_f(gpen['min_total'])} | "
          f"{pc('guard_h_pen')} |",
          f"| (4a) mean agent time ratio <= 1.2, rows used | {_f(g['time']['ratio'])} ({_f(g['time']['mean_b'], 1)} s / "
          f"{_f(g['time']['mean_a'], 1)} s) | {pc('time')} | n/a | |",
          f"| (4b) infrastructure-error rows, all attempts, stratified by job batch: {g['arm_b']} {h['k_candidate']} of "
          f"{h['n']} | exact p = {hp} (> 0.05) | {pc('harness')} | n/a | |",
          '', f"Tasks: {g['n_tasks_planned']} in the panel, {g['n_tasks_after_a0']} after A0, {g['n_tasks_used']} used; "
              f"repeats {g['repeats']} per arm and task.",
          f"Repository-equal estimate: {_f(core.get('repo_equal'))}. Sum of task gains: {_f(core.get('sum_gain'))}.", '']
    if bt:
        L += [f"Bootstrap: B = {bt['n_boot']}, seed {g['thresholds']['seed']}, G = {bt['n_clusters']} repositories, "
              f"{bt['n_clusters_nonzero']} with a nonzero total (smallest attainable exact p {bt['min_attainable_p_exact']:.3g}); "
              f"CR1 se = {_f(bt['se'])}; exact enumeration p = {_f(bt.get('p_exact_enumeration'))} (diagnostic); "
              f"d_pen exact enumeration p = {_f(btp.get('p_exact_enumeration'))}."]
        for name, s in core.get('sensitivity', {}).items():
            L.append(f"Sensitivity (not the decision) {name}: p = {_f(s['p'])}, t = {_f(s['t'])}.")
        L.append('')
    L += ['## (4b) by job batch (A8: each batch is tested against its own job shares)', '',
          f"| batch (run tag) | jobs {g['arm_a']} | jobs {g['arm_b']} | candidate share | infrastructure rows {g['arm_a']} | "
          f"infrastructure rows {g['arm_b']} |", '|---|---|---|---|---|---|']
    for s in h['strata']:
        L.append(f"| {s['run_tag']} | {s['jobs_baseline']} | {s['jobs_candidate']} | {s['share_candidate']} | "
                 f"{s['k_baseline']} | {s['k_candidate']} |")
    L += ['', f"## d_pen (A8): {dp['n_penalised']} candidate slot(s) scored unsolved "
              f"({dp['n_penalised_solved']} of them had a solved used row)", '']
    if dp['penalised_slots']:
        L.append(f"- {[(x['task'], x['repeat'], x['ids'], x['used_row_solved']) for x in dp['penalised_slots'][:40]]}")
        L.append('')
    L += _md_v21_common_sections(g)
    sus = g['harness']['silent_verification_suspects']
    if any(sus.values()):
        L.append(f"Diagnostic: used unsolved rows with an unusual pytest exit code: "
                 f"{ {a: [(x['task'], x['repeat'], x['test_exit']) for x in v][:20] for a, v in sus.items()} }")
        L.append('')
    if 'secondary' in g:
        L.append(md_secondary(g['secondary']))
    return '\n'.join(L)


def md_gate_v21_blind(g):
    L = [f"# E7 gate (rules v2.1): {g['arm_b']} vs {g['arm_a']} on panel {g['panel']} (block {g['block']}): BLIND REPORT", '',
         f"## Decision: **{g['decision']}** ({g['decision_basis']})", '',
         f"{g['blind_note']}: no d, p, per-repository table, guard H, time ratio, (4b) p or secondary metrics.", '']
    for r in g['reasons']:
        L.append(f"> {'INPUT (clearable)' if r['clearable'] else 'INCONCLUSIVE'} [{r['kind']}]: {r['text']}")
    L += ['', f"**Next step (A8, pre-registered):** {g['next_step']}", '']
    for n in g['notes']:
        L.append(f'> {n}')
    L.append('')
    L += _md_v21_common_sections(g)
    return '\n'.join(L)


def md_screen_v21(s):
    L = [f"# E7 screening on panel {s['panel']} (rules v2.1; exploratory)", '',
         f"**{s['decision']}**. Baseline {s['baseline']}; {s['n_tasks']} tasks (of {s['n_tasks_planned']}) x "
         f"{s['repeats']} repeats planned per arm. Frozen settings: {_f(s['frozen_settings'])}.", '']
    if s.get('next_step'):
        L += [f"**Next step:** {s['next_step']}", '']
    for t in s['tie_log'] + s['notes']:
        L.append(f'> {t}')
    L += ['', '| rank | arm | gain (sum over tasks) | tasks used | mean gain/task | repos +/- | edit-call success | mean agent time (s) |',
          '|---|---|---|---|---|---|---|---|']
    rows = {r['arm']: r for r in s['arms_table']}
    for i, a in enumerate(s['ranking'], 1):
        r = rows[a]
        L.append(f"| {i} | {a} | {_f(r['gain'], 2)} ({r['gain_exact']}) | {r['tasks_used']} | {_f(r['mean_gain_per_task'])} | "
                 f"{r['repos_positive']}/{r['repos_negative']} | {_f(r['edit_success_float'])} ({r['edit_success']}) | {_f(r['mean_agent_time'], 1)} |")
    L += ['', '| arm | planned | used | lacking | infrastructure-error rows (all attempts) | solves | outcomes | traces found/missing | writes (edit_file, write_file, shell) [attempts, failed] |',
          '|---|---|---|---|---|---|---|---|---|']
    for a, v in s['run_summary'].items():
        L.append(f"| {a} | {v['planned']} | {v['used']} | {v['lacking']} | {v['infra_rows']} {v['infra_by_id']} | {v['solves']} | "
                 f"{v['outcome_counts']} | {v['traces_found']}/{v['traces_missing']} | {v['edit_channels']} |")
    sl = s['slots']
    L += ['', f"Slots: planned {sl['n_planned']}, with a used row {sl['n_with_used_row']}, without {len(sl['lacking'])}; "
              f"rows per slot {sl['attempts_hist']}; attempt gaps {len(sl['gaps'])}; ignored rows {s['data']['ignored_rows']}."]
    if sl['lacking']:
        L.append(f"Slots without a valid row (first 40): {[(x['arm'], x['task'], x['repeat'], x['reason']) for x in sl['lacking'][:40]]}")
    pc = s['provenance_check']
    L.append(f"Provenance (A3, diagnostic for screening): {'all rows agree' if pc['ok'] else 'PROBLEMS'}; missing "
             f"{[(m['field'], m['arm'], m['rows']) for m in pc['missing']]}; disagreements {pc['disagreements']}")
    rf = s['required_fields']
    L.append(f"Required fields: {'complete' if rf['ok'] else 'PROBLEMS'} {[(m['field'], m['scope'], m['arm'], m['rows']) for m in rf['missing']][:20]}")
    man = s['manifest']
    L.append(f"Block manifest: {man.get('path') or 'none'}; input problems {man.get('clearable')}; record problems {man.get('data')}")
    L.append(f"A0: {s['a0']['n_dropped']} of {s['a0']['n_panel']} tasks dropped: {s['a0']['dropped_in_panel']}; drop-file "
             f"sha256 {s['a0'].get('drop_file_sha256')}")
    for r in s['arms_table']:
        if r['tasks_excluded']:
            L.append(f"{r['arm']}: tasks excluded from the gain (no valid row in one arm): {r['tasks_excluded']}")
    return '\n'.join(L) + '\n'


def md_stop_v21(res):
    L = [f"# E7 {res['command']} (rules v2.1): STOPPED, no decision", '', f"**{res['reason']}**", '',
         'A8: a STOP report is blind (slots, records and job ids only).', '']
    for d in res['details'][:60]:
        L.append(f'- {d}')
    return '\n'.join(L) + '\n'


def md_reruns_v21(r):
    L = [f"# E7 next batch (rules v2.1), block {r['block']}: {r['batch_kind']} batch", '',
         f"{len(r['jobs'])} job(s) to run; {len(r['waiting_for_extra_batch'])} slot(s) wait for the extra batch; "
         f"{len(r['exhausted'])} slot(s) exhausted after {MAX_ATTEMPTS_V21} attempts (the gate becomes a FAIL if any "
         f"is left after the extra batch). Planned slots {r['n_planned']}, with a used row {r['n_with_used_row']}.", '',
         r['note'], '']
    for j in r['jobs'][:200]:
        L.append(f"- {j['job_id']}")
    if r['exhausted']:
        L += ['', f"Exhausted: {[(x['arm'], x['task'], x['repeat']) for x in r['exhausted']]}"]
    if r['manifest_data_problems']:
        L += ['', f"Manifest record problems (they keep the gate INCONCLUSIVE): {r['manifest_data_problems']}"]
    return '\n'.join(L) + '\n'


def md_parity_v21(p):
    return '\n'.join([f"# LoRA-server parity (A8, before the D block): {p['arm']}", '',
                      f"Mean agent time {_f(p['mean_lora'], 1)} s on the LoRA-enabled server ({p['n_lora']} rows) vs "
                      f"{_f(p['mean_plain'], 1)} s on the plain server ({p['n_plain']} rows): ratio {_f(p['ratio'])} "
                      f"(limit {p['limit']}). Investigate before D: {_f(p['investigate'])}.", '', p['note'], '']) + '\n'


def _provenance_v21(args, extra=None):
    p = _provenance_v2(args)
    p.update(analysis=ANALYSIS_ID_V21, rules='v2.1', error_table_sha256=ERROR_TABLE_SHA256_V21, d_pen_ids=list(D_PEN_IDS),
             max_attempts=MAX_ATTEMPTS_V21)
    if extra:
        p.update(extra)
    return p


_REASON_LABEL_V22 = {'input': 'INPUT (clearable)', 'record': 'RECORD (void)', 'slots': 'SLOTS'}


def _md_reasons_v22(reasons):
    return [f"> {_REASON_LABEL_V22.get(r.get('category'), 'REASON')} [{r['kind']}]: {r['text']}" for r in reasons]


def _md_v22_sections(g):
    """A9 sections shared by every v2.2 report (no outcome information)."""
    L = []
    ss = g.get('server_settings') or {}
    L += ['## Server settings (A9: max_model_len and the LoRA settings of every run in the block manifest)', '',
          f"{'complete and consistent' if ss.get('ok') else 'PROBLEMS'}; values {ss.get('values')}"]
    for p in ss.get('problems', []):
        L.append(f'- record problem: {p}')
    for n in ss.get('notes', []):
        L.append(f'- note: {n}')
    jf = (g.get('a0') or {}).get('job_files') or {}
    L += ['', (f"A0 job files (A9): {jf.get('n_job_entries')} job entries of the batch job lists and "
               f"{jf.get('n_started_records')} started records checked against the drop-file sha256" if jf.get('checked')
               else f"A0 job files (A9): {jf.get('note')}")]
    vc = g.get('void_chain') or {}
    L += ['', f"## Block id (A9): {vc.get('block')} (panel {vc.get('panel')}); replaces {vc.get('replaces')}; "
              f"void report {vc.get('void_report')}", '']
    for p in vc.get('problems', []):
        L.append(f'- input problem: {p}')
    st = g.get('started') or {}
    if st.get('n_duplicate_job_ids'):
        L.append(f"- started records repeating a job_id of their run directory: {st['n_duplicate_job_ids']} "
                 f"(e.g. {st['duplicate_job_ids'][:5]})")
    sl = g.get('slots') or {}
    if 'infra_class_mismatches' in sl:
        m = sl['infra_class_mismatches']
        L.append(f"- recorded vs computed error class, infrastructure-involved only (A9; agent ids masked): {m['n']} {m['by_kind']}")
    L.append('')
    return L


def md_gate_v22(g):
    if g.get('blind'):
        return md_gate_v22_blind(g)
    md = md_gate_v21({k: v for k, v in g.items() if k != 'secondary'})
    md = md.replace('# E7 gate (rules v2.1):', '# E7 gate (rules v2.2):', 1)
    md = md.replace('Rules v2.1 (PROTOCOL.md with PROTOCOL_AMENDMENTS.md A0-A8)',
                    'Rules v2.2 (PROTOCOL.md with PROTOCOL_AMENDMENTS.md A0-A9)', 1)
    L = [md.rstrip('\n'), ''] + _md_v22_sections(g)
    if 'secondary' in g:
        L.append(md_secondary(g['secondary']))
    return '\n'.join(L)


def md_gate_v22_blind(g):
    L = [f"# E7 gate (rules v2.2): {g['arm_b']} vs {g['arm_a']} on panel {g['panel']} (block {g['block']}): BLIND REPORT", '',
         f"## Decision: **{g['decision']}** ({g['decision_basis']})", '',
         f"{g['blind_note']}: no d, p, per-repository table, guard H, time ratio, (4b) p or secondary metrics.", '']
    L += _md_reasons_v22(g['reasons'])
    L += ['', f"**Next step (A8/A9, pre-registered):** {g['next_step']}", '']
    for n in g['notes']:
        L.append(f'> {n}')
    L.append('')
    L += _md_v21_common_sections(g)
    L += _md_v22_sections(g)
    return '\n'.join(L)


def _md_blind_body_v22(s):
    L = []
    sl = s['slots']
    L.append(f"Slots: planned {sl['n_planned']}, with a used row {sl['n_with_used_row']}, without {len(sl['lacking'])}; "
             f"rows per slot {sl['attempts_hist']}; attempt gaps {len(sl['gaps'])}; ignored rows {s['data']['ignored_rows']}.")
    if sl['lacking']:
        L.append(f"Slots without a valid row (first 40): "
                 f"{[(x['arm'], x['task'], x['repeat'], x['attempts'], x['classes']) for x in sl['lacking'][:40]]}")
    for a, v in s['harness']['by_arm'].items():
        L.append(f"Infrastructure-error rows of {a}: {v['rows']}; by table id {v['by_id']}; by batch {v['by_batch']}")
    pc = s['provenance_check']
    L.append(f"Provenance (A3): {'all rows agree' if pc['ok'] else 'PROBLEMS'}; missing "
             f"{[(m['field'], m['arm'], m['rows']) for m in pc['missing']]}; disagreements {pc['disagreements']}")
    rf = s['required_fields']
    L.append(f"Required fields: {'complete' if rf['ok'] else 'PROBLEMS'} "
             f"{[(m['field'], m['scope'], m['arm'], m['rows']) for m in rf['missing']][:20]}; job_id mismatches {rf['n_malformed']}")
    st = s['started']
    L.append(f"Started records: directories without started.jsonl {st['no_started_file']}; rows without a started record "
             f"{st['n_rows_without_started']}; started records without job_id {st['n_started_without_job_id']}")
    man = s['manifest']
    L.append(f"Block manifest: {man.get('path') or 'none'}; input problems {man.get('clearable')}; record problems {man.get('data')}")
    L.append(f"A0: {s['a0']['n_dropped']} of {s['a0']['n_panel']} tasks dropped: {s['a0']['dropped_in_panel']}; drop-file "
             f"sha256 {s['a0'].get('drop_file_sha256')}")
    if s.get('load_problems'):
        L.append(f"Load problems: {len(s['load_problems'])}: {s['load_problems'][:10]}")
    L.append('')
    return L


def md_screen_v22(s):
    if s.get('blind'):
        L = [f"# E7 screening on panel {s['panel']} (rules v2.2; exploratory; block {s['block']}): BLIND REPORT", '',
             f"**{s['decision']}** ({s['decision_basis']}). {s['blind_note']}.", '']
        L += _md_reasons_v22(s['reasons'])
        L += ['', f"**Next step:** {s['next_step']}", '']
        for n in s['notes']:
            L.append(f'> {n}')
        L.append('')
        L += _md_blind_body_v22(s)
        L += _md_v22_sections(s)
        return '\n'.join(L)
    md = md_screen_v21(s).replace('(rules v2.1; exploratory)', '(rules v2.2; exploratory)', 1)
    return md.rstrip('\n') + '\n\n' + '\n'.join(_md_v22_sections(s)) + '\n'


def md_records_v22(r):
    L = [f"# E7 record check (rules v2.2, A9; blind) of block {r['block']} on panel {r['panel']}: **{r['decision']}**", '',
         f"Arms {r['arms_analysed']}; repeats {r['repeats']}; frozen settings {_f(r['frozen_settings'])}. {r['note']}.", '']
    L += _md_reasons_v22(r['reasons'])
    if r.get('next_step'):
        L += ['', f"**Next step:** {r['next_step']}"]
    if r.get('slot_step'):
        L += ['', f"Slots: {r['slot_step']}"]
    L.append('')
    for n in r['notes']:
        L.append(f'> {n}')
    L.append('')
    L += _md_blind_body_v22(r)
    L += _md_v22_sections(r)
    return '\n'.join(L)


def md_stop_v22(res):
    L = [f"# E7 {res['command']} (rules v2.2): STOPPED, no decision", '', f"**{res['reason']}**", '',
         'A8/A9: a STOP report is blind (slots, records and job ids only).', '']
    for d in res['details'][:60]:
        L.append(f'- {d}')
    return '\n'.join(L) + '\n'


def md_reruns_v22(r):
    if r.get('decision') == 'VOID':
        L = [f"# E7 next batch (rules v2.2), block {r['block']}: none, the block is VOID", '', r['note'], '']
        L += _md_reasons_v22(r['reasons'])
        L += ['', f"**Next step:** {r['next_step']}", '']
        return '\n'.join(L) + '\n'
    L = [f"# E7 next batch (rules v2.2), block {r['block']}: {r['batch_kind']} batch", '',
         f"{len(r['jobs'])} job(s) to run; {len(r['waiting_for_extra_batch'])} slot(s) wait for the extra batch; "
         f"{len(r['exhausted'])} slot(s) exhausted after {MAX_ATTEMPTS_V21} attempts (the gate becomes a FAIL if any "
         f"is left after the extra batch). Planned slots {r['n_planned']}, with a used row {r['n_with_used_row']}.", '',
         r['note'], '']
    for j in r['jobs'][:200]:
        L.append(f"- {j['job_id']}")
    if r['exhausted']:
        L += ['', f"Exhausted: {[(x['arm'], x['task'], x['repeat']) for x in r['exhausted']]}"]
    return '\n'.join(L) + '\n'


def md_parity_v22(p):
    L = [f"# LoRA-server parity (A8/A9, before the D block): {p['arm']}", '',
         f"Paired by task (A9) on {p['n_tasks_common']} common task(s): mean of the task means {_f(p['mean_lora'], 1)} s on "
         f"the LoRA-enabled server vs {_f(p['mean_plain'], 1)} s on the plain server: ratio {_f(p['ratio'])} "
         f"({p['ratio_exact']}; limit {p['limit']}). Investigate before D: {_f(p['investigate'])}. Unpaired ratio "
         f"(v2.1, diagnostic): {_f(p['ratio_unpaired'])}. Rows: {p['n_lora']} LoRA-server, {p['n_plain']} plain.", '']
    for w in p['warnings']:
        L.append(f'> WARNING: {w}')
    L += ['', p['method'] + '.', p['note'], '']
    return '\n'.join(L) + '\n'


def _provenance_v22(args, extra=None):
    p = _provenance_v2(args)
    p.update(analysis=ANALYSIS_ID_V22, rules='v2.2', error_table_sha256=ERROR_TABLE_SHA256_V22, d_pen_ids=list(D_PEN_IDS_V22),
             max_attempts=MAX_ATTEMPTS_V21)
    if extra:
        p.update(extra)
    return p


def _main_v22(a, ap):
    """CLI under rules v2.2 (A0-A9)."""
    if not a.drop_file:
        ap.error('rules v2.2 need --drop-file (A8: the A0 list frozen before the block; an empty list if nothing is dropped)')
    if a.cmd in ('gate', 'secondary', 'reruns', 'records') and not a.manifest:
        ap.error('rules v2.2 need --manifest for gate, secondary, reruns and records (A8: the block manifest written by the '
                 'runner)')
    panel = load_panel(a.panel)
    task_repo = load_task_repos(a.tasks, needed=set(panel['tasks']))
    check_panel_repos(panel, task_repo)
    if a.cmd == 'records' and not a.arms and panel.get('panel') != 'S':
        ap.error('records needs --arms (the arms of the block, e.g. bundle0 bundle0_adapter_A); panel S defaults to the six frozen arms')
    dropped, drop_sha = load_drop_file(a.drop_file)
    drop_source = dict(drop_file=os.path.abspath(a.drop_file), drop_file_sha256=drop_sha, n_dropped_in_file=len(dropped))
    manifest = load_manifest_v21(a.manifest) if a.manifest else None
    void_report = load_void_report_v22(a.void_report) if a.void_report else None
    load = load_runs_v21(a.runs, arm_from_repeat=a.arm_from_repeat, arm_tag_from_dir=a.arm_tag_from_dir, dedupe_started=True)
    store = TraceStore()
    common = dict(dropped=dropped, drop_sha256=drop_sha, drop_source=drop_source, block=a.block, load=load,
                  manifest=manifest, void_report=void_report)
    rc = 0
    try:
        if a.cmd == 'screen':
            res = screening_v22(load['runs'], panel, task_repo, baseline=a.baseline, arms=a.arms, repeats=a.repeats,
                                traces=store, **common)
            md = md_screen_v22(res)
        elif a.cmd == 'gate':
            res = gate_v22(load['runs'], panel, task_repo, arm_a=a.arm_a, arm_b=a.arm_b, repeats=a.repeats, n_boot=a.draws,
                           seed=a.seed, traces=store, with_secondary=not a.no_secondary, **common)
            md = md_gate_v22(res)
        elif a.cmd == 'reruns':
            res = rerun_jobs_v22(load['runs'], panel, a.arms, repeats=a.repeats, **common)
            md = md_reruns_v22(res)
            os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
            with open(a.out + '.jobs.json', 'w', encoding='utf-8') as f:
                json.dump(res['jobs'], f)
        elif a.cmd == 'records':
            res = records_v22(load['runs'], panel, a.arms or SCREEN_ARMS, repeats=a.repeats, **common)
            md = md_records_v22(res)
            rc = 0 if res['decision'] == 'RECORDS COMPLETE' else 3
        else:
            g = gate_v22(load['runs'], panel, task_repo, arm_a=a.arm_a, arm_b=a.arm_b, repeats=a.repeats, traces=store,
                         with_secondary=True, **common)
            if g['blind']:
                res = dict(kind='secondary', rules='v2.2', blind=True, decision=g['decision'],
                           reasons=g['inconclusive_reasons'],
                           note='A8/A9: no secondary metrics while the gate is INCONCLUSIVE, VOID or a FAIL by INCONCLUSIVE')
                md = (f"# E7 secondary metrics (rules v2.2): BLIND\n\nThe gate is {g['decision']} ({g['decision_basis']}); "
                      'A8/A9 withhold secondary metrics until its reasons are cleared.\n')
            else:
                res = dict(g['secondary'], rules='v2.2', blind=False, gate_decision=g['decision'])
                md = md_secondary(res)
    except AnalysisStop as e:
        res = dict(kind='error', rules='v2.2', command=a.cmd, decision='STOPPED', reason=str(e), blind=True,
                   details=[_jsonable(d) if isinstance(d, dict) else str(d) for d in e.details][:500])
        md = md_stop_v22(res)
        rc = 2
    res['load_problems'] = load['problems']
    res['load_notes'] = load['notes']
    res['run_dirs'] = [os.path.abspath(d) for d in a.runs]
    res['provenance'] = _provenance_v22(a, dict(manifest_sha256=(manifest or {}).get('sha256'),
                                                void_report_sha256=(void_report or {}).get('sha256')))
    if a.label:
        res['label'] = a.label
        md = f'**{a.label}**\n\n' + md
    _write(a.out, res, md)
    print(json.dumps({k: res.get(k) for k in ('kind', 'rules', 'decision', 'finalist', 'panel')}, default=str))
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('screen', 'gate', 'secondary', 'reruns', 'records'):
        p = sub.add_parser(name)
        p.add_argument('--runs', nargs='+', required=True, help='run directories (with results.jsonl or out/results.jsonl)')
        p.add_argument('--panel', required=True)
        p.add_argument('--tasks', default=DEFAULT_TASKS)
        p.add_argument('--repeats', type=int, default=None)
        p.add_argument('--out', required=True, help='output prefix; writes <out>.json and <out>.md')
        p.add_argument('--arm-from-repeat', action='store_true', help='smoke tests only: repeat k of arm X -> arm X#rk')
        p.add_argument('--arm-tag-from-dir', action='store_true', help='smoke tests only: arm X of run dir L -> arm X@L')
        p.add_argument('--label', default='', help='free text added to the report title (e.g. SMOKE TEST)')
        p.add_argument('--rules', choices=('v1', 'v2', 'v2.1', 'v2.2'), default=None,
                       help='decision rules: v2.2 (default; PROTOCOL_AMENDMENTS.md A0-A9), v2.1 (A0-A8), v2 (A0-A7) or v1 '
                            '(PROTOCOL.md as first implemented)')
        p.add_argument('--drop-file', default=None, help='v2/v2.1: A0 list of tasks dropped after the final Docker '
                                                         'revalidation (v2.1: required; its sha256 must match the rows)')
        p.add_argument('--block', default=None, help='v2/v2.1: block label the rows must carry (default: the panel name)')
        p.add_argument('--manifest', default=None, help='v2.1/v2.2: the block manifest written by the runner (A8; required '
                                                        'for gate, secondary, reruns and records)')
        p.add_argument('--void-report', default=None, help='v2.2: the VOID report of the block that this re-run block '
                                                           'replaces (A9; needed when --block is <panel>_v<n>)')
        if name == 'screen':
            p.add_argument('--baseline', default=BASELINE)
            p.add_argument('--arms', nargs='*', default=None, help='v2.1 default: the six frozen screening arms')
        elif name == 'reruns':
            p.add_argument('--arms', nargs='+', required=True, help='the arms of the block, e.g. bundle0 bundle0_adapter_A')
        elif name == 'records':
            p.add_argument('--arms', nargs='*', default=None, help='v2.2: the arms of the block (panel S: default the six '
                                                                   'frozen arms)')
        else:
            p.add_argument('--arm-a', default=BASELINE)
            p.add_argument('--arm-b', required=True)
        if name == 'gate':
            p.add_argument('--draws', type=int, default=N_BOOT)
            p.add_argument('--seed', type=int, default=BOOT_SEED)
            p.add_argument('--no-secondary', action='store_true')
    p = sub.add_parser('error-table', help='print the frozen error-class table (Markdown)')
    p.add_argument('--rules', choices=('v2', 'v2.1', 'v2.2'), default='v2.2')
    p = sub.add_parser('parity', help='v2.1/v2.2 (A8, before the D block): agent-time ratio, LoRA-enabled vs plain server '
                                      '(v2.2: paired by task, A9)')
    p.add_argument('--rules', choices=('v2.1', 'v2.2'), default=None)
    p.add_argument('--runs-plain', nargs='+', required=True)
    p.add_argument('--runs-lora', nargs='+', required=True)
    p.add_argument('--arm', default=BASELINE)
    p.add_argument('--out', required=True)
    p.add_argument('--label', default='')
    a = ap.parse_args(argv)
    if a.cmd == 'error-table':
        if a.rules == 'v2':
            print(f'A1 error-class table, sha256 {ERROR_TABLE_SHA256}\n')
            print(error_table_markdown())
        elif a.rules == 'v2.2':
            print(f'A1/A8/A9 error-class table (rules v2.2), sha256 {ERROR_TABLE_SHA256_V22}\n')
            print(error_table_markdown_v22())
            print(f'd_pen classes (A8, A9): {", ".join(D_PEN_IDS_V22)}; v2.1 -> v2.2 id changes: {ERROR_ID_MAP_V21_TO_V22}')
        else:
            print(f'A1/A8 error-class table (rules v2.1), sha256 {ERROR_TABLE_SHA256_V21}\n')
            print(error_table_markdown_v21())
            print(f'd_pen classes (A8): {", ".join(D_PEN_IDS)}; v2 -> v2.1 id changes: {ERROR_ID_MAP_V2_TO_V21}')
        return 0
    if a.cmd == 'parity':
        lp = load_runs_v21(a.runs_plain)
        ll = load_runs_v21(a.runs_lora)
        if (a.rules or DEFAULT_RULES) == 'v2.1':
            res = parity_ratio_v21(lp['runs'], ll['runs'], arm=a.arm)
            res['run_dirs'] = dict(plain=[os.path.abspath(d) for d in a.runs_plain], lora=[os.path.abspath(d) for d in a.runs_lora])
            res['provenance'] = _provenance_v21(a)
            md = md_parity_v21(res)
        else:
            res = parity_ratio_v22(lp['runs'], ll['runs'], arm=a.arm)
            res['run_dirs'] = dict(plain=[os.path.abspath(d) for d in a.runs_plain], lora=[os.path.abspath(d) for d in a.runs_lora])
            res['provenance'] = _provenance_v22(a)
            md = md_parity_v22(res)
        if a.label:
            res['label'] = a.label
            md = f'**{a.label}**\n\n' + md
        _write(a.out, res, md)
        print(json.dumps({k: res.get(k) for k in ('kind', 'ratio', 'investigate')}, default=str))
        return 0
    rules = a.rules or DEFAULT_RULES
    if rules != 'v2.2' and (a.cmd == 'records' or a.void_report):
        ap.error('records and --void-report exist only under --rules v2.2')
    if rules == 'v2.2':
        return _main_v22(a, ap)
    if rules == 'v1':
        if a.cmd == 'reruns' or a.drop_file or a.block or a.manifest:
            ap.error('reruns, --drop-file, --block and --manifest exist only under --rules v2 / v2.1')
        panel = load_panel(a.panel)
        task_repo = load_task_repos(a.tasks, needed=set(panel['tasks']))
        check_panel_repos(panel, task_repo)
        runs, problems = load_runs(a.runs, arm_from_repeat=a.arm_from_repeat, arm_tag_from_dir=a.arm_tag_from_dir)
        store = TraceStore()
        if a.cmd == 'screen':
            res = screening(runs, panel, task_repo, baseline=a.baseline, arms=a.arms, repeats=a.repeats, traces=store)
            md = md_screen(res)
        elif a.cmd == 'gate':
            res = gate(runs, panel, task_repo, arm_a=a.arm_a, arm_b=a.arm_b, repeats=a.repeats, n_boot=a.draws,
                       seed=a.seed, traces=store, with_secondary=not a.no_secondary)
            md = md_gate(res)
        else:
            res = secondary(runs, store, panel, task_repo, arm_a=a.arm_a, arm_b=a.arm_b, repeats=a.repeats)
            md = md_secondary(res)
        res['load_problems'] = problems
        res['run_dirs'] = [os.path.abspath(d) for d in a.runs]
        res['provenance'] = _provenance(a)
        if a.label:
            res['label'] = a.label
            md = f'**{a.label}**\n\n' + md
        _write(a.out, res, md)
        print(json.dumps({k: res.get(k) for k in ('kind', 'decision', 'finalist', 'panel')}, default=str))
        return 0
    if rules == 'v2':
        if a.manifest:
            ap.error('--manifest exists only under --rules v2.1')
        # ---------------------------------------------------------------- rules v2 (unchanged)
        panel = load_panel(a.panel)
        task_repo = load_task_repos(a.tasks, needed=set(panel['tasks']))
        check_panel_repos(panel, task_repo)
        dropped, drop_source = None, None
        if a.drop_file:
            dropped, sha = load_drop_file(a.drop_file)
            drop_source = dict(drop_file=os.path.abspath(a.drop_file), drop_file_sha256=sha, n_dropped_in_file=len(dropped))
        runs, problems, load_notes = load_runs_v2(a.runs, arm_from_repeat=a.arm_from_repeat, arm_tag_from_dir=a.arm_tag_from_dir)
        store = TraceStore()
        try:
            if a.cmd == 'screen':
                res = screening_v2(runs, panel, task_repo, baseline=a.baseline, arms=a.arms, repeats=a.repeats, traces=store,
                                   dropped=dropped, drop_source=drop_source, block=a.block, load_problems=problems)
                md = md_screen_v2(res)
            elif a.cmd == 'gate':
                res = gate_v2(runs, panel, task_repo, arm_a=a.arm_a, arm_b=a.arm_b, repeats=a.repeats, n_boot=a.draws,
                              seed=a.seed, traces=store, with_secondary=not a.no_secondary, dropped=dropped,
                              drop_source=drop_source, block=a.block, load_problems=problems)
                md = md_gate_v2(res)
            elif a.cmd == 'reruns':
                res = rerun_jobs_v2(runs, panel, a.arms, repeats=a.repeats, block=a.block, dropped=dropped)
                md = md_reruns(res)
                with open(a.out + '.jobs.json', 'w', encoding='utf-8') as f:
                    json.dump(res['jobs'], f)
            else:
                tasks, a0 = apply_drops(panel, dropped)
                repeats = a.repeats or DEFAULT_REPEATS.get(panel.get('panel'), 4)
                asm = assemble_v2(runs, tasks, [a.arm_a, a.arm_b], repeats, block=a.block or panel.get('panel'), dropped=dropped)
                res = secondary(runs, store, dict(panel=panel.get('panel'), tasks=tasks), task_repo, arm_a=a.arm_a,
                                arm_b=a.arm_b, repeats=repeats, asm=asm, status=_status_v2_metric)
                res['rules'] = 'v2'
                md = md_secondary(res)
            rc = 0
        except AnalysisStop as e:
            res = dict(kind='error', rules='v2', command=a.cmd, decision='STOPPED', reason=str(e),
                       details=[_jsonable(d) if isinstance(d, dict) else str(d) for d in e.details][:500])
            md = md_stop(res)
            rc = 2
        res['load_problems'] = problems
        res['load_notes'] = load_notes
        res['run_dirs'] = [os.path.abspath(d) for d in a.runs]
        res['provenance'] = _provenance_v2(a)
        if a.label:
            res['label'] = a.label
            md = f'**{a.label}**\n\n' + md
        _write(a.out, res, md)
        print(json.dumps({k: res.get(k) for k in ('kind', 'rules', 'decision', 'finalist', 'panel')}, default=str))
        return rc
    # ---------------------------------------------------------------- rules v2.1 (A0-A8)
    if not a.drop_file:
        ap.error('rules v2.1 need --drop-file (A8: the A0 list frozen before the block; an empty list if nothing is dropped)')
    if a.cmd in ('gate', 'secondary', 'reruns') and not a.manifest:
        ap.error('rules v2.1 need --manifest for gate, secondary and reruns (A8: the block manifest written by the runner)')
    panel = load_panel(a.panel)
    task_repo = load_task_repos(a.tasks, needed=set(panel['tasks']))
    check_panel_repos(panel, task_repo)
    dropped, drop_sha = load_drop_file(a.drop_file)
    drop_source = dict(drop_file=os.path.abspath(a.drop_file), drop_file_sha256=drop_sha, n_dropped_in_file=len(dropped))
    manifest = load_manifest_v21(a.manifest) if a.manifest else None
    load = load_runs_v21(a.runs, arm_from_repeat=a.arm_from_repeat, arm_tag_from_dir=a.arm_tag_from_dir)
    store = TraceStore()
    common = dict(dropped=dropped, drop_sha256=drop_sha, drop_source=drop_source, block=a.block, load=load,
                  manifest=manifest)
    try:
        if a.cmd == 'screen':
            res = screening_v21(load['runs'], panel, task_repo, baseline=a.baseline, arms=a.arms, repeats=a.repeats,
                                traces=store, **common)
            md = md_screen_v21(res)
        elif a.cmd == 'gate':
            res = gate_v21(load['runs'], panel, task_repo, arm_a=a.arm_a, arm_b=a.arm_b, repeats=a.repeats, n_boot=a.draws,
                           seed=a.seed, traces=store, with_secondary=not a.no_secondary, **common)
            md = md_gate_v21(res)
        elif a.cmd == 'reruns':
            res = rerun_jobs_v21(load['runs'], panel, a.arms, repeats=a.repeats, block=a.block, dropped=dropped,
                                 drop_sha256=drop_sha, load=load, manifest=manifest)
            md = md_reruns_v21(res)
            os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
            with open(a.out + '.jobs.json', 'w', encoding='utf-8') as f:
                json.dump(res['jobs'], f)
        else:
            g = gate_v21(load['runs'], panel, task_repo, arm_a=a.arm_a, arm_b=a.arm_b, repeats=a.repeats, traces=store,
                         with_secondary=True, **common)
            if g['blind']:
                res = dict(kind='secondary', rules='v2.1', blind=True, decision=g['decision'],
                           reasons=g['inconclusive_reasons'],
                           note='A8: no secondary metrics while the gate is INCONCLUSIVE (or a FAIL by INCONCLUSIVE)')
                md = (f"# E7 secondary metrics (rules v2.1): BLIND\n\nThe gate is {g['decision']} ({g['decision_basis']}); "
                      'A8 withholds secondary metrics until its reasons are cleared.\n')
            else:
                res = dict(g['secondary'], rules='v2.1', blind=False, gate_decision=g['decision'])
                md = md_secondary(res)
        rc = 0
    except AnalysisStop as e:
        res = dict(kind='error', rules='v2.1', command=a.cmd, decision='STOPPED', reason=str(e), blind=True,
                   details=[_jsonable(d) if isinstance(d, dict) else str(d) for d in e.details][:500])
        md = md_stop_v21(res)
        rc = 2
    res['load_problems'] = load['problems']
    res['load_notes'] = load['notes']
    res['run_dirs'] = [os.path.abspath(d) for d in a.runs]
    res['provenance'] = _provenance_v21(a, dict(manifest_sha256=(manifest or {}).get('sha256')))
    if a.label:
        res['label'] = a.label
        md = f'**{a.label}**\n\n' + md
    _write(a.out, res, md)
    print(json.dumps({k: res.get(k) for k in ('kind', 'rules', 'decision', 'finalist', 'panel')}, default=str))
    return rc

if __name__ == '__main__':
    sys.exit(main())
