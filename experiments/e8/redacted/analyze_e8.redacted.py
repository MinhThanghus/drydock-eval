#!/usr/bin/env python3
# Copyright 2026 The E8 authors
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
"""E8 analysis (PROTOCOL.md sections 8-12; LEAD_DECISIONS.md items 1, 3-5 and 8).

Subcommands (each writes <out>.json and <out>.md):
  report   Blind while any reason is open (slots open for re-runs before the re-run cut-off, a stage set that is still
           provisional, missing or inconsistent manipulation-check records, record, manifest or provenance problems,
           arm-configuration mismatches). Otherwise the unblinded analysis: for each trap arm X in R, C, H, T the
           task-weighted solve-rate difference d_X = X - F on the analysis set, the exact two-sided task-level sign-flip
           p, Holm over the four trap arms (the claim rule), the 99,999-draw Monte Carlo check, the inverted sign-flip
           95% and simultaneous (Bonferroni 98.75%) intervals, the repository-cluster secondary (analyze_e7's exact
           Rademacher wild cluster bootstrap-t, inverted; its Monte Carlo and Webb checks; CR1; per-repository table,
           repository-equal estimate and leave-one-repository-out range), the sensitivities (all rows if any slot is
           missing, time-cap rows as missing, d_pen_X and d_pen_F, worst-case bounds) and the secondary metrics.
  check    Always blind (the same reasons plus slots, infrastructure rows, manipulation checks, provenance and the load
           monitor). Use it during the block and for the smoke and capture blocks (--smoke).
  reruns   The next blind re-run batch (attempts 2-3, then the extra batch 4-5) for the analysis-set slots without a
           valid row, as a job file with the E8 arm fields (make_jobs_e8.job_entry).

Rows are analyze_e7 rules v2.2 rows (RUN_RECORD_SPEC_pinned.md, the vendored v2.2 spec) plus the E8 fields (PROTOCOL.md
section 6.3). The E7 code is reused, not copied, from the vendored byte copy analyze_e7_pinned.py (sha256 e86ccd81...,
rules v2.2 as verified; E7 amendment A10; LEAD_DECISIONS.md item 11): classify_row_v22, load_runs_v21 (dedupe_started, A9), assemble_v21, rerun_jobs_v21, the manifest, batch,
started-record, required-field, server-setting and provenance checks, a0_jobfile_stops_v22, d_pen_task_diffs,
exact_rademacher_p, wild_cluster_bootstrap_t, cluster_mean_ci, gate_core and secondary. A row whose manipulation check
failed is an infrastructure error of class M01 (LEAD_DECISIONS.md item 5): analyze_e8 passes classify_row_e8 (M01 on top
of classify_row_v22) to every analyze_e7 function that takes a classifier, and while it runs analyze_e7 functions it also
substitutes classify_row_e8 for analyze_e7.classify_row_v21 and classify_row_v22 (context manager e8_classes), so every
E7 rule (slots, re-runs, (4b) strata, d_pen, metrics) sees M01 as infrastructure. Record problems void the block (E7 A9:
status VOID, blind, no re-run batch).

Pin (PROTOCOL.md section 8.6): analyze_e7_pinned.py must be a regular file (not a symbolic link) whose bytes hash to
PINNED_ANALYZE_E7_SHA256; analyze_e8 executes exactly those bytes (compile + exec, no import system, no bytecode cache)
and never reads the E7 folder. Otherwise every subcommand refuses (exit 3); --allow-unpinned-e7 lets `check` and `reruns`
run on the copy present (outputs marked non-frozen) and is refused with `report`.

Time (PROTOCOL.md section 11.6): every decision uses the wall clock (UTC). `report` and `check` accept --as-of only as
YYYY-MM-DDTHH:MM:SS[.ffffff] followed by 'Z' or '+00:00', not later than the wall clock (an earlier time can only keep a
report blind); anything else is refused. `reruns` ignores --as-of and uses the wall clock. Every output records the
wall-clock time; a refusal writes <out>.refused.json.

Frozen hashes (PROTOCOL.md sections 6.4 and 15; LEAD_DECISIONS.md items 14 and 17): FROZEN_CHECKER_SHA256 is the sha256
of the manipulation checker manip_check_e8.py (this folder), filled in at freeze (None until then). analyze_e8 executes
exactly the checker bytes it hashed (compile + exec, like the analyze_e7 pin) and, at every run of every subcommand,
recomputes each row's manipulation record from the row's raw request dump (found by its llm_dump_sha256 in the run
directories given); no stored manipulation record is read or trusted. Every subcommand refuses (exit 3) when the
checker file is missing, a symbolic link, or, once a hash is frozen, hashes to anything else. `report` and `reruns`
refuse while FROZEN_CHECKER_SHA256 is None, and require --checker-sha256 equal to it and --analyze-e8-sha256 equal to the
sha256 of this file (both values as printed in the frozen PROTOCOL.md); `check` accepts both flags and verifies them
when given. Records carry the sha256 of the checker bytes that computed them; a record of any other checker version is
an input problem (not attached, so it cannot make an M01 row).

Dumps (PROTOCOL.md sections 6.2(f), 6.3 and 6.4; LEAD_DECISIONS.md item 23): a row's llm_dump_sha256 names the worker
dump written by the E8 wrapper W (the outermost acompletion wrapper, above every recorder): one record per call through
W, so its records with a response equal e8_hook_calls and its records with an error equal e8_hook_calls_failed by
construction. The D4 runner's forwarder capture is secondary evidence (the capture block, PROTOCOL.md 7.2), never the
named dump: a record computed from a dump in any other format (the capture, or a dump mixing formats) is not attached
and is a record problem (dump_e8; the block is VOID, PROTOCOL.md 11.6).

Hook calls (PROTOCOL.md section 6.2; LEAD_DECISIONS.md item 12): e8_hook_calls counts the calls through the E8 wrapper
that returned a response, e8_hook_calls_failed the calls that raised (context-length 400s, retried transient errors,
cancellations). Manipulation checks apply to returned responses only (record field n_requests = returned calls checked;
n_requests_failed = failed calls dumped, not checked), so a context-length run stays an agent outcome (A18), not M01.
The per-job container runner's own row for a job that died without a results row ('g4s-d<N> infra: ', I09) cannot know
the hook facts: it carries the four hook keys (null allowed) and is exempt from the hook check.

Runner tamper rows (PROTOCOL.md section 6.3; LEAD_DECISIONS.md item 25): when a job wrote anything that is not exactly
what the worker and PID 1 write, the runner replaces the job's row by its own row with error 'g4s-d<N> tamper: <what
it refused>' (class I12 under classify_row_v22). Such a row is treated like the runner's I09 row: the four hook keys
need only be present (null allowed), no hook check and no manipulation check applies (it gets no record; a record for
it is ignored, so it stays I12 and is never M01), it is an infrastructure row whose slot is re-run (A8), and it never
voids the block by itself. Its tamper evidence (the refusal kinds in the error text, the sha256 of the untrusted row
the runner set aside) is reported only after unblinding (results.tamper_rows); blind outputs count it as an I12 row
and show only the 'g4s-d<N> tamper: ' prefix of its error text.

usage: analyze_e8.py report|check|reruns --runs DIR [DIR ...] --manifest M --drop-file A0
       --tokenizer TOKENIZER_JSON --template-kaggle KAGGLE_V2_JINJA --template-t HF_CANONICAL_JINJA [--workers N]
       [--metrics VLLM_METRICS ...] [--as-of ISO_UTC] [--smoke] [--block E8]
       [--checker-sha256 HEX --analyze-e8-sha256 HEX (required by report and reruns)] --out PREFIX
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import datetime as _dt
import hashlib
import json
import os
import pathlib
import re
import sys
import types
from fractions import Fraction

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

# analyze_e7 is pinned (PROTOCOL.md section 8.6; E7 PROTOCOL_AMENDMENTS.md A10): the vendored byte copy of the analyze_e7
# version verified for rules v2.2. Later E7-only fixes do not change it.
PINNED_ANALYZE_E7_SHA256 = 'e86ccd811424e98462f9696d1e977d780108dc68b9dea6cea47d88053613b50c'
A7_PATH = HERE / 'analyze_e7_pinned.py'
MJ_PATH = HERE / 'make_jobs_e8.py'
# The run-record spec E8 follows (LEAD_DECISIONS.md item 13): the byte copy RUN_RECORD_SPEC_pinned.md of E7's
# RUN_RECORD_SPEC.md as it stood for rules v2.2. analyze_e8 does not read it; the tests check the copy's hash.
PINNED_RUN_RECORD_SPEC_SHA256 = '258c9246c8461aa57b2045ee6a95443eb4e9035b014f76391e8d405f5aa897f8'
# The manipulation checker manip_check_e8.py's sha256 (PROTOCOL.md sections 6.4 and 15; LEAD_DECISIONS.md items 14 and
# 17). FILLED IN AT FREEZE, after the checker's independent verification; the same value is printed in the frozen
# PROTOCOL.md. While it is None, `report` and `reruns` refuse. Filling it in is the only change to this file at freeze.
FROZEN_CHECKER_SHA256 = 'b075ed73acea92e95ef2ac91d9bf1d00d115fd957dae7e0baaa5e065651d0c16'
CHECKER_PATH = HERE / 'manip_check_e8.py'
_HEX64_RE = re.compile(r'[0-9a-f]{64}')


class Refusal(Exception):
    """The CLI refuses (exit 3) and writes only <out>.refused.json."""


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha_file(path):
    return _sha_bytes(pathlib.Path(path).read_bytes())


def _read_regular(path):
    """The bytes of a regular file (not a symbolic link). Raises Refusal."""
    p = pathlib.Path(path)
    if p.is_symlink():
        raise Refusal(f'{p} is a symbolic link, not a byte copy')
    if not p.is_file():
        raise Refusal(f'{p} does not exist')
    return p.read_bytes()


def exec_module_bytes(name, path, data):
    """Execute exactly `data` (the bytes that were hashed) as module `name`: compile + exec, so no import-system cache
    (__pycache__) can substitute other code, and no time passes between hashing and loading."""
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    try:
        exec(compile(data, str(path), 'exec', dont_inherit=True), mod.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


try:
    A7_BYTES, A7_LOAD_ERROR = _read_regular(A7_PATH), None
except Refusal as _e:
    A7_BYTES, A7_LOAD_ERROR = None, str(_e)
A7_SHA256 = _sha_bytes(A7_BYTES) if A7_BYTES is not None else None
A7_PINNED = A7_SHA256 == PINNED_ANALYZE_E7_SHA256
A7 = None                                   # bound by _bind_e7: at import when pinned, else only by the CLI's override
_ORIG_CLASSIFY = None
E8_D_PEN_IDS = None
MJ_BYTES = _read_regular(MJ_PATH)
MJ_SHA256 = _sha_bytes(MJ_BYTES)
MJ = exec_module_bytes('make_jobs_e8', MJ_PATH, MJ_BYTES)


def _bind_e7(data=None, path=A7_PATH):
    """Execute the analyze_e7 copy (its hashed bytes) and bind the names that depend on it."""
    global A7, _ORIG_CLASSIFY, E8_D_PEN_IDS
    A7 = exec_module_bytes('analyze_e7', path, A7_BYTES if data is None else data)
    _ORIG_CLASSIFY = A7.classify_row_v22
    E8_D_PEN_IDS = tuple(A7.D_PEN_IDS_V22) + ('M01',)
    return A7


if A7_PINNED:
    _bind_e7()


# this file's sha256 when it was loaded (report and reruns compare it with --analyze-e8-sha256, the frozen value)
ANALYZE_E8_SHA256 = _sha_file(__file__)

# the manipulation checker (LEAD_DECISIONS.md item 17): its bytes are read once and hashed; checker() executes exactly
# those bytes (compile + exec, no import system), and only when they hash to FROZEN_CHECKER_SHA256 (or no hash is
# frozen yet, `check` only)
try:
    CHECKER_BYTES, CHECKER_LOAD_ERROR = _read_regular(CHECKER_PATH), None
except Refusal as _e:
    CHECKER_BYTES, CHECKER_LOAD_ERROR = None, str(_e)
CHECKER_SHA256 = _sha_bytes(CHECKER_BYTES) if CHECKER_BYTES is not None else None
MC = None


def checker_refusal():
    """None if the checker may run, else the reason every subcommand refuses."""
    if CHECKER_BYTES is None:
        return f'the manipulation checker cannot be loaded: {CHECKER_LOAD_ERROR}'
    if FROZEN_CHECKER_SHA256 is not None and CHECKER_SHA256 != FROZEN_CHECKER_SHA256:
        return (f'{CHECKER_PATH.name} has sha256 {CHECKER_SHA256}, not the frozen FROZEN_CHECKER_SHA256 '
                f'{FROZEN_CHECKER_SHA256} (PROTOCOL.md 6.4; LEAD_DECISIONS.md items 14, 17, 20)')
    return None


def checker():
    """The checker module, executed from CHECKER_BYTES (raises Refusal when checker_refusal() gives a reason)."""
    global MC
    why = checker_refusal()
    if why:
        raise Refusal(why)
    if MC is None:
        MC = exec_module_bytes('manip_check_e8', CHECKER_PATH, CHECKER_BYTES)
    return MC


def code_sha256():
    return dict(analyze_e8=ANALYZE_E8_SHA256, make_jobs_e8=MJ_SHA256, analyze_e7_pinned=A7_SHA256,
                analyze_e7_pinned_expected=PINNED_ANALYZE_E7_SHA256, analyze_e7_pinned_ok=A7_PINNED,
                manip_check_e8=CHECKER_SHA256, checker_frozen=FROZEN_CHECKER_SHA256,
                checker_ok=checker_refusal() is None and FROZEN_CHECKER_SHA256 is not None)


def runtime():
    """The interpreter that ran the analysis (recorded in every output; the protocol's commands use python -I)."""
    return dict(python=sys.version.split()[0], executable=sys.executable, isolated=bool(sys.flags.isolated),
                dont_write_bytecode=bool(sys.dont_write_bytecode), numpy=np.__version__)


# ------------------------------------------------------------------------------------------------ wall clock
_UTC_RE = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{1,6})?(Z|\+00:00)')


def _system_clock():
    return _dt.datetime.now(_dt.timezone.utc)


_wall_clock = _system_clock                 # tests replace it to simulate the wall clock; the CLI never takes a time


def _iso(d):
    return d.astimezone(_dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S') + (
        f'.{d.microsecond:06d}' if d.microsecond else '') + 'Z'


def parse_utc_strict(s, what='--as-of'):
    """Strict ISO 8601 UTC: YYYY-MM-DDTHH:MM:SS[.ffffff] then 'Z' or '+00:00'. Raises Refusal on anything else (naive
    times, other offsets, dates only, epoch numbers, words such as 'now', a lowercase 'z', blanks)."""
    if not isinstance(s, str) or not _UTC_RE.fullmatch(s):
        raise Refusal(f'{what} {s!r} is not a UTC time of the form YYYY-MM-DDTHH:MM:SS[.ffffff]Z (or +00:00)')
    try:
        d = _dt.datetime.fromisoformat(s[:-1] + '+00:00' if s.endswith('Z') else s)
    except ValueError as e:
        raise Refusal(f'{what} {s!r} is not a valid time ({e})')
    return d.astimezone(_dt.timezone.utc)


def resolve_clock(as_of=None, now=None):
    """(evaluation time, wall clock, clock source, as-of source). `now` (tests only) simulates the wall clock. --as-of
    must not be later than the wall clock: an earlier time can only keep a report blind (the provisional analysis set
    and the re-run window both close later), so no as-of can unblind what the wall clock would not."""
    if now is None:
        wall, source = _wall_clock(), ('system' if _wall_clock is _system_clock else 'simulated (tests)')
    else:
        wall, source = (now if isinstance(now, _dt.datetime) else parse_utc_strict(now, 'now')), 'simulated (tests)'
    if wall.tzinfo is None:
        raise Refusal('the wall clock is not timezone-aware')
    if as_of is None:
        return wall, wall, source, 'wall clock'
    at = parse_utc_strict(as_of)
    if at > wall:
        raise Refusal(f'--as-of {as_of} is later than the wall clock {_iso(wall)}: refused (PROTOCOL.md section 11.6)')
    return at, wall, source, '--as-of'


# ------------------------------------------------------------------------------------------------ frozen settings
ANALYSIS_ID = ('analyze_e8 v5 (2026-11-10; PROTOCOL.md; analyze_e7 rules v2.2 from analyze_e7_pinned.py; manipulation '
               'checks recomputed from the dumps by manip_check_e8.py; LEAD_DECISIONS.md items 1-25)')
ARMS = tuple(MJ.ARMS)                      # F, R, C, H, T
REFERENCE = MJ.REFERENCE_ARM               # F
FAMILY = tuple(MJ.FAMILY)                  # R, C, H, T: the Holm family (LEAD_DECISIONS.md item 1)
REPEATS = MJ.REPEATS                       # 4
BLOCK = 'E8'
ALPHA = Fraction(1, 20)                    # two-sided, family-wise over the four trap arms (Holm)
N_MC = 99_999                              # task-level sign-flip Monte Carlo check and interval draws
MC_SEED = 20261109                         # = analyze_e7.BOOT_SEED
N_BOOT_REPO = 9_999                        # repository-cluster Monte Carlo check (analyze_e7.N_BOOT)
CUTOFF_UTC = '2026-11-11T06:00:00Z'        # LEAD_DECISIONS.md item 3: stages fully dispatched before it are analysed
RERUN_CUTOFF_UTC = '2026-11-12T00:00:00Z'  # PROTOCOL.md section 11.3 (confirmed, LEAD_DECISIONS.md item 20)
LOAD_T_AGENT_S = 240                       # LEAD_DECISIONS.md item 8
LOAD_FLAG_SHARE = Fraction(1, 20)          # load monitor flag: more than 5% of rows above 240 s (no rule changes)
TIME_CAP_ID = 'A05'                        # agent time cap (LEAD_DECISIONS.md item 4: sensitivity as missing)
# E7 A9 (rules v2.2): the server settings are read from every run entry of the block manifest (required, one value per
# block). E8 serves no adapter, so LoRA is off.
E8_SERVER_SETTINGS = dict(max_model_len=32768, enable_lora=False, max_lora_rank=None, served_adapter_names=[])
STAGE_SETS = {1: 'B', 2: 'A', 3: 'C'}      # cumulative panel after stage s
# 80% detectable even change (percentage points) under Holm with one affected arm, simulation range over repeat
# correlation 0.5-0.85, base rate 0.15-0.25 and two difficulty models, drops and gains together (PROTOCOL.md section 9;
# power/sim_main.json, gains of 7 and 8 points from power/sim_gain_supplement.json).
MDE_HOLM_ONE = {'B': 'about 4.3 to 7.4', 'A': 'about 3.4 to 6.4', 'C': 'below 3 to about 5.2'}
INTERVAL_GRID = 0.01                       # inversion grid for intervals, refined by bisection
INTERVAL_TOL = 1e-4
REPO_INTERVAL_TOL = 1e-3

# manipulation-check records (PROTOCOL.md section 6.4): one per row, computed at every run by manip_check_e8.py from the
# row's dump (LEAD_DECISIONS.md item 17). n_requests = the dumped requests that returned a response (the ones checked);
# n_requests_failed = the dumped requests whose call raised (not checked: LEAD_DECISIONS.md item 12)
MANIP_REQUIRED = ('job_id', 'arm', 'pass', 'checker_sha256', 'dump_sha256', 'n_requests', 'n_requests_failed',
                  'n_prompt_tokens_reproduced', 'n_ids_checked', 'n_ids_reproduced', 'factor_present',
                  'other_differences', 'dump_readable', 'hook_calls_match')
MANIP_KEY = '_e8_manip'                    # set on in-memory rows by analyze_e8 only

# E8 record fields (PROTOCOL.md section 6.3), on top of RUN_RECORD_SPEC_pinned.md (rules v2.2)
REQ_ROW_E8 = ('stage', 'bundle', 'e8_arms_sha256', 'e8_arm_config_sha256', 'history_reasoning',
              'server_chat_template_sha256', 'trust_request_chat_template', 'adk_version',
              'e8_hook_bound')                                                             # non-empty, every row (*)
REQ_ROW_E8_KEYS = ('compaction_config', 'request_chat_template_sha256', 'llm_dump_sha256', 'summaries',
                   'max_prompt', 'e8_hook_calls', 'e8_hook_calls_failed', 'e8_hook_rebound')  # key present, every row
# (*) the per-job container runner's own rows cannot know the hook facts: its row for a job that died without a results
# row (exc 'g4s-d<N> infra: ', I09) and its tamper row (error 'g4s-d<N> tamper: ', I12; LEAD_DECISIONS.md item 25). On
# them these fields need only the key (null allowed), and the hook check and the manipulation check skip them
HOOK_FIELDS = ('e8_hook_bound', 'e8_hook_rebound', 'e8_hook_calls', 'e8_hook_calls_failed')
REQ_STARTED_E8 = ('stage', 'e8_arms_sha256', 'e8_arm_config_sha256', 'request_model')
E8_BLOCK_FIELDS = ('bundle', 'bundle_sha256', 'e8_arms_sha256', 'server_chat_template_sha256',
                   'trust_request_chat_template', 'adk_version', 'served_model_name')
E8_ARM_FIELDS = ('request_model', 'compaction_config', 'history_reasoning', 'request_chat_template_sha256',
                 'e8_arm_config_sha256')
# the acompletion names the hook must hold (PROTOCOL.md section 6.2); a row listing others is a record problem
HOOK_NAMES = ('google.adk.models.lite_llm.acompletion', 'litellm.acompletion')
# the runner's tamper row (LEAD_DECISIONS.md item 25): run_berth.py tamper_row() writes resolved false and error
# 'g4s-d<N> tamper: <the kinds of what it refused>' (no exc), which classify_row_v22 makes I12 (unknown error text)
RUNNER_TAMPER_PATTERN = r'^g4s-d[0-9]+ tamper: '
_RUNNER_TAMPER_RE = re.compile(RUNNER_TAMPER_PATTERN)

# Reason categories (E7 A9, rules v2.2): 'input' -> INCONCLUSIVE until the analysis inputs are fixed; 'record' -> the
# block is VOID (blind; no re-run batch); 'slots' -> re-runs (A1/A8); 'wait' -> the analysis set is provisional until
# the cut-off; 'final' -> no analysis set (E8 makes no claim).
E8_EXTRA_CATEGORY = dict(required_fields_e8='record', provenance_e8='record', arm_config='record', sampling_e8='record',
                         hook_e8='record', dump_e8='record', manipulation_missing='input', manipulation_records='input',
                         restart_chain='input', analysis_set='wait', analysis_set_empty='final')


# ------------------------------------------------------------------------------------------------ classification
def classify_row_e8(row):
    """analyze_e7.classify_row_v22 (rules v2.2: I09 also for the per-job container runner's 'g4s-d<N> infra: ' rows,
    I13 anchored) plus M01: a row whose blind manipulation check failed is an infrastructure error (LEAD_DECISIONS.md
    item 5); its slot is missing and is re-run. Started jobs without a row stay I09."""
    m = row.get(MANIP_KEY)
    if m is not None and m.get('pass') is False and not row.get('_synthetic'):
        # the detail never names the row's own class: blind reports list infrastructure details, and the class of a
        # failed-check row (A00 = solved) would reveal its outcome
        why = '; '.join(str(x) for x in (m.get('reasons') or [])) or 'manipulation check failed'
        return dict(klass='infra', category='manipulation_check_failed', id='M01', detail=why[:160])
    return _ORIG_CLASSIFY(row)


def _status_e8_metric(row):
    """analyze_e7._status_v22_metric under classify_row_e8 (secondary metrics)."""
    c = classify_row_e8(row)
    if c['klass'] == 'infra':
        return 'harness_error', c['detail']
    return A7._V21_METRIC_STATUS.get(c['category'], c['category']), c['detail']


@contextlib.contextmanager
def e8_classes():
    """Run analyze_e7 functions with classify_row_e8 in place of classify_row_v21 and classify_row_v22 (restored on
    exit), for the analyze_e7 functions that look the classifier up by name."""
    prev21, prev22 = A7.classify_row_v21, A7.classify_row_v22
    A7.classify_row_v21 = A7.classify_row_v22 = classify_row_e8
    try:
        yield
    finally:
        A7.classify_row_v21, A7.classify_row_v22 = prev21, prev22


# ------------------------------------------------------------------------------------------------ small helpers
def parse_utc(s):
    """Row and record times: ISO 8601 ('...Z' or an offset) -> aware datetime; None if unparsable. (Never used for the
    time a report is made for: see resolve_clock.)"""
    if not isinstance(s, str) or not s.strip():
        return None
    t = s.strip()
    if t.endswith('Z'):
        t = t[:-1] + '+00:00'
    try:
        d = _dt.datetime.fromisoformat(t)
    except ValueError:
        return None
    if d.tzinfo is None:
        return None
    return d.astimezone(_dt.timezone.utc)


def _frac_str(x):
    return None if x is None else f'{x.numerator}/{x.denominator}'


def _nonneg_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _runner_row(row):
    """True for the per-job container runner's own row for a job that died without a results row: exc starts with
    'g4s-d<N> infra: ' (analyze_e7.RUNNER_NO_ROW_PATTERN; classify_row_v22 makes it I09). The worker never wrote it, so
    it cannot carry the hook facts (PROTOCOL.md section 6.3)."""
    exc = row.get('exc')
    return isinstance(exc, str) and re.match(A7.RUNNER_NO_ROW_PATTERN, exc) is not None


def _tamper_row(row):
    """True for the per-job container runner's tamper row (LEAD_DECISIONS.md item 25): error 'g4s-d<N> tamper: ...'
    on a row that classify_row_v22 makes I12 (no exc or tb, resolved false). The runner writes it in place of the job's
    row when the job wrote anything that is not exactly what the worker and PID 1 write, so the worker's hook facts are
    not on it (PROTOCOL.md section 6.3). A row of any other class (an exc row, a resolved row) is not a tamper row."""
    err = row.get('error')
    return (isinstance(err, str) and _RUNNER_TAMPER_RE.match(err) is not None
            and _ORIG_CLASSIFY(row)['id'] == 'I12')


def _runner_own_row(row):
    """The runner's own rows, which the worker never wrote: its I09 row for a job that died without a results row and
    its tamper row (I12). On both the hook keys need only be present (null allowed); neither gets a hook check or a
    manipulation record; both are infrastructure rows whose slots are re-run (PROTOCOL.md section 6.3)."""
    return _runner_row(row) or _tamper_row(row)


def _solved(r):
    return A7.solved(r['row'])


def _canon(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)


def _strip_template(sampling):
    """sampling_sent without the per-request chat template (arm T's only sampling difference)."""
    if not isinstance(sampling, dict):
        return sampling
    s = copy.deepcopy(sampling)
    s.pop('chat_template', None)
    eb = s.get('extra_body')
    if isinstance(eb, dict):
        eb.pop('chat_template', None)
    return s


# ------------------------------------------------------------------------------------------------ primary statistics
def signflip_exact(K):
    """Exact two-sided task-level sign-flip test of sum(K) for integer task differences K (LEAD_DECISIONS.md item 1):
    p = #{eps in {-1, +1}^n : |sum_i eps_i |K_i|| >= |sum_i K_i|} / 2^n over the n tasks with K_i != 0 (a zero
    difference does not change the statistic). Counted exactly by subset sums (integers, no rounding)."""
    K = [int(k) for k in K]
    S = sum(K)
    a = [abs(k) for k in K if k]
    n = len(a)
    if S == 0:
        return dict(p=Fraction(1), count=2 ** n, total=2 ** n, n_nonzero=n, S=0)
    T = sum(a)
    c = [0] * (T + 1)
    c[0] = 1
    top = 0
    for ai in a:
        top += ai
        for w in range(top, ai - 1, -1):
            c[w] += c[w - ai]
    # the statistic sum(eps_i a_i) = 2 W - T, W = the sum of a_i over the tasks with eps_i = +1
    count = sum(cw for w, cw in enumerate(c) if abs(2 * w - T) >= abs(S))
    return dict(p=Fraction(count, 2 ** n), count=count, total=2 ** n, n_nonzero=n, S=S)


def _sign_chunks(n_tasks, n_draws, seed, chunk=10_000):
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_draws:
        m = min(chunk, n_draws - done)
        yield rng.integers(0, 2, size=(m, n_tasks), dtype=np.int8) * 2 - 1
        done += m


class SignFlipMC:
    """Frozen Monte Carlo sign draws for one set of task differences, with p(shift) memoised, so that the Monte Carlo p
    and both inverted intervals use the same draws and share their grid evaluations."""

    def __init__(self, D, n_draws=N_MC, seed=MC_SEED):
        self.D = np.asarray(D, dtype=float)
        self.n_draws, self.seed = n_draws, seed
        self.chunks = [E.astype(np.float64) for E in _sign_chunks(self.D.size, n_draws, seed)]
        self._memo = {}

    def count(self, shift=0.0):
        key = round(float(shift), 12)
        if key not in self._memo:
            x = self.D - key
            S = abs(float(x.sum()))
            a = np.abs(x)
            thr = S - 1e-9 * max(1.0, S)
            self._memo[key] = sum(int(np.count_nonzero(np.abs(E @ a) >= thr)) for E in self.chunks)
        return self._memo[key]

    def p(self, shift=0.0):
        return (1 + self.count(shift)) / (self.n_draws + 1)


def signflip_mc(D, n_draws=N_MC, seed=MC_SEED, mc=None):
    """Monte Carlo two-sided sign-flip p, (1 + #{|S*| >= |S|}) / (B + 1), frozen seed (the check beside the exact p)."""
    mc = mc or SignFlipMC(D, n_draws, seed)
    return dict(p=mc.p(0.0), count=mc.count(0.0), n_draws=mc.n_draws, seed=mc.seed)


def _invert(accept, center, lo_bound=-1.0, hi_bound=1.0, step=INTERVAL_GRID, tol=INTERVAL_TOL):
    """Smallest and largest delta accepted on a grid of `step` over [lo_bound, hi_bound] (plus `center`), each refined
    by bisection to `tol` against its rejected grid neighbour. Returns (lo, hi) or (None, None)."""
    n = int(round((hi_bound - lo_bound) / step))
    grid = sorted(set([round(lo_bound + i * step, 10) for i in range(n + 1)] + [float(center)]))
    acc = [g for g in grid if accept(g)]
    if not acc:
        return None, None
    lo, hi = acc[0], acc[-1]

    def refine(inside, outside):
        while abs(outside - inside) > tol:
            mid = 0.5 * (inside + outside)
            if accept(mid):
                inside = mid
            else:
                outside = mid
        return inside
    below = [g for g in grid if g < lo]
    above = [g for g in grid if g > hi]
    if below:
        lo = refine(lo, below[-1])
    if above:
        hi = refine(hi, above[0])
    return lo, hi


def signflip_interval(D, level=0.95, n_draws=N_MC, seed=MC_SEED, mc=None):
    """Task-level interval by inverting the two-sided sign-flip test under a constant shift (D_i - delta symmetric about
    0), with the same frozen Monte Carlo draws for every delta."""
    D = np.asarray(D, dtype=float)
    if D.size == 0:
        return dict(lo=None, hi=None, level=level)
    mc = mc or SignFlipMC(D, n_draws, seed)
    alpha = 1.0 - level
    lo, hi = _invert(lambda delta: mc.p(delta) > alpha + 1e-12, float(D.mean()))
    return dict(lo=lo, hi=hi, level=level, n_draws=mc.n_draws, seed=mc.seed,
                method='inverted two-sided sign-flip test under a constant shift (frozen Monte Carlo draws)')


def holm_adjust(pvals, alpha=ALPHA):
    """Holm step-down adjusted p-values for {arm: p} (Fractions stay exact); ties keep the FAMILY order."""
    order = sorted(pvals, key=lambda a: (pvals[a], FAMILY.index(a) if a in FAMILY else 99, a))
    m = len(order)
    out, running = {}, Fraction(0)
    for j, a in enumerate(order):
        adj = min(Fraction(1), (m - j) * Fraction(pvals[a]))
        running = max(running, adj)
        out[a] = running
    return dict(adjusted=out, reject={a: out[a] <= alpha for a in out}, alpha=alpha, m=m)


# ------------------------------------------------------------------------------------------------ repository clusters
def repo_cluster(K, L, repos, n_boot=N_BOOT_REPO, seed=MC_SEED, interval=True, step=INTERVAL_GRID,
                 tol=REPO_INTERVAL_TOL):
    """Secondary (LEAD_DECISIONS.md item 1): analyze_e7's null-imposed Rademacher wild cluster bootstrap-t over
    repositories, exact by enumeration. Two-sided p = min(1, 2 min(p_upper, p_lower)); the 95% interval is the set of
    delta that neither one-sided exact test rejects at 0.025. Checks: the 9,999-draw Monte Carlo version and Webb
    weights (both two-sided), the CR1 interval with t(G - 1). Descriptives from analyze_e7.gate_core: per-repository
    table, repository-equal estimate, leave-one-repository-out estimates."""
    D = np.array([int(k) for k in K], dtype=float) / L
    names = sorted(set(repos))
    G = len(names)
    pos = {r: i for i, r in enumerate(names)}
    idx = np.array([pos[r] for r in repos], dtype=np.int64)
    out = dict(n_tasks=int(D.size), n_repos=G)
    if G < 2:
        out['note'] = 'fewer than 2 repositories: no repository-cluster inference'
        return out
    pu, k = A7.exact_rademacher_p(D, idx, G)
    pl, _ = A7.exact_rademacher_p(-D, idx, G)
    out.update(p_upper_exact=pu, p_lower_exact=pl, n_clusters_nonzero=k,
               p_two_exact=None if pu is None or pl is None else min(1.0, 2.0 * min(pu, pl)),
               min_attainable_p_two=min(1.0, 2.0 * 2.0 ** (-k)) if k else 1.0)
    mu = A7.wild_cluster_bootstrap_t(D, idx, G, n_boot=n_boot, seed=seed)
    ml = A7.wild_cluster_bootstrap_t(-D, idx, G, n_boot=n_boot, seed=seed)
    wu = A7.wild_cluster_bootstrap_t(D, idx, G, n_boot=n_boot, seed=seed + 1, weights='webb')
    wl = A7.wild_cluster_bootstrap_t(-D, idx, G, n_boot=n_boot, seed=seed + 1, weights='webb')
    out.update(p_two_mc=min(1.0, 2.0 * min(mu['p'], ml['p'])), p_two_webb=min(1.0, 2.0 * min(wu['p'], wl['p'])),
               mc_draws=n_boot, mc_seed=seed, webb_seed=seed + 1, cr1=A7.cluster_mean_ci(D.tolist(), list(repos)))
    if interval:
        def accept(delta):
            a, _ = A7.exact_rademacher_p(D - delta, idx, G)
            b, _ = A7.exact_rademacher_p(delta - D, idx, G)
            return a is not None and b is not None and a > 0.025 and b > 0.025
        lo, hi = _invert(accept, float(D.mean()), step=step, tol=tol)
        out['interval95'] = dict(lo=lo, hi=hi, method=f'inverted exact Rademacher wild cluster bootstrap-t (grid {step}, '
                                                      f'refined to {tol})')
    core = A7.gate_core(K, L, list(repos), n_boot=n_boot, seed=seed, sensitivities=False, exact=False)
    loo = [v['loo'] for v in (core.get('repos') or {}).values() if v.get('loo') is not None]
    out.update(repo_equal=core.get('repo_equal'), repo_table=core.get('repos'),
               loo_min=min(loo) if loo else None, loo_max=max(loo) if loo else None,
               n_repos_positive=core.get('n_repos_positive'), n_repos_negative=core.get('n_repos_negative'))
    return out


# ------------------------------------------------------------------------------------------------ loading E8 inputs
def index_manip_records(records, source='analyze_e8 (recomputed)'):
    """Index manipulation-check records by job_id. Returns dict(by_job, problems, checker_sha256s, n_records, files).
    A record without a required field, or two different records for one job, is a problem (an input problem)."""
    by_job, problems, shas = {}, [], collections.Counter()
    n = 0
    for li, rec in enumerate(records):
        n += 1
        if not isinstance(rec, dict):
            problems.append(f'{source}:{li + 1}: manipulation record is not an object')
            continue
        miss = [k for k in MANIP_REQUIRED if k not in rec]
        if miss:
            problems.append(f'{source}:{li + 1}: manipulation record without {miss}')
            continue
        shas[str(rec['checker_sha256'])] += 1
        jid = rec['job_id']
        if jid in by_job and _canon(by_job[jid]) != _canon(rec):
            problems.append(f'job {jid}: two manipulation records that differ')
            by_job[jid] = dict(by_job[jid], _conflict=True)
            continue
        by_job.setdefault(jid, rec)
    return dict(by_job=by_job, problems=problems, checker_sha256s=dict(shas), n_records=n, files={})


DUMP_PATTERNS = ('capture/*.jsonl.gz', 'capture/*.jsonl', 'out/llm/**/*.jsonl.gz', 'out/llm/**/*.jsonl')


def index_dumps(run_dirs):
    """{sha256: path} of the request dumps under the run directories (the worker's out/llm/ and the D4 runner's
    capture/), regular files only. A row's dump is found by its llm_dump_sha256 (content-addressed). The field must
    name the worker dump (LEAD_DECISIONS.md item 23); captures are indexed too, so that a row naming one is reported
    as such (dump_e8, a record problem) rather than as a dump not found."""
    import glob
    idx = {}
    for d in run_dirs:
        for pat in DUMP_PATTERNS:
            for p in sorted(glob.glob(os.path.join(str(d), pat), recursive=True)):
                if os.path.isfile(p) and not os.path.islink(p):
                    idx.setdefault(_sha_file(p), p)
    return idx


def manip_targets(load, block):
    """The rows the checker judges: every worker row of an E8 arm in the block (all attempts). The runner's own I09
    row has no dump and gets no record (PROTOCOL.md 6.3); started jobs without a row have none either. The runner's
    tamper row (I12) gets none even when it names the job's worker dump: no manipulation check applies to it
    (LEAD_DECISIONS.md item 25)."""
    out = []
    for r in load['runs']:
        row = r['row']
        if r.get('synthetic') or r['arm'] not in ARMS or row.get('block') not in (block, None) or _runner_own_row(row):
            continue
        out.append(r)
    return out


_MANIP_STATE = {}


def _manip_init(asset_paths, arms_doc):
    mc = checker()
    _MANIP_STATE.update(assets=mc.Assets(*asset_paths), rules=mc.arm_rules(arms_doc), mc=mc)


def _manip_dump_part(job):
    """One (arm, dump path or None) -> the checker's dump-level result (runs in a worker process or in-process)."""
    arm, path = job
    st = _MANIP_STATE
    data = None if path is None else pathlib.Path(path).read_bytes()
    return st['mc'].check_dump(arm, data, st['rules'], st['assets'])


def compute_manip(load, arms_doc, asset_paths, block=BLOCK, workers=1):
    """LEAD_DECISIONS.md item 17: every row's manipulation record, recomputed now from its raw dump by the checker
    executed from CHECKER_BYTES; records carry CHECKER_SHA256 (the bytes that ran), never a value the checker reports.
    `asset_paths` = (Kaggle-v2 template, HF canonical template, tokenizer.json); the checker refuses other hashes.
    Rows whose dump is not found get no record (an input problem listed in `dumps_not_found`)."""
    mc = checker()
    try:
        _manip_init(asset_paths, arms_doc)
    except (OSError, mc.CheckerError) as e:
        raise Refusal(f'the manipulation checker cannot load its inputs: {e}')
    targets = manip_targets(load, block)
    dirs = sorted({load['batches'][r['batch']]['dir'] for r in targets if r.get('batch') in load['batches']})
    idx = index_dumps(dirs)
    jobs, not_found = {}, []
    for r in targets:
        sha = r['row'].get('llm_dump_sha256')
        if sha is None:
            jobs[(r['arm'], None)] = None
        elif isinstance(sha, str) and sha in idx:
            jobs[(r['arm'], idx[sha])] = None
        else:
            not_found.append(dict(job_id=r.get('job_id'), arm=r['arm'], ref=A7._ref(r)))
    keys = sorted(jobs, key=lambda k: (k[0], k[1] or ''))
    if workers > 1 and len(keys) > 1:
        import concurrent.futures
        import multiprocessing
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork'),
                                                    initializer=_manip_init, initargs=(asset_paths, arms_doc)) as ex:
            for k, part in zip(keys, ex.map(_manip_dump_part, keys, chunksize=4)):
                jobs[k] = part
    else:
        for k in keys:
            jobs[k] = _manip_dump_part(k)
    nf = {x['ref'] for x in not_found}
    records = []
    for r in targets:
        if A7._ref(r) in nf:
            continue
        sha = r['row'].get('llm_dump_sha256')
        part = jobs[(r['arm'], None if sha is None else idx[sha])]
        records.append(mc.finish_record(r['row'], part, CHECKER_SHA256))
    out = index_manip_records(records)
    out.update(dumps_not_found=not_found, n_dumps=len(idx), assets=_MANIP_STATE['assets'].describe(),
               records_sha256=_sha_bytes(''.join(_canon(x) + '\n' for x in sorted(records, key=lambda x: _canon(
                   x.get('job_id')))).encode('utf-8')), records=records, workers=workers)
    return out


def manip_record_problem(rec):
    """None if the record is internally consistent, else a reason. pass must equal the pass formula: every dumped request
    that returned a response (n_requests) reproduced (prompt tokens and, where logged, ids), the arm's factor present, no
    other difference, the dump readable and the row's e8_hook_calls equal to n_requests. Failed calls
    (n_requests_failed) are counted, never checked (LEAD_DECISIONS.md item 12)."""
    if rec.get('_conflict'):
        return 'conflicting records'
    if not isinstance(rec.get('pass'), bool):
        return 'pass is not a boolean'
    for k in ('n_requests', 'n_requests_failed', 'n_prompt_tokens_reproduced'):
        if not _nonneg_int(rec.get(k)):
            return f'{k} is not a non-negative integer'
    if not isinstance(rec.get('other_differences'), list):
        return 'other_differences is not a list'
    for k in ('n_ids_checked', 'n_ids_reproduced'):
        if not _nonneg_int(rec.get(k)):
            return f'{k} is not a non-negative integer'
    if rec['pass']:
        if rec['n_prompt_tokens_reproduced'] != rec['n_requests']:
            return 'pass true but not every request reproduced'
        if rec.get('factor_present') is not True:
            return 'pass true but the factor is not present'
        if rec['other_differences']:
            return 'pass true with other differences'
        if rec.get('n_ids_reproduced') != rec.get('n_ids_checked'):
            return 'pass true but token ids not reproduced'
    # the pass formula (manip_check_e8.record_pass): a record fails only for what its returned calls, its dump or its
    # hook count show, so a row whose returned calls pass is never M01 (LEAD_DECISIONS.md items 12 and 17)
    expected = (rec.get('dump_readable') is True and rec['n_prompt_tokens_reproduced'] == rec['n_requests']
                and rec['n_ids_reproduced'] == rec['n_ids_checked'] and rec.get('factor_present') is True
                and not rec['other_differences'] and rec.get('hook_calls_match') is True)
    if rec['pass'] != expected:
        return f"pass {rec['pass']} disagrees with the record's own fields (the pass formula gives {expected})"
    return None


def attach_manip(runs, manip, in_plan_refs, checker_sha256=None):
    """Attach each record to its row (MANIP_KEY). Returns the per-row summary used by the reasons and the report.
    `checker_sha256`: the frozen checker hash (FROZEN_CHECKER_SHA256); a record of any other checker version is listed
    as wrong_checker and not attached, so it can never make an M01 row (LEAD_DECISIONS.md item 14). A record with pass
    true must have checked exactly the row's returned calls: n_requests equal to the row's e8_hook_calls (item 12).
    A record computed from a dump other than the worker dump W writes (dump_format not 'worker'; None for a row
    without a dump or an empty dump) is listed as not_worker_dump and not attached (LEAD_DECISIONS.md item 23)."""
    by_job = manip['by_job']
    out = dict(missing_agent=[], missing_infra=[], wrong_arm=[], wrong_dump=[], wrong_checker=[], inconsistent=[],
               not_worker_dump=[], failed=[], passed=0, checked=0, unknown_jobs=[])
    used_jobs = set()
    for r in runs:
        if r.get('synthetic') or A7._ref(r) not in in_plan_refs:
            continue
        row = r['row']
        jid = r.get('job_id')
        rec = by_job.get(jid) if jid else None
        ref = dict(job_id=jid, arm=r['arm'], ref=A7._ref(r))
        if _runner_own_row(row):
            # the runner's own row for a job that died without a results row has no dump, and its tamper row is not
            # checked (LEAD_DECISIONS.md item 25): never checked, never M01 (it stays I09 or I12); a record for it is
            # ignored
            out['missing_infra'].append(ref)
            if rec is not None:
                used_jobs.add(jid)
            continue
        if rec is None:
            klass = _ORIG_CLASSIFY(row)['klass']
            (out['missing_agent'] if klass == 'agent' else out['missing_infra']).append(ref)
            continue
        used_jobs.add(jid)
        if checker_sha256 is not None and rec.get('checker_sha256') != checker_sha256:
            out['wrong_checker'].append(dict(ref, record_checker_sha256=str(rec.get('checker_sha256'))[:64]))
            continue
        prob = manip_record_problem(rec)
        if rec.get('arm') != r['arm']:
            out['wrong_arm'].append(dict(ref, record_arm=rec.get('arm')))
            continue
        if rec.get('dump_sha256') != row.get('llm_dump_sha256'):
            out['wrong_dump'].append(ref)
            continue
        if rec.get('dump_format') not in ('worker', None):
            # the row names the forwarder capture (or a dump mixing formats), not the worker dump: its returned calls
            # need not match e8_hook_calls, so a time-cap row could become M01 (LEAD_DECISIONS.md item 23)
            out['not_worker_dump'].append(dict(ref, dump_format=str(rec.get('dump_format'))[:40]))
            continue
        # (a row whose e8_hook_calls is not a count is a hook record problem, reported by e8_provenance)
        if prob is None and rec['pass'] and _nonneg_int(row.get('e8_hook_calls')) and \
                row['e8_hook_calls'] != rec['n_requests']:
            prob = "pass true but n_requests (returned calls checked) differs from the row's e8_hook_calls"
        if prob:
            out['inconsistent'].append(dict(ref, problem=prob))
            continue
        out['checked'] += 1
        row[MANIP_KEY] = {'pass': rec['pass'], 'reasons': list(rec.get('reasons') or [])}
        if rec['pass']:
            out['passed'] += 1
        else:
            out['failed'].append(dict(ref, reasons=list(rec.get('reasons') or [])[:5]))
    out['unknown_jobs'] = sorted(set(by_job) - used_jobs)[:200]
    return out


def load_metrics(paths):
    """vLLM /metrics samples (JSON lines {"t_utc": ..., "vllm:num_preemptions_total": v}) -> pre-emption counts.
    A counter that decreases is a server restart: the value after it counts from 0 (and the restart is listed)."""
    samples, problems = [], []
    for p in paths or []:
        for li, line in enumerate(pathlib.Path(p).read_text(encoding='utf-8').splitlines()):
            if not line.strip():
                continue
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f'{p}:{li + 1}: unparsable')
                continue
            v = s.get('vllm:num_preemptions_total', s.get('num_preemptions_total'))
            t = parse_utc(s.get('t_utc'))
            if t is None or not isinstance(v, (int, float)) or isinstance(v, bool):
                problems.append(f'{p}:{li + 1}: sample without t_utc or a numeric pre-emption counter')
                continue
            samples.append((t, float(v)))
    samples.sort()
    total, restarts, incs = 0.0, 0, []
    for (t0, v0), (t1, v1) in zip(samples, samples[1:]):
        inc = v1 - v0 if v1 >= v0 else v1
        if v1 < v0:
            restarts += 1
        total += inc
        incs.append((t1, inc))
    return dict(n_samples=len(samples), preemptions=total if samples else None, counter_resets=restarts,
                first=samples[0][0].isoformat() if samples else None, last=samples[-1][0].isoformat() if samples else None,
                increments=incs, problems=problems)


# ------------------------------------------------------------------------------------------------ analysis set
def stage_tasks(panel_doc, dropped=()):
    dropped = set(dropped or ())
    return {int(s): [t for t in panel_doc['stages'][str(s)] if t not in dropped] for s in (1, 2, 3)}


def dispatch_status(panel_doc, load, block, cutoff, dropped=(), repeats=REPEATS):
    """LEAD_DECISIONS.md item 3: stage s is analysed iff every planned attempt-1 job of stages 1..s started before the
    cut-off (started records, or the row's t_start_utc). Outcome-blind: reads job ids and start times only."""
    cut = parse_utc(cutoff)
    st = stage_tasks(panel_doc, dropped)
    starts = {}
    for s in (load or {}).get('started', []):
        rec = s['rec']
        if A7._as_int(rec.get('attempt')) != 1 or rec.get('block') != block:
            continue
        t = parse_utc(rec.get('t_start_utc'))
        if t is not None and rec.get('job_id'):
            starts[rec['job_id']] = min(starts.get(rec['job_id'], t), t)
    for r in (load or {}).get('runs', []):
        row = r['row']
        if r.get('attempt') != 1 or row.get('block') != block or not r.get('job_id'):
            continue
        t = parse_utc(row.get('t_start_utc'))
        if t is not None:
            starts[r['job_id']] = min(starts.get(r['job_id'], t), t)
    info, s_star, ok_so_far = {}, 0, True
    for s in (1, 2, 3):
        planned = [MJ.job_id(block, a, t, r, 1) for t in st[s] for r in range(1, repeats + 1) for a in ARMS]
        before = sum(1 for j in planned if j in starts and starts[j] < cut)
        after = sum(1 for j in planned if j in starts and starts[j] >= cut)
        full = before == len(planned)
        info[s] = dict(n_tasks=len(st[s]), n_jobs=len(planned), started_before_cutoff=before,
                       started_after_cutoff=after, fully_dispatched=full)
        if ok_so_far and full:
            s_star = s
        else:
            ok_so_far = False
    tasks = [t for s in range(1, s_star + 1) for t in st[s]]
    return dict(stages=info, s_star=s_star, panel_set=STAGE_SETS.get(s_star), tasks=tasks, cutoff=cutoff)


# ------------------------------------------------------------------------------------------------ E8 record checks
def e8_required_fields(planned_rows, started_in_plan):
    missing, examples = collections.Counter(), {}
    for arm, rows in planned_rows.items():
        for r in rows:
            if r.get('synthetic'):
                continue
            row = r['row']
            runner = _runner_own_row(row)                   # its I09 row or its tamper row (LEAD item 25)
            for f in REQ_ROW_E8:
                if runner and f in HOOK_FIELDS:
                    if f not in row:                         # the runner's row: key present, null allowed
                        missing[(f, 'runner row (key)', arm)] += 1
                        examples.setdefault((f, 'runner row (key)', arm), A7._ref(r))
                elif A7._empty(row.get(f)):
                    missing[(f, 'result row', arm)] += 1
                    examples.setdefault((f, 'result row', arm), A7._ref(r))
            for f in REQ_ROW_E8_KEYS:
                if f not in row:
                    missing[(f, 'result row (key)', arm)] += 1
                    examples.setdefault((f, 'result row (key)', arm), A7._ref(r))
    for s in started_in_plan:
        for f in REQ_STARTED_E8:
            if A7._empty(s['rec'].get(f)):
                key = (f, 'started record', s['rec'].get('arm'))
                missing[key] += 1
                examples.setdefault(key, s['ref'])
    return [dict(field=f, scope=sc, arm=a, rows=n, example=examples[(f, sc, a)])
            for (f, sc, a), n in sorted(missing.items(), key=lambda kv: tuple(str(x) for x in kv[0]))]


def e8_provenance(planned_rows, used_rows, arms_doc, arms_sha256):
    """Block-wide E8 fields take one value (bundle, e8 arm file, server template, ADK version, served names); every
    row's arm fields equal the frozen arm file's entry; the sampling sent agrees across arms once the request template
    (arm T's factor) is removed. Returns dict(block_disagreements, arm_mismatches, sampling_disagreement, values)."""
    values, block_dis, arm_mis = {}, [], []
    for f in E8_BLOCK_FIELDS:
        c = collections.Counter(_canon(r['row'].get(f)) for rows in planned_rows.values() for r in rows
                                if not r.get('synthetic') and not A7._empty(r['row'].get(f)))
        values[f] = dict(c)
        if len(c) > 1:
            block_dis.append(dict(field=f, values=dict(c)))
    expect_block = dict(e8_arms_sha256=arms_sha256, bundle=arms_doc.get('bundle'),
                        served_model_name=arms_doc.get('served_model_names'),
                        server_chat_template_sha256=arms_doc.get('server_chat_template_sha256'),
                        adk_version=arms_doc.get('adk_version'),
                        trust_request_chat_template=arms_doc.get('trust_request_chat_template'))
    for f, want in expect_block.items():
        bad = [k for k in values.get(f, {}) if k != _canon(want)]
        if bad:
            block_dis.append(dict(field=f, expected=want, found=bad[:3]))
    for arm, rows in planned_rows.items():
        entry = arms_doc['arms'].get(arm)
        if entry is None:
            arm_mis.append(dict(arm=arm, problem='arm not in the arm file'))
            continue
        want = dict(entry, e8_arm_config_sha256=MJ.arm_config_sha256(arms_doc, arm))
        for r in rows:
            if r.get('synthetic'):
                continue
            for f in E8_ARM_FIELDS:
                if f in r['row'] and _canon(r['row'].get(f)) != _canon(want.get(f)):
                    arm_mis.append(dict(arm=arm, field=f, expected=want.get(f), found=r['row'].get(f), ref=A7._ref(r)))
            rm = str(r['row'].get('request_model') or '')
            if arm != 'R' and 'gemma4' in rm.lower():
                arm_mis.append(dict(arm=arm, field='request_model', problem="'gemma4' outside arm R", ref=A7._ref(r)))
    # the hook (PROTOCOL.md section 6.2): bound on every row the worker wrote, holding both acompletion names (other
    # rebound names are listed for the lead), with both call counts as non-negative integers (returned calls and
    # failed calls, LEAD_DECISIONS.md item 12; the manipulation checker compares the returned calls with the dump). The
    # runner's own row for a job that died without a results row (I09) cannot know these facts and is skipped; so is
    # its tamper row (I12, LEAD_DECISIONS.md item 25), counted apart and pooled over arms (blind).
    hook_bad, rebound, n_runner, n_tamper = [], collections.Counter(), 0, 0
    for rows in planned_rows.values():
        for r in rows:
            if r.get('synthetic'):
                continue
            row = r['row']
            if _runner_row(row):
                n_runner += 1
                continue
            if _tamper_row(row):
                n_tamper += 1
                continue
            names = row.get('e8_hook_rebound')
            if isinstance(names, list):
                rebound.update(str(x) for x in names)
            if not any(f in row for f in HOOK_FIELDS):
                continue                                    # a missing field is a required-field problem
            probs = []
            if row.get('e8_hook_bound') is not True:
                probs.append('e8_hook_bound is not true')
            if not isinstance(names, list) or any(n not in names for n in HOOK_NAMES):
                probs.append('e8_hook_rebound lacks a required acompletion name')
            for f in ('e8_hook_calls', 'e8_hook_calls_failed'):
                if not _nonneg_int(row.get(f)):
                    probs.append(f'{f} is not a non-negative integer')   # the value is not shown (blind)
            if probs:
                hook_bad.append(dict(arm=r['arm'], ref=A7._ref(r), problems=probs,
                                     e8_hook_rebound=names if isinstance(names, list) else repr(names)[:80]))
    samp = collections.Counter(_canon(_strip_template(r['row'].get('sampling_sent'))) for rows in used_rows.values()
                               for r in rows if not A7._empty(r['row'].get('sampling_sent')))
    return dict(block_disagreements=block_dis, arm_mismatches=arm_mis[:200], n_arm_mismatches=len(arm_mis),
                sampling_disagreement=len(samp) > 1, sampling_values=dict(samp), values=values,
                hook_problems=hook_bad[:100], n_hook_problems=len(hook_bad), hook_rebound_names=dict(rebound),
                hook_other_names=sorted(n for n in rebound if n not in HOOK_NAMES),
                n_runner_rows_without_hook_check=n_runner, n_tamper_rows_without_hook_check=n_tamper)


TAMPER_BLIND_NOTE = '(evidence after unblinding; LEAD item 25)'


def tamper_evidence(load, block, asm, task_set, repeats=REPEATS):
    """LEAD_DECISIONS.md item 25: the runner's tamper rows of the block (every E8 arm, all attempts), reported only after
    unblinding (results.tamper_rows; never in a blind output). Per arm the number of rows; per row its job, attempt and
    batch, whether it lies in the analysis set and whether its slot has a valid row, the kinds of what the runner
    refused (its error text after 'g4s-d<N> tamper: ', which the runner cuts at 200 characters) and, from the runner's
    d4 record, the number of refused items and the sha256 of the untrusted result row it set aside. Descriptive: no
    rule depends on it (the rows themselves are I12 infrastructure rows, re-run under A8)."""
    per_arm = {a: 0 for a in ARMS}
    rows = []
    for r in (load or {}).get('runs', []):
        row = r['row']
        if r.get('synthetic') or r['arm'] not in ARMS or row.get('block') not in (block, None) or not _tamper_row(row):
            continue
        per_arm[r['arm']] += 1
        d4 = row.get('d4') if isinstance(row.get('d4'), dict) else {}
        err = row['error']
        rep = r.get('repeat')
        in_set = r['task'] in task_set and isinstance(rep, int) and 1 <= rep <= repeats
        rows.append(dict(job_id=r.get('job_id'), arm=r['arm'], task=r['task'], repeat=rep, attempt=r.get('attempt'),
                         batch=r.get('batch'), in_analysis_set=in_set,
                         slot_has_valid_row=in_set and (r['arm'], r['task'], rep) in asm['chosen'],
                         refused=[x.strip() for x in err[_RUNNER_TAMPER_RE.match(err).end():].split(';') if x.strip()],
                         n_refused=d4.get('refused'), untrusted_row_sha256=d4.get('untrusted_row_sha256')))
    rows.sort(key=lambda x: (str(x['arm']), str(x['task']), str(x['repeat']), str(x['attempt'])))
    return dict(n_rows=len(rows), rows_per_arm=per_arm, rows=rows[:500],
                note='the runner\'s tamper rows (I12, error "g4s-d<N> tamper: ..."; LEAD_DECISIONS.md item 25): '
                     'infrastructure rows, re-run under A8, never a hook or manipulation check, never VOID by '
                     'themselves; their evidence is reported after unblinding only. Descriptive, no rule.')


def _blind_tamper_details(asm):
    """asm with the detail of each runner tamper row (I12) cut to its 'g4s-d<N> tamper: ' prefix and TAMPER_BLIND_NOTE:
    the tamper evidence is reported after unblinding only (LEAD_DECISIONS.md item 25). Blind outputs still count such a
    row as an I12 infrastructure row of its arm, attempt and batch, like any other infrastructure row."""
    def mask(x):
        m = _RUNNER_TAMPER_RE.match(str(x.get('detail') or '')) if x.get('id') == 'I12' else None
        return dict(x, detail=m.group(0) + TAMPER_BLIND_NOTE) if m else x
    return dict(asm, infra={a: [mask(x) for x in lst] for a, lst in asm['infra'].items()})


def load_monitor(planned_rows, metrics):
    """LEAD_DECISIONS.md item 8, blind: pooled over arms (never per arm): the share of rows with t_agent above 240 s
    (all attempts) and vLLM pre-emptions, overall and per batch. A batch is shown on its own only when every arm has
    the same number of rows in it (a balanced batch, such as a complete main or resume batch): a re-run batch of one
    arm (for example the M01 re-runs of H) would show that arm's time-cap rows. Unbalanced batches enter only the
    overall figures."""
    rows = [r for rs in planned_rows.values() for r in rs if not r.get('synthetic')]
    per_batch = collections.defaultdict(lambda: [0, 0])
    arms_in = collections.defaultdict(collections.Counter)
    n, k = 0, 0
    for r in rows:
        arms_in[r.get('batch')][r['arm']] += 1
        t = r['row'].get('t_agent')
        if isinstance(t, (int, float)) and not isinstance(t, bool):
            n += 1
            per_batch[r.get('batch')][0] += 1
            if t > LOAD_T_AGENT_S:
                k += 1
                per_batch[r.get('batch')][1] += 1
    balanced = {b for b, c in arms_in.items() if set(c) == set(ARMS) and len(set(c.values())) == 1}
    share = Fraction(k, n) if n else None
    shown = {b: v for b, v in per_batch.items() if b in balanced}
    flagged = [str(b) for b, (nb, kb) in shown.items() if nb and Fraction(kb, nb) > LOAD_FLAG_SHARE]
    windows = {}
    for b in shown:
        ts = [parse_utc(r['row'].get('t_start_utc')) for r in rows if r.get('batch') == b]
        te = [parse_utc(r['row'].get('t_end_utc')) for r in rows if r.get('batch') == b]
        ts, te = [x for x in ts if x], [x for x in te if x]
        if ts and te:
            windows[b] = (min(ts), max(te))
    pre_batch = {}
    if metrics and metrics.get('increments'):
        for b, (t0, t1) in windows.items():
            pre_batch[str(b)] = sum(inc for t, inc in metrics['increments'] if t0 <= t <= t1)
    return dict(threshold_s=LOAD_T_AGENT_S, rows_with_t_agent=n, rows_above=k,
                share_above=None if share is None else float(share),
                overall_flagged=bool(share is not None and share > LOAD_FLAG_SHARE),
                per_batch={str(b): dict(rows=v[0], above=v[1]) for b, v in shown.items()},
                n_batches_not_shown=len(set(arms_in) - balanced),
                flag_share=float(LOAD_FLAG_SHARE), batches_flagged=flagged,
                preemptions=None if not metrics else metrics.get('preemptions'),
                preemptions_per_batch=pre_batch, counter_resets=None if not metrics else metrics.get('counter_resets'),
                metric_samples=None if not metrics else metrics.get('n_samples'),
                metric_problems=[] if not metrics else metrics.get('problems', [])[:20],
                note=('pooled over arms (blind); per-batch figures only for batches in which every arm has the same '
                      'number of rows; agent-time secondaries are measured under mixed load'))


# ------------------------------------------------------------------------------------------------ assembly and reasons
def _panels_for_a0(panel_doc):
    st = panel_doc['stages']
    acc, out = [], []
    for s in (1, 2, 3):
        acc = acc + list(st[str(s)])
        out.append(dict(panel=f'E8-{STAGE_SETS[s]}', tasks=list(acc)))
    return out


def e8_server_settings(manifest, planned_rows):
    """E7 A9 (rules v2.2): analyze_e7.check_server_settings_v22 (every run entry of the block manifest carries
    max_model_len, enable_lora, max_lora_rank and served_adapter_names, one value per block) plus E8's values
    (E8_SERVER_SETTINGS). Returns the analyze_e7 result with the E8 problems added."""
    srv = A7.check_server_settings_v22(manifest, list(ARMS), planned_rows, adapter_arms=())
    extra = []
    for f, want in E8_SERVER_SETTINGS.items():
        for k in (srv.get('values') or {}).get(f, {}):
            if k != _canon(want):
                extra.append(f'server setting {f} is {k} in the block manifest, not {_canon(want)} (PROTOCOL.md 4, 6.3)')
    srv = dict(srv, problems=list(srv['problems']) + extra, ok=srv['ok'] and not extra, e8_expected=E8_SERVER_SETTINGS)
    return srv


def restart_chain(block, manifest):
    """LEAD_DECISIONS.md item 6 and E7 A9: a block restarted after an abort or a void (E8b, E8c, ...) names the block it
    replaces in its manifest: "replaces": {"block": <an earlier E8 block id>, "reason": "VOID" | "abort",
    "record_sha256": <sha256 of that block's VOID report or the lead's abort record>}. The first block (E8) names none.
    Returns a list of input problems (clearable)."""
    m = (manifest or {}).get('data')
    if not isinstance(m, dict):
        return []                                               # reported as a manifest problem
    rep = m.get('replaces')
    if block == BLOCK:
        return [] if rep is None else [f'the manifest of block {BLOCK} names a block it replaces ({rep!r})']
    if not MJ.BLOCK_RE.fullmatch(str(block)):
        return [f'block id {block!r} is not E8 or E8b, E8c, ...']
    if not isinstance(rep, dict):
        return [f'block {block} restarts E8: its manifest must carry "replaces": {{"block", "reason", "record_sha256"}}']
    out = []
    rb = str(rep.get('block'))
    if not (MJ.BLOCK_RE.fullmatch(rb) and (rb == BLOCK or rb < str(block))):
        out.append(f'replaces.block {rb!r} is not an earlier E8 block than {block}')
    if rep.get('reason') not in ('VOID', 'abort'):
        out.append(f"replaces.reason {rep.get('reason')!r} is not 'VOID' or 'abort'")
    if not (isinstance(rep.get('record_sha256'), str) and A7.HEX64.match(rep['record_sha256'])):
        out.append('replaces.record_sha256 is not a sha256')
    return out


def attempt1_batches(load, block):
    """Blind: the batches whose job lists hold attempt-1 jobs of the block (the main batch and any resume batch,
    PROTOCOL.md section 11.4), with their job counts."""
    out = {}
    for b in (load or {}).get('batches', {}).values():
        n = sum(1 for j in b.get('jobs') or [] if j.get('block', block) == block and A7._as_int(j.get('attempt')) == 1)
        if n:
            out[b['run_tag']] = n
    return out


def prepare(load, panel_doc, arms_doc, arms_sha256, manip, dropped, drop_sha256, manifest, at, block=BLOCK,
            cutoff=CUTOFF_UTC, rerun_cutoff=RERUN_CUTOFF_UTC, smoke_tasks=None, repeats=REPEATS, metrics=None,
            checker_sha256=None):
    """Everything up to the decision whether the report may be unblinded, evaluated at the time `at` (an aware
    datetime, never later than the wall clock: resolve_clock). Raises analyze_e7.AnalysisStop on A0/A2/A8/A9 stops.
    `checker_sha256`: the frozen manipulation-checker hash to enforce (the CLI passes FROZEN_CHECKER_SHA256).
    Must run inside e8_classes()."""
    runs = load['runs']
    if smoke_tasks is not None:
        a0_panels = A7.check_drops_all_panels_v21(dropped, [dict(panel='E8-smoke', tasks=list(smoke_tasks))])
        tasks = [t for t in smoke_tasks if t not in set(dropped or ())]
        disp = dict(stages={}, s_star=None, panel_set='smoke', tasks=tasks, cutoff=None)
    else:
        a0_panels = A7.check_drops_all_panels_v21(dropped, _panels_for_a0(panel_doc))
        disp = dispatch_status(panel_doc, load, block, cutoff, dropped, repeats)
        tasks = disp['tasks']
    A7.a0_row_stops_v21(runs, block, dropped, drop_sha256)
    a0_jobs = A7.a0_jobfile_stops_v22(load, block, drop_sha256)
    # manipulation records attach to the rows of the plan (all attempts)
    task_set = set(tasks)
    in_plan_refs = {A7._ref(r) for r in runs if r['arm'] in ARMS and r['task'] in task_set
                    and 1 <= r['repeat'] <= repeats and r['row'].get('block') in (block, None)}
    msum = attach_manip(runs, manip, in_plan_refs, checker_sha256=checker_sha256)
    asm = A7.assemble_v21(runs, tasks, list(ARMS), repeats, block=block, classify=classify_row_e8)
    reasons, parts = A7._collect_reasons_v21(asm, list(ARMS), tasks, repeats, block, load, manifest, adapter_arms=(),
                                             drop_given=drop_sha256 is not None, kind='gate', classify=classify_row_e8)
    reasons = [r for r in reasons if r['kind'] != 'slots']          # E8's own slot rule below
    started_in_plan = [s for s in load.get('started', [])
                       if s['rec'].get('arm') in ARMS and s['rec'].get('task') in task_set
                       and A7._as_int(s['rec'].get('repeat_id', s['rec'].get('seed'))) in range(1, repeats + 1)
                       and s['rec'].get('block', block) == block]
    # E7 A9 (rules v2.2): server settings from the manifest, duplicated started records
    server = e8_server_settings(manifest, parts['planned_rows'])
    for t in server['problems']:
        reasons.append(A7._reason('server_settings', t))
    dup = (load or {}).get('flags', {}).get('duplicate_started_job_ids', [])
    if dup:
        reasons.append(A7._reason('started_records', f'{len(dup)} started record(s) repeat a job_id of their run '
                                                     f'directory (A9: a record problem): {dup[:5]}'))
    if smoke_tasks is None:
        for t in restart_chain(block, manifest):
            reasons.append(A7._reason('restart_chain', t, clearable=True))
    req8 = e8_required_fields(parts['planned_rows'], started_in_plan)
    if req8:
        reasons.append(A7._reason('required_fields_e8', f'{len(req8)} E8 required field group(s) missing '
                                                        '(PROTOCOL.md section 6.3)'))
    prov8 = e8_provenance(parts['planned_rows'], parts['used_rows'], arms_doc, arms_sha256)
    if prov8['block_disagreements']:
        reasons.append(A7._reason('provenance_e8', f"{len(prov8['block_disagreements'])} block-wide E8 field(s) disagree "
                                                   'or differ from the frozen arm file'))
    if prov8['n_arm_mismatches']:
        reasons.append(A7._reason('arm_config', f"{prov8['n_arm_mismatches']} row field(s) differ from the frozen arm "
                                                'file (stop rule: find the cause; the block is VOID)'))
    if prov8['sampling_disagreement']:
        reasons.append(A7._reason('sampling_e8', 'the sampling sent differs between arms beyond the request template'))
    if prov8['n_hook_problems']:
        reasons.append(A7._reason('hook_e8', f"{prov8['n_hook_problems']} row(s) whose E8 hook was not bound to both "
                                             f'acompletion names {list(HOOK_NAMES)} or lacks its call counts '
                                             '(PROTOCOL.md section 6.2)'))
    # manipulation-check records (LEAD_DECISIONS.md item 5)
    if msum['not_worker_dump']:
        reasons.append(A7._reason('dump_e8', f"{len(msum['not_worker_dump'])} row(s) whose llm_dump_sha256 names a "
                                             "dump other than the worker dump W writes (the forwarder capture or a "
                                             'dump mixing formats): the field must name the worker dump (PROTOCOL.md '
                                             'section 6.3, LEAD_DECISIONS.md item 23)'))
    if msum['missing_agent']:
        reasons.append(A7._reason('manipulation_missing', f"{len(msum['missing_agent'])} agent-outcome row(s) without a "
                                                          "manipulation-check record: the row's request dump "
                                                          '(llm_dump_sha256) was not found in the run directories given',
                                                          clearable=True))
    for k, txt in (('wrong_arm', 'whose arm differs from the row'), ('wrong_dump', 'whose dump sha256 differs from the '
                   "row's llm_dump_sha256"), ('inconsistent', 'that are internally inconsistent')):
        if msum[k]:
            reasons.append(A7._reason('manipulation_records', f'{len(msum[k])} manipulation record(s) {txt}: find the '
                                                              'cause (the checker computed them from the dumps now)',
                                                              clearable=True))
    if manip['problems']:
        reasons.append(A7._reason('manipulation_records', f"{len(manip['problems'])} unreadable or conflicting "
                                                          'manipulation record(s)', clearable=True))
    if len(manip['checker_sha256s']) > 1:
        reasons.append(A7._reason('manipulation_records', 'manipulation records come from more than one checker '
                                                          'version', clearable=True))
    if msum['wrong_checker']:
        reasons.append(A7._reason('manipulation_records', f"{len(msum['wrong_checker'])} manipulation record(s) come "
                                                          'from a checker other than the frozen one '
                                                          f'({checker_sha256}); they are not used: run the frozen '
                                                          'checker (PROTOCOL.md section 6.4, LEAD_DECISIONS.md item '
                                                          '14)', clearable=True))
    # A failed check (M01) is an infrastructure row: its slot is open (a 'slots' reason) until a re-run gives it a
    # valid row or the re-run cut-off passes; the stop rule (the lead classifies the cause blind) is in next_step.
    # the analysis set (outcome-blind stopping), evaluated at `at` (<= the wall clock)
    cut_t = parse_utc(cutoff) if cutoff else None
    if smoke_tasks is None:
        before_cut = cut_t is not None and at < cut_t
        if disp['s_star'] == 0 and not before_cut:
            reasons.append(A7._reason('analysis_set_empty', 'no stage was fully dispatched before the cut-off: E8 has no '
                                                            'analysis set and makes no claim'))
        elif disp['s_star'] == 0:
            reasons.append(A7._reason('analysis_set', f'no stage fully dispatched yet; the cut-off {cutoff} has not '
                                                      'passed', clearable=True))
        elif disp['s_star'] < 3 and before_cut:
            reasons.append(A7._reason('analysis_set', f"provisional: stages 1-{disp['s_star']} dispatched, the cut-off "
                                                      f'{cutoff} has not passed', clearable=True))
    # slots (A1/A8) with E8's re-run cut-off
    rr_t = parse_utc(rerun_cutoff) if rerun_cutoff else None
    open_slots = [x for x in asm['lacking'] if x['max_attempt'] < A7.MAX_ATTEMPTS_V21
                  and not x['reason'].startswith('valid row only')]
    rerun_window = rr_t is None or at < rr_t
    if open_slots and rerun_window:
        reasons.append(A7._reason('slots', f'{len(open_slots)} planned slot(s) without a valid row can still get an '
                                           'attempt (A1/A8)'))
    for r in reasons:
        r['category'] = E8_EXTRA_CATEGORY.get(r['kind']) or A7.REASON_CATEGORY_V22.get(r['kind']) or \
            ('input' if r['clearable'] else 'record')
    final_missing = [] if (open_slots and rerun_window) else list(asm['lacking'])
    load_mon = load_monitor(parts['planned_rows'], metrics)
    # the runner's tamper rows (LEAD_DECISIONS.md item 25): kept for the unblinded report only
    tamper = tamper_evidence(load, block, asm, task_set, repeats)
    return dict(asm=asm, tasks=tasks, dispatch=disp, reasons=reasons, parts=parts, a0_panels=a0_panels, a0_jobs=a0_jobs,
                tamper=tamper,
                manip=msum, manip_files=manip.get('files'), checker_sha256s=manip['checker_sha256s'],
                manip_computed=dict(dumps_not_found=(manip.get('dumps_not_found') or [])[:50],
                                    n_dumps_not_found=len(manip.get('dumps_not_found') or []),
                                    n_dumps_indexed=manip.get('n_dumps'), assets=manip.get('assets'),
                                    records_sha256=manip.get('records_sha256'), workers=manip.get('workers'),
                                    checker=CHECKER_PATH.name, checker_sha256=CHECKER_SHA256),
                checker_sha256_enforced=checker_sha256, req_e8=req8,
                prov_e8=prov8, server=server, duplicate_started=dup, open_slots=open_slots, final_missing=final_missing,
                load_monitor=load_mon, at=at, rerun_cutoff=rerun_cutoff, rerun_window_open=rerun_window,
                repeats=repeats, block=block, attempt1_batches=attempt1_batches(load, block))


VOID_STEP = ('the block is VOID (E7 A9: record problems no re-run can clear). Archive its run directories unopened (no '
             'outcome is computed or shown), log the deviation, find and fix the cause (a code or image fix follows the '
             'recovery rule, LEAD_DECISIONS.md item 6), and, if time remains before the cut-off, restart all arms and '
             'slots under a new block id (E8b, E8c, ...) whose manifest names this block in "replaces" with the sha256 '
             'of this report; otherwise E8 is INCONCLUSIVE and makes no claim')


def decide(reasons):
    """E7 A9 order: input problems -> INCONCLUSIVE (fix the inputs); else record problems -> VOID; else no analysis
    set after the cut-off -> INCONCLUSIVE (no claim); else wait / slots -> INCONCLUSIVE (blind). None: no reason."""
    if not reasons:
        return None
    cats = {r['category'] for r in reasons}
    if 'input' in cats:
        return 'INCONCLUSIVE'
    if 'record' in cats:
        return 'VOID'
    return 'INCONCLUSIVE'


def next_step(reasons, open_slots, rerun_window):
    if not reasons:
        return None
    kinds = {r['kind'] for r in reasons}
    cats = {r['category'] for r in reasons}
    m01_open = [x for x in open_slots if 'M01' in x.get('classes', [])]
    if 'arm_config' in kinds:
        return ('stop rule: pause the block; the lead classifies the cause blind (arm, fields, error texts). ' + VOID_STEP)
    if 'input' in cats:
        return ('fix the inputs listed as clearable (run directories, manifest, drop file, manipulation-check records, '
                'the restart chain) or wait for the cut-off, then run the analysis again; no runs are needed for them')
    if 'record' in cats:
        return VOID_STEP
    if m01_open and rerun_window:
        return ('stop rule: pause the block; the lead classifies the cause blind (arm, failed manipulation checks, '
                'error texts). A fix that changes code or the image aborts the block (LEAD_DECISIONS.md item 6); a '
                'transient cause is re-run by `analyze_e8.py reruns`')
    if 'final' in cats:
        return 'none: no stage was fully dispatched before the cut-off; E8 is INCONCLUSIVE and makes no claim'
    if open_slots and rerun_window:
        return f'`analyze_e8.py reruns`: the next blind batch for {len(open_slots)} slot(s); then run the report again'
    return 'wait for the cut-off (the analysis set is provisional), then run the report again'


# ------------------------------------------------------------------------------------------------ comparisons
def _diffs(asm, tasks, arm, ref=REFERENCE, drop_ids=()):
    """Per task: mean solved of `arm` minus mean solved of `ref` over the used rows (rows whose class is in drop_ids
    are left out). Tasks lacking a row in either arm are skipped. Returns (tasks, diffs as Fractions)."""
    ts, ds = [], []
    for t in tasks:
        ra = [r for r in asm['cells'][(arm, t)] if classify_row_e8(r['row'])['id'] not in drop_ids]
        rf = [r for r in asm['cells'][(ref, t)] if classify_row_e8(r['row'])['id'] not in drop_ids]
        if ra and rf:
            ts.append(t)
            ds.append(Fraction(sum(_solved(r) for r in ra), len(ra)) - Fraction(sum(_solved(r) for r in rf), len(rf)))
    return ts, ds


def task_level(diffs, tasks, n_draws=N_MC, seed=MC_SEED, intervals=True):
    if not diffs:
        return dict(n_tasks=0, d=None, p_exact=None)
    K, L = A7._common_denominator(diffs)
    ex = signflip_exact(K)
    D = [float(x) for x in diffs]
    out = dict(n_tasks=len(diffs), d=float(Fraction(sum(K), L * len(K))), d_exact=_frac_str(Fraction(sum(K), L * len(K))),
               p_exact=ex['p'], p_exact_fraction=_frac_str(ex['p']), n_nonzero=ex['n_nonzero'],
               denominator=L, sum_K=sum(K))
    if intervals:
        mc = SignFlipMC(D, n_draws, seed)
        out.update(p_mc=mc.p(0.0), mc_draws=n_draws, mc_seed=seed,
                   interval95=signflip_interval(D, 0.95, mc=mc),
                   interval_simultaneous=signflip_interval(D, 1.0 - float(ALPHA) / len(FAMILY), mc=mc),
                   t_interval95=A7.cluster_mean_ci(D, list(tasks)))
        del mc
    return out


def comparison(asm, tasks, task_repo, arm, final_missing, ref=REFERENCE, repeats=REPEATS, n_draws=N_MC, seed=MC_SEED,
               repo_interval=True):
    """One trap arm against F (LEAD_DECISIONS.md items 1 and 4). Primary set: the analysis-set tasks with all `repeats`
    valid slots in both arms."""
    complete = [t for t in tasks if len(asm['cells'][(arm, t)]) == repeats and len(asm['cells'][(ref, t)]) == repeats]
    ts, ds = _diffs(asm, complete, arm, ref)
    prim = task_level(ds, ts, n_draws, seed)
    K, L = A7._common_denominator(ds) if ds else ([], 1)
    prim['repository'] = repo_cluster(K, L, [task_repo[t] for t in ts], interval=repo_interval) if ds else None
    missing = [x for x in final_missing if x['arm'] in (arm, ref)]
    out = dict(arm=arm, reference=ref, n_tasks_analysis_set=len(tasks), n_tasks_primary=len(ts), primary=prim,
               n_missing_slots=len(missing), incomplete=bool(missing),
               missing_slots=[dict(arm=x['arm'], task=x['task'], repeat=x['repeat'], classes=x['classes'])
                              for x in missing][:100])
    sens = {}
    if missing:
        ta, da = _diffs(asm, tasks, arm, ref)
        sens['all_rows'] = task_level(da, ta, n_draws, seed, intervals=False)
        # worst-case bounds over every analysis-set task: the missing slots of both arms imputed in opposite
        # directions (upper: missing X solved, missing F unsolved; lower: missing X unsolved, missing F solved)
        upper, lower = [], []
        for t in tasks:
            rx, rf = asm['cells'][(arm, t)], asm['cells'][(ref, t)]
            sx, sf = sum(_solved(r) for r in rx), sum(_solved(r) for r in rf)
            mx, mf = repeats - len(rx), repeats - len(rf)
            upper.append(Fraction(sx + mx, repeats) - Fraction(sf, repeats))
            lower.append(Fraction(sx, repeats) - Fraction(sf + mf, repeats))
        sens['bounds_upper'] = float(sum(upper) / len(upper)) if upper else None
        sens['bounds_lower'] = float(sum(lower) / len(lower)) if lower else None
        sens['bounds_note'] = ('every analysis-set task; upper: missing X slots solved and missing F slots unsolved; '
                               'lower: the reverse')
    a05 = [r for t in ts for a in (arm, ref) for r in asm['cells'][(a, t)]
           if classify_row_e8(r['row'])['id'] == TIME_CAP_ID]
    if a05:
        tt, dt = _diffs(asm, ts, arm, ref, drop_ids=(TIME_CAP_ID,))
        sens['time_cap_as_missing'] = dict(task_level(dt, tt, n_draws, seed, intervals=False), n_time_cap_rows=len(a05))
    if ts:
        px, pen_x = A7.d_pen_task_diffs(asm, ts, ref, arm, pen_ids=E8_D_PEN_IDS, classify=classify_row_e8)
        pf_raw, pen_f = A7.d_pen_task_diffs(asm, ts, arm, ref, pen_ids=E8_D_PEN_IDS, classify=classify_row_e8)
        pf = [-x for x in pf_raw]
        sens['d_pen_x'] = dict(task_level(px, ts, n_draws, seed, intervals=False), n_penalised=len(pen_x))
        sens['d_pen_f'] = dict(task_level(pf, ts, n_draws, seed, intervals=False), n_penalised=len(pen_f))
    out['sensitivity'] = sens
    return out


def solve_rates(asm, tasks, task_repo, repeats=REPEATS):
    out = {}
    for a in ARMS:
        vals, cl, per_rep = [], [], collections.Counter()
        for t in tasks:
            rs = asm['cells'][(a, t)]
            if rs:
                vals.append(sum(_solved(r) for r in rs) / len(rs))
                cl.append(task_repo[t])
            for r in rs:
                per_rep[r['repeat']] += _solved(r)
        out[a] = dict(task_weighted=A7.cluster_mean_ci(vals, cl), solved_rows=sum(per_rep.values()),
                      solved_per_repeat={str(k): per_rep.get(k, 0) for k in range(1, repeats + 1)},
                      used_rows=sum(len(asm['cells'][(a, t)]) for t in tasks))
    return out


def e8_row_secondary(asm, tasks, task_repo, arm, ref=REFERENCE):
    """Row-field secondary measures (PROTOCOL.md section 8.5): paired task differences X - F with CR1 repository
    intervals, plus per-arm descriptives. Exploratory."""
    def per_run(r):
        row = r['row']
        cid = classify_row_e8(row)['id']
        num = lambda k: float(row[k]) if isinstance(row.get(k), (int, float)) and not isinstance(row.get(k), bool) else None
        s = num('summaries')
        # A18 rows are the UNSOLVED context-length runs: a run whose working tree passes after the 400 is resolved
        # and has no error (swegemma verification.py:697), so it is A00
        # llm_calls_failed: calls through the E8 wrapper that raised (e8_hook_calls_failed; context-length 400s,
        # retried transient errors), unblinded only: per arm it would reveal context-length runs
        return dict(summaries=s, any_compaction=None if s is None else float(s > 0), max_prompt=num('max_prompt'),
                    context_length_unsolved=float(cid == 'A18'), time_cap=float(cid == TIME_CAP_ID),
                    tool_budget=float(cid == 'A02'), no_submit=float(cid == 'A06'), t_agent=num('t_agent'),
                    llm_calls=num('llm_calls'), llm_calls_failed=num('e8_hook_calls_failed'),
                    tool_calls=num('tool_calls'), tok_in=num('tok_in'), tok_out=num('tok_out'))
    keys = ('summaries', 'any_compaction', 'max_prompt', 'context_length_unsolved', 'time_cap', 'tool_budget',
            'no_submit', 't_agent', 'llm_calls', 'llm_calls_failed', 'tool_calls', 'tok_in', 'tok_out')
    per = {a: {t: [per_run(r) for r in asm['cells'][(a, t)]] for t in tasks} for a in (ref, arm)}
    arms_out = {}
    for a in (ref, arm):
        runs = [m for t in tasks for m in per[a][t]]
        arms_out[a] = {k: A7._describe([m[k] for m in runs]) for k in keys}
        ta = [m['t_agent'] for m in runs if m['t_agent'] is not None]
        arms_out[a]['t_agent_p90'] = A7.percentile(ta, 90) if ta else None
    paired = {}
    for k in keys:
        vals, cl = [], []
        for t in tasks:
            xa = [m[k] for m in per[arm][t] if m[k] is not None]
            xf = [m[k] for m in per[ref][t] if m[k] is not None]
            if xa and xf:
                vals.append(sum(xa) / len(xa) - sum(xf) / len(xf))
                cl.append(task_repo[t])
        paired[k] = A7.cluster_mean_ci(vals, cl)
    return dict(arms=arms_out, paired_differences=paired)


def manip_exposure(asm, tasks, manip_by_job):
    """Per arm, from the manipulation-check records of the rows used (unblinded only): requests checked (returned
    calls), requests whose prompt the manipulation changed (n_requests_changed, when the checker reports it), token ids
    checked, failed calls dumped (not checked, LEAD_DECISIONS.md item 12) and the rows whose dumped failed calls differ
    from the row's e8_hook_calls_failed (0 by construction with the worker dump W writes, LEAD_DECISIONS.md item 23; a
    non-zero count shows a dump that missed a failed call; descriptive, no rule)."""
    out = {}
    for a in ARMS:
        n_req = n_chg = n_ids = n_rows = n_fail = n_fail_mismatch = 0
        has_chg = True
        for t in tasks:
            for r in asm['cells'][(a, t)]:
                rec = manip_by_job.get(r.get('job_id')) or {}
                n_rows += 1
                n_req += int(rec.get('n_requests') or 0)
                n_fail += int(rec.get('n_requests_failed') or 0)
                n_ids += int(rec.get('n_ids_checked') or 0)
                if rec and rec.get('n_requests_failed') != r['row'].get('e8_hook_calls_failed'):
                    n_fail_mismatch += 1
                if isinstance(rec.get('n_requests_changed'), int):
                    n_chg += rec['n_requests_changed']
                else:
                    has_chg = False
        out[a] = dict(rows=n_rows, requests=n_req, requests_changed=n_chg if has_chg else None,
                      share_changed=(n_chg / n_req) if has_chg and n_req else None, token_ids_checked=n_ids,
                      failed_calls_dumped=n_fail, rows_failed_calls_not_matching_dump=n_fail_mismatch)
    return out


def claim_text(arm, c, holm_adj, holm_rej, panel_set):
    prim = c['primary']
    n = prim['n_tasks']
    if prim.get('d') is None:
        return f'{arm}: no task with complete slots in both arms; no estimate.'
    iv = prim.get('interval95') or {}
    rng = f"[{iv.get('lo'):+.4f}, {iv.get('hi'):+.4f}]" if iv.get('lo') is not None else '[n/a]'
    if holm_rej:
        direction = 'raises' if prim['d'] > 0 else 'lowers'
        return (f'{arm} {direction} E4B\'s solve rate against the scorer path F on these {n} tasks: d = '
                f"{prim['d']:+.4f} (95% interval {rng}), exact two-sided p = {float(prim['p_exact']):.3g}, Holm-adjusted "
                f'p = {float(holm_adj):.3g}.')
    return (f'{arm}: no difference from F detected on these {n} tasks; d = {prim["d"]:+.4f}, 95% interval {rng}; '
            f'Holm-adjusted p = {float(holm_adj):.3g}. The design detects an even change of '
            f'{MDE_HOLM_ONE.get(panel_set, "n/a")} points with 80% power (simulation range, PROTOCOL.md section 9). '
            'This is not evidence of no effect.')


def unblinded(prep, task_repo, manip_by_job, traces=True, n_draws=N_MC, seed=MC_SEED, repo_interval=True):
    asm, tasks = prep['asm'], prep['tasks']
    comps = {a: comparison(asm, tasks, task_repo, a, prep['final_missing'], n_draws=n_draws, seed=seed,
                           repo_interval=repo_interval) for a in FAMILY}
    pvals = {a: (comps[a]['primary']['p_exact'] if comps[a]['primary'].get('p_exact') is not None else Fraction(1))
             for a in FAMILY}
    holm = holm_adjust(pvals)
    for a in FAMILY:
        c = comps[a]
        c['holm_adjusted_p'] = holm['adjusted'][a]
        c['claim'] = bool(holm['reject'][a])
        flips = []
        for key in ('d_pen_x', 'd_pen_f'):
            s = c['sensitivity'].get(key)
            if s and s.get('p_exact') is not None:
                alt = dict(pvals, **{a: s['p_exact']})
                if holm_adjust(alt)['reject'][a] != c['claim']:
                    flips.append(key)
        c['sensitive_to_infrastructure_errors'] = bool(flips)
        c['sensitivity_flips'] = flips
        c['statement'] = claim_text(a, c, holm['adjusted'][a], c['claim'], prep['dispatch'].get('panel_set'))
        c['row_secondary'] = e8_row_secondary(asm, [t for t in tasks if asm['cells'][(a, t)] and
                                                    asm['cells'][(REFERENCE, t)]], task_repo, a)
        if traces:
            store = A7.TraceStore()
            both = [t for t in tasks if asm['cells'][(a, t)] and asm['cells'][(REFERENCE, t)]]
            c['trace_secondary'] = A7.secondary([], store, dict(panel='E8', tasks=both), task_repo, REFERENCE, a,
                                                prep['repeats'], asm=asm, status=_status_e8_metric)
    return dict(comparisons=comps, holm=dict(adjusted={a: holm['adjusted'][a] for a in FAMILY},
                                             reject=holm['reject'], alpha=float(ALPHA), family=list(FAMILY)),
                solve_rates=solve_rates(asm, tasks, task_repo, prep['repeats']),
                exposure=manip_exposure(asm, tasks, manip_by_job),
                tamper_rows=prep['tamper'])


# ------------------------------------------------------------------------------------------------ blind sections
def blind_sections(prep):
    asm = prep['asm']
    per_arm = {}
    for a in ARMS:
        per_arm[a] = dict(planned=len(prep['tasks']) * prep['repeats'],
                          with_valid_row=sum(len(asm['cells'][(a, t)]) for t in prep['tasks']),
                          lacking=sum(1 for x in asm['lacking'] if x['arm'] == a),
                          m01_rows=sum(1 for x in asm['infra'][a] if x['id'] == 'M01'))
    # E7 A9 (rules v2.2): blind reports show only class mismatches that involve an infrastructure id, agent ids masked
    # (analyze_e7.blind_class_mismatches_v22); M01 rows are expected mismatches and left out
    mm = A7.blind_class_mismatches_v22([x for x in asm['class_mismatches'] if x['computed'] != 'M01'])
    infra_share = {}
    for x in FAMILY:
        strata_x, strata_f = [], []
        for key, (cnt, _) in prep['parts']['used_batches'].items():
            kx = sum(1 for y in asm['infra'][x] if y['batch'] == key)
            kf = sum(1 for y in asm['infra'][REFERENCE] if y['batch'] == key)
            strata_x.append(dict(batch=key, jobs_baseline=cnt.get(REFERENCE, 0), jobs_candidate=cnt.get(x, 0),
                                 k_baseline=kf, k_candidate=kx))
            strata_f.append(dict(batch=key, jobs_baseline=cnt.get(x, 0), jobs_candidate=cnt.get(REFERENCE, 0),
                                 k_baseline=kx, k_candidate=kf))
        infra_share[x] = dict(x_more_than_f=A7.harness_4b_stratified(strata_x),
                              f_more_than_x=A7.harness_4b_stratified(strata_f))
    slots = A7._blind_slots(asm)
    slots.pop('n_class_mismatches', None)
    return dict(slots=dict(slots, per_arm=per_arm, infra_class_mismatches=mm, n_class_mismatches=mm['n'],
                           class_mismatch_note='infrastructure-involved mismatches only, agent ids masked (E7 A9)'),
                infra_share_tests=dict(tests=infra_share, note='diagnostic only: one-sided exact tests stratified by '
                                                               'job batch (analyze_e7.harness_4b_stratified), all '
                                                               'attempts; no rule depends on them'),
                # the runner's tamper rows count like any I12 row; their evidence waits for unblinding (LEAD item 25)
                infrastructure=A7._blind_infra(_blind_tamper_details(asm), list(ARMS)),
                manipulation_checks=dict(rows_checked=prep['manip']['checked'], passed=prep['manip']['passed'],
                                         failed=prep['manip']['failed'][:100],
                                         failed_by_arm=dict(collections.Counter(x['arm'] for x in prep['manip']['failed'])),
                                         missing_on_agent_rows=prep['manip']['missing_agent'][:50],
                                         n_missing_on_agent_rows=len(prep['manip']['missing_agent']),
                                         n_missing_on_infra_rows=len(prep['manip']['missing_infra']),
                                         wrong_arm=prep['manip']['wrong_arm'][:20],
                                         wrong_dump=prep['manip']['wrong_dump'][:20],
                                         not_worker_dump=prep['manip']['not_worker_dump'][:20],
                                         n_not_worker_dump=len(prep['manip']['not_worker_dump']),
                                         wrong_checker=prep['manip']['wrong_checker'][:20],
                                         n_wrong_checker=len(prep['manip']['wrong_checker']),
                                         inconsistent=prep['manip']['inconsistent'][:20],
                                         records_for_unknown_jobs=prep['manip']['unknown_jobs'][:50],
                                         checker_sha256s=prep['checker_sha256s'],
                                         checker_sha256_enforced=prep['checker_sha256_enforced'],
                                         files=prep.get('manip_files'), computed=prep.get('manip_computed')),
                provenance=dict(e7=prep['parts']['provenance'], e8=prep['prov_e8']),
                required_fields=dict(e7=prep['parts']['required_fields'], e8=prep['req_e8']),
                manifest={k: v for k, v in prep['parts']['manifest'].items() if k != 'ok'},
                server_settings=prep['server'],
                started=dict(prep['parts']['started'], duplicate_job_ids=prep['duplicate_started'][:50],
                             n_duplicate_job_ids=len(prep['duplicate_started'])),
                batch_problems=prep['parts']['batch_problems'], attempt1_batches=prep['attempt1_batches'],
                load_monitor=prep['load_monitor'],
                analysis_set=dict(prep['dispatch'], n_tasks=len(prep['tasks'])),
                a0=dict(panels=prep['a0_panels'], job_files=prep['a0_jobs']),
                ignored_rows=dict(collections.Counter(x['reason'] for x in asm['ignored'])),
                deviations=asm['deviations'][:100], attempt_gaps=asm['gaps'][:100])


# ------------------------------------------------------------------------------------------------ Markdown
def _f(x, nd=4):
    return A7._f(float(x) if isinstance(x, Fraction) else x, nd)


def md_blind(res):
    b = res['blind_sections']
    L = [f"# E8 report{' (' + res['label'] + ')' if res.get('label') else ''}: {res['status']} (BLIND)", '',
         f"analysis {res['analysis']}; block {res['block']}; as of {res['as_of']} ({res['as_of_source']}); wall clock "
         f"{res['wall_clock_utc']} ({res['clock']}); analyze_e7 sha256 {str(res['analyze_e7_sha256'])[:16]}... "
         f"({'pinned' if res['analyze_e7_pinned'] else 'NOT PINNED'})", '',
         'This report is blind (LEAD_DECISIONS.md item 5, E7 A8/A9): slots, infrastructure rows, manipulation checks, '
         'provenance, records and the pooled load monitor only. No solve count, difference, p-value, interval, '
         'repository table or secondary metric.', '', '## Reasons', '']
    for r in res['reasons']:
        L.append(f"- [{r['kind']}, {r['category']}{', clearable' if r['clearable'] else ''}] {r['text']}")
    if not res['reasons']:
        L.append('- none (the `report` command would unblind)')
    L += ['', f"**Next step:** {res.get('next_step') or 'none'}", '', '## Analysis set', '']
    a = b['analysis_set']
    L.append(f"cut-off {a.get('cutoff')}; stages fully dispatched: 1-{a.get('s_star')} (panel {a.get('panel_set')}); "
             f"{a['n_tasks']} tasks")
    for s, v in (a.get('stages') or {}).items():
        L.append(f"- stage {s}: {v['n_tasks']} tasks, {v['n_jobs']} attempt-1 jobs, started before the cut-off "
                 f"{v['started_before_cutoff']}, after {v['started_after_cutoff']}, fully dispatched "
                 f"{_f(v['fully_dispatched'])}")
    L += ['', '## Slots per arm', '', '| arm | planned | valid row | lacking | M01 rows |', '|---|---|---|---|---|']
    for arm, v in b['slots']['per_arm'].items():
        L.append(f"| {arm} | {v['planned']} | {v['with_valid_row']} | {v['lacking']} | {v['m01_rows']} |")
    L += ['', f"attempts histogram {b['slots']['attempts_hist']}; attempt gaps {len(b['attempt_gaps'])}; deviations "
              f"{len(b['deviations'])}", '', '## Infrastructure rows (all attempts)', '',
          '| arm | rows | by class | by batch |', '|---|---|---|---|']
    for arm, v in b['infrastructure'].items():
        L.append(f"| {arm} | {v['rows']} | {v['by_id']} | {v['by_batch']} |")
    L += ['', 'Diagnostic (no rule): one-sided exact tests of each trap arm\'s share of infrastructure rows against F, '
              'stratified by batch: ' + '; '.join(
                  f"{x}: more than F p = {_f(t['x_more_than_f'].get('p'))}, fewer p = {_f(t['f_more_than_x'].get('p'))}"
                  for x, t in b['infra_share_tests']['tests'].items())]
    m = b['manipulation_checks']
    L += ['', '## Manipulation checks', '',
          f"rows checked {m['rows_checked']}, passed {m['passed']}, failed {len(m['failed'])} (by arm "
          f"{m['failed_by_arm']}); agent rows without a record {m['n_missing_on_agent_rows']}; infrastructure rows "
          f"without a record {m['n_missing_on_infra_rows']}; checker sha256 {list(m['checker_sha256s'])}; frozen "
          f"checker enforced {m['checker_sha256_enforced'] or 'none (not frozen or not passed)'}; records of another "
          f"checker (not used) {m['n_wrong_checker']}"]
    for x in m['failed'][:20]:
        L.append(f"- failed: {x['job_id']} ({x['arm']}): {'; '.join(x.get('reasons') or [])}")
    lm = b['load_monitor']
    L += ['', '## Load monitor (pooled over arms)', '',
          f"rows with t_agent > {lm['threshold_s']} s: {lm['rows_above']} of {lm['rows_with_t_agent']} "
          f"({_f(lm['share_above'])}); balanced batches above {lm['flag_share']:.0%}: "
          f"{lm['batches_flagged'] or 'none'} ({lm['n_batches_not_shown']} unbalanced batch(es) only in the overall "
          f"figure); vLLM pre-emptions {lm['preemptions']} (counter resets {lm['counter_resets']})", '',
          '## Records and provenance', '',
          f"E7 required fields missing: {len(b['required_fields']['e7']['missing'])} group(s); E8: "
          f"{len(b['required_fields']['e8'])} group(s); E7 provenance ok {_f(b['provenance']['e7']['ok'])}; E8 block "
          f"disagreements {len(b['provenance']['e8']['block_disagreements'])}; arm-file mismatches "
          f"{b['provenance']['e8']['n_arm_mismatches']}; hook problems {b['provenance']['e8']['n_hook_problems']} "
          f"(other rebound names {b['provenance']['e8']['hook_other_names'] or 'none'}); runner rows without a hook "
          f"check {b['provenance']['e8']['n_runner_rows_without_hook_check']} (I09) and "
          f"{b['provenance']['e8']['n_tamper_rows_without_hook_check']} (tamper, I12; evidence after unblinding); "
          f"sampling differs across arms "
          f"{_f(b['provenance']['e8']['sampling_disagreement'])}; server settings ok {_f(b['server_settings']['ok'])} "
          f"(problems {b['server_settings']['problems'][:3]}); duplicate started records "
          f"{b['started']['n_duplicate_job_ids']}; infrastructure class mismatches "
          f"{b['slots']['infra_class_mismatches']['n']}; manifest clearable {b['manifest'].get('clearable')}, data "
          f"{b['manifest'].get('data')}; batches holding attempt-1 jobs {b['attempt1_batches']}"]
    if res.get('void'):
        L += ['', '## VOID', '', res['void']['instructions']]
    return '\n'.join(L) + '\n'


def md_full(res):
    u = res['results']
    a = res['blind_sections']['analysis_set']
    L = [f"# E8 report{' (' + res['label'] + ')' if res.get('label') else ''}: UNBLINDED", '',
         f"analysis {res['analysis']}; block {res['block']}; as of {res['as_of']} ({res['as_of_source']}); wall clock "
         f"{res['wall_clock_utc']} ({res['clock']}); analysis set: stages 1-{a.get('s_star')} (panel "
         f"{a.get('panel_set')}), {a['n_tasks']} tasks; analyze_e7 sha256 {str(res['analyze_e7_sha256'])[:16]}...; "
         f"protocol settings {res['protocol_settings']}", '',
         'Primary: exact two-sided task-level sign-flip test, Holm over R, C, H, T at 0.05; claims are about these '
         'tasks only (LEAD_DECISIONS.md item 1). Absolute solve rates are not hosted scores.', '', '## Claims', '']
    for arm in FAMILY:
        c = u['comparisons'][arm]
        tag = ' [INCOMPLETE]' if c['incomplete'] else ''
        tag += ' [sensitive to infrastructure errors]' if c['sensitive_to_infrastructure_errors'] else ''
        L.append(f"- {c['statement']}{tag}")
    L += ['', '## Primary (task level)', '',
          '| arm | tasks | d | exact p | Holm p | claim | MC p | 95% interval | 98.75% simultaneous | t interval |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for arm in FAMILY:
        c = u['comparisons'][arm]
        p = c['primary']
        iv, si, ti = p.get('interval95') or {}, p.get('interval_simultaneous') or {}, p.get('t_interval95') or {}
        L.append(f"| {arm} | {p['n_tasks']} | {_f(p.get('d'))} | {_f(p.get('p_exact'), 6)} | "
                 f"{_f(c['holm_adjusted_p'], 6)} | {_f(c['claim'])} | {_f(p.get('p_mc'), 6)} | "
                 f"[{_f(iv.get('lo'))}, {_f(iv.get('hi'))}] | [{_f(si.get('lo'))}, {_f(si.get('hi'))}] | "
                 f"[{_f(ti.get('lo'))}, {_f(ti.get('hi'))}] |")
    L += ['', '## Repository-cluster secondary ("repositories like these")', '',
          '| arm | exact two-sided p | 95% interval | MC p | Webb p | CR1 interval | repository-equal | LOO range |',
          '|---|---|---|---|---|---|---|---|']
    for arm in FAMILY:
        r = u['comparisons'][arm]['primary'].get('repository') or {}
        iv, cr = r.get('interval95') or {}, r.get('cr1') or {}
        L.append(f"| {arm} | {_f(r.get('p_two_exact'), 6)} | [{_f(iv.get('lo'))}, {_f(iv.get('hi'))}] | "
                 f"{_f(r.get('p_two_mc'), 6)} | {_f(r.get('p_two_webb'), 6)} | [{_f(cr.get('lo'))}, {_f(cr.get('hi'))}] | "
                 f"{_f(r.get('repo_equal'))} | [{_f(r.get('loo_min'))}, {_f(r.get('loo_max'))}] |")
    L += ['', '## Sensitivities', '', '| arm | sensitivity | tasks | d | exact p | note |', '|---|---|---|---|---|---|']
    for arm in FAMILY:
        for k, s in u['comparisons'][arm]['sensitivity'].items():
            if isinstance(s, dict):
                note = f"penalised {s.get('n_penalised')}" if 'n_penalised' in s else (
                    f"time-cap rows {s.get('n_time_cap_rows')}" if 'n_time_cap_rows' in s else '')
                L.append(f"| {arm} | {k} | {s.get('n_tasks')} | {_f(s.get('d'))} | {_f(s.get('p_exact'), 6)} | {note} |")
            elif isinstance(s, str):
                continue
            else:
                L.append(f'| {arm} | {k} | | {_f(s)} | | |')
    L += ['', '## Solve rates per arm (task-weighted, CR1 repository interval)', '',
          '| arm | rate | interval | solved rows | per repeat | used rows |', '|---|---|---|---|---|---|']
    for arm, v in u['solve_rates'].items():
        tw = v['task_weighted']
        L.append(f"| {arm} | {_f(tw.get('mean'))} | [{_f(tw.get('lo'))}, {_f(tw.get('hi'))}] | {v['solved_rows']} | "
                 f"{v['solved_per_repeat']} | {v['used_rows']} |")
    L += ['', '## Exposure (manipulation-check records)', '',
          '| arm | rows | requests checked | changed | share | failed calls dumped | rows whose failed calls differ from '
          'the dump |', '|---|---|---|---|---|---|---|']
    for arm, v in u['exposure'].items():
        L.append(f"| {arm} | {v['rows']} | {v['requests']} | {v['requests_changed']} | {_f(v['share_changed'])} | "
                 f"{v['failed_calls_dumped']} | {v['rows_failed_calls_not_matching_dump']} |")
    tr = u.get('tamper_rows') or {}
    L += ['', "## Tamper rows (the runner's I12 rows, all attempts; LEAD_DECISIONS.md item 25; descriptive)", '',
          f"{tr.get('n_rows', 0)} row(s); per arm {tr.get('rows_per_arm')}"]
    for x in (tr.get('rows') or [])[:50]:
        L.append(f"- {x['job_id']} ({x['arm']}, attempt {x['attempt']}, batch {x['batch']}; analysis set "
                 f"{_f(x['in_analysis_set'])}, slot has a valid row {_f(x['slot_has_valid_row'])}): refused "
                 f"{'; '.join(x['refused']) or 'n/a'} (items {x['n_refused']}; untrusted row sha256 "
                 f"{str(x['untrusted_row_sha256'])[:16]})")
    L += ['', '## Row secondary measures (paired X - F, CR1 repository interval; exploratory)', '']
    for arm in FAMILY:
        ps = u['comparisons'][arm]['row_secondary']['paired_differences']
        L.append(f"- {arm}: " + '; '.join(f"{k} {_f(v.get('mean'))} [{_f(v.get('lo'))}, {_f(v.get('hi'))}]"
                                          for k, v in ps.items()))
    L += ['', '## Slots per arm' + md_blind(res).split('## Slots per arm', 1)[1]]
    return '\n'.join(L) + '\n'


# ------------------------------------------------------------------------------------------------ drivers
def load_runs(dirs):
    """analyze_e7.load_runs_v21 as rules v2.2 call it (dedupe_started: a started record repeating a job_id of its run
    directory is listed as a record problem instead of becoming a synthetic row, E7 A9)."""
    return A7.load_runs_v21(dirs, dedupe_started=True)


def _inputs(a):
    panel_doc = MJ.load_panel(a.panel)
    arms_bytes = pathlib.Path(a.arms_file).read_bytes()
    arms_doc = json.loads(arms_bytes)
    if arms_bytes != MJ.arms_bytes():
        raise A7.AnalysisStop('the arm file differs from make_jobs_e8.arms_document(): not the frozen arm file')
    dropped, drop_sha = A7.load_drop_file(a.drop_file)
    manifest = A7.load_manifest_v21(a.manifest) if a.manifest else None
    load = load_runs(a.runs)
    # LEAD_DECISIONS.md item 17: every record is recomputed now from the dumps; no stored record is read
    manip = compute_manip(load, arms_doc, (a.template_kaggle, a.template_t, a.tokenizer), block=a.block,
                          workers=a.workers)
    metrics = load_metrics(a.metrics) if a.metrics else None
    return panel_doc, arms_doc, hashlib.sha256(arms_bytes).hexdigest(), dropped, drop_sha, manifest, load, manip, metrics


def _protocol_settings(n_draws, seed, cutoff, rerun_cutoff, repeats, smoke_tasks, clock, checker_sha256=None):
    return (n_draws == N_MC and seed == MC_SEED and cutoff == CUTOFF_UTC and rerun_cutoff == RERUN_CUTOFF_UTC
            and repeats == REPEATS and A7_PINNED and smoke_tasks is None and clock == 'system'
            and FROZEN_CHECKER_SHA256 is not None and checker_sha256 == FROZEN_CHECKER_SHA256)


FREEZE_NOTE = ('protocol_settings says only that the statistical settings, cut-offs, pinned analyze_e7, the frozen '
               'checker hash and the system clock are the protocol\'s. On the command line, report and reruns run only '
               'when --analyze-e8-sha256 equals this file\'s sha256 and --checker-sha256 equals FROZEN_CHECKER_SHA256, '
               'both as printed in the frozen PROTOCOL.md (sections 6.4 and 15); code_sha256 records the hashes '
               '(PROTOCOL.md section 11.6)')


def run_analysis(load, panel_doc, arms_doc, arms_sha256, manip, dropped, drop_sha256, manifest, task_repo, mode,
                 block=BLOCK, as_of=None, now=None, cutoff=CUTOFF_UTC, rerun_cutoff=RERUN_CUTOFF_UTC, smoke_tasks=None,
                 repeats=REPEATS, metrics=None, traces=True, n_draws=N_MC, seed=MC_SEED, repo_interval=True,
                 label='', checker_sha256=None):
    """mode 'report' (may unblind) or 'check' (always blind). Every decision uses the wall clock (`now` simulates it in
    tests only); `as_of`, if given, must be a strict UTC time not later than it (resolve_clock). `checker_sha256`: the
    frozen manipulation-checker hash to enforce (the CLI passes FROZEN_CHECKER_SHA256; records of other versions are not
    used). Raises Refusal on a bad as_of, analyze_e7.AnalysisStop on stops."""
    if A7 is None:
        raise Refusal(f'analyze_e7 is not loaded: {A7_LOAD_ERROR or "the copy is not the pinned version"}')
    at, wall, clock, as_of_source = resolve_clock(as_of, now)
    with e8_classes():
        prep = prepare(load, panel_doc, arms_doc, arms_sha256, manip, dropped, drop_sha256, manifest, at, block=block,
                       cutoff=cutoff, rerun_cutoff=rerun_cutoff, smoke_tasks=smoke_tasks, repeats=repeats,
                       metrics=metrics, checker_sha256=checker_sha256)
        decision = decide(prep['reasons'])
        res = dict(kind='e8_report', mode=mode, rules='v2.2', analysis=ANALYSIS_ID, block=block, panel='E8',
                   as_of=_iso(at), as_of_source=as_of_source, wall_clock_utc=_iso(wall), clock=clock, cutoff=cutoff,
                   rerun_cutoff=rerun_cutoff, repeats=repeats, family=list(FAMILY), arms_analysed=list(ARMS),
                   alpha=float(ALPHA), n_mc=n_draws, mc_seed=seed, analyze_e7_sha256=A7_SHA256,
                   analyze_e7_pinned=A7_PINNED, analyze_e7_path=str(A7_PATH), code_sha256=code_sha256(),
                   e8_arms_sha256=arms_sha256, drop_file_sha256=drop_sha256, a0=dict(drop_file_sha256=drop_sha256),
                   d_pen_ids=list(E8_D_PEN_IDS), error_table_sha256=A7.ERROR_TABLE_SHA256_V22,
                   smoke_tasks=list(smoke_tasks) if smoke_tasks is not None else None, reasons=prep['reasons'],
                   next_step=next_step(prep['reasons'], prep['open_slots'], prep['rerun_window_open']),
                   blind_sections=blind_sections(prep), label=label, checker_sha256_enforced=checker_sha256,
                   runtime=runtime())
        res['protocol_settings'] = _protocol_settings(n_draws, seed, cutoff, rerun_cutoff, repeats, smoke_tasks, clock,
                                                      checker_sha256)
        res['freeze_note'] = FREEZE_NOTE
        if decision == 'VOID':
            res['void'] = dict(block=block, record_problems=[r['text'] for r in prep['reasons']
                                                             if r['category'] == 'record'], instructions=VOID_STEP)
        if decision is not None or mode == 'check':
            res.update(status=decision or 'CHECK PASSED', decision=decision, blind=True)
            return res
        res.update(status='UNBLINDED', decision=None, blind=False,
                   results=unblinded(prep, task_repo, manip['by_job'], traces=traces, n_draws=n_draws, seed=seed,
                                     repo_interval=repo_interval))
        return res


def run_reruns(load, panel_doc, arms_doc, arms_sha256, manip, dropped, drop_sha256, manifest, block=BLOCK, now=None,
               cutoff=CUTOFF_UTC, rerun_cutoff=RERUN_CUTOFF_UTC, smoke_tasks=None, repeats=REPEATS, checker_sha256=None):
    """The next blind batch (analyze_e7.rerun_jobs_v21 under the E8 classes) with the E8 job fields, decided at the
    wall clock only (`now` simulates it in tests; there is no as-of). Refuses (AnalysisStop) on input problems,
    missing manipulation records, records of a checker other than the frozen one (`checker_sha256`) and after the
    re-run cut-off; a block with record problems is VOID: no batch."""
    if A7 is None:
        raise Refusal(f'analyze_e7 is not loaded: {A7_LOAD_ERROR or "the copy is not the pinned version"}')
    at, wall, clock, _ = resolve_clock(None, now)
    with e8_classes():
        prep = prepare(load, panel_doc, arms_doc, arms_sha256, manip, dropped, drop_sha256, manifest, at, block=block,
                       cutoff=cutoff, rerun_cutoff=rerun_cutoff, smoke_tasks=smoke_tasks, repeats=repeats,
                       checker_sha256=checker_sha256)
        kinds = {r['kind'] for r in prep['reasons']}
        cats = {r['category'] for r in prep['reasons']}
        base = dict(kind='e8_reruns', rules='v2.2', analysis=ANALYSIS_ID, block=block, wall_clock_utc=_iso(wall),
                    clock=clock, rerun_cutoff=rerun_cutoff, code_sha256=code_sha256(), reasons=prep['reasons'],
                    analysis_set=dict(prep['dispatch'], n_tasks=len(prep['tasks'])), blind=True,
                    checker_sha256_enforced=checker_sha256, runtime=runtime())
        if prep['manip']['missing_agent'] or {'manipulation_records'} & kinds:
            raise A7.AnalysisStop('reruns refused: manipulation-check records are missing (a dump not found), '
                                  'inconsistent or not from the frozen checker, so failed checks (M01) cannot be '
                                  'known; give every run directory with its dumps')
        inputs = [r['text'] for r in prep['reasons'] if r['category'] == 'input']
        if inputs:
            raise A7.AnalysisStop('reruns refused: the analysis inputs are incomplete (A8/A9); a re-run list made from '
                                  'them could repeat or skip attempts', details=inputs)
        if 'record' in cats:
            recs = [r['text'] for r in prep['reasons'] if r['category'] == 'record']
            return dict(base, decision='VOID', batch_kind=None, jobs=[], waiting_for_extra_batch=[], exhausted=[],
                        void=dict(block=block, record_problems=recs, instructions=VOID_STEP), next_step=VOID_STEP,
                        note='blind (E7 A9): the block is VOID; no re-run batch is made for a void block')
        if not prep['rerun_window_open']:
            raise A7.AnalysisStop(f'reruns refused: the re-run cut-off {rerun_cutoff} has passed (wall clock '
                                  f'{_iso(wall)}; missing slots are handled by the missing-slot rule)')
        panel = dict(panel='E8', tasks=list(prep['tasks']))
        all_panels = ([dict(panel='E8-smoke', tasks=list(smoke_tasks))] if smoke_tasks is not None
                      else _panels_for_a0(panel_doc))
        rr = A7.rerun_jobs_v21(load['runs'], panel, list(ARMS), repeats=repeats, block=block, dropped=dropped,
                               drop_sha256=drop_sha256, load=load, manifest=manifest, all_panels=all_panels)
    stage_of = {}
    if smoke_tasks is None:
        for s, ts in panel_doc['stages'].items():
            for t in ts:
                stage_of[t] = int(s)
    jobs = []
    for j in rr['jobs']:
        e = MJ.job_entry(block, j['arm'], j['task'], j['repeat_id'], j['attempt'], stage_of.get(j['task'], 0),
                         arms_doc, arms_sha256)
        e.update(a0_drop_sha256=j['a0_drop_sha256'], batch_kind=j['batch_kind'])
        jobs.append(e)
    refused = [A7.job_refusal_v22(j) for j in jobs if A7.job_refusal_v22(j)]
    if refused:
        raise A7.AnalysisStop('reruns refused: the next batch would hold jobs the runner must refuse (A9)',
                              details=refused[:20])
    rr['jobs'] = jobs
    rr.update(base, decision=None, rerun_builder='analyze_e7.rerun_jobs_v21 under classify_row_e8 (e8_classes)')
    return rr


def _write(prefix, obj, md):
    os.makedirs(os.path.dirname(os.path.abspath(prefix)) or '.', exist_ok=True)
    with open(prefix + '.json', 'w', encoding='utf-8') as f:
        json.dump(A7._jsonable(obj) if A7 is not None else obj, f, indent=1, default=str)
    with open(prefix + '.md', 'w', encoding='utf-8') as f:
        f.write(md)


def _refuse(a, argv, wall, clock, reason):
    """Exit 3: print the reason and write only <out>.refused.json (no outcome, no analysis)."""
    print(f'refused: {reason}', file=sys.stderr)
    rec = dict(kind='e8_refusal', command=a.cmd, status='REFUSED', blind=True, reason=reason,
               wall_clock_utc=_iso(wall), clock=clock, code_sha256=code_sha256(), runtime=runtime(),
               analysis=ANALYSIS_ID, argv=list(argv if argv is not None else sys.argv[1:]))
    try:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
        with open(a.out + '.refused.json', 'w', encoding='utf-8') as f:
            json.dump(rec, f, indent=1, default=str)
    except OSError as e:
        print(f'(could not write {a.out}.refused.json: {e})', file=sys.stderr)
    return 3


def frozen_hash_refusal(cmd, checker_given, analyze_e8_given):
    """LEAD_DECISIONS.md item 14 (PROTOCOL.md sections 6.4 and 15): report and reruns run only with the frozen hashes.
    Returns the refusal reason, or None. `check` needs neither flag; a flag given to it is verified the same way."""
    required = cmd in ('report', 'reruns')
    if required and FROZEN_CHECKER_SHA256 is None:
        return (f'{cmd} refused: the manipulation checker sha256 is not frozen yet (analyze_e8.FROZEN_CHECKER_SHA256 is '
                'None; it is filled in at freeze, PROTOCOL.md section 6.4)')
    for flag, given, want in (('--checker-sha256', checker_given, FROZEN_CHECKER_SHA256),
                              ('--analyze-e8-sha256', analyze_e8_given, ANALYZE_E8_SHA256)):
        if given is None:
            if required:
                return f'{cmd} requires {flag} (the frozen value printed in PROTOCOL.md)'
            continue
        if not (isinstance(given, str) and _HEX64_RE.fullmatch(given)):
            return f'{flag} {given!r} is not a lowercase sha256'
        if want is None:
            return f'{flag} given, but no checker hash is frozen in this analyze_e8.py (FROZEN_CHECKER_SHA256 is None)'
        if given != want:
            what = ('the frozen checker hash in this analyze_e8.py' if flag == '--checker-sha256'
                    else 'the sha256 of the analyze_e8.py that runs')
            return f'{flag} {given} differs from {what} ({want}): not the frozen version (PROTOCOL.md 6.4, 15)'
    return None


def main(argv=None):
    wall = _wall_clock()
    clock = 'system' if _wall_clock is _system_clock else 'simulated (tests)'
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('report', 'check', 'reruns'):
        p = sub.add_parser(name)
        p.add_argument('--runs', nargs='+', required=True)
        p.add_argument('--manifest', required=True)
        p.add_argument('--drop-file', required=True)
        p.add_argument('--tokenizer', required=True, help="the model's tokenizer.json (sha256 cc8d3a0c...)")
        p.add_argument('--template-kaggle', required=True, help='the Kaggle-v2 chat template (sha256 33204f1a...)')
        p.add_argument('--template-t', required=True, help='the HF canonical E4B template, arm T (sha256 0a2c8073...)')
        p.add_argument('--workers', type=int, default=4, help='processes for the manipulation checks (default 4)')
        p.add_argument('--metrics', nargs='*', default=[], help='vLLM pre-emption samples (JSON lines)')
        p.add_argument('--panel', default=str(HERE / 'panel_E8.json'))
        p.add_argument('--arms-file', default=str(HERE / 'e8_arms.json'))
        p.add_argument('--tasks', default=None, help='Berth tasks.jsonl (default: analyze_e7.DEFAULT_TASKS)')
        p.add_argument('--block', default=BLOCK)
        p.add_argument('--as-of', default=None,
                       help=('report/check only: a UTC time YYYY-MM-DDTHH:MM:SS[.ffffff]Z not later than the wall '
                             'clock (default: the wall clock); reruns ignores it'))
        p.add_argument('--smoke', action='store_true', help='the smoke or capture block: the 2 smoke tasks, 1 repeat')
        p.add_argument('--allow-unpinned-e7', action='store_true',
                       help='check and reruns only: run on an analyze_e7 copy that is not the pinned one (non-frozen)')
        p.add_argument('--checker-sha256', default=None,
                       help=('the frozen manipulation-checker sha256 as printed in the frozen PROTOCOL.md (section 6.4); '
                             'required by report and reruns, must equal FROZEN_CHECKER_SHA256'))
        p.add_argument('--analyze-e8-sha256', default=None,
                       help=('the frozen analyze_e8.py sha256 as printed in the frozen PROTOCOL.md (section 15); '
                             'required by report and reruns, must equal the sha256 of the analyze_e8.py that runs'))
        p.add_argument('--label', default='')
        p.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    # the pin (PROTOCOL.md section 8.6): report never runs unpinned; the override is for check and reruns only
    if a.cmd == 'report' and a.allow_unpinned_e7:
        return _refuse(a, argv, wall, clock, '--allow-unpinned-e7 is not accepted by report (PROTOCOL.md 8.6, 11.6)')
    if not A7_PINNED:
        why = (A7_LOAD_ERROR or f'analyze_e7 copy {A7_PATH} has sha256 {A7_SHA256}, not the pinned '
                                f'{PINNED_ANALYZE_E7_SHA256}')
        if not a.allow_unpinned_e7:
            return _refuse(a, argv, wall, clock, why)
        if A7_BYTES is None:
            return _refuse(a, argv, wall, clock, why + ' (nothing to run with --allow-unpinned-e7)')
        _bind_e7()
    if a.smoke and a.cmd == 'report':
        return _refuse(a, argv, wall, clock, 'smoke and capture outcomes are never opened; use `check --smoke`')
    why = frozen_hash_refusal(a.cmd, a.checker_sha256, a.analyze_e8_sha256) or checker_refusal()
    if why:
        return _refuse(a, argv, wall, clock, why)
    if not 1 <= a.workers <= 64:
        return _refuse(a, argv, wall, clock, f'--workers {a.workers} is not between 1 and 64')
    as_of_ignored = None
    if a.cmd == 'reruns':
        as_of_ignored, a_as_of = a.as_of, None          # reruns: the wall clock only (PROTOCOL.md section 11.3)
    else:
        a_as_of = a.as_of
        try:
            resolve_clock(a_as_of, wall)
        except Refusal as e:
            return _refuse(a, argv, wall, clock, str(e))
    if as_of_ignored is not None:
        print(f'note: reruns ignores --as-of ({as_of_ignored!r}); it uses the wall clock {_iso(wall)}', file=sys.stderr)
    repeats = 1 if a.smoke else REPEATS
    try:
        panel_doc, arms_doc, arms_sha, dropped, drop_sha, manifest, load, manip, metrics = _inputs(a)
        try:
            smoke = list(MJ.smoke_tasks_for(dropped)) if a.smoke else None
        except ValueError as e:
            raise A7.AnalysisStop(str(e))
        needed = set(panel_doc['tasks']) | {t for t, _ in MJ.SMOKE_CANDIDATES}
        task_repo = A7.load_task_repos(a.tasks or A7.DEFAULT_TASKS, needed=needed)
        task_repo.update({t: r for t, r in panel_doc.get('task_repo', {}).items() if t not in task_repo})
        if a.cmd == 'reruns':
            res = run_reruns(load, panel_doc, arms_doc, arms_sha, manip, dropped, drop_sha, manifest, block=a.block,
                             now=wall if clock != 'system' else None, smoke_tasks=smoke, repeats=repeats,
                             checker_sha256=FROZEN_CHECKER_SHA256)
            res['as_of_ignored'] = as_of_ignored
            if res.get('decision') == 'VOID':
                md = f"# E8 next batch (blind): none, the block is VOID\n\n{res['void']['instructions']}\n"
            else:
                md = (f"# E8 next batch (blind): {res['batch_kind']}, {len(res['jobs'])} job(s)\n\nwall clock "
                      f"{res['wall_clock_utc']}; waiting for the extra batch {len(res['waiting_for_extra_batch'])}; "
                      f"exhausted {len(res['exhausted'])}\n\n{res['note']}\n")
                with open(a.out + '.jobs.json', 'w', encoding='utf-8') as f:
                    json.dump(res['jobs'], f)
            rc = 0
        else:
            res = run_analysis(load, panel_doc, arms_doc, arms_sha, manip, dropped, drop_sha, manifest, task_repo,
                               a.cmd, block=a.block, as_of=a_as_of, now=wall if clock != 'system' else None,
                               smoke_tasks=smoke, repeats=repeats, metrics=metrics, label=a.label,
                               checker_sha256=FROZEN_CHECKER_SHA256)
            md = md_full(res) if not res['blind'] else md_blind(res)
            if not res['blind']:
                # the recomputed records are written only with an unblinded report (failed-call counts per row would
                # reveal context-length runs); blind outputs carry only their sha256 (manipulation_checks.computed)
                with open(a.out + '.manip.jsonl', 'w', encoding='utf-8') as f:
                    for rec in sorted(manip['records'], key=lambda x: str(x.get('job_id'))):
                        f.write(_canon(rec) + '\n')
            rc = 0
    except Refusal as e:
        return _refuse(a, argv, wall, clock, str(e))
    except A7.AnalysisStop as e:
        res = dict(kind='e8_error', command=a.cmd, status='STOPPED', blind=True, reason=str(e),
                   wall_clock_utc=_iso(wall), clock=clock, code_sha256=code_sha256(), runtime=runtime(),
                   as_of_ignored=as_of_ignored,
                   details=[A7._jsonable(d) if isinstance(d, dict) else str(d) for d in e.details][:500])
        md = f"# E8 analysis STOPPED (blind)\n\nwall clock {_iso(wall)}\n\n{e}\n"
        rc = 2
    res['inputs'] = dict(runs=[os.path.abspath(d) for d in a.runs], manifest=os.path.abspath(a.manifest),
                         drop_file=os.path.abspath(a.drop_file), tokenizer=os.path.abspath(a.tokenizer),
                         template_kaggle=os.path.abspath(a.template_kaggle),
                         template_t=os.path.abspath(a.template_t), workers=a.workers,
                         manip_check_e8_sha256=CHECKER_SHA256,
                         metrics=[os.path.abspath(p) for p in a.metrics], panel=os.path.abspath(a.panel),
                         arms_file=os.path.abspath(a.arms_file), analyze_e8_sha256=ANALYZE_E8_SHA256,
                         checker_sha256_given=a.checker_sha256, analyze_e8_sha256_given=a.analyze_e8_sha256,
                         make_jobs_e8_sha256=MJ_SHA256, analyze_e7_sha256=A7_SHA256,
                         argv=list(argv if argv is not None else sys.argv[1:]))
    if not A7_PINNED:
        res['non_frozen'] = 'analyze_e7 is not the pinned version (--allow-unpinned-e7; check and reruns only)'
    _write(a.out, res, md)
    print(json.dumps({k: res.get(k) for k in ('kind', 'status', 'blind', 'batch_kind', 'wall_clock_utc')}, default=str))
    return rc


if __name__ == '__main__':
    sys.exit(main())
