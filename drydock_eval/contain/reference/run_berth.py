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
"""Berth evaluation runner, Track D4 (deployment directory <deploy>/evald): ONE locked-down container per job (D3's design), plus
the SSRF filter, block-device I/O caps, strict ingestion, RUN_RECORD_SPEC v2.1/v2.2 records (a job that dies without a
results row gets the runner's own row, exc 'g4s-d4 infra: <why>' = class I09, spec v2.2 section 5.4) and per-job arm configs.

Each job runs in its own container of image g4s-sandbox (by image ID). PID 1 is docker/init.py; it runs the E8 worker
(D4.2: docker/e8/worker_berth_e8.py, the e6 Berth worker plus the E8 hook; D4/D4.1 ran the e6 worker through docker/arm.py)
with the job's arm config, so Phase 1 (agent) and Phase 2 (verification) both happen inside that container, in Kaggle's layout.

Isolation (D3, all flags in config.json and per row): --network none (the only way out is PID 1's relay 127.0.0.1:8000 ->
a per-job unix socket that this runner's HttpForwarder (lib/g4s_d4.py) parses: only POST /v1/chat/completions with
text-only JSON passes, to vLLM on 127.0.0.1 only), --read-only root, --cap-drop ALL, no-new-privileges, uid 2000,
tmpfs-only writable paths with size= and nr_inodes=, memory = memory+swap, pids, cpus, cpu-shares, ulimits (core 1:1,
fsize, memlock 64 KiB), io.max read/write caps on the disk behind the bind mounts, --log-driver none, oom-score-adj 500,
read-only bind mounts only, no Docker socket, no GPU. Results come back only as PID 1's nonce-framed tar on stdout.

Agent mode (model-issued commands) is refused unless (a) the GPU lock <deploy>/gpu.lock is held by an ancestor of
this process (flock -w 21600 ... run_berth.py), (b) the canary gate binds this configuration: canary_passed.json (or,
before it exists, --gate precanary:<canary tag> = a D4 canary record whose checks all passed with the same runner,
library and image), (c) every served name is free of "gemma4" except the single E8 arm-R alias allowed by the logged
runner exception (block E8*, frozen arm file, R jobs only). There is no default mode.

Track D4.1 (verifier findings of D4): the forwarder type-checks and size-caps every request field it keeps or records and caps
the distinct values per job (lib); ingestion is exception-safe (every JSON member parsed with a size and nesting cap, worker
row value types checked, PID 1's init.json = the tar's last member, any failure -> a tamper row (I12) with the evidence
recorded, or the runner's own I09 row; a job that was started always gets exactly one row); job containers get the D4.1
seccomp profile (Docker's default without Linux native AIO; lib/seccomp_g4s_d41.json); summary.json counts started jobs
without a row (started_without_row) separately from the runner's I09 rows (runner_rows).

Track D4.2 (D4.1 verifiers' findings and the E8 worker of E8 PROTOCOL.md 6.2): every job runs the E8 worker
(docker/e8/worker_berth_e8.py = the e6 worker plus the hook of docker/e8/e8_hook.py; arm.py is no longer used); the
runner writes the job's arm config to /tmp/e6/arm.json and accepts the four hook row fields; its own rows (I09 runner
rows and I12 tamper rows) carry them with null; for E8 jobs llm_dump_sha256 names the worker's request dump (which holds
the hook's failed calls) and E8 agent blocks need --dump 1. The runner refuses to start unless its cgroup has a memory
ceiling (a user-level systemd scope with MemoryMax, see launch.sh) and starts vLLM in a scope of its own. The forwarder's
parse path is bounded (lib). Gates: agent mode needs --block; --gate precanary:<tag> runs only named validation blocks
(SMOKE_*, CAPTURE_*, CANARY_*, E8smoke, E8capture) and needs a complete canary record (every part) bound to the image,
runner, library and harness files; every gate applies the scope, and a batch is in the E8 family when its block id
starts with E8, an arm file is given, or any job carries E8 fields or an arm config other than F's. E8 recovery block ids
(E8b, E8c, ...) are E8 blocks. For E8 blocks the runner log and summary.json do not show outcomes (resolved, agent
classes); infrastructure counts stay.

Track D4.3 (D4.2 escape verifier's two runner MEDIUMs): the forwarder's body scanner is linear (lib json_outside_strings);
canary mode may serve the real model (--canary-vllm) only in a CANARY_* block and only under canary_passed.json's binding
of all 8 keys of this configuration (and its scope; the flags key is the job flags, i.e. the canary run's flags minus its
one read-only mount of the canary program, whose tree sha256 config.json records), refused before the GPU lock otherwise
when the block or gate kind is wrong; --upstream-port must be a listener of the canary host (lib upstream_owner_problem);
a canary run with a non-CANARY_* block needs its own --manifest.

usage: run_berth.py --jobs <file>.json --tag <name> --mode agent|gold|empty|canary [--block <id>] [--drop-file <A0 list>]
                    [--arms-file e8_arms.json] [--request-template templates/<file>.jinja] [--workers 8] [--dump 1]
                    [--deadline-s 43200] [--job-deadline-s 2400] [--gate canary_passed|precanary:<tag>] ...
"""
import argparse, collections, contextlib, fcntl, hashlib, importlib.util, json, os, pathlib, queue, re, secrets, shutil, signal, socket
import statistics, subprocess, sys, tempfile, threading, time, urllib.request
# D4.2: glibc keeps freed memory in up to 8 x cores per-thread malloc arenas; the forwarder's connection threads then hold
# hundreds of MB after bursts of large bodies (measured: 2.1 GB vs 0.3 GB with 8 attacking jobs). The runner re-executes
# itself once (same PID) with at most 2 arenas; vLLM is started without the variable (lib.vllm_scope_argv).
if os.environ.get('MALLOC_ARENA_MAX') != '2':
    os.execve(sys.executable, sys.orig_argv, dict(os.environ, MALLOC_ARENA_MAX='2'))

ROOT = pathlib.Path(__file__).resolve().parent            # <deploy>/evald
EVAL = ROOT.parent / 'eval'                               # read-only: data, hostenv2
sys.path.insert(0, str(ROOT / 'lib'))
import g4s_d4 as D  # noqa: E402

MIN_FREE_MIB = 20 * 1024
MAX_UTIL = 0.55
MAX_WORKERS = 8
INIT = ['python3', '-I', '-S', '/usr/local/lib/g4s/init.py']
GPU_LOCK = ROOT.parent / 'gpu.lock'
HOST_LABEL = os.environ.get('DRYDOCK_CONTAIN_HOST_LABEL', 'host')   # 'host' field of summary.json / startup.json
E8_BLOCK_RE = re.compile(r'E8(?:[a-z]|smoke|capture)?')    # D4.2: E8, its recovery blocks E8b, E8c, ... (make_jobs_e8.BLOCK_RE), smoke, capture
VALIDATION_BLOCK_RE = re.compile(r'(?:SMOKE|CAPTURE|CANARY)_[A-Za-z0-9_.-]{1,40}|E8smoke|E8capture')   # --gate precanary only
CANARY_BLOCK_RE = re.compile(r'CANARY_[A-Za-z0-9_.-]{1,40}')     # D4.3: the only blocks in which canary mode may serve the model
GATE_KEYS = ('image_id', 'runner_sha256', 'lib_sha256', 'flags_sha256', 'options_sha256', 'harness_sha256', 'model_sha256',
             'vllm_args_sha256')                                   # the 8 keys canary_passed.json binds
PRECANARY_PARTS = ('battery', 'd4', 'd41', 'd42', 'd43', 'kills', 'backstop', 'sigkill', 'pdeath', 'gates', 'reaper', 'static')
HOOK_KEYS = ('e8_hook_bound', 'e8_hook_rebound', 'e8_hook_calls', 'e8_hook_calls_failed')


def is_e8_block(b):
    return b is not None and E8_BLOCK_RE.fullmatch(str(b)) is not None
A7_PATH = ROOT / 'tools' / 'analyze_e7.py'               # pinned copy (sha256 recorded in harness_sha256)

ap = argparse.ArgumentParser()
ap.add_argument('--jobs', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--mode', required=True, choices=['agent', 'gold', 'empty', 'canary'])
ap.add_argument('--block', default=None, help='block id written into every row (S, C, D, E8, ... or a validation block)')
ap.add_argument('--drop-file', default=None, help='A0 drop list; its sha256 is a0_drop_sha256 (jobs carrying one must match)')
ap.add_argument('--manifest', default=None, help='block manifest (default runs/manifests/<block>_manifest.json)')
ap.add_argument('--arms-file', default=None, help='frozen E8 arm file (enables the E8 fields and the arm-R alias exception)')
ap.add_argument('--request-template', default=None, help='E8 arm T: local template file (sha256 must equal the arm file)')
ap.add_argument('--gate', default='canary_passed', help='agent mode: canary_passed | precanary:<canary tag>')
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--image', default='g4s-sandbox:v8')
ap.add_argument('--job-deadline-s', type=int, default=2400)
ap.add_argument('--kill-grace-s', type=int, default=60)
ap.add_argument('--deadline-s', type=int, default=12 * 3600)
ap.add_argument('--memory', default='10g')
ap.add_argument('--cpus', default='2')
ap.add_argument('--cpu-shares', default='128')
ap.add_argument('--pids', type=int, default=4096)
ap.add_argument('--tmp-size', default='4g')
ap.add_argument('--tmp-inodes', type=int, default=131072)
ap.add_argument('--out-size', default='64m')
ap.add_argument('--out-inodes', type=int, default=8192)
ap.add_argument('--vartmp-size', default='512m')
ap.add_argument('--shm-size', default='256m')
ap.add_argument('--fsize', type=int, default=4 * 1024 ** 3)
ap.add_argument('--uid', type=int, default=2000)
ap.add_argument('--hostname', default=D.HOSTNAME)
ap.add_argument('--io-read-bps', type=int, default=64 * 1024 * 1024)
ap.add_argument('--io-read-iops', type=int, default=4000)
ap.add_argument('--io-write-bps', type=int, default=16 * 1024 * 1024)
ap.add_argument('--io-write-iops', type=int, default=1000)
ap.add_argument('--io-weight', type=int, default=10)
ap.add_argument('--frame-cap', type=int, default=96 * 1024 * 1024)
ap.add_argument('--gpu-util', type=float, default=0.55)
ap.add_argument('--port', type=int, default=8001, help='host vLLM port')
ap.add_argument('--relay-port', type=int, default=8000, help='in-container port of PID 1 relay (Kaggle serves on 8000)')
ap.add_argument('--upstream-port', type=int, default=0, help='canary mode only: forwarder upstream (a dummy server)')
ap.add_argument('--canary-vllm', type=int, default=0, help='canary mode only: start vLLM exactly as agent mode does (GPU lock '
                'required) and make it the forwarder upstream (the live SSRF canary)')
ap.add_argument('--min-free-gib', type=float, default=150.0)
ap.add_argument('--model-dir', default=str(ROOT.parent / 'models/served_model'))
ap.add_argument('--served-model-name', default=D.KAGGLE_MODEL)
ap.add_argument('--model-name', default='gemma-4-e4b-it-qat-w4a16-ct')
ap.add_argument('--max-model-len', type=int, default=32768)
ap.add_argument('--max-lora-rank', type=int, default=16)
ap.add_argument('--lora-bundles', default='', help='comma-separated bundles whose adapters the server serves even when no job '
                'of this batch uses them (a base-only re-run batch must be served with the same server settings as the other batches of its block)')
ap.add_argument('--vllm-language-only', type=int, default=1, help='vLLM --language-model-only (every multimodal limit 0)')
ap.add_argument('--time-scale', type=float, default=1.0)
ap.add_argument('--hold-before-jobs-s', type=int, default=0, help='wait after vLLM is up before the first job (probe window)')
ap.add_argument('--dump', type=int, default=0, help="worker-side request dumps (the e6 worker's --dump_llm)")
ap.add_argument('--capture', type=int, default=1, help='host-side request capture with prompt token ids (return_token_ids)')
ap.add_argument('--capture-bodies', type=int, default=0, help='host capture also keeps messages and tools')
ap.add_argument('--vllm-log-requests', type=int, default=0)
ap.add_argument('--vllm-debug', type=int, default=0)
ap.add_argument('--comp', default=str(EVAL / 'data/berth'))
ap.add_argument('--pyds', default=str(EVAL / 'data/pyds'))
ap.add_argument('--shims', default=str(EVAL / 'data/shims'))
ap.add_argument('--wheelhouse', default=str(ROOT.parent / 'wheelhouse'))
ap.add_argument('--worker-env', default=str(EVAL / 'hostenv2'))
ap.add_argument('--base-env', default=str(ROOT.parent / 'env-serve'))
ap.add_argument('--cache', default=str(ROOT / 'cache/work_d3'))
a = ap.parse_args()
if not 1 <= a.workers <= MAX_WORKERS:
    sys.exit(f'--workers must be 1..{MAX_WORKERS}')
if not 0 < a.gpu_util <= MAX_UTIL:
    sys.exit(f'--gpu-util must be in (0, {MAX_UTIL}]')
if os.nice(0) < 10:
    sys.exit('start the runner with nice -n 10 (see launch.sh)')
if not re.fullmatch(r'[A-Za-z0-9_.-]+', a.tag) or (a.block is not None and not re.fullmatch(r'[A-Za-z0-9_.-]+', a.block)):
    sys.exit('--tag and --block must match [A-Za-z0-9_.-]+')
if 'gemma4' in a.served_model_name.lower():
    sys.exit('--served-model-name contains "gemma4": google-adk would send tool results with role "tool_responses" (Kaggle: "tool")')
if (a.upstream_port or a.canary_vllm) and a.mode != 'canary':
    sys.exit('--upstream-port and --canary-vllm are for canary mode only')
if a.upstream_port and a.canary_vllm:
    sys.exit('--upstream-port and --canary-vllm exclude each other')
SERVE = a.mode == 'agent' or bool(a.canary_vllm)
if a.uid < 1001 or a.uid == os.getuid():
    sys.exit('--uid must be a dedicated uid (not the host account)')
if a.block == 'D' and a.deadline_s < 43200:
    sys.exit('the D block needs --deadline-s >= 43200 (A8)')
if a.mode == 'agent' and a.drop_file is None:
    sys.exit('agent mode needs --drop-file (the A0 list; [] for validation blocks)')
if a.mode == 'agent' and a.block is None:
    sys.exit('agent mode needs --block (D4.2: every agent row belongs to a named block and its manifest)')
if a.mode == 'agent' and a.gate.startswith('precanary:') and not VALIDATION_BLOCK_RE.fullmatch(a.block):
    sys.exit(f'refused: --gate precanary runs only named validation blocks (SMOKE_*, CAPTURE_*, CANARY_*, E8smoke, E8capture), '
             f'not block {a.block!r}')
# D4.3 (D4.2 escape verifier MEDIUM): canary mode serves the real model (--canary-vllm) only in a CANARY_* block and only
# under canary_passed.json's 8-key binding of this configuration (checked below with the agent gate, before any server);
# --upstream-port only reaches a listener of the canary host itself (lib.upstream_owner_problem); and a canary run never
# writes into a block's default manifest. Every refusal here comes before the GPU lock is looked at.
if a.mode == 'canary' and a.canary_vllm:
    if a.block is None or not CANARY_BLOCK_RE.fullmatch(a.block):
        sys.exit(f'refused: --canary-vllm serves the real model only in a CANARY_* block, not block {a.block!r}')
    if a.gate != 'canary_passed':
        sys.exit('refused: --canary-vllm needs the gate canary_passed.json (its 8-key binding); a canary record (precanary) binds 4 keys')
if a.mode == 'canary' and a.block is not None and not CANARY_BLOCK_RE.fullmatch(a.block):
    _default_manifest = (ROOT / 'runs' / 'manifests' / f'{a.block}_manifest.json').resolve()
    if a.manifest is None or pathlib.Path(a.manifest).resolve() == _default_manifest:
        sys.exit(f'refused: canary mode with block {a.block!r} needs its own --manifest (canary rows never enter the default manifest '
                 f'of a block; CANARY_* blocks are canary-only)')
if a.upstream_port:
    _up = D.upstream_owner_problem(a.upstream_port)
    if _up:
        sys.exit(f'refused: --upstream-port {a.upstream_port}: {_up}')

# ------------------------------------------------------------------------------------------------ agent-mode gates (D4: in the runner itself)
GATE = {'mode': a.mode, 'gate': a.gate if SERVE else None, 'serve': SERVE}
if SERVE:
    held, detail = D.lock_held_by_ancestor(GPU_LOCK)
    GATE['gpu_lock'] = detail
    if not held:
        sys.exit(f'refused: agent mode and --canary-vllm need the GPU lock held by an ancestor (flock -w 21600 {GPU_LOCK} ...): {detail}')
# D4.2: the runner's own memory ceiling (a user-level systemd scope with MemoryMax and no swap; launch.sh and the canaries
# start it so); vLLM gets a scope of its own below, so this ceiling bounds the runner, its docker CLIs and job frames only
MEMORY = D.cgroup_memory()
_mp = D.memory_ceiling_problem(MEMORY)
if _mp:
    sys.exit(f'refused: the runner must run under a memory ceiling (systemd-run --user --scope -p MemoryMax=... -p '
             f'MemorySwapMax=0, see launch.sh): {_mp}')
GATE['memory'] = {k: MEMORY.get(k) for k in ('cgroup', 'memory.max', 'memory.swap.max')}

RUN = ROOT / 'runs' / a.tag
OUT = RUN / 'out'
if RUN.exists() and any(RUN.iterdir()):
    sys.exit(f'{RUN} exists; use a new --tag')
for d in (RUN, OUT, RUN / 'logs', RUN / 'jobs', RUN / 'env', RUN / 'arm', RUN / 'capture', RUN / 'refused'):
    d.mkdir(parents=True, exist_ok=True)
os.chmod(RUN / 'env', 0o700)
(RUN / 'tmp').mkdir(exist_ok=True)
CACHE_HOST = ROOT / 'cache'
HOST_ENV = {'TRANSFORMERS_NO_TF': '1', 'TRANSFORMERS_NO_FLAX': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_OFFLINE': '1', 'VLLM_NO_USAGE_STATS': '1',
            'DO_NOT_TRACK': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_ENGINE_READY_TIMEOUT_S': '1200',
            'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1', 'NCCL_DEBUG': 'WARN', 'LITELLM_LOCAL_MODEL_COST_MAP': 'True',
            'LITELLM_TELEMETRY': 'False', 'LITELLM_LOG': 'WARNING', 'OPENAI_API_KEY': 'EMPTY', 'LOCAL_API_KEY': 'EMPTY',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'OTEL_SDK_DISABLED': 'true', 'OTEL_PYTHON_DISABLED': 'true',
            'CUDA_VISIBLE_DEVICES': '0', 'TMPDIR': str(RUN / 'tmp'), 'XDG_CACHE_HOME': str(CACHE_HOST / 'xdg'),
            'XDG_CONFIG_HOME': str(CACHE_HOST / 'config'), 'VLLM_CACHE_ROOT': str(CACHE_HOST / 'vllm'),
            'TRITON_CACHE_DIR': str(CACHE_HOST / 'triton'), 'TORCHINDUCTOR_CACHE_DIR': str(CACHE_HOST / 'inductor'),
            'FLASHINFER_WORKSPACE_BASE': str(CACHE_HOST / 'flashinfer'), 'CUDA_CACHE_PATH': str(CACHE_HOST / 'nv'),
            'HF_HOME': str(CACHE_HOST / 'hf'), 'MPLCONFIGDIR': str(CACHE_HOST / 'mpl'), 'PIP_DISABLE_PIP_VERSION_CHECK': '1',
            'VLLM_MEDIA_URL_ALLOW_REDIRECTS': '0'}          # D4: vLLM follows no redirect when it fetches media (it fetches none)
os.environ.update(HOST_ENV)
os.environ.pop('VLLM_ALLOW_RUNTIME_LORA_UPDATING', None)      # no runtime LoRA load/unload endpoints
tempfile.tempdir = None
for _k in ('XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'FLASHINFER_WORKSPACE_BASE',
           'CUDA_CACHE_PATH', 'HF_HOME', 'MPLCONFIGDIR'):
    pathlib.Path(HOST_ENV[_k]).mkdir(parents=True, exist_ok=True)
T0 = time.time()
LOGF = open(RUN / 'runner.log', 'a', buffering=1)
LOGLOCK = threading.Lock()


def log(*x):
    msg = f'[{time.time() - T0:7.0f}s] ' + ' '.join(str(v) for v in x)
    with LOGLOCK:
        print(msg, file=LOGF, flush=True)
        print(msg, flush=True)


def refuse(why):
    STATE['reason'] = f'abort: {why}'
    log('REFUSED:', why)
    raise SystemExit(f'refused: {why}')


DOCKER_CONFIG = ROOT / 'cache/dockercli'
DOCKER_CONFIG.mkdir(parents=True, exist_ok=True)
DENV = D.denv(DOCKER_CONFIG)
RUNNER_SHA = D.sha256_file(__file__)
LIB_SHA = D.sha256_file(ROOT / 'lib/g4s_d4.py')
TOKEN = D.owner_token()
(RUN / 'owner.json').write_text(json.dumps({'token': TOKEN, 'pid': os.getpid(), 'runner_sha256': RUNNER_SHA, 'lib_sha256': LIB_SHA,
                                            'argv': sys.argv}, indent=1))
os.environ['GEMMA4_EVAL_RUN'] = str(RUN)
UMASK = os.umask(0)
os.umask(UMASK)
STATE = {'srv': None, 'vllm_pids': [], 'reason': 'finished', 'stop': False, 'running': {}, 'cleaned': False}
ROWLOCK = threading.Lock()
IO_DEV, IO_MM = D.disk_of([a.comp, a.wheelhouse, a.pyds, a.shims, a.worker_env, a.base_env, a.cache, ROOT / 'bundles'])
SECCOMP = D.seccomp_profile(ROOT / 'lib')                  # D4.1: pinned profile (Docker default minus Linux AIO)
C = {'seccomp': SECCOMP, 'memory': a.memory, 'pids': a.pids, 'cpus': a.cpus, 'cpu_shares': a.cpu_shares, 'fsize': a.fsize, 'hostname': a.hostname,
     'tmp_size': a.tmp_size, 'tmp_inodes': a.tmp_inodes, 'out_size': a.out_size, 'out_inodes': a.out_inodes,
     'vartmp_size': a.vartmp_size, 'shm_size': a.shm_size, 'io_device': IO_DEV, 'io_weight': a.io_weight,
     'read_bps': a.io_read_bps, 'read_iops': a.io_read_iops, 'write_bps': a.io_write_bps, 'write_iops': a.io_write_iops}
WORKER_PY = str(pathlib.Path(a.worker_env) / 'bin/python')
CACHE = pathlib.Path(a.cache)
TASKS = {}
for line in open(pathlib.Path(a.comp) / 'tasks.jsonl', encoding='utf-8'):
    r = json.loads(line)
    TASKS[r['instance_id']] = r


def on_signal(signum, frame):
    STATE['reason'] = f'signal {signum}'
    STATE['stop'] = True
    raise SystemExit(128 + signum)


for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(s, on_signal)


def free_gib(path):
    return shutil.disk_usage(path).free / 2 ** 30


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A7 = load_module('analyze_e7_pinned', A7_PATH)          # classify_row_v22/v21, ERROR_TABLE_SHA256_V22/V21, job_id_v2
# RUN_RECORD_SPEC v2.2 (A9): error_class/error_class_id/error_table_sha256 from classify_row_v22 when the deployed analysis
# has it; the v2.1 class is recorded as well (error_class_id_v21) for analyses pinned to v2.1 (E8)
CLASSIFY = getattr(A7, 'classify_row_v22', A7.classify_row_v21)
TABLE_SHA = getattr(A7, 'ERROR_TABLE_SHA256_V22', A7.ERROR_TABLE_SHA256_V21)


# ------------------------------------------------------------------------------------------------ provenance (computed once)
def site_packages(env_root):
    c = sorted(pathlib.Path(env_root).glob('lib/python3*/site-packages'))
    return c[0] if c else None


def harness_hashes():
    """RUN_RECORD_SPEC section 2 harness_sha256, D4 paths: the files every job container imports or runs (image files by
    their host sources, which build.sh checked equal to the image; env files from the read-only mounted envs) and the
    pinned analyze_e7.py that classifies rows."""
    sp = site_packages(a.base_env)
    sp2 = site_packages(a.worker_env)
    h = {'lib/worker_berth_e8.py': D.sha256_file(ROOT / 'docker/e8/worker_berth_e8.py'),
         'lib/e8_hook.py': D.sha256_file(ROOT / 'docker/e8/e8_hook.py'),
         'lib/sandbox_berth.py': D.sha256_file(ROOT / 'docker/e6/sandbox_berth.py'),
         'docker/init.py': D.sha256_file(ROOT / 'docker/init.py'),
         'docker/Dockerfile': D.sha256_file(ROOT / 'docker/Dockerfile'), 'lib/g4s_d4.py': LIB_SHA, 'run_berth.py': RUNNER_SHA,
         'analyze_e7.py': D.sha256_file(A7_PATH)}
    for rel in ('swegemma/evaluate.py', 'swegemma/harness/agent_runner.py', 'swegemma/harness/verification.py',
                'swegemma/harness/container_setup.py', 'swegemma/sandbox/subprocess.py', 'swegemma/sandbox/base.py',
                'swegemma/tools/execution.py', 'swegemma/tools/workspace.py', 'swegemma/config.py',
                'adk_eval_core/sandbox/subprocess_sandbox.py', 'google/adk/models/lite_llm.py', 'google/adk/apps/compaction.py',
                'adk_submission/server.py', 'adk_submission/registry.py'):
        for base in (sp2, sp):
            if base is not None and (base / rel).is_file():
                h[rel] = D.sha256_file(base / rel)
                break
    seen = set()
    for base in (sp2, sp):                     # the worker env (hostenv2) shadows env-serve, as on the worker's sys.path
        if base is None:
            continue
        for rec in sorted(base.glob('*.dist-info/RECORD')):
            pkg = rec.parent.name.split('-')[0].lower()
            if pkg in ('swegemma', 'google_adk', 'litellm', 'openai', 'adk_submission', 'vllm') and pkg not in seen:
                seen.add(pkg)
                h[f'{rec.parent.name}/RECORD'] = D.sha256_file(rec)
    return h


def model_hash(model_dir):
    """sha256 of the model directory: tree hash with a cache for large files keyed by (path, size, mtime_ns, inode)."""
    cache_f = CACHE_HOST / 'sha_cache.json'
    try:
        cache = json.load(open(cache_f))
    except (OSError, ValueError):
        cache = {}
    lines = []
    for p in sorted(pathlib.Path(model_dir).rglob('*')):
        if p.is_symlink() or not p.is_file() or '__pycache__' in p.parts:
            continue
        st = p.stat()
        key = f'{p}|{st.st_size}|{st.st_mtime_ns}|{st.st_ino}'
        if key not in cache:
            cache[key] = D.sha256_file(p)
        lines.append(f'{cache[key]}  {p.relative_to(model_dir).as_posix()}\n')
    tmp = cache_f.with_suffix('.tmp')
    tmp.write_text(json.dumps(cache))
    os.replace(tmp, cache_f)
    return hashlib.sha256(''.join(lines).encode()).hexdigest()


def read_agent_model(bundle_dir):
    """(model, adapter) declared by the bundle's agent.yaml (`model:` and the optional `adapter:` line of export_adapter.py)."""
    t = (bundle_dir / 'agent.yaml').read_text()
    m = re.search(r'^model:\s*(\S+)\s*$', t, re.M)
    ad = re.search(r'^adapter:\s*(\S+)\s*$', t, re.M)
    return (m.group(1) if m else None), (ad.group(1) if ad else None)


# ------------------------------------------------------------------------------------------------ jobs, blocks, arms
JOBS_RAW = json.load(open(a.jobs))
if not isinstance(JOBS_RAW, list) or not JOBS_RAW or not all(isinstance(j, dict) for j in JOBS_RAW):
    sys.exit('--jobs must be a non-empty JSON list of job objects')
DROP_SHA = D.sha256_file(a.drop_file) if a.drop_file else None
ARMS_DOC = json.load(open(a.arms_file)) if a.arms_file else None
ARMS_SHA = D.sha256_file(a.arms_file) if a.arms_file else None
TEMPLATE_TEXT, TEMPLATE_SHA = None, None
if a.request_template:
    TEMPLATE_TEXT = pathlib.Path(a.request_template).read_text(encoding='utf-8')
    TEMPLATE_SHA = D.sha256_bytes(TEMPLATE_TEXT.encode())
    if pathlib.Path(a.request_template).resolve().parent != (ROOT / 'templates').resolve():
        sys.exit('--request-template must be a file in evald/templates (mounted read-only into T jobs)')
SERVER_TEMPLATE_SHA = D.sha256_file(pathlib.Path(a.model_dir) / 'chat_template.jinja') if (pathlib.Path(a.model_dir) / 'chat_template.jinja').is_file() else None
DEFAULT_COMPACTION = {'compaction_interval': 15, 'overlap_size': 2, 'token_threshold': 14336, 'event_retention_size': 5}
WORKER_SRC = (ROOT / 'docker/e8/worker_berth_e8.py').read_text()
assert ('EventsCompactionConfig(compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5)' in WORKER_SRC), \
    "the worker's literal compaction config is not the default arm config"
EXCEPTIONS = []


def normalise(job, idx):
    """A job entry with every record field the runner needs (D4): block, repeat, attempt, job_id (checked against
    analyze_e7.job_id_v2), a0_drop_sha256 (checked against --drop-file), and the resolved arm config."""
    j = dict(job)
    for k in ('arm', 'task'):
        if not isinstance(j.get(k), str) or not re.fullmatch(r'[A-Za-z0-9_.@+=,-]{1,120}', j[k]):
            sys.exit(f'job {idx}: bad {k!r}')
    if a.mode != 'canary' and j['task'] not in TASKS:
        sys.exit(f'job {idx}: unknown task {j["task"]}')
    rep = j.get('repeat_id', j.get('seed', 1))
    if j.get('seed', rep) != rep or not isinstance(rep, int) or isinstance(rep, bool) or rep < 1:
        sys.exit(f'job {idx}: seed/repeat_id must be one positive integer')
    j['seed'] = j['repeat_id'] = rep
    att = j.get('attempt', 1)
    if not isinstance(att, int) or isinstance(att, bool) or not 1 <= att <= 5:
        sys.exit(f'job {idx}: attempt must be 1..5')
    j['attempt'] = att
    blk = j.get('block', a.block)
    if a.block is not None and blk != a.block:
        sys.exit(f'job {idx}: block {blk!r} differs from --block {a.block!r}')
    if blk is None:
        blk = f'X{a.mode}'
    j['block'] = blk
    want = A7.job_id_v2(blk, j['arm'], j['task'], rep, att)
    if j.get('job_id', want) != want:
        sys.exit(f'job {idx}: job_id {j.get("job_id")!r} != {want!r}')
    j['job_id'] = want
    if a.mode == 'agent':
        why = A7.job_refusal_v22(j) if hasattr(A7, 'job_refusal_v22') else (
            None if re.fullmatch(r'[0-9a-f]{64}', str(j.get('a0_drop_sha256', ''))) else 'no a0_drop_sha256')
        if why:
            sys.exit(f'refused (A9): job {idx}: {why}')
    if DROP_SHA is not None:
        if j.get('a0_drop_sha256', DROP_SHA) != DROP_SHA:
            sys.exit(f'job {idx}: a0_drop_sha256 differs from --drop-file (A8)')
        j['a0_drop_sha256'] = DROP_SHA
    elif not j.get('a0_drop_sha256'):
        j['a0_drop_sha256'] = f'none:{a.mode}'
    # arm config (E8 fields when given; E7/P1 jobs: the bundle = the arm, F's settings)
    bundle = j.get('bundle', j['arm'] if a.mode != 'canary' else 'bundle0')
    if not (ROOT / 'bundles' / bundle / 'agent.yaml').is_file():
        sys.exit(f'job {idx}: no bundle {bundle}')
    j['bundle'] = bundle
    cc = j.get('compaction_config', DEFAULT_COMPACTION)
    hr = j.get('history_reasoning', 'echo')
    ts = j.get('request_chat_template_sha256')
    if cc is not None and cc != DEFAULT_COMPACTION:
        sys.exit(f'job {idx}: compaction_config must be null or the worker literal {DEFAULT_COMPACTION}')
    if hr not in ('echo', 'strip'):
        sys.exit(f'job {idx}: history_reasoning must be echo|strip')
    if ts is not None and (ts != TEMPLATE_SHA or not ARMS_DOC or not is_e8_block(blk)):
        sys.exit(f'job {idx}: a per-request chat template needs block E8*, --arms-file and --request-template with sha {ts}')
    adapters = sorted(p.name for p in (ROOT / 'bundles' / bundle / 'adapters').iterdir() if p.is_dir()) \
        if (ROOT / 'bundles' / bundle / 'adapters').is_dir() else []
    agent_model, agent_adapter = read_agent_model(ROOT / 'bundles' / bundle)
    if agent_model != a.model_name:
        sys.exit(f'job {idx}: bundle {bundle} agent model {agent_model} != --model-name {a.model_name}')
    if adapters or agent_adapter:
        if agent_adapter not in adapters:
            sys.exit(f'job {idx}: bundle {bundle} declares adapter {agent_adapter} but has adapters {adapters}')
        req_model, served = f'openai/{agent_adapter}', a.served_model_name
    else:
        req_model = j.get('request_model', f'openai/{a.served_model_name}')
        served = req_model[len('openai/'):] if req_model.startswith('openai/') else None
        if served is None:
            sys.exit(f'job {idx}: request_model must start with openai/')
    if j.get('request_model', req_model) != req_model:
        sys.exit(f'job {idx}: request_model {j.get("request_model")!r} != what the runner would send {req_model!r}')
    if 'gemma4' in req_model.lower():
        alias = served
        ok = (ARMS_DOC is not None and is_e8_block(blk) and j['arm'] == 'R' and alias in (ARMS_DOC.get('served_model_names') or [])
              and alias != a.served_model_name and ARMS_DOC['arms'].get('R', {}).get('request_model') == req_model)
        if not ok:
            sys.exit(f'job {idx}: request model {req_model!r} contains "gemma4" (allowed only for E8 arm R with the frozen arm file)')
    arm_cfg = {'bundle': bundle, 'request_model': req_model, 'compaction_config': cc, 'history_reasoning': hr,
               'request_chat_template_sha256': ts}
    j['_served'] = served
    j['_adapters'] = adapters
    j['_agent_model'] = agent_adapter or agent_model
    j['arm_config'] = arm_cfg
    j['arm_config_sha256'] = D.canonical_sha256(arm_cfg)
    if ARMS_DOC is not None and is_e8_block(blk):
        entry = ARMS_DOC['arms'].get(j['arm'])
        if entry is None:
            sys.exit(f'job {idx}: arm {j["arm"]} not in the arm file')
        if j.get('e8_arms_sha256', ARMS_SHA) != ARMS_SHA:
            sys.exit(f'job {idx}: e8_arms_sha256 differs from --arms-file')
        cfg_sha = D.canonical_sha256(entry)
        if j.get('e8_arm_config_sha256', cfg_sha) != cfg_sha:
            sys.exit(f'job {idx}: e8_arm_config_sha256 differs from the arm file entry')
        for k in ('request_model', 'compaction_config', 'history_reasoning', 'request_chat_template_sha256'):
            if D.canonical(entry.get(k)) != D.canonical(arm_cfg.get(k)):
                sys.exit(f'job {idx}: arm {j["arm"]} {k} {arm_cfg.get(k)!r} != arm file {entry.get(k)!r}')
        if bundle != ARMS_DOC.get('bundle'):
            sys.exit(f'job {idx}: bundle {bundle} != arm file bundle {ARMS_DOC.get("bundle")}')
        j['e8_arms_sha256'] = ARMS_SHA
        j['e8_arm_config_sha256'] = cfg_sha
        j.setdefault('stage', 0)
    return j


JOBS = [normalise(j, i) for i, j in enumerate(JOBS_RAW)]
dup = [k for k, n in collections.Counter(j['job_id'] for j in JOBS).items() if n > 1]
if dup:
    sys.exit(f'duplicate job_id in the job file: {dup[:3]}')
SERVED = [a.served_model_name] + sorted({j['_served'] for j in JOBS if not j['_adapters'] and j['_served'] != a.served_model_name})
if len(SERVED) > 2:
    sys.exit(f'at most one extra served name (the E8 arm-R alias): {SERVED}')
if len(SERVED) == 2:
    EXCEPTIONS.append({'kind': 'E8 arm-R served-name alias', 'alias': SERVED[1], 'block': a.block, 'arms_sha256': ARMS_SHA,
                       'jobs': sum(1 for j in JOBS if j['_served'] == SERVED[1])})
TRUST_TEMPLATE = any(j['arm_config']['request_chat_template_sha256'] for j in JOBS)
if TRUST_TEMPLATE:
    EXCEPTIONS.append({'kind': 'E8 arm-T --trust-request-chat-template', 'template_sha256': TEMPLATE_SHA, 'block': a.block,
                       'jobs': sum(1 for j in JOBS if j['arm_config']['request_chat_template_sha256'])})
ADAPTERS = {}
for j in JOBS:
    for n in j['_adapters']:
        if 'gemma4' in n.lower() or 'gemma4' in j['bundle'].lower():
            sys.exit(f'adapter or bundle name contains "gemma4": {n} / {j["bundle"]}')
        p = ROOT / 'bundles' / j['bundle'] / 'adapters' / n
        if n in ADAPTERS and ADAPTERS[n]['path'] != str(p):
            sys.exit(f'two bundles define adapter {n}')
        ADAPTERS[n] = {'path': str(p), 'sha256': D.tree_hash(p), 'bundle': j['bundle']}
for lb in [x for x in a.lora_bundles.split(',') if x]:
    adir = ROOT / 'bundles' / lb / 'adapters'
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', lb) or not adir.is_dir():
        sys.exit(f'--lora-bundles: no adapters in bundle {lb}')
    for n in sorted(x.name for x in adir.iterdir() if x.is_dir()):
        if 'gemma4' in n.lower():
            sys.exit(f'adapter name contains "gemma4": {n}')
        if n in ADAPTERS and ADAPTERS[n]['path'] != str(adir / n):
            sys.exit(f'two bundles define adapter {n}')
        ADAPTERS[n] = {'path': str(adir / n), 'sha256': D.tree_hash(adir / n), 'bundle': lb}
LORA = {n: v['path'] for n, v in ADAPTERS.items()}
SERVER_SETTINGS = {'max_model_len': a.max_model_len, 'enable_lora': bool(LORA) and SERVE,
                   'max_lora_rank': a.max_lora_rank if (LORA and SERVE) else None,
                   'served_adapter_names': sorted(LORA) if SERVE else []}
BUNDLE_SHA = {b: D.tree_hash(ROOT / 'bundles' / b) for b in sorted({j['bundle'] for j in JOBS})}
SERVED_RECORD = SERVED[0] if len(SERVED) == 1 else SERVED
if ARMS_DOC is not None and is_e8_block(a.block):
    SERVED_RECORD = list(ARMS_DOC.get('served_model_names') or SERVED)   # E8 rows: the block-wide list (PROTOCOL 6.3)
    if a.mode == 'agent' and SERVED != SERVED_RECORD and len(SERVED) == 2:
        sys.exit(f'served names {SERVED} differ from the arm file {SERVED_RECORD}')
    if a.mode == 'agent' and len(SERVED) == 1:
        SERVED = SERVED_RECORD                              # the E8 server always carries both names (one server for all arms)
        EXCEPTIONS.append({'kind': 'E8 arm-R served-name alias (served; no R job in this batch)', 'alias': SERVED[1], 'block': a.block})
    # D4.2: the block-wide template setting applies to every batch of an E8 block, also a batch without a T job and also in
    # canary mode (whose records mirror agent mode's); without --request-template such a batch's rows would say false and
    # void the block (analyze_e8 provenance), so it is refused
    if ARMS_DOC.get('trust_request_chat_template') and a.mode in ('agent', 'canary') and not TEMPLATE_SHA:
        sys.exit('refused: E8 blocks need --request-template (the arm file serves every batch with --trust-request-chat-template)')
    if a.mode in ('agent', 'canary') and ARMS_DOC.get('trust_request_chat_template') and not TRUST_TEMPLATE and TEMPLATE_SHA:
        TRUST_TEMPLATE = True
        EXCEPTIONS.append({'kind': 'E8 --trust-request-chat-template (block-wide; no T job in this batch)', 'block': a.block})
for e in EXCEPTIONS:
    log('RUNNER EXCEPTION (logged):', json.dumps(e))
# D4.2: the experiment family of this batch, from the block id AND the job content (a batch with an E8 arm config or E8
# fields under another block id is E8 too); every gate applies the scope to it
F_ARM = {'compaction_config': DEFAULT_COMPACTION, 'history_reasoning': 'echo', 'request_chat_template_sha256': None}
E8_JOB_KEYS = ('stage', 'e8_arms_sha256', 'e8_arm_config_sha256')
FAMILY = 'E8' if (is_e8_block(a.block) or ARMS_DOC is not None
                  or any(any(k in j for k in E8_JOB_KEYS) for j in JOBS_RAW)
                  or any(any(D.canonical(j['arm_config'].get(k)) != D.canonical(v) for k, v in F_ARM.items())
                         or 'gemma4' in j['arm_config']['request_model'].lower() for j in JOBS)) else 'E7/P1'
BLIND = FAMILY == 'E8'               # E8 PROTOCOL 7.1/11.6: the runner shows no outcome of an E8 batch
if FAMILY == 'E8' and a.mode == 'agent' and not a.dump:
    sys.exit('refused: E8 batches need --dump 1 (the worker request dump holds the hook\'s failed calls; PROTOCOL 6.2(f), 6.4)')


# ------------------------------------------------------------------------------------------------ flags
def canary_mount():
    return D.ro(ROOT / 'tools/d4canary', '/tmp/e6/canary')


def static_flags(uid, with_canary=None):
    """The flags every job/canary container of this run gets (also recorded in config.json and in every results row).
    Canary mode adds one read-only mount, the canary program (with_canary=False gives the job flags without it, D4.3)."""
    f = D.base_flags(C, uid, uid)
    f += D.ro(a.comp, '/tmp/berth') + D.ro(a.wheelhouse, '/tmp/wheelhouse') + D.ro(a.pyds, '/tmp/pyds') + D.ro(a.shims, '/tmp/shims')
    f += D.ro(a.worker_env) + D.ro(a.base_env) + D.ro(CACHE / 'py313', '/tmp/work/py313')
    if (a.mode == 'canary') if with_canary is None else with_canary:
        f += canary_mount()
    return f


def deps_tasks(job):
    if a.mode == 'canary':
        return list(job.get('deps_tasks', []))
    return [job['task']]


def deps_keys(job):
    return sorted({f'deps_{D.deps_key(TASKS[t]["requirements"])}' for t in deps_tasks(job)})


def dyn_mounts(job):
    """Per-job read-only mounts: the deps folders, the job's OWN bundle at /tmp/e6/bundles/<arm> (no other arm's files are
    visible), the request-template directory for T jobs."""
    m = []
    for k in deps_keys(job):                 # tasks with identical requirements share one deps folder: mount it once
        m += D.ro(CACHE / k, f'/tmp/work/{k}')
    m += D.ro(ROOT / 'bundles' / job['bundle'], f'/tmp/e6/bundles/{job["arm"]}')
    if job['arm_config']['request_chat_template_sha256']:
        m += D.ro(ROOT / 'templates', '/tmp/e6/templates')
    return m


def prep(tasks):
    """Build py313 and the deps folders of `tasks` that are not in the cache yet, in an isolated prep container (uid of
    this runner, which owns the cache directory; trusted input only: tasks.jsonl requirements + the local wheel pool)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    missing = sorted({t for t in tasks if not (CACHE / f'deps_{D.deps_key(TASKS[t]["requirements"])}' / '.done').exists()})
    if not missing and (CACHE / 'py313/python/bin/python3.13').exists():
        return {'missing': 0}
    name = f'g4s-{a.tag}-prep-{secrets.token_hex(3)}'
    nonce = secrets.token_hex(16)
    uid = os.getuid()
    cfg = {'nonce': nonce, 'mode': 'prep', 'deadline_s': 3600, 'umask': '0022', 'nice': 10,
           'mkdirs': ['/kaggle/working/out', '/tmp/home'], 'files': {'/tmp/e6/prep_tasks.txt': '\n'.join(missing) + '\n'},
           'argv': ['/usr/bin/python3', '-I', '/usr/local/lib/g4s/prep.py', '/tmp/e6/prep_tasks.txt'],
           'env': {'PATH': '/usr/sbin:/usr/bin:/sbin:/bin', 'HOME': '/tmp/home', 'TMPDIR': '/tmp', 'LANG': 'C.UTF-8',
                   'BERTH_SHIMS': '/tmp/shims', 'PIP_DISABLE_PIP_VERSION_CHECK': '1'},
           'statfs': D.STATFS, 'max_file_bytes': 8 << 20, 'max_total_bytes': 16 << 20}
    envf = RUN / 'env' / f'{name}.env'
    D.write_env_file(envf, cfg)
    argv = [D.DOCKER, 'run', '--name', name, '--label', D.LABEL, '--label', f'g4s.run={RUN}', '--label', f'g4s.owner={TOKEN}',
            '--label', 'g4s.kind=prep', *D.base_flags(dict(C, io_device=None), uid, uid, skip=('/tmp/work',)),
            '--mount', f'type=bind,src={CACHE},dst=/tmp/work', *D.ro(a.comp, '/tmp/berth'), *D.ro(a.pyds, '/tmp/pyds'),
            *D.ro(a.shims, '/tmp/shims'), '--env-file', str(envf), IMAGE_ID, *INIT]
    log('prep: building deps for', len(missing), 'tasks in', name)
    r = D.run_container(DENV, argv, name, 3700, 32 << 20, 64 << 10, log=log)
    envf.unlink(missing_ok=True)
    payload, err = D.parse_frame(r['stdout'], nonce)
    res = {'missing': len(missing), 'rc': r['rc'], 'frame': err or 'ok', 'wall_s': r['wall_s'], 'gone': r['gone']}
    if payload:
        for rel, data in D.tar_members(payload, 8 << 20, 16 << 20)[0]:
            if rel == 'out/prep.json':
                res['prep'] = json.loads(data)
            elif rel == 'worker.log':
                D.write_new(RUN / 'logs', f'{name}.log', data)
    still = sorted({t for t in tasks if not (CACHE / f'deps_{D.deps_key(TASKS[t]["requirements"])}' / '.done').exists()})
    res['still_missing'] = still
    if still or not (CACHE / 'py313/python/bin/python3.13').exists():
        STATE['reason'] = 'abort: prep failed'
        raise SystemExit(f'prep failed: {still[:5]} {err} rc {r["rc"]}')
    return res


def job_cfg(slot, job, nonce):
    mkdirs = ['/kaggle/working/out', '/tmp/home', '/tmp/e6/lib']
    copy = [['/usr/local/lib/g4s/e8/worker_berth_e8.py', '/tmp/e6/lib/worker_berth_e8.py'],
            ['/usr/local/lib/g4s/e8/e8_hook.py', '/tmp/e6/lib/e8_hook.py'],
            ['/usr/local/lib/g4s/e6/lib/sandbox_berth.py', '/tmp/e6/lib/sandbox_berth.py']]
    env = D.worker_env(C, RUN)
    armc = dict(job['arm_config'])
    if armc['request_chat_template_sha256']:
        armc['request_chat_template_path'] = f'/tmp/e6/templates/{pathlib.Path(a.request_template).name}'
    wjob = {k: job[k] for k in ('arm', 'task', 'seed', 'repeat_id', 'attempt', 'block', 'job_id')}
    files = {'/tmp/e6/jobs.jsonl': json.dumps(wjob) + '\n', '/tmp/e6/arm.json': json.dumps(armc)}
    wargv = [WORKER_PY, '-u', '/tmp/e6/lib/worker_berth_e8.py', '--jobs', '/tmp/e6/jobs.jsonl', '--arm_config', '/tmp/e6/arm.json',
             '--out', '/kaggle/working/out', '--bundles', '/tmp/e6/bundles', '--comp', '/tmp/berth', '--extra', '/tmp', '--pyds', '/tmp/pyds',
             '--work', '/tmp/work', '--worker', str(slot), '--served', job['_served'], '--model', a.model_name, '--emulate_l4', '0',
             '--dump_llm', str(a.dump), '--time_scale', str(a.time_scale), '--mode', a.mode if a.mode != 'canary' else 'agent',
             '--port', str(a.relay_port)]
    if a.mode == 'canary':
        # D4.2: a canary job may run the real E8 worker (argv = agent mode's) against the canary upstream (a stub model)
        cargs = dict(job.get('canary', {}), worker_argv=wargv)
        argv = [WORKER_PY, '-u', '/tmp/e6/canary/canary_inner.py', '--part', job['task'], '--args', json.dumps(cargs),
                '--slot', str(slot)]
    else:
        argv = wargv
    deadline = job.get('canary', {}).get('pid1_deadline_s', a.job_deadline_s) if a.mode == 'canary' else a.job_deadline_s
    return {'nonce': nonce, 'mode': 'canary' if a.mode == 'canary' else 'job', 'deadline_s': deadline, 'umask': f'{UMASK:04o}',
            'nice': 10, 'relay_port': a.relay_port, 'relay_uds': '/run/g4s/vllm.sock', 'relay_max_conns': 32,
            'mkdirs': mkdirs, 'copy': copy, 'files': files, 'argv': argv, 'env': env, 'out_root': '/kaggle/working',
            'statfs': D.STATFS, 'max_file_bytes': 32 << 20, 'max_total_bytes': 80 << 20}


# ------------------------------------------------------------------------------------------------ records
def record_base(job):
    """The RUN_RECORD_SPEC v2.1 fields every started record and row of this job carries (D4)."""
    r = {'job_id': job['job_id'], 'block': job['block'], 'arm': job['arm'], 'task': job['task'], 'seed': job['seed'],
         'repeat_id': job['repeat_id'], 'attempt': job['attempt'], 'docker_image_id': IMAGE_ID, 'harness_sha256': HARNESS,
         'vllm_version': VLLM_VERSION, 'served_model_name': SERVED_RECORD, 'request_model': job['arm_config']['request_model'],
         'bundle': job['bundle'], 'bundle_sha256': BUNDLE_SHA[job['bundle']], 'a0_drop_sha256': job['a0_drop_sha256'],
         'adapter_name': job['_agent_model'] if job['_adapters'] else None,
         'adapter_sha256': ADAPTERS[job['_agent_model']]['sha256'] if job['_adapters'] else None,
         'arm_config': job['arm_config'], 'arm_config_sha256': job['arm_config_sha256'],
         'compaction_config': job['arm_config']['compaction_config'], 'history_reasoning': job['arm_config']['history_reasoning'],
         'request_chat_template_sha256': job['arm_config']['request_chat_template_sha256'],
         'server_chat_template_sha256': SERVER_TEMPLATE_SHA, 'trust_request_chat_template': TRUST_TEMPLATE,
         'adk_version': ADK_VERSION, 'runner_sha256': RUNNER_SHA, 'lib_sha256': LIB_SHA, 'flags_sha256': FLAGS_SHA,
         'options_sha256': OPTIONS_SHA, 'vllm_args_sha256': VLLM_ARGS_SHA, 'model_sha256': MODEL_SHA, 'run_tag': a.tag,
         'runner_exceptions': [e['kind'] for e in EXCEPTIONS] or None}
    for k in ('stage', 'e8_arms_sha256', 'e8_arm_config_sha256'):
        if k in job:
            r[k] = job[k]
    return r


def append_jsonl(path, rec):
    with ROWLOCK:
        with open(path, 'a') as f:
            f.write(json.dumps(rec, default=str) + '\n')


def started(slot, job, name):
    rec = record_base(job)
    rec.update(worker=slot, t_start_utc=D.utc_now(), t=round(time.time(), 1), job_container=name)
    append_jsonl(OUT / 'started.jsonl', rec)
    return rec


def tamper_row(job, slot, reasons):
    """D4: refused rows/artifacts make the job's row an infrastructure error (resolved False, an error text no harness
    source writes -> analyze_e7 class I12 'unknown_error_text': infrastructure, penalised by d_pen, listed in reports)."""
    return {'arm': job['arm'], 'task': job['task'], 'seed': job['seed'], 'repeat_id': job['repeat_id'], 'worker': slot,
            'resolved': False, 'error': ('g4s-d4 tamper: ' + '; '.join(reasons))[:200], 'test_exit': None, 'test_summary': '',
            't_agent': None, 't_setup': None, 't_verify': None, 't_phase1': None, 'sampling_sent': None,
            **{k: None for k in HOOK_KEYS}}


def died_row(job, slot, why):
    """RUN_RECORD_SPEC v2.2 section 5.4 (A9): the runner's own row for a job that died without a results row (job deadline,
    no results row, frame error): exc 'g4s-d4 infra: <why>' -> analyze_e7 class I09 (infrastructure, re-run, counted in (4b),
    penalised by d_pen). It gets every required field below (record_base etc.); error/test_exit/test_summary are absent
    because exc is set; the phase clocks and sampling_sent are present with null."""
    return {'arm': job['arm'], 'task': job['task'], 'seed': job['seed'], 'repeat_id': job['repeat_id'], 'worker': slot,
            'resolved': False, 'exc': ('g4s-d4 infra: ' + why)[:300], 'tb': None,
            't_agent': None, 't_setup': None, 't_verify': None, 't_phase1': None, 'sampling_sent': None,
            **{k: None for k in HOOK_KEYS}}     # D4.2: the runner cannot know the hook facts (E8 PROTOCOL 6.3)


def _d(x):
    """x if it is a dict, else {} (fields of PID 1's init.json are read only through this)."""
    return x if isinstance(x, dict) else {}


def ingest(slot, idx, job, name, r, payload, ferr, fwd, srec, jstate):
    """D4.1: exception-safe. Every parse is capped (D.loads_capped) and anything a job wrote that is not exactly what the
    worker and PID 1 write is refused and recorded as tamper evidence (the row becomes 'g4s-d4 tamper: ...', I12). If an
    unexpected error still escapes, the runner writes its own I09 row ('g4s-d4 infra: ingest error <type>') with every
    required field, unless this job's row was already appended (jstate['row'])."""
    try:
        return _ingest(slot, idx, job, name, r, payload, ferr, fwd, srec, jstate)
    except Exception as e:  # noqa: BLE001
        log('slot', slot, 'job', idx, 'ingest error', repr(e)[:300])
        if jstate.get('row'):
            return [], None
        return fallback_row(slot, idx, job, name, srec, f'ingest error {type(e).__name__}', jstate, r=r, fwd=fwd)


def fallback_row(slot, idx, job, name, srec, why, jstate, r=None, fwd=None):
    """The runner's own row (RUN_RECORD_SPEC 5.4, class I09) when ingestion or the job's runner path failed, written once."""
    row = died_row(job, slot, why)
    row.update(record_base(job))
    row.update(worker=slot, t_start_utc=(srec or {}).get('t_start_utc') or D.utc_now(), t_end_utc=D.utc_now(), job_container=name)
    fw = fwd if isinstance(fwd, dict) else {}
    row.update(sampling_http=fw.get('sampling_http'), request_models_seen=fw.get('models'), tool_result_roles=fw.get('tool_result_roles'),
               history_reasoning_chars_sent=fw.get('history_reasoning_chars'), capture_sha256=fw.get('capture_sha256'),
               n_requests=fw.get('forwarded'), forwarder_refused=fw.get('refused'), history_reasoning_messages_stripped=None,
               arm_shim_calls=None, worker_dump_sha256=None,
               llm_dump_sha256=fw.get('capture_sha256') if (a.capture_bodies and FAMILY != 'E8') else None,
               summaries=None, max_prompt=None)
    cls = CLASSIFY(row)
    row.update(error_class=cls['klass'], error_class_id=cls['id'], error_detail=cls['detail'], error_table_sha256=TABLE_SHA,
               error_class_id_v21=A7.classify_row_v21(row)['id'])
    row['d4'] = {'job_container': name, 'runner_fallback': why, 'container_rc': (r or {}).get('rc'),
                 'killed_by_orchestrator': (r or {}).get('killed_by_orchestrator'), 'container_gone': (r or {}).get('gone'),
                 'wall_s': (r or {}).get('wall_s')}
    append_jsonl(OUT / 'results.jsonl', row)
    jstate['row'] = True
    append_jsonl(RUN / 'infra_events.jsonl', {'job_id': job['job_id'], 'container': name, 'class': 'I09 (runner row g4s-d4 infra)',
                                              'why': why, 't_utc': row['t_end_utc'], 'd4': row['d4']})
    return [row], why


def _ingest(slot, idx, job, name, r, payload, ferr, fwd, srec, jstate):
    members, skipped = D.tar_members(payload, 32 << 20, 90 << 20) if payload else ([], [])
    init, res_data, st_data, written, refused, artifacts, stats = {}, None, None, 0, [], {}, {}
    allowed = D.job_artifacts(job['arm'], job['task'], job['seed'], a.mode)
    last = len(members) - 1
    for i, (rel, data) in enumerate(members):
        try:
            if rel == 'init.json' and i == last:
                # PID 1 appends its init.json as the LAST tar member after every other process is dead (docker/init.py)
                obj, err = D.loads_capped(data, D.MAX_INIT_JSON)
                if err or not isinstance(obj, dict):
                    refused.append({'what': 'init.json of PID 1 unreadable', 'why': err or 'not an object', 'bytes': len(data),
                                    'sha256': D.sha256_bytes(data)})
                else:
                    init = obj
            elif rel == 'init.json':
                refused.append({'what': 'init.json written in the job', 'bytes': len(data), 'sha256': D.sha256_bytes(data)})
            elif rel == 'worker.log':
                D.write_new(RUN / 'logs', f'{name}.log', data)
            elif rel == 'out/results.jsonl':
                res_data = data
            elif rel == 'out/started.jsonl':
                st_data = data
            elif rel == 'out/arm_stats.json':
                obj, err = D.loads_capped(data, D.MAX_ARM_STATS)
                bad = err or ('not an object' if not isinstance(obj, dict) else
                              next((f'{k} not an integer' for k in ('calls', 'history_reasoning_messages_stripped')
                                    if obj.get(k) is not None and D._jtype(obj.get(k)) != 'int'), None))
                if bad:
                    refused.append({'what': 'arm_stats.json invalid', 'why': bad, 'bytes': len(data), 'sha256': D.sha256_bytes(data)})
                else:
                    stats = obj
                    D.write_new(RUN / 'arm', f'{name}.json', data)
            elif rel.startswith('out/') and (rel[4:] in allowed or (a.mode == 'canary' and re.fullmatch(r'out/canary_[a-z0-9_]+\.json', rel))):
                D.write_new(OUT, rel[4:], data)
                artifacts[rel[4:]] = D.sha256_bytes(data)
                written += 1
            else:
                refused.append({'what': 'artifact', 'path': rel[:200], 'bytes': len(data), 'sha256': D.sha256_bytes(data)})
        except Exception as e:  # noqa: BLE001  (D4.1: any error on one member is recorded, never fatal)
            skipped.append((rel[:120], repr(e)[:120]))
    if members and members[-1][0] != 'init.json':
        refused.append({'what': 'tar does not end with PID 1 init.json', 'last_member_sha256': D.sha256_bytes(members[-1][0].encode())})
    # result rows: exactly one, matching the dispatched job (line length capped before parsing)
    cand = []
    if res_data is not None:
        if len(res_data) > D.MAX_RESULTS_FILE:
            refused.append({'what': 'results.jsonl', 'bytes': len(res_data), 'sha256': D.sha256_bytes(res_data[:D.MAX_RESULTS_FILE])})
        else:
            lines, over = D.split_lines(res_data)
            for n, sha in over:
                refused.append({'what': 'result line over cap', 'bytes': n, 'sha256': sha})
            for l in lines:
                row, err = D.loads_capped(l, D.MAX_LINE)
                if err:
                    refused.append({'what': 'unparsable result line', 'why': err, 'bytes': len(l), 'sha256': D.sha256_bytes(l)})
                    continue
                problems = []
                if not isinstance(row, dict):
                    problems.append('not an object')
                else:
                    extra = sorted(set(row) - D.WORKER_ROW_KEYS)
                    if extra:
                        problems.append('unexpected keys ' + ','.join(str(x)[:30] for x in extra[:5]))
                    for k, v in (('arm', job['arm']), ('task', job['task']), ('seed', job['seed']), ('repeat_id', job['repeat_id']),
                                 ('worker', slot)):
                        if row.get(k) != v or D._jtype(row.get(k)) != D._jtype(v):
                            problems.append(f'{k} {str(row.get(k))[:40]!r} != dispatched {v!r}')
                    for k in ('job_id', 'attempt', 'block'):
                        if k in row and row[k] != job[k]:
                            problems.append(f'{k} != dispatched')
                    tb = D.row_type_problems(row)
                    if tb:
                        problems.append('value types ' + ','.join(tb[:6]))
                if problems:
                    what = ('result row with unexpected value types' if all(p.startswith('value types') for p in problems)
                            else 'result row of another job')
                    refused.append({'what': what, 'bytes': len(l), 'sha256': D.sha256_bytes(l), 'problems': problems[:5]})
                else:
                    cand.append(row)
            if len(cand) > 1:
                refused.append({'what': f'{len(cand)} result rows for one job', 'sha256': [D.sha256_bytes(json.dumps(c).encode()) for c in cand]})
    if st_data is not None:
        if len(st_data) > D.MAX_RESULTS_FILE:
            refused.append({'what': 'started.jsonl', 'bytes': len(st_data), 'sha256': D.sha256_bytes(st_data[:D.MAX_RESULTS_FILE])})
        else:
            lines, over = D.split_lines(st_data)
            ok = 0
            for l in lines:
                s, err = D.loads_capped(l, D.MAX_LINE)
                if (not err and isinstance(s, dict) and set(s) <= {'arm', 'task', 'repeat_id', 'worker', 't'} and s.get('arm') == job['arm']
                        and s.get('task') == job['task'] and s.get('repeat_id') == job['repeat_id'] and s.get('worker') == slot):
                    ok += 1
                else:
                    refused.append({'what': 'started line of another job' if not err else 'unparsable started line',
                                    'bytes': len(l), 'sha256': D.sha256_bytes(l)})
            if over or ok > 1:
                refused.append({'what': 'started.jsonl: lines over cap or more than one', 'n_over': len(over), 'n_ok': ok})
    cg = _d(init.get('cgroup'))
    d4 = {'job_container': name, 'flags': None, 'deps_mounts': deps_keys(job), 'container_rc': r['rc'], 'frame': ferr or 'ok',
          'reason': init.get('reason'), 'deadline_hit': init.get('deadline_hit'), 'killed_by_orchestrator': r['killed_by_orchestrator'],
          'container_gone': r['gone'], 'wall_s': r['wall_s'], 'child_s': init.get('t_child_s'), 'child_status': init.get('child_status'),
          'processes_killed_at_end': init.get('processes_killed_at_end'), 'stdout_bytes': r['stdout_total'],
          'memory_peak': cg.get('memory.peak'), 'memory_events': cg.get('memory.events'),
          'pids_peak': cg.get('pids.peak'), 'io_stat': cg.get('io.stat'),
          'relay': init.get('relay'), 'files_written': written, 'skipped': len(skipped), 'refused': len(refused),
          'forwarder': {k: v for k, v in (fwd or {}).items() if k not in ('refusal_log', 'prompt_tokens', 'sampling_http')},
          'forwarder_refusals': (fwd or {}).get('refusal_log', [])[:10], 'seccomp_sha256': D.SECCOMP_SHA256}
    t_end = D.utc_now()
    rows = []
    if refused:
        with contextlib.suppress(Exception):
            D.write_new(RUN / 'refused', f'{name}.json', json.dumps({'job_id': job['job_id'], 'refused': refused[:200]}, indent=1,
                                                                      default=str).encode())
        append_jsonl(RUN / 'tamper.jsonl', {'job_id': job['job_id'], 'container': name, 'n_refused': len(refused),
                                            'kinds': sorted({x['what'] for x in refused}), 't_utc': t_end,
                                            'refused_sha256': [x.get('sha256') for x in refused[:50] if isinstance(x.get('sha256'), str)]})
        base = tamper_row(job, slot, sorted({x['what'] for x in refused}))
        if len(cand) == 1:
            d4['untrusted_row_sha256'] = D.sha256_bytes(json.dumps(cand[0], sort_keys=True).encode())
        rows = [base]
    elif len(cand) == 1:
        rows = [cand[0]]
    why = None
    if not rows:
        why = 'job deadline' if init.get('deadline_hit') is True or r['killed_by_orchestrator'] else (ferr or 'no results row')
        rows = [died_row(job, slot, why)]
    out_rows = []
    for row in rows:
        row = dict(row)
        row.update(record_base(job))
        row.update(worker=slot, t_start_utc=srec['t_start_utc'], t_end_utc=t_end, job_container=name)
        fw = fwd or {}
        row.update(sampling_http=fw.get('sampling_http'), request_models_seen=fw.get('models'),
                   tool_result_roles=fw.get('tool_result_roles'), history_reasoning_chars_sent=fw.get('history_reasoning_chars'),
                   capture_sha256=fw.get('capture_sha256'), n_requests=fw.get('forwarded'),
                   forwarder_refused=fw.get('refused'))
        row['history_reasoning_messages_stripped'] = stats.get('history_reasoning_messages_stripped')
        row['arm_shim_calls'] = stats.get('calls')
        dump = f'llm/{job["arm"]}/{job["task"]}_s{job["seed"]}.jsonl.gz'
        row['worker_dump_sha256'] = artifacts.get(dump)
        # the job's request dump: the host capture (what was sent over HTTP, with vLLM's prompt token ids; written by this
        # runner, not reachable from the container) when it keeps the messages, else the worker's own dump
        # D4.2: an E8 job's request dump is the worker's (it holds the hook's failed calls, PROTOCOL 6.2(f), 6.4)
        row['llm_dump_sha256'] = artifacts.get(dump) if (FAMILY == 'E8' or not a.capture_bodies) else fw.get('capture_sha256')
        for k in ('summaries', 'max_prompt'):
            row.setdefault(k, None)
        cls = CLASSIFY(row)
        row.update(error_class=cls['klass'], error_class_id=cls['id'], error_detail=cls['detail'], error_table_sha256=TABLE_SHA,
                   error_class_id_v21=A7.classify_row_v21(row)['id'])
        row['d4'] = d4
        out_rows.append(row)
    for row in out_rows:
        append_jsonl(OUT / 'results.jsonl', row)
        jstate['row'] = True
    if why is not None:
        append_jsonl(RUN / 'infra_events.jsonl', {'job_id': job['job_id'], 'container': name, 'class': 'I09 (runner row g4s-d4 infra)',
                                                  'why': why, 't_utc': t_end, 'd4': d4})
    try:
        detail = {'job': {k: v for k, v in job.items() if not k.startswith('_')}, 'slot': slot, 'idx': idx, 'd4': d4, 'init': init,
                  'stderr_head': r['stderr'][:4000].decode('utf-8', 'replace'), 'stderr_total': r['stderr_total'],
                  'skipped': skipped[:50], 'refused': refused[:50], 'forwarder': fwd}
        D.write_new(RUN / 'jobs', f'{name}.json', json.dumps(detail, indent=1, default=str).encode())
    except Exception as e:  # noqa: BLE001  (the row is written; a missing detail file is logged only)
        log('slot', slot, 'job', idx, 'job detail not written', repr(e)[:200])
    return out_rows, why


def fwd_policy(job):
    pol = {'ctk_sha256': [D.canonical_sha256(x) for x in D.DEFAULT_CTK], 'chat_template_sha256': [],
           'return_token_ids': bool(a.capture) and SERVE, 'dump_bodies': bool(a.capture_bodies)}
    if job['arm_config']['request_chat_template_sha256']:
        pol['chat_template_sha256'] = [job['arm_config']['request_chat_template_sha256']]
    return pol


def run_job(slot, idx, job):
    """One job. D4.1: the started record is written first; from then on any error of the runner's own path (socket, env
    file, docker CLI, frame, ingestion) still ends with exactly one row for the job (fallback_row: 'g4s-d4 infra: runner
    error <type>', class I09)."""
    if STATE['stop']:
        return
    name = f'g4s-{a.tag}-j{idx:04d}-{secrets.token_hex(3)}'
    nonce = secrets.token_hex(16)
    jstate = {'row': False}
    srec, r, fwd, opened = None, None, None, False
    envf = RUN / 'env' / f'{name}.env'
    try:
        srec = started(slot, job, name)      # RUN_RECORD_SPEC 3.2 (A9): written on the host before the container starts
        STATE['running'][name] = time.time()
        sock = FWD.open(name, fwd_policy(job))
        opened = True
        D.write_env_file(envf, job_cfg(slot, job, nonce))
        dyn = ['--name', name, '--label', D.LABEL, '--label', f'g4s.run={RUN}', '--label', f'g4s.owner={TOKEN}', '--label', 'g4s.kind=job',
               '--label', f'g4s.job={idx}']
        dyn += dyn_mounts(job)
        dyn += D.ro(sock, '/run/g4s/vllm.sock')
        argv = [D.DOCKER, 'run', *dyn, *FLAGS, '--env-file', str(envf), IMAGE_ID, *INIT]
        try:
            r = D.run_container(DENV, argv, name, a.job_deadline_s + a.kill_grace_s, a.frame_cap, 256 << 10, log=log)
        finally:
            STATE['running'].pop(name, None)
            fwd = FWD.close(name) if opened else None
            opened = False
            envf.unlink(missing_ok=True)
        payload, ferr = D.parse_frame(r['stdout'], nonce)
        rows, why = ingest(slot, idx, job, name, r, payload, ferr, fwd, srec, jstate)
        log('job', idx, job['job_id'], 'resolved', ['(E8: not shown)' if BLIND else x.get('resolved') for x in rows], 'class',
            [('agent' if BLIND and x.get('error_class') == 'agent' else x.get('error_class_id')) for x in rows],
            'frame', ferr or 'ok', 'wall', r['wall_s'], 'runner row (I09): ' + why if why else '')
    except Exception as e:  # noqa: BLE001
        log('slot', slot, 'job', idx, 'runner error', repr(e)[:300])
        STATE['running'].pop(name, None)
        if opened:
            with contextlib.suppress(Exception):
                fwd = FWD.close(name)
        with contextlib.suppress(OSError):
            envf.unlink(missing_ok=True)
        if srec is not None and not jstate['row']:
            fallback_row(slot, idx, job, name, srec, f'runner error {type(e).__name__}', jstate, r=r, fwd=fwd)


def slot_loop(slot, q, end):
    while not STATE['stop']:
        try:
            idx, job = q.get_nowait()
        except queue.Empty:
            return
        if time.time() > end:
            STATE['reason'] = 'deadline'
            return
        try:
            run_job(slot, idx, job)
        except Exception as e:  # noqa: BLE001
            log('slot', slot, 'job', idx, 'error', repr(e)[:300])


def kill_run_containers():
    p = D.docker(DENV, 'ps', '-a', '--filter', f'label=g4s.run={RUN}', '--filter', f'label=g4s.owner={TOKEN}',
                 '--filter', f'label={D.LABEL}', '--format', '{{.Names}}', timeout=60)
    names = [n for n in p.stdout.split() if n.startswith('g4s-')]
    for n in names:
        D.docker(DENV, 'kill', '--signal', 'KILL', n, timeout=60)
    for n in names:
        D.ensure_gone(DENV, n)
    return names


def cleanup():
    if STATE['cleaned']:
        return
    STATE['cleaned'] = True
    STATE['stop'] = True
    killed = kill_run_containers()
    for t in THREADS:
        t.join(timeout=120)
    killed += kill_run_containers()          # a slot that was between its stop check and docker run
    for t in THREADS:
        t.join(timeout=60)
    srv = STATE['srv']
    if srv is not None:
        try:
            srv.stop(timeout=30)
        except Exception as e:  # noqa: BLE001
            log('vllm stop error', repr(e)[:200])
    FWD.close_all()
    gpu_left = []
    if STATE['vllm_pids']:
        for _ in range(60):
            q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True, timeout=60)
            apps = {int(x) for x in q.stdout.split() if x.isdigit()}
            gpu_left = sorted(apps & set(STATE['vllm_pids']))
            if not gpu_left:
                break
            time.sleep(1)
    D.smi_snapshot(RUN / 'gpu_after.txt')
    reap_end = D.reap(DENV, ROOT / 'runs', log=log)
    left = D.docker(DENV, 'ps', '-aq', '--filter', f'label=g4s.run={RUN}', timeout=60).stdout.split()
    STATE['cleanup'] = {'containers_killed_at_cleanup': killed, 'containers_left': left, 'vllm_pids_on_gpu': gpu_left,
                        'reap_end': reap_end}
    log('cleanup done:', json.dumps(STATE['cleanup']))


def manifest_path():
    return pathlib.Path(a.manifest) if a.manifest else ROOT / 'runs' / 'manifests' / f'{a.block}_manifest.json'


def manifest_check_or_append(entry=None):
    """A8/A9 block manifest (RUN_RECORD_SPEC 3.3): one entry per run of the block, with the server settings the runner
    started the server with. Called twice: early without an entry (read-only: refuses job ids already started in a run
    the manifest lists, so no slot attempt runs twice, and a manifest of another block) and again just before the first
    job starts, which appends the entry (a run that refused or whose server failed to start is never listed)."""
    mp = manifest_path()
    mp.parent.mkdir(parents=True, exist_ok=True)
    with open(mp.with_suffix('.lock'), 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        m = json.load(open(mp)) if mp.exists() else {'block': a.block, 'runs': []}
        if m.get('block') != a.block:
            refuse(f'manifest {mp} is for block {m.get("block")}')
        if any(e.get('run_tag') == a.tag for e in m['runs']):
            refuse(f'manifest {mp} already lists run {a.tag}')
        seen = set()
        for e in m['runs']:
            sp = ROOT / 'runs' / e['run_tag'] / 'out' / 'started.jsonl'
            if sp.exists():
                for l in open(sp):
                    with contextlib.suppress(ValueError):
                        seen.add(json.loads(l).get('job_id'))
        again = sorted({j['job_id'] for j in JOBS} & seen)
        if again:
            refuse(f'{len(again)} job ids already started in this block (e.g. {again[0]}); re-runs need the next attempt index')
        for e in m['runs']:
            prev = {k: e.get(k) for k in SERVER_SETTINGS}
            if entry is not None and prev != SERVER_SETTINGS:
                refuse(f'server settings {SERVER_SETTINGS} differ from run {e.get("run_tag")} of this block {prev} (A9)')
        if entry is not None:
            m['runs'].append(dict(entry, **SERVER_SETTINGS))
            # block-level copy of the server settings (equal for every run of the block, see the refusal above): E8's
            # analyze_e8.py reads max_model_len at the top of the manifest, E7's analyze_e7.py in each run entry (A9)
            m.update(SERVER_SETTINGS)
            tmp = mp.with_suffix('.tmp')
            tmp.write_text(json.dumps(m, indent=1))
            os.replace(tmp, mp)
    return str(mp)


THREADS = []
summary = {'tag': a.tag, 'mode': a.mode, 'block': a.block, 'workers': a.workers, 'jobs': len(JOBS), 'host': HOST_LABEL,
           't_start_unix': round(T0, 1), 't_start_utc': D.utc_now(), 'runner_sha256': RUNNER_SHA, 'lib_sha256': LIB_SHA,
           'owner': TOKEN, 'served_model_names': SERVED, 'track': 'D4.3', 'runner_exceptions': EXCEPTIONS,
           'seccomp_sha256': D.SECCOMP_SHA256, 'family': FAMILY, 'blind': BLIND,
           'memory_ceiling': {k: MEMORY.get(k) for k in ('cgroup', 'memory.max', 'memory.swap.max')}}
FWD = D.HttpForwarder(RUN / 'sock', upstream_port=(a.port if SERVE else (a.upstream_port or None)),
                      capture_dir=RUN / 'capture', log=log)
IMAGE_ID = FLAGS = FLAGS_SHA = OPTIONS_SHA = VLLM_ARGS_SHA = HARNESS = MODEL_SHA = None
VLLM_VERSION = f'none:{a.mode}'
ADK_VERSION = None
try:
    summary['reap_start'] = D.reap(DENV, ROOT / 'runs', log=log)
    p = D.docker(DENV, 'image', 'inspect', a.image, '--format', '{{.Id}}')
    if p.returncode:
        refuse(f'image {a.image} not available')
    IMAGE_ID = p.stdout.strip()
    FLAGS = static_flags(a.uid)
    FLAGS_SHA = hashlib.sha256(json.dumps([IMAGE_ID, FLAGS]).encode()).hexdigest()
    # D4.3: the job flags without canary mode's read-only mount of the canary program; the gate binds these, and a canary
    # run's flags must be exactly these plus that one mount
    FLAGS_JOB = static_flags(a.uid, with_canary=False)
    FLAGS_JOB_SHA = hashlib.sha256(json.dumps([IMAGE_ID, FLAGS_JOB]).encode()).hexdigest()
    assert FLAGS == FLAGS_JOB + (canary_mount() if a.mode == 'canary' else []), 'flags differ from the job flags by more than the canary mount'
    OPTIONS = {k: getattr(a, k) for k in ('job_deadline_s', 'kill_grace_s', 'frame_cap', 'time_scale', 'max_model_len', 'gpu_util',
                                          'model_name', 'served_model_name', 'relay_port', 'max_lora_rank', 'vllm_language_only', 'capture')}
    OPTIONS_SHA = D.canonical_sha256(OPTIONS)
    HARNESS = harness_hashes()
    ADK_VERSION = next((re.match(r'google_adk-(.+)\.dist-info/RECORD$', k).group(1) for k in HARNESS
                        if re.match(r'google_adk-(.+)\.dist-info/RECORD$', k)), None)
    MODEL_SHA = model_hash(a.model_dir) if SERVE else None
    vllm_base = D.vllm_safety_args(a.vllm_language_only)
    VLLM_ARGS_SHA = D.canonical_sha256({'base_args': vllm_base, 'env': {'VLLM_MEDIA_URL_ALLOW_REDIRECTS': '0'},
                                        'tool_call_parser': 'gemma4', 'reasoning_parser': 'gemma4', 'dtype': 'bfloat16',
                                        'max_model_len': a.max_model_len, 'gpu_util': a.gpu_util, 'tp': 1})
    if SERVE:
        # ---- canary gate (D4: in the runner): the configuration must be the one the canaries passed; D4.3: also for
        # --canary-vllm (canary mode serving the real model), with canary_passed.json only (checked above)
        cur = {'image_id': IMAGE_ID, 'runner_sha256': RUNNER_SHA, 'lib_sha256': LIB_SHA, 'flags_sha256': FLAGS_JOB_SHA,
               'options_sha256': OPTIONS_SHA, 'harness_sha256': D.canonical_sha256(HARNESS), 'model_sha256': MODEL_SHA,
               'vllm_args_sha256': VLLM_ARGS_SHA}
        assert tuple(cur) == GATE_KEYS
        GATE['current'] = cur
        if a.mode == 'canary':
            GATE['canary_flags_sha256'] = FLAGS_SHA                       # the job flags (bound) + the canary program's mount
            GATE['canary_program_tree_sha256'] = D.tree_hash(ROOT / 'tools/d4canary')
        if any(not isinstance(v, str) or not v for v in cur.values()):
            refuse(f'the configuration has no value for {sorted(k for k, v in cur.items() if not isinstance(v, str) or not v)}')
        if a.gate == 'canary_passed':
            cpf = ROOT / 'canary_passed.json'
            if not cpf.is_file():
                refuse('agent mode and --canary-vllm need canary_passed.json (D4 canaries, controls and validation passed)')
            cp = json.load(open(cpf))
            diff = {k: (cp.get(k), v) for k, v in cur.items() if cp.get(k) != v}
            if diff:
                refuse(f'canary_passed.json does not bind this configuration: {sorted(diff)}')
            # D4.1: a gate file names the experiment families it was validated for ('scope'); D4.2: the scope is required
            # and the family comes from the block id and the jobs (FAMILY)
            scope = cp.get('scope')
            if not isinstance(scope, list) or FAMILY not in scope:
                refuse(f'canary_passed.json scope {scope} does not include {FAMILY} (block {a.block})')
            GATE['canary_passed_sha256'] = D.sha256_file(cpf)
            GATE['scope'] = scope
            GATE['family'] = FAMILY
            if a.mode == 'canary':
                log('GATE: canary mode serves the model in block', a.block, 'under canary_passed.json', GATE['canary_passed_sha256'][:16])
        elif a.gate.startswith('precanary:'):
            ctag = a.gate.split(':', 1)[1]
            cf = ROOT / 'runs' / f'{ctag}_canary_d4.json'
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', ctag) or not cf.is_file():
                refuse(f'no D4 canary record {cf}')
            cr = json.load(open(cf))
            s = cr.get('summary') or {}
            if s.get('failed') or not s.get('checks') or s.get('passed') != s.get('checks'):
                refuse(f'canary record {ctag} has failures: {s.get("failed")}')
            # D4.2: only a complete canary record (every part) counts, and only for named validation blocks (checked above)
            if sorted(cr.get('parts') or []) != sorted(PRECANARY_PARTS):
                refuse(f'canary record {ctag} ran parts {cr.get("parts")}, not all of {list(PRECANARY_PARTS)}')
            b = dict(cr.get('binding') or {})
            if isinstance(b.get('harness_sha256'), dict):          # the canary record keeps the harness map; compare its canonical sha
                b['harness_sha256'] = D.canonical_sha256(b['harness_sha256'])
            diff = sorted(k for k in ('image_id', 'runner_sha256', 'lib_sha256', 'harness_sha256') if b.get(k) != cur[k])
            if diff:
                refuse(f'canary record {ctag} was made with another {diff}')
            if not VALIDATION_BLOCK_RE.fullmatch(str(a.block)):
                refuse(f'--gate precanary runs only named validation blocks, not {a.block}')
            GATE['precanary'] = {'tag': ctag, 'sha256': D.sha256_file(cf), 'checks': s.get('checks'), 'scope': 'validation blocks',
                                 'family': FAMILY}
            log('GATE: pre-validation run under the D4 canary record', ctag, '(canary_passed.json not used)')
        else:
            refuse(f'unknown --gate {a.gate}')
    JOBS_FILE_BYTES = json.dumps([{k: v for k, v in j.items() if not k.startswith('_')} for j in JOBS]).encode()
    (RUN / 'jobs.json').write_bytes(JOBS_FILE_BYTES)
    JOBS_SHA = D.sha256_bytes(JOBS_FILE_BYTES)
    (RUN / 'config.json').write_text(json.dumps({'args': vars(a), 'image_id': IMAGE_ID, 'flags': FLAGS, 'flags_sha256': FLAGS_SHA,
                                                 'flags_job_sha256': FLAGS_JOB_SHA,
                                                 'options': OPTIONS, 'options_sha256': OPTIONS_SHA, 'harness_sha256': HARNESS,
                                                 'model_sha256': MODEL_SHA, 'vllm_args_sha256': VLLM_ARGS_SHA,
                                                 'runner_sha256': RUNNER_SHA, 'lib_sha256': LIB_SHA, 'umask': f'{UMASK:04o}',
                                                 'worker_env': D.worker_env(C, RUN), 'python': sys.executable, 'nice': os.nice(0),
                                                 'owner': TOKEN, 'gate': GATE, 'io_device': [IO_DEV, IO_MM], 'served': SERVED,
                                                 'served_record': SERVED_RECORD, 'exceptions': EXCEPTIONS, 'bundles': BUNDLE_SHA,
                                                 'adapters': ADAPTERS, 'drop_sha256': DROP_SHA, 'arms_sha256': ARMS_SHA,
                                                 'request_template_sha256': TEMPLATE_SHA, 'jobs_sha256': JOBS_SHA,
                                                 'forwarder_caps': FWD.caps, 'server_settings': SERVER_SETTINGS,
                                                 'classifier': CLASSIFY.__name__, 'error_table_sha256': TABLE_SHA,
                                                 'seccomp_profile': SECCOMP, 'seccomp_sha256': D.SECCOMP_SHA256,
                                                 'family': FAMILY, 'blind': BLIND, 'memory': MEMORY,
                                                 'malloc_arena_max': os.environ.get('MALLOC_ARENA_MAX')}, indent=1, default=str))
    if a.block is not None:
        manifest_check_or_append()
    if free_gib(RUN) < a.min_free_gib:
        refuse(f'only {free_gib(RUN):.0f} GiB free under {RUN}')
    if a.mode != 'canary':
        summary['prep'] = prep(sorted({t for j in JOBS for t in deps_tasks(j)}))
    else:
        summary['prep'] = prep(sorted({t for j in JOBS for t in deps_tasks(j)})) if any(deps_tasks(j) for j in JOBS) else {'missing': 0}
    D.smi_snapshot(RUN / 'gpu_before.txt')
    startup = None
    if SERVE:
        with socket.socket() as sk:
            if sk.connect_ex(('127.0.0.1', a.port)) == 0:
                refuse(f'port {a.port} in use')
        q = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=60)
        free = [int(x) for x in q.stdout.split() if x.isdigit()]
        log('gpu free MiB', free)
        if not free or free[0] < MIN_FREE_MIB:
            refuse(f'GPU free {free} MiB < {MIN_FREE_MIB}')
        from adk_submission import VllmConfig, VllmServer

        class PdeathVllmServer(VllmServer):
            def build_cmd(self):
                return D.vllm_scope_argv(a.tag, sys.executable, super().build_cmd())   # D4.2: own user scope + pdeath

        extra = ['--served-model-name', *SERVED] + vllm_base + (['--trust-request-chat-template'] if TRUST_TEMPLATE else []) \
            + (['--enable-log-requests'] if a.vllm_log_requests else [])
        lora = LORA
        cfg = VllmConfig(model=a.model_dir, port=a.port, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4',
                         max_model_len=a.max_model_len, dtype='bfloat16', gpu_memory_utilization=a.gpu_util, enable_auto_tool_choice=True,
                         enable_lora=bool(lora), max_loras=max(1, len(lora)), max_lora_rank=a.max_lora_rank,
                         scheduler_reserve_full_isl=False, tensor_parallel_size=1, startup_timeout=60 * 20, extra_args=extra)
        srv = PdeathVllmServer(cfg, lora_modules=lora) if lora else PdeathVllmServer(cfg)
        STATE['srv'] = srv
        cmd = srv.build_cmd()
        if lora and ('--max-lora-rank' not in cmd or cmd[cmd.index('--max-lora-rank') + 1] != str(a.max_lora_rank)):
            refuse(f'vLLM LoRA rank is not {a.max_lora_rank}: {cmd[cmd.index("--max-lora-rank") + 1] if "--max-lora-rank" in cmd else None}')
        got = {'max_model_len': int(cmd[cmd.index('--max-model-len') + 1]), 'enable_lora': '--enable-lora' in cmd,
               'max_lora_rank': int(cmd[cmd.index('--max-lora-rank') + 1]) if '--max-lora-rank' in cmd else None,
               'served_adapter_names': sorted(x.split('=', 1)[0] for x in cmd[cmd.index('--lora-modules') + 1:]
                                              if '=' in x and not x.startswith('-')) if '--lora-modules' in cmd else []}
        if got != SERVER_SETTINGS:
            refuse(f'the vLLM command {got} differs from the server settings recorded for the manifest {SERVER_SETTINGS}')
        summary['vllm_cmd_tail'] = [x for x in cmd[3:] if not x.startswith('/home/')][:80]
        summary['vllm_cmd_sha256'] = D.canonical_sha256(cmd[4:])
        log('vllm args:', ' '.join(map(str, cmd[5:]))[-900:])
        os.link(srv.log_path, RUN / 'vllm.log')
        if a.vllm_debug:
            os.environ['VLLM_LOGGING_LEVEL'] = 'DEBUG'
        t = time.time()
        srv.start()
        os.environ.pop('VLLM_LOGGING_LEVEL', None)
        startup = round(time.time() - t, 1)
        kids = subprocess.run(['pgrep', '-P', str(srv.process.pid)], capture_output=True, text=True).stdout.split()
        STATE['vllm_pids'] = sorted({srv.process.pid, *map(int, kids)})
        with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/version', timeout=30) as resp:
            VLLM_VERSION = str(json.loads(resp.read()).get('version'))
        with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/v1/models', timeout=30) as resp:
            summary['v1_models'] = sorted(m.get('id') for m in json.loads(resp.read()).get('data', []))
        log('vllm up', startup, 's; version', VLLM_VERSION, '; models', summary['v1_models'])
    summary['vllm_version'] = VLLM_VERSION
    summary['startup_s'] = startup
    (RUN / 'startup.json').write_text(json.dumps({'startup_s': startup, 'mode': a.mode, 'host': HOST_LABEL, 'served_model_names': SERVED,
                                                  'vllm_version': VLLM_VERSION, 'v1_models': summary.get('v1_models')}))
    if SERVE and a.hold_before_jobs_s > 0:
        log('holding', a.hold_before_jobs_s, 's before the first job')
        time.sleep(min(a.hold_before_jobs_s, 600))
    if a.block is not None:
        summary['manifest'] = manifest_check_or_append({'run_tag': a.tag, 'jobs_sha256': JOBS_SHA, 'deadline_s': a.deadline_s,
                                                        'job_file': os.path.basename(a.jobs), 'n_jobs': len(JOBS), 't_start_utc': D.utc_now(),
                                                        'mode': a.mode, 'attempts': sorted({j['attempt'] for j in JOBS})})
    (OUT / 'started.jsonl').touch()
    (OUT / 'results.jsonl').touch()          # a run whose every job died still has a (possibly empty) results file
    q = queue.Queue()
    for i, j in enumerate(JOBS):
        q.put((i, j))
    end = T0 + a.deadline_s
    t_workers = time.time()
    for i in range(a.workers):
        t = threading.Thread(target=slot_loop, args=(i, q, end), daemon=True)
        t.start()
        THREADS.append(t)
    log('slots started', a.workers, 'jobs', len(JOBS), 'mode', a.mode, 'block', a.block, 'image', IMAGE_ID[:19], 'flags', FLAGS_SHA[:12])
    last_log = last_smi = 0
    smi_csv = open(RUN / 'gpu_samples.csv', 'a', buffering=1)
    smi_csv.write('t_s,memory_used_mib,memory_free_mib,util_pct\n')
    while any(t.is_alive() for t in THREADS):
        now = time.time()
        if now > end:
            STATE['reason'] = 'deadline'
            log('run deadline reached; stopping')
            break
        if free_gib(RUN) < a.min_free_gib:
            STATE['reason'] = f'disk: free space below {a.min_free_gib} GiB'
            log(STATE['reason'])
            break
        srvp = STATE['srv'].process if STATE['srv'] is not None else None
        if srvp is not None and srvp.poll() is not None:
            STATE['reason'] = 'vllm exited'
            log(STATE['reason'], srvp.returncode)
            break
        if SERVE and now - last_smi >= 30:
            last_smi = now
            q2 = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=60)
            for r in q2.stdout.strip().splitlines():
                smi_csv.write(f'{now - T0:.0f},{r.replace(" ", "")}\n')
        if now - last_log >= 300:
            last_log = now
            n = sum(1 for _ in open(OUT / 'results.jsonl')) if (OUT / 'results.jsonl').exists() else 0
            log('finished jobs', n, '/', len(JOBS), '; running', len(STATE['running']))
        time.sleep(2)
    summary['wall_workers_s'] = round(time.time() - t_workers, 1)
finally:
    try:
        cleanup()
    except Exception as e:  # noqa: BLE001
        log('cleanup error', repr(e)[:300])
    def _jl(f):
        out = []
        if f.exists():
            for l in open(f):
                if l.strip():
                    with contextlib.suppress(ValueError):
                        out.append(json.loads(l))
        return [x for x in out if isinstance(x, dict)]

    def _nums(xs):
        return [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    try:
        rows = _jl(OUT / 'results.jsonl')
        started_recs = _jl(OUT / 'started.jsonl')
        infra_ev = _jl(RUN / 'infra_events.jsonl')
        tam = _jl(RUN / 'tamper.jsonl')
        row_ids = {r.get('job_id') for r in rows}
        summary.update(reason=STATE['reason'], finished=len(rows), started=len(started_recs),
                       solved=None if BLIND else sum(r.get('resolved') is True for r in rows),
                       exceptions=sum(1 for r in rows if r.get('exc')), setup_fallback=sum(_nums(r.get('setup_fallback') for r in rows)),
                       error_classes=dict(collections.Counter(('agent' if BLIND and r.get('error_class') == 'agent' else str(r.get('error_class_id')))
                                                              for r in rows)),
                       infra_rows=sum(1 for r in rows if r.get('error_class') == 'infra'),
                       # D4.1: started jobs that have no row at all (must be 0) vs the runner's own I09 rows (D4 called these
                       # 'started_without_row')
                       started_without_row=sum(1 for x in started_recs if x.get('job_id') not in row_ids),
                       runner_rows=len(infra_ev), runner_fallback_rows=sum(1 for r in rows if (r.get('d4') or {}).get('runner_fallback')),
                       tamper_jobs=len(tam), forwarder_refused=sum(_nums((r.get('forwarder_refused') for r in rows))),
                       deadline_hits=sum(1 for r in rows if (r.get('d4') or {}).get('deadline_hit') is True),
                       frame_errors=sum(1 for r in rows if (r.get('d4') or {}).get('frame') not in (None, 'ok')),
                       cleanup=STATE.get('cleanup'), wall_total_s=round(time.time() - T0, 1), t_end_utc=D.utc_now())
        ag = _nums(r.get('t_agent') for r in rows)
        try:
            summary['memory_end'] = {k: v for k, v in D.cgroup_memory().items() if k in ('memory.peak', 'memory.current', 'memory.events')}
            summary['runner_vmhwm_bytes'] = int(next(l.split()[1] for l in open('/proc/self/status') if l.startswith('VmHWM'))) * 1024
        except Exception:  # noqa: BLE001
            pass
        if ag:
            summary.update(mean_t_agent=round(statistics.mean(ag), 1), median_t_agent=round(statistics.median(ag), 1), max_t_agent=max(ag))
        if rows:
            walls = _nums(r.get('wall') for r in rows)
            cw = _nums((r.get('d4') or {}).get('wall_s') for r in rows)
            summary['mean_job_wall'] = round(statistics.mean(walls), 1) if walls else None
            summary['mean_container_wall'] = round(statistics.mean(cw), 1) if cw else None
        if rows and started_recs:
            st_by = {x.get('job_id'): x.get('t') for x in started_recs if _nums([x.get('t')])}
            ends = [st_by[r['job_id']] + ((r.get('d4') or {}).get('wall_s') or 0) for r in rows if r.get('job_id') in st_by]
            if ends:
                first = min(st_by.values())
                summary['span_first_start_to_last_finish_s'] = round(max(ends) - first, 1)
                summary['runs_per_hour'] = round(len(rows) / max(1e-6, max(ends) - first) * 3600, 1)
    except Exception as e:  # noqa: BLE001
        summary['summary_error'] = repr(e)[:300]
    (RUN / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    log('summary', json.dumps(summary, default=str)[:3000])
