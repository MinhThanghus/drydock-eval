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
"""Track D4 orchestrator library (host side, trusted). Used by run_berth.py, tools/reap.py and the canaries.

D3's design is kept: model-influenced files never live on, and are never touched by, the host. Every job runs in ONE
container (image g4s-sandbox, PID 1 = docker/init.py) whose writable paths are tmpfs mounts capped in bytes and inodes;
host paths are mounted read-only only. Results leave a container only as a nonce-framed tar on its stdout, read here
with a byte cap, parsed in memory, and written as data into files whose names this code validates.

D4 additions:
  * HttpForwarder (replaces D3's byte Forwarder): the per-job unix socket now carries HTTP/1.1 that is PARSED here. Only
    `POST /v1/chat/completions` with a JSON body of text-only chat messages (no image/audio/video/file part of any kind,
    no unknown key) is sent on to the one upstream (vLLM on 127.0.0.1); `chat_template` and `chat_template_kwargs` pass
    only with a sha256 approved for the job's block; body size, request rate, connection rate and bytes per job are
    capped; every refusal is logged. The forwarder also asks vLLM for the prompt token ids (`return_token_ids`), removes
    them from the response before the client sees it, and keeps them in a host-side capture file per job.
  * block-device I/O caps (io.max read/write bps and iops on the disk behind the bind mounts, blkio weight 10), memlock
    64 KiB, neutral hostname.
  * strict ingestion helpers (one row per job, line cap before json.loads, artifact allowlist per job).
  * the reaper also removes stale per-job socket and env files of dead runners.

D4.1 additions (verifier findings of D4):
  * every model-controlled request field the forwarder keeps or records has a type and size rule (SAMPLING_RULES covers
    every SAMPLING_KEYS key, plus `user`; `stop` at most 8 strings of at most 128 bytes, `tool_choice` none/auto/required
    or one named function); a body may hold at most 200,000 JSON containers and nest at most 64 levels; a job may use at
    most 8 distinct sampling variants and 4 distinct model strings (the next distinct value is refused, not counted);
    distinct refusal kinds are capped; any other exception while one request is handled becomes a counted 400 refusal;
    Content-Length must be ASCII digits.
  * JSON members of a job's result tar are parsed by loads_capped (size cap, nesting depth 64, RecursionError and every
    other parse error caught) and worker rows must have the value types the e6 worker writes (ROW_TYPES).
  * job containers get a seccomp profile derived from Docker's default that denies Linux native AIO and io_uring
    (seccomp_profile(); profile pinned by SECCOMP_SHA256).
  * OwnerPatterns: owner names are kept only as salted sha256 values (local/owner_patterns_hashed.json); scans and
    redaction hash the substring of each pattern length at every character position.
  * utc_now reads the clock once.

D4.2 additions (D4.1 verifiers' findings and the E8 worker):
  * the forwarder's parse path is bounded: body cap 1 MiB (the largest real request of D4's GPU validation was 190,017 B);
    before json.loads, scan_json_shape() checks the body's bytes (strings removed by one linear regex) for nesting depth,
    JSON containers and structural values, so no object is built for a body over a cap; one process-wide semaphore
    (PARSE_SLOTS) admits at most 2 parses and checks at a time; json_depth() pushes containers only; at most 2 concurrent
    connections per job (every real job used 1) and 64 over all jobs; 64 MiB up and 64 MiB down per job (D4's 83 agent
    jobs: at most 3.19 MB up and 172 KB down);
  * worker rows: the E8 worker's four hook fields (PROTOCOL.md 6.2(g)) are accepted with their types; every string field
    has the e6/E8 worker's length cap and every number must be finite and within +-2**53 (a longer error text could stall
    the v2.1 classifier's I13 regex; NaN/Infinity broke summary.json), else the row is refused as tamper evidence (I12);
  * memory ceiling: cgroup_memory() reads the memory.max / memory.swap.max / memory.peak of this process's cgroup; the
    runner refuses to start unless it runs in a cgroup with memory.max <= RUNNER_MEMORY_MAX_LIMIT and no swap (a user-level
    systemd scope, scope_argv()); vLLM runs in its own user scope (vllm_scope_argv()), so the runner's ceiling never
    counts it.

D4.3 additions (D4.2 escape verifier's two runner MEDIUMs):
  * scan_json_shape() removes string literals with json_outside_strings(), a byte scan of a fixed number of linear passes
    (bytes.replace / count / split / join; no regex): D4.2's regex took time quadratic in the length of an unterminated
    string of escaped quotes and held the GIL (128 KB froze the runner for 27 s; 1 MiB about 30 min). Two bounds no valid
    body within the D4.2 caps can exceed are added: at most max_values + max_containers + 1 strings (from the quote count,
    before the text is cut into pieces) and at most max_containers closing brackets. Decisions on valid bodies are those of
    D4.2 (local and host checks: 20,012 valid bodies, structural bytes and decisions identical);
  * upstream_owner_problem(): canary mode's --upstream-port must be a listener of the canary host itself (the runner's
    ancestors below PID 1 / the systemd manager, and their direct children), never another process's server.
"""
import collections
import contextlib
import gzip
import hashlib
import io
import json
import os
import re
import resource
import secrets
import signal
import socket
import stat
import subprocess
import tarfile
import threading
import time

DOCKER = '/usr/bin/docker'
LABEL = 'gemma4swe=eval'
KAGGLE_MODEL = '/kaggle/input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2'
FRAME_RE_TMPL = rb'\nG4S-FRAME-BEGIN %s (\d{1,10}) ([0-9a-f]{64})\n'
SAFE_PART = re.compile(r'^[A-Za-z0-9_.@+=,-]{1,160}$')
HOSTNAME = 'sandbox'                       # neutral container hostname (D3 used the host's name)
MEDIA_BLOCK_DOMAIN = 'g4s-media-disabled.invalid'   # vLLM --allowed-media-domains: a name that never resolves


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def canonical(obj):
    """Canonical JSON (sorted keys, no spaces; = make_jobs_e8.canonical): the bytes hashed for arm-config hashes."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def canonical_sha256(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def tree_hash(root):
    """RUN_RECORD_SPEC section 2: sha256("".join(f"{sha256(file)}  {path}\\n" for path in sorted(paths))) over every regular
    file below root (paths relative, '/'-separated, __pycache__ excluded). Symlinks are refused."""
    root = os.path.realpath(root)
    paths = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != '__pycache__')
        for n in files:
            p = os.path.join(base, n)
            st = os.lstat(p)
            if stat.S_ISLNK(st.st_mode):
                raise ValueError(f'symlink in hashed tree: {p}')
            if stat.S_ISREG(st.st_mode):
                paths.append(os.path.relpath(p, root).replace(os.sep, '/'))
    return hashlib.sha256(''.join(f'{sha256_file(os.path.join(root, p))}  {p}\n' for p in sorted(paths)).encode()).hexdigest()


def utc_now():
    t = time.time()                          # D4.1: one clock read (D4 read it twice: up to 1 s early near a second boundary)
    return time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(t)) + f'.{int(t * 1000) % 1000:03d}Z'


# ------------------------------------------------------------------------------------------------ bounded JSON (D4.1)
MAX_JSON_DEPTH = 64                # nesting levels of any JSON value the runner parses from a job (real maximum: 7)
MAX_BODY_CONTAINERS = 20_000       # D4.2: JSON containers in one request body, counted outside strings (real maximum 1,239)
MAX_BODY_VALUES = 50_000           # D4.2: ',' and ':' outside strings in one request body (bounds the scalars parsed)
PARSE_SLOTS = threading.BoundedSemaphore(2)    # D4.2: request bodies parsed and checked at once, over all jobs
_NOT_STRUCT = bytes(b for b in range(256) if b not in b'[]{},:')


def json_outside_strings(body, max_strings=None):
    """D4.3: the bytes of a JSON text that lie outside its string literals, by a byte scan of a fixed number of passes, each
    linear in len(body) (bytes.replace, bytes.count, bytes.split, bytes.join; no regex, no backtracking, no per-byte Python
    loop):
      1. delete every escaped backslash, i.e. each backslash pair of a backslash run, paired from the left as JSON reads it
         (a run inside a string always starts after a non-backslash byte); a run keeps at most one backslash;
      2. delete every escaped quote (a backslash, then '"'); the byte before such a backslash is never a backslash, so no
         new pair forms. Every '"' left now opens or closes a string;
      3. if max_strings is given and more than 2 * max_strings quotes are left, return None (before any piece is made);
      4. cut the text at those quotes and keep every second piece (the pieces outside strings).
    For a text json.loads accepts, this is exactly the text with its string literals removed (the D4.2 regex gave the same
    structural bytes). A text with an odd number of delimiting quotes ends in an unterminated string: its bytes are kept, as
    in D4.2, and json.loads refuses such a body. In any other text json.loads stops at its first error, and up to that byte
    the strings found here are json's strings."""
    t = body.replace(b'\\\\', b'').replace(b'\\"', b'')
    if max_strings is not None and t.count(b'"') > 2 * max_strings:
        return None
    parts = t.split(b'"')
    out = b''.join(parts[0::2])
    if not len(parts) % 2:                  # odd number of quotes: the last string is unterminated; keep its bytes
        out += parts[-1]
    return out


def scan_json_shape(body, max_depth=MAX_JSON_DEPTH, max_containers=MAX_BODY_CONTAINERS, max_values=MAX_BODY_VALUES):
    """D4.2: None, or the reason a JSON body may not be parsed, from its bytes alone (no object is built): the string
    literals are removed, then the structural bytes outside strings bound the containers, the nesting depth and the values.
    D4.3: the strings are removed by json_outside_strings() (linear byte scan; D4.2's regex took time quadratic in the length
    of an unterminated string of escaped quotes: 27 s for 128 KB, about 30 min for 1 MiB). Two bounds are added that no
    valid body within the D4.2 caps can exceed, so no decision on such a body changes: (1) at most
    max_values + max_containers + 1 strings (each string of a valid text is an object key, i.e. one ':', or a value, and a
    text has at most #',' + #containers + 1 values), checked from the quote count before the text is cut into pieces (the
    pieces of 1 MiB of quotes would cost about 55 MB); (2) closing brackets are bounded like opening ones (a valid text has
    as many of each), so the depth loop below runs over at most 2 * max_containers + max_values bytes."""
    outside = json_outside_strings(body, max_values + max_containers + 1)
    if outside is None:
        return f'body has over {max_values + max_containers + 1} JSON strings'
    struct = outside.translate(None, _NOT_STRUCT)
    n_open = struct.count(b'[') + struct.count(b'{')
    if n_open > max_containers:
        return f'body has over {max_containers} JSON containers'
    if struct.count(b']') + struct.count(b'}') > max_containers:
        return f'body has over {max_containers} closing JSON brackets'
    if struct.count(b',') + struct.count(b':') > max_values:
        return f'body has over {max_values} JSON values'
    d = mx = 0
    for b in struct:
        if b in (91, 123):          # [ {
            d += 1
            if d > mx:
                mx = d
                if mx > max_depth:
                    return f'body nesting deeper than {max_depth}'
        elif b in (93, 125):        # ] }
            d -= 1
    return None


def json_depth(obj, limit=None):
    """Nesting depth of a parsed JSON value (a scalar is 0), iterative; D4.2: only containers are pushed (D4.1 pushed a
    tuple for every value); stops early above `limit`."""
    if not isinstance(obj, (dict, list)):
        return 0
    mx, st = 0, [(obj, 1)]
    while st:
        x, d = st.pop()
        if d > mx:
            mx = d
            if limit is not None and mx > limit:
                return mx
        for v in (x.values() if isinstance(x, dict) else x):
            if isinstance(v, (dict, list)):
                st.append((v, d + 1))
    return mx


def loads_capped(data, max_bytes=1 << 20, max_depth=MAX_JSON_DEPTH):
    """(value, None) or (None, reason) for bytes or str from a job: size cap before parsing, every parse error caught
    (ValueError, UnicodeDecodeError, RecursionError from deep nesting, MemoryError, ...), nesting depth capped after."""
    try:
        if len(data) > max_bytes:
            return None, f'over {max_bytes} bytes'
        obj = json.loads(data)
    except RecursionError:
        return None, 'nesting too deep to parse'
    except Exception as e:  # noqa: BLE001
        return None, f'unparsable ({type(e).__name__})'
    if json_depth(obj, max_depth) > max_depth:
        return None, f'nesting deeper than {max_depth}'
    return obj, None


def marker(s):
    return f'<owner process/container detail redacted; sha256:{hashlib.sha256(s.encode()).hexdigest()[:12]}>'


# ------------------------------------------------------------------------------------------------ docker CLI
def denv(docker_config):
    return {'PATH': '/usr/bin:/bin', 'HOME': str(docker_config), 'DOCKER_CONFIG': str(docker_config), 'LANG': 'C.UTF-8'}


def docker(env, *args, timeout=120):
    return subprocess.run([DOCKER, *args], capture_output=True, text=True, timeout=timeout, env=env, stdin=subprocess.DEVNULL)


def container_exists(env, name):
    p = docker(env, 'ps', '-aq', '--filter', f'name=^/{name}$', timeout=60)
    return p.returncode != 0 or bool(p.stdout.strip())


def ensure_gone(env, name, wait_s=30.0):
    """Our container (unique g4s- name) must not exist afterwards: wait for --rm, then docker rm -f."""
    if not name.startswith('g4s-'):
        raise ValueError(name)
    t = time.time()
    while time.time() - t < wait_s:
        with contextlib.suppress(Exception):
            if not container_exists(env, name):
                return True
        time.sleep(0.25)
    with contextlib.suppress(Exception):
        docker(env, 'rm', '-f', name, timeout=120)
    with contextlib.suppress(Exception):
        return not container_exists(env, name)
    return False


# ------------------------------------------------------------------------------------------------ ownership
def start_ticks(pid):
    try:
        with open(f'/proc/{pid}/stat') as f:
            rest = f.read().rsplit(')', 1)[1].split()
        return None if rest[0] == 'Z' else int(rest[19])
    except (OSError, IndexError, ValueError):
        return None


def owner_token(pid=None):
    pid = pid or os.getpid()
    return f'{pid}:{start_ticks(pid)}'


def owner_alive(token):
    try:
        pid, st = token.split(':')
        return start_ticks(int(pid)) == int(st)
    except (ValueError, AttributeError):
        return False


def ancestors(pid):
    out = set()
    while pid and pid > 1 and pid not in out:
        out.add(pid)
        try:
            with open(f'/proc/{pid}/stat') as f:
                pid = int(f.read().rsplit(')', 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
    return out


def _stat_fields(pid):
    with open(f'/proc/{pid}/stat') as f:
        return f.read().rsplit(')', 1)[1].split()


def listen_inodes(port):
    """D4.3: socket inodes of the TCP listeners (state 0A) that accept a connection to 127.0.0.1:<port> in this network
    namespace: bound to 127.0.0.1 or 0.0.0.0, or (IPv6) to ::, ::ffff:127.0.0.1 or ::ffff:0.0.0.0."""
    want = {'/proc/net/tcp': {'0100007F', '00000000'},
            '/proc/net/tcp6': {'0' * 32, '0000000000000000FFFF00000100007F', '0000000000000000FFFF000000000000'}}
    out = set()
    for fn, addrs in want.items():
        try:
            lines = open(fn).read().splitlines()[1:]
        except OSError:
            continue
        for l in lines:
            x = l.split()
            if len(x) < 10 or x[3] != '0A':
                continue
            addr, _, p = x[1].partition(':')
            if addr in addrs and int(p, 16) == port:
                out.add(x[9])
    return out


def upstream_owner_problem(port, pid=None):
    """D4.3: None if every TCP listener behind 127.0.0.1:<port> is held by this process's canary host, else the reason.
    The canary host is the chain of this process's ancestors (flock, a canary script, an interactive shell) up to, not
    including, the first one that is PID 1, a systemd manager or not of this uid, plus the direct children of those
    ancestors (a stub model the canary script started). A listener of any other process (a vLLM another runner started, a
    server of another session or of the owner) is refused. Only the fd links of these processes of our own uid are read;
    nothing about any process is recorded."""
    if not isinstance(port, int) or not 0 < port < 65536:
        return f'port {port!r} out of range'
    inodes = listen_inodes(port)
    if not inodes:
        return 'no TCP listener accepts 127.0.0.1 on that port'
    uid = os.getuid()
    host, cur = [], os.getppid() if pid is None else pid
    while cur and cur > 1 and cur not in host:
        try:
            if os.stat(f'/proc/{cur}').st_uid != uid or open(f'/proc/{cur}/comm').read().strip() == 'systemd':
                break
            host.append(cur)
            cur = int(_stat_fields(cur)[1])
        except (OSError, IndexError, ValueError):
            break
    if not host:
        return 'this runner has no canary host (its parent is PID 1, a systemd manager or another user): start it from the canary script'
    cand = set(host)
    for h in host:
        try:
            for tid in os.listdir(f'/proc/{h}/task'):
                with open(f'/proc/{h}/task/{tid}/children') as f:
                    cand.update(int(c) for c in f.read().split())
        except (OSError, ValueError):
            pass
    holders = set()
    for c in cand:
        try:
            if os.stat(f'/proc/{c}').st_uid != uid:
                continue
            for fd in os.listdir(f'/proc/{c}/fd'):
                with contextlib.suppress(OSError):
                    ln = os.readlink(f'/proc/{c}/fd/{fd}')
                    if ln.startswith('socket:[') and ln[8:-1] in inodes:
                        holders.add(ln[8:-1])
        except OSError:
            continue
    if holders != inodes:
        return (f'{len(inodes) - len(holders)} of {len(inodes)} listener(s) on that port are not held by the canary host (its '
                f'{len(host)} ancestor(s) or their children): canary jobs reach only a stub model of the canary host')
    return None


def lock_held_by_ancestor(lock_path, pid=None):
    """True if an flock/POSIX lock on lock_path (by inode) is held by this process or one of its ancestors (flock(1) holds
    the lock in its own process and runs the command as its child). Returns (held, detail)."""
    try:
        st = os.stat(lock_path)
    except OSError as e:
        return False, f'no lock file: {e!r}'
    anc = ancestors(pid or os.getpid())
    dev_ino = f':{st.st_ino} '
    holders = []
    with open('/proc/locks') as f:
        for line in f:
            x = line.split()
            # "1: FLOCK  ADVISORY  WRITE 12345 103:02:224135356 0 EOF"
            if len(x) >= 6 and x[5].endswith(f':{st.st_ino}') and x[1] in ('FLOCK', 'POSIX', 'OFDLCK') and '->' not in x[1]:
                try:
                    holders.append(int(x[4]))
                except ValueError:
                    pass
    mine = [h for h in holders if h in anc]
    return bool(mine), {'lock_inode': st.st_ino, 'holders': len(holders), 'held_by_ancestor': mine}


# ------------------------------------------------------------------------------------------------ reaper
def reap(env, runs_dir, log=print, kill_wait_s=5.0):
    """Remove every g4s container / network / volume (label gemma4swe=eval AND name g4s-*) whose owner (label g4s.owner =
    '<pid>:<start ticks>' of the orchestrator that created it) is no longer alive; kill our own processes (this uid,
    environment entry GEMMA4_EVAL_RUN=<runs_dir>/<tag>) whose run's owner (runs/<tag>/owner.json) is dead; and (D4) remove
    the per-job socket files (runs/<tag>/sock/*.sock) and env files (runs/<tag>/env/*.env) of runs whose owner is dead.
    Never touches an object without both markers, an object of a live owner, this process or its ancestors, another uid's
    process, or a file outside runs/<tag>/{sock,env}."""
    runs_dir = str(runs_dir)
    res = {'containers_removed': [], 'containers_live_owner': 0, 'legacy_left': [], 'networks_removed': [],
           'volumes_removed': [], 'processes_killed': [], 'stale_files_removed': 0, 'errors': []}
    p = docker(env, 'ps', '-a', '--no-trunc', '--filter', f'label={LABEL}', '--format',
               '{{.ID}}\t{{.Names}}\t{{.Label "g4s.owner"}}\t{{.State}}', timeout=120)
    if p.returncode:
        res['errors'].append('docker ps: ' + p.stderr.strip()[:200])
    for line in p.stdout.splitlines():
        parts = line.split('\t')
        if len(parts) != 4 or not parts[1].startswith('g4s-'):
            continue
        cid, name, owner, state = parts
        if not owner:
            if state != 'running':
                docker(env, 'rm', '-f', cid, timeout=120)
                res['containers_removed'].append(name)
            else:
                res['legacy_left'].append(name)
            continue
        if owner_alive(owner):
            res['containers_live_owner'] += 1
            continue
        docker(env, 'rm', '-f', cid, timeout=120)
        res['containers_removed'].append(name)
    for kind, key in (('network', 'networks_removed'), ('volume', 'volumes_removed')):
        fmt = '{{.Name}}\t{{.Label "g4s.owner"}}'
        p = docker(env, kind, 'ls', '--filter', f'label={LABEL}', '--format', fmt, timeout=120)
        for line in p.stdout.splitlines():
            parts = line.split('\t')
            if len(parts) != 2 or not parts[0].startswith('g4s-'):
                continue
            if parts[1] and owner_alive(parts[1]):
                continue
            r = docker(env, kind, 'rm', parts[0], timeout=120)
            (res[key] if r.returncode == 0 else res['errors']).append(parts[0] if r.returncode == 0 else f'{kind} rm {parts[0]}: {r.stderr.strip()[:120]}')
    # processes
    me, uid = os.getpid(), os.getuid()
    keep = ancestors(me) | {me}
    prefix = b'GEMMA4_EVAL_RUN=' + runs_dir.encode() + b'/'
    victims = []
    owners = {}
    for d in os.listdir('/proc'):
        if not d.isdigit() or int(d) in keep:
            continue
        try:
            if os.stat('/proc/' + d).st_uid != uid:
                continue
            envs = open(f'/proc/{d}/environ', 'rb').read().split(b'\0')
        except OSError:
            continue
        hit = [e for e in envs if e.startswith(prefix)]
        if not hit:
            continue
        run = hit[0][len(b'GEMMA4_EVAL_RUN='):].decode(errors='replace')
        if run not in owners:
            try:
                owners[run] = json.load(open(os.path.join(run, 'owner.json'))).get('token')
            except (OSError, ValueError):
                owners[run] = None
        tok = owners[run]
        if tok and not owner_alive(tok):
            victims.append((int(d), start_ticks(int(d))))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, st in victims:
            if start_ticks(pid) == st and st is not None:
                with contextlib.suppress(OSError):
                    os.kill(pid, sig)
        if victims and sig == signal.SIGTERM:
            time.sleep(kill_wait_s)
    res['processes_killed'] = [pid for pid, _ in victims]
    # stale per-job socket / env files of dead runners (D3 residual risk 8)
    try:
        tags = sorted(os.listdir(runs_dir))
    except OSError:
        tags = []
    for tag in tags:
        rd = os.path.join(runs_dir, tag)
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', tag) or os.path.islink(rd) or not os.path.isdir(rd):
            continue
        sub = [(os.path.join(rd, 'sock'), '.sock', stat.S_ISSOCK), (os.path.join(rd, 'env'), '.env', stat.S_ISREG)]
        if not any(os.path.isdir(d) and not os.path.islink(d) and os.listdir(d) for d, _, _ in sub):
            continue
        try:
            tok = json.load(open(os.path.join(rd, 'owner.json'))).get('token')
        except (OSError, ValueError):
            tok = None
        if tok and owner_alive(tok):
            continue
        for d, suffix, kind in sub:
            if not os.path.isdir(d) or os.path.islink(d):
                continue
            for n in os.listdir(d):
                p = os.path.join(d, n)
                try:
                    st = os.lstat(p)
                except OSError:
                    continue
                if n.startswith('g4s-') and n.endswith(suffix) and kind(st.st_mode) and st.st_uid == uid:
                    with contextlib.suppress(OSError):
                        os.unlink(p)
                        res['stale_files_removed'] += 1
    if any(res[k] for k in ('containers_removed', 'networks_removed', 'volumes_removed', 'processes_killed', 'legacy_left',
                            'stale_files_removed')):
        log('reaper:', json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in res.items()}))
    return res


# ------------------------------------------------------------------------------------------------ HTTP forwarder (SSRF filter)
ALLOWED_TOP_KEYS = frozenset({'model', 'messages', 'tools', 'tool_choice', 'temperature', 'top_p', 'top_k', 'max_tokens',
                              'max_completion_tokens', 'chat_template_kwargs', 'chat_template', 'thinking_token_budget',
                              'stream', 'n', 'stop', 'seed', 'presence_penalty', 'frequency_penalty', 'user',
                              'parallel_tool_calls', 'skip_special_tokens'})
ALLOWED_ROLES = frozenset({'system', 'developer', 'user', 'assistant', 'tool', 'tool_responses'})
ALLOWED_MSG_KEYS = frozenset({'role', 'content', 'name', 'tool_calls', 'tool_call_id', 'reasoning', 'reasoning_content'})
SAMPLING_KEYS = ('temperature', 'top_p', 'top_k', 'max_tokens', 'max_completion_tokens', 'chat_template_kwargs',
                 'thinking_token_budget', 'n', 'stop', 'seed', 'presence_penalty', 'frequency_penalty', 'tool_choice',
                 'parallel_tool_calls', 'skip_special_tokens', 'stream')
DEFAULT_CTK = [{'enable_thinking': True}]          # the harness's chat_template_kwargs (approved in every block)


def vllm_safety_args(language_only=True):
    """vLLM 0.19.1 options of every D4 server (verified in its source): --allowed-media-domains with a name that never
    resolves (MediaConnector._assert_url_in_allowed_media_domains raises for every http(s) media URL before any fetch;
    file:// needs --allowed-local-media-path, which stays unset; data: URLs fetch nothing) and --language-model-only (every
    multimodal limit 0: media items are refused, encoders not loaded). The environment adds
    VLLM_MEDIA_URL_ALLOW_REDIRECTS=0. image/audio embeddings stay refused (--enable-mm-embeds unset)."""
    return ['--allowed-media-domains', MEDIA_BLOCK_DOMAIN] + (['--language-model-only'] if language_only else [])


def _num(v, lo, hi, integer=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (integer and not isinstance(v, int)):
        return False
    return lo <= v <= hi


# D4.1: a type and size rule for EVERY key the forwarder keeps or records (the sampling keys, a Counter key per job) and
# for `user` (forwarded only). The harness sends only temperature, top_p, max_completion_tokens, thinking_token_budget,
# chat_template_kwargs and (arm T) chat_template (1,602 captured D4 requests); the rules leave room for OpenAI-style values.
MAX_STOP_ITEMS, MAX_STOP_BYTES, MAX_USER_BYTES = 8, 128, 128
FUNC_NAME = re.compile(r'^[A-Za-z0-9_.:-]{1,64}$')


def _short_str(v, n):
    return isinstance(v, str) and len(v.encode('utf-8', 'surrogatepass')) <= n


def _stop_ok(v):
    if v is None or _short_str(v, MAX_STOP_BYTES):
        return True
    return isinstance(v, list) and len(v) <= MAX_STOP_ITEMS and all(_short_str(x, MAX_STOP_BYTES) for x in v)


def _tool_choice_ok(v):
    if v is None or v in ('none', 'auto', 'required'):
        return True
    if not isinstance(v, dict) or set(v) != {'type', 'function'} or v.get('type') != 'function':
        return False
    f = v.get('function')
    return isinstance(f, dict) and set(f) == {'name'} and isinstance(f.get('name'), str) and bool(FUNC_NAME.match(f['name']))


SAMPLING_RULES = {
    'temperature': lambda v: v is None or _num(v, 0, 2), 'top_p': lambda v: v is None or _num(v, 0, 1),
    'top_k': lambda v: v is None or _num(v, -1, 1000, True), 'max_tokens': lambda v: v is None or _num(v, 1, 32768, True),
    'max_completion_tokens': lambda v: v is None or _num(v, 1, 32768, True),
    'chat_template_kwargs': lambda v: isinstance(v, dict),               # and an approved sha256 (check_chat_body)
    'thinking_token_budget': lambda v: v is None or _num(v, 0, 32768, True), 'n': lambda v: v is None or _num(v, 1, 1, True),
    'stop': _stop_ok, 'seed': lambda v: v is None or _num(v, -2 ** 63, 2 ** 63, True),
    'presence_penalty': lambda v: v is None or _num(v, -2, 2), 'frequency_penalty': lambda v: v is None or _num(v, -2, 2),
    'tool_choice': _tool_choice_ok, 'parallel_tool_calls': lambda v: v is None or isinstance(v, bool),
    'skip_special_tokens': lambda v: v is None or isinstance(v, bool), 'stream': lambda v: v is None or v is False}
OTHER_RULES = {'user': lambda v: v is None or _short_str(v, MAX_USER_BYTES)}
assert set(SAMPLING_RULES) == set(SAMPLING_KEYS), 'every recorded sampling key needs a type/size rule'
assert set(ALLOWED_TOP_KEYS) - set(SAMPLING_KEYS) == {'model', 'messages', 'tools', 'chat_template', 'user'}
MAX_SAMPLING_VARIANTS = 8     # distinct sampling dicts per job (D4 real maximum: 2 = agent + compaction summary)
MAX_MODEL_STRINGS = 4         # distinct model strings per job (real: 1)
MAX_REFUSAL_KINDS = 64


def check_chat_body(obj, policy):
    """Reasons (list of short strings) why this chat body may not reach vLLM; [] = allowed. Text-only messages; every
    key on an allowlist; chat_template / chat_template_kwargs only with an approved sha256."""
    why = []
    if not isinstance(obj, dict):
        return ['body is not a JSON object']
    extra = sorted(k for k in obj if k not in ALLOWED_TOP_KEYS)
    if extra:
        why.append('unknown keys ' + ','.join(x[:40] for x in extra[:8]))
    if not isinstance(obj.get('model'), str) or not 0 < len(obj['model']) <= 512:
        why.append('model missing or not a short string')
    msgs = obj.get('messages')
    if not isinstance(msgs, list) or not 0 < len(msgs) <= 4096:
        why.append('messages missing or not a list of 1..4096')
        msgs = []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            why.append(f'message {i} not an object')
            continue
        bad = sorted(k for k in m if k not in ALLOWED_MSG_KEYS)
        if bad:
            why.append(f'message {i} keys ' + ','.join(x[:40] for x in bad[:6]))
        if m.get('role') not in ALLOWED_ROLES:
            why.append(f'message {i} role {str(m.get("role"))[:30]!r}')
        c = m.get('content')
        if isinstance(c, list):
            for j, p in enumerate(c):
                if not isinstance(p, dict) or p.get('type') != 'text' or set(p) - {'type', 'text'} or not isinstance(p.get('text'), str):
                    t = p.get('type') if isinstance(p, dict) else type(p).__name__
                    why.append(f'message {i} part {j} type {str(t)[:30]!r} (only text parts pass)')
        elif c is not None and not isinstance(c, str):
            why.append(f'message {i} content type {type(c).__name__}')
        for k in ('reasoning', 'reasoning_content', 'name', 'tool_call_id'):
            if m.get(k) is not None and not isinstance(m.get(k), str):
                why.append(f'message {i} {k} not a string')
        tc = m.get('tool_calls')
        if tc is not None:
            if not isinstance(tc, list):
                why.append(f'message {i} tool_calls not a list')
            else:
                for t in tc:
                    f = t.get('function') if isinstance(t, dict) else None
                    if (not isinstance(t, dict) or set(t) - {'id', 'type', 'function', 'index'} or not isinstance(f, dict)
                            or set(f) - {'name', 'arguments'} or not all(isinstance(f.get(k), (str, type(None))) for k in ('name', 'arguments'))):
                        why.append(f'message {i} malformed tool_call')
                        break
    tools = obj.get('tools')
    if tools is not None:
        if not isinstance(tools, list) or len(tools) > 128:
            why.append('tools not a list of <= 128')
        else:
            for t in tools:
                if not isinstance(t, dict) or set(t) - {'type', 'function'} or t.get('type') != 'function' or not isinstance(t.get('function'), dict):
                    why.append('tool entry not a function definition')
                    break
    if 'chat_template' in obj:
        ct = obj['chat_template']
        if not isinstance(ct, str) or sha256_bytes(ct.encode()) not in policy.get('chat_template_sha256', ()):
            why.append('chat_template not approved for this block')
    if 'chat_template_kwargs' in obj:
        k = obj['chat_template_kwargs']
        if not isinstance(k, dict) or canonical_sha256(k) not in policy.get('ctk_sha256', ()):
            why.append('chat_template_kwargs not approved for this block')
    for k, lo, hi, integer in (('max_tokens', 1, 32768, True), ('max_completion_tokens', 1, 32768, True), ('n', 1, 1, True),
                               ('temperature', 0, 2, False), ('top_p', 0, 1, False), ('top_k', -1, 1000, True),
                               ('thinking_token_budget', 0, 32768, True), ('presence_penalty', -2, 2, False),
                               ('frequency_penalty', -2, 2, False)):
        if k in obj and obj[k] is not None and not _num(obj[k], lo, hi, integer):
            why.append(f'{k} out of range')
    if obj.get('stream') not in (None, False):
        why.append('stream must be false')
    if 'seed' in obj and obj['seed'] is not None and not _num(obj['seed'], -2 ** 63, 2 ** 63, True):
        why.append('seed not an integer')
    # D4.1: every kept/recorded key (and `user`) has an allowed type and size; keys checked above keep their message
    for k, rule in list(SAMPLING_RULES.items()) + list(OTHER_RULES.items()):
        if k in obj and not rule(obj[k]) and not any(w.startswith(k) for w in why):
            why.append(f'{k} has a type or size that is not allowed')
    return why


def body_summary(obj):
    """What the capture keeps of a request: sampling keys (chat_template replaced by its sha256), roles, reasoning
    characters in assistant messages, tool-result roles, counts. No message text."""
    s = {k: obj.get(k) for k in SAMPLING_KEYS if k in obj}
    if 'chat_template' in obj:
        s['chat_template'] = 'sha256:' + sha256_bytes(str(obj['chat_template']).encode())
    msgs = obj.get('messages') or []
    roles = [m.get('role') for m in msgs if isinstance(m, dict)]
    rc = sum(len(m.get('reasoning') or m.get('reasoning_content') or '') for m in msgs
             if isinstance(m, dict) and m.get('role') == 'assistant')
    rmsgs = sum(1 for m in msgs if isinstance(m, dict) and m.get('role') == 'assistant' and (m.get('reasoning') or m.get('reasoning_content')))
    return {'model': obj.get('model'), 'sampling': s, 'n_messages': len(msgs), 'roles': roles,
            'tool_result_roles': dict(collections.Counter(r for r in roles if r in ('tool', 'tool_responses'))),
            'history_reasoning_chars': rc, 'history_reasoning_messages': rmsgs,
            'tools_sha256': sha256_bytes(canonical(obj.get('tools')).encode()) if obj.get('tools') is not None else None}


class _Refuse(Exception):
    def __init__(self, status, reason, close=True):
        super().__init__(reason)
        self.status, self.reason, self.close = status, reason, close


_STATUS_TEXT = {200: 'OK', 400: 'Bad Request', 403: 'Forbidden', 404: 'Not Found', 405: 'Method Not Allowed', 411: 'Length Required',
                413: 'Payload Too Large', 417: 'Expectation Failed', 429: 'Too Many Requests', 431: 'Request Header Fields Too Large',
                501: 'Not Implemented', 502: 'Bad Gateway', 504: 'Gateway Timeout'}


class HttpForwarder:
    """Per-job unix sockets in a 0700 directory (one per job, bind-mounted read-only into its container at
    /run/g4s/vllm.sock). Each accepted connection carries HTTP/1.1 requests; each one is parsed and checked here and, if
    it passes, sent on a fresh connection to the ONE upstream 127.0.0.1:<upstream_port> (Connection: close; only the
    Content-Type and Content-Length headers are passed on). With upstream_port None every request is refused.
    Caps (per job): concurrent connections, new connections per minute, requests per minute, requests in total, request
    body bytes, header bytes, bytes up and down in total; (all jobs) concurrent connections."""

    def __init__(self, sock_dir, upstream_port=None, capture_dir=None, log=print, per_job_conns=2, conns_per_min=120,
                 req_per_min=60, req_total=2000, body_cap=1 << 20, header_cap=32 << 10, up_cap=64 << 20,
                 down_cap=64 << 20, total_conns=64, idle_s=1800, upstream_timeout_s=1800, resp_cap=64 << 20):
        self.dir = str(sock_dir)
        os.makedirs(self.dir, mode=0o700, exist_ok=True)
        os.chmod(self.dir, 0o700)
        self.capture_dir = str(capture_dir) if capture_dir else None
        if self.capture_dir:
            os.makedirs(self.capture_dir, mode=0o700, exist_ok=True)
        self.upstream_port = upstream_port
        self.log = log
        self.caps = dict(per_job_conns=per_job_conns, conns_per_min=conns_per_min, req_per_min=req_per_min,
                         req_total=req_total, body_cap=body_cap, header_cap=header_cap, up_cap=up_cap, down_cap=down_cap,
                         total_conns=total_conns, idle_s=idle_s, upstream_timeout_s=upstream_timeout_s, resp_cap=resp_cap)
        self.jobs, self.total = {}, 0
        self.lock = threading.Lock()

    def open(self, job, policy=None):
        """policy: {'chat_template_sha256': [...], 'ctk_sha256': [...], 'return_token_ids': bool, 'dump_bodies': bool,
        'inject': {key: value} (never used by the runner; reserved), 'templates': {sha: text}}"""
        path = os.path.join(self.dir, job + '.sock')
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(path)
        os.chmod(path, 0o666)   # the job uid (2000) must be able to connect; the 0700 directory keeps other host users out
        s.listen(16)
        st = {'listener': s, 'conns': set(), 'path': path, 'closed': False, 'policy': dict(policy or {}),
              'accepted': 0, 'refused_conns': 0, 'requests': 0, 'forwarded': 0, 'refused': 0, 'refusals': collections.Counter(),
              'refusal_log': [], 'up_bytes': 0, 'down_bytes': 0, 'upstream_errors': 0, 'statuses': collections.Counter(),
              'conn_times': collections.deque(), 'req_times': collections.deque(), 'models': collections.Counter(),
              'tool_result_roles': collections.Counter(), 'history_reasoning_chars': 0, 'history_reasoning_messages': 0,
              'sampling': collections.Counter(), 'prompt_tokens': [], 'cap_lock': threading.Lock(), 'cap_f': None,
              'cap_path': None, 'n_capture': 0,
              # D4.1: distinct values admitted so far (reserved before forwarding; bounded by MAX_* above)
              'var_lock': threading.Lock(), 'model_set': set(), 'sampling_set': set(), 'variant_refusals': 0}
        if self.capture_dir:
            st['cap_path'] = os.path.join(self.capture_dir, job + '.jsonl.gz')
            fd = os.open(st['cap_path'], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            st['cap_f'] = gzip.open(os.fdopen(fd, 'wb'), 'wt')
        with self.lock:
            self.jobs[job] = st
        threading.Thread(target=self._accept, args=(job, st), daemon=True).start()
        return path

    # -------------------------------------------------------------------------------------------- bookkeeping
    def _refusal(self, st, reason, detail=None):
        st['refused'] += 1
        kind = reason.split(':')[0][:60]
        if kind not in st['refusals'] and len(st['refusals']) >= MAX_REFUSAL_KINDS:
            kind = 'other'                         # D4.1: distinct refusal kinds per job are capped
        st['refusals'][kind] += 1
        if len(st['refusal_log']) < 200:
            st['refusal_log'].append({'t_utc': utc_now(), 'reason': reason[:300], **({'detail': detail} if detail else {})})

    def _rate(self, dq, per_min):
        now = time.time()
        while dq and now - dq[0] > 60:
            dq.popleft()
        if len(dq) >= per_min:
            return False
        dq.append(now)
        return True

    def _accept(self, job, st):
        while not st['closed']:
            try:
                c, _ = st['listener'].accept()
            except OSError:
                return
            with self.lock:
                ok = (not st['closed'] and len(st['conns']) < self.caps['per_job_conns'] and self.total < self.caps['total_conns']
                      and self._rate(st['conn_times'], self.caps['conns_per_min']))
                if ok:
                    st['conns'].add(c)
                    st['accepted'] += 1
                    self.total += 1
                else:
                    st['refused_conns'] += 1
            if not ok:
                self._refusal(st, 'connection cap (concurrent, per minute or total)')
                with contextlib.suppress(OSError):
                    c.close()
                continue
            threading.Thread(target=self._serve, args=(st, c), daemon=True).start()

    # -------------------------------------------------------------------------------------------- one client connection
    def _recv(self, st, c, buf, n_needed=None, until=None, cap=None):
        """Read from c into buf until `until` is in buf or len(buf) >= n_needed; counts and caps the job's bytes up."""
        while True:
            if until is not None and until in buf:
                return buf
            if n_needed is not None and len(buf) >= n_needed:
                return buf
            if cap is not None and len(buf) > cap:
                raise _Refuse(431 if until else 413, 'request too large')
            d = c.recv(65536)
            if not d:
                raise ConnectionError('client closed')
            st['up_bytes'] += len(d)
            if st['up_bytes'] > self.caps['up_cap']:
                raise _Refuse(429, 'job byte cap (up) reached')
            buf += d

    def _send_client(self, st, c, status, body, close):
        head = (f'HTTP/1.1 {status} {_STATUS_TEXT.get(status, "Status")}\r\ncontent-type: application/json\r\n'
                f'content-length: {len(body)}\r\nconnection: {"close" if close else "keep-alive"}\r\n\r\n').encode()
        data = head + body
        st['down_bytes'] += len(data)
        c.sendall(data)

    def _serve(self, st, c):
        buf = b''
        try:
            c.settimeout(self.caps['idle_s'])
            while not st['closed']:
                try:
                    try:
                        buf = self._recv(st, c, buf, until=b'\r\n\r\n', cap=self.caps['header_cap'])
                    except ConnectionError:
                        return
                    head, buf = buf.split(b'\r\n\r\n', 1)
                    keep = self._one(st, c, head, buf)
                    buf = keep if keep is not None else b''
                    if keep is None:
                        return
                except OSError:
                    raise                     # client gone or idle timeout: closed without an answer, not a refusal
                except Exception as e:  # noqa: BLE001  (_Refuse; D4.1: also header-phase caps and any other error)
                    r = e if isinstance(e, _Refuse) else _Refuse(400, f'malformed request ({type(e).__name__})')
                    self._refusal(st, r.reason)
                    st['statuses'][r.status] += 1
                    body = json.dumps({'error': {'message': f'g4s forwarder refused: {r.reason}', 'type': 'g4s_refused',
                                                 'code': r.status}}).encode()
                    with contextlib.suppress(OSError):
                        self._send_client(st, c, r.status, body, True)
                    return
        except OSError:
            pass
        finally:
            with contextlib.suppress(OSError):
                c.close()
            with self.lock:
                st['conns'].discard(c)
                self.total -= 1

    def _one(self, st, c, head, rest):
        """One request whose header block is `head` and whose body starts in `rest`. Returns the bytes after this request
        (keep-alive) or None (close)."""
        try:
            text = head.decode('latin-1')
        except UnicodeDecodeError:
            raise _Refuse(400, 'undecodable header')
        lines = text.split('\r\n')
        parts = lines[0].split(' ')
        if len(parts) != 3 or not parts[2].startswith('HTTP/1.'):
            raise _Refuse(400, 'malformed request line')
        method, target = parts[0], parts[1]
        hdr = {}
        for l in lines[1:]:
            if ':' not in l:
                raise _Refuse(400, 'malformed header line')
            k, v = l.split(':', 1)
            k = k.strip().lower()
            if k in hdr and k in ('content-length', 'transfer-encoding', 'host'):
                raise _Refuse(400, f'duplicate header {k}')
            hdr[k] = v.strip()
        st['requests'] += 1
        if not self._rate(st['req_times'], self.caps['req_per_min']) or st['requests'] > self.caps['req_total']:
            raise _Refuse(429, 'request rate or total cap reached')
        if method != 'POST' or target != '/v1/chat/completions':
            raise _Refuse(404 if method in ('GET', 'POST', 'HEAD') else 405, f'path refused: {method[:10]} {target[:80]}')
        if 'transfer-encoding' in hdr:
            raise _Refuse(501, 'transfer-encoding refused (Content-Length only)')
        if 'expect' in hdr:
            raise _Refuse(417, 'expect header refused')
        if 'content-length' not in hdr or not re.fullmatch(r'[0-9]{1,10}', hdr['content-length']):
            raise _Refuse(411, 'content-length missing or invalid')      # D4.1: ASCII digits only (D4: str.isdigit)
        n = int(hdr['content-length'])
        if n > self.caps['body_cap']:
            raise _Refuse(413, f'body over {self.caps["body_cap"]} bytes')
        if not hdr.get('content-type', '').lower().startswith('application/json'):
            raise _Refuse(400, 'content-type must be application/json')
        try:
            rest = self._recv(st, c, rest, n_needed=n)
        except ConnectionError:
            return None
        body, after = rest[:n], rest[n:]
        with PARSE_SLOTS:                         # D4.2: at most 2 bodies are scanned, parsed and checked at once
            shape = scan_json_shape(body)         # D4.2: depth, containers and values from the bytes, before any object
            if shape:
                raise _Refuse(413, shape)
            try:
                obj = json.loads(body.decode('utf-8'))
            except RecursionError:
                raise _Refuse(400, 'body nesting too deep')
            except (UnicodeDecodeError, ValueError):
                raise _Refuse(400, 'body is not UTF-8 JSON')
            if json_depth(obj, MAX_JSON_DEPTH) > MAX_JSON_DEPTH:
                raise _Refuse(400, f'body nesting deeper than {MAX_JSON_DEPTH}')
            why = check_chat_body(obj, st['policy'])
            if why:
                raise _Refuse(400, 'body refused: ' + '; '.join(why)[:280])
            if self.upstream_port is None:
                raise _Refuse(403, 'no model endpoint in this mode')
            summ = body_summary(obj)
        mkey, skey = str(obj.get('model'))[:200], canonical(summ['sampling'])
        with st['var_lock']:                      # D4.1: the per-job distinct values the forwarder keeps are capped
            if mkey not in st['model_set'] and len(st['model_set']) >= MAX_MODEL_STRINGS:
                st['variant_refusals'] += 1
                raise _Refuse(400, f'body refused: more than {MAX_MODEL_STRINGS} distinct model strings in this job')
            if skey not in st['sampling_set'] and len(st['sampling_set']) >= MAX_SAMPLING_VARIANTS:
                st['variant_refusals'] += 1
                raise _Refuse(400, f'body refused: more than {MAX_SAMPLING_VARIANTS} distinct sampling variants in this job')
            st['model_set'].add(mkey)
            st['sampling_set'].add(skey)
        send = dict(obj)
        if st['policy'].get('return_token_ids'):
            send['return_token_ids'] = True
        out = json.dumps(send, ensure_ascii=False, separators=(',', ':')).encode()
        t0 = time.time()
        status, rbody, err = self._upstream(out)
        dt = time.time() - t0
        if err:
            st['upstream_errors'] += 1
            self._refusal(st, 'upstream error: ' + err)
            st['statuses'][502] += 1
            self._send_client(st, c, 502, json.dumps({'error': {'message': 'g4s forwarder: upstream error', 'type': 'g4s_upstream',
                                                                 'code': 502}}).encode(), True)
            self._capture(st, summ, body, obj, None, 502, dt, err)
            return None
        ptoks, ctoks, usage, finish = None, None, None, None
        if st['policy'].get('return_token_ids') and status == 200:
            try:
                r = json.loads(rbody.decode('utf-8'))
                ptoks = r.pop('prompt_token_ids', None)
                ctoks = []
                for ch in r.get('choices') or []:
                    if isinstance(ch, dict):
                        ctoks.append(ch.pop('token_ids', None))
                usage = r.get('usage')
                finish = [ch.get('finish_reason') for ch in r.get('choices') or [] if isinstance(ch, dict)]
                rbody = json.dumps(r, ensure_ascii=False, separators=(',', ':')).encode()
            except (UnicodeDecodeError, ValueError, AttributeError):
                pass
        st['forwarded'] += 1
        st['statuses'][status] += 1
        st['models'][mkey] += 1
        st['tool_result_roles'].update(summ['tool_result_roles'])
        st['history_reasoning_chars'] += summ['history_reasoning_chars']
        st['history_reasoning_messages'] += summ['history_reasoning_messages']
        st['sampling'][skey] += 1
        if usage and isinstance(usage, dict) and isinstance(usage.get('prompt_tokens'), int):
            st['prompt_tokens'].append(usage.get('prompt_tokens'))
        if st['down_bytes'] + len(rbody) > self.caps['down_cap']:
            raise _Refuse(429, 'job byte cap (down) reached')
        close = hdr.get('connection', '').lower() == 'close'
        self._send_client(st, c, status, rbody, close)
        self._capture(st, summ, body, obj, {'prompt_token_ids': ptoks, 'completion_token_ids': ctoks, 'usage': usage,
                                            'finish_reason': finish}, status, dt, None)
        return None if close else after

    def _upstream(self, body):
        req = (b'POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:%d\r\ncontent-type: application/json\r\n'
               b'content-length: %d\r\nconnection: close\r\n\r\n' % (int(self.upstream_port), len(body))) + body
        u = None
        try:
            u = socket.create_connection(('127.0.0.1', int(self.upstream_port)), timeout=10)
            u.settimeout(self.caps['upstream_timeout_s'])
            u.sendall(req)
            data = bytearray()
            while True:
                d = u.recv(1 << 20)
                if not d:
                    break
                data += d
                if len(data) > self.caps['resp_cap']:
                    return None, None, 'response over cap'
        except OSError as e:
            return None, None, repr(e)[:120]
        finally:
            if u is not None:
                with contextlib.suppress(OSError):
                    u.close()
        data = bytes(data)
        if b'\r\n\r\n' not in data:
            return None, None, 'malformed upstream response'
        head, body = data.split(b'\r\n\r\n', 1)
        lines = head.decode('latin-1').split('\r\n')
        try:
            status = int(lines[0].split(' ')[1])
        except (IndexError, ValueError):
            return None, None, 'malformed upstream status line'
        hdr = {}
        for l in lines[1:]:
            if ':' in l:
                k, v = l.split(':', 1)
                hdr[k.strip().lower()] = v.strip()
        if 'chunked' in hdr.get('transfer-encoding', '').lower():
            out, pos = bytearray(), 0
            while True:
                eol = body.find(b'\r\n', pos)
                if eol < 0:
                    return None, None, 'bad chunked encoding'
                size = int(body[pos:eol].split(b';')[0] or b'0', 16)
                pos = eol + 2
                if size == 0:
                    break
                out += body[pos:pos + size]
                pos += size + 2
            body = bytes(out)
        elif 'content-length' in hdr and hdr['content-length'].isdigit():
            body = body[:int(hdr['content-length'])]
        return status, body, None

    def _capture(self, st, summ, raw, obj, resp, status, dt, err):
        if st['cap_f'] is None:
            return
        rec = {'i': st['n_capture'], 't_utc': utc_now(), 'status': status, 'upstream_s': round(dt, 3), 'error': err,
               'body_sha256': sha256_bytes(raw), 'body_bytes': len(raw), **summ, **(resp or {})}
        if st['policy'].get('dump_bodies'):
            o = dict(obj)
            if 'chat_template' in o:
                o['chat_template'] = 'sha256:' + sha256_bytes(str(o['chat_template']).encode())
            rec['messages'] = o.get('messages')
            rec['tools'] = o.get('tools')
        with st['cap_lock']:
            st['n_capture'] += 1
            st['cap_f'].write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')

    def close(self, job):
        with self.lock:
            st = self.jobs.pop(job, None)
        if st is None:
            return None
        st['closed'] = True
        with contextlib.suppress(OSError):
            st['listener'].close()
        for s in list(st['conns']):
            with contextlib.suppress(OSError):
                s.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(OSError):
                s.close()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(st['path'])
        cap_sha = None
        with st['cap_lock']:
            if st['cap_f'] is not None:
                with contextlib.suppress(Exception):
                    st['cap_f'].close()
                st['cap_f'] = None
                with contextlib.suppress(OSError):
                    cap_sha = sha256_file(st['cap_path'])
        return {'accepted_conns': st['accepted'], 'refused_conns': st['refused_conns'], 'requests': st['requests'],
                'forwarded': st['forwarded'], 'refused': st['refused'], 'refusals': dict(st['refusals']),
                'refusal_log': st['refusal_log'][:50], 'up_bytes': st['up_bytes'], 'down_bytes': st['down_bytes'],
                'upstream_errors': st['upstream_errors'], 'statuses': {str(k): v for k, v in st['statuses'].items()},
                'models': dict(st['models']), 'tool_result_roles': dict(st['tool_result_roles']),
                'history_reasoning_chars': st['history_reasoning_chars'],
                'history_reasoning_messages': st['history_reasoning_messages'],
                'sampling_http': [json.loads(k) for k in st['sampling']], 'n_sampling_variants': len(st['sampling']),
                'variant_refusals': st['variant_refusals'],
                'prompt_tokens': [p for p in st['prompt_tokens'] if p is not None],
                'capture_path': os.path.basename(st['cap_path']) if st['cap_path'] else None, 'capture_sha256': cap_sha,
                'n_capture': st['n_capture']}

    def close_all(self):
        for job in list(self.jobs):
            self.close(job)


# ------------------------------------------------------------------------------------------------ container flags
_RLIMITS = {'core': 'RLIMIT_CORE', 'cpu': 'RLIMIT_CPU', 'data': 'RLIMIT_DATA', 'fsize': 'RLIMIT_FSIZE', 'locks': 'RLIMIT_LOCKS',
            'memlock': 'RLIMIT_MEMLOCK', 'msgqueue': 'RLIMIT_MSGQUEUE', 'nice': 'RLIMIT_NICE', 'nofile': 'RLIMIT_NOFILE',
            'nproc': 'RLIMIT_NPROC', 'rss': 'RLIMIT_RSS', 'rtprio': 'RLIMIT_RTPRIO', 'rttime': 'RLIMIT_RTTIME',
            'sigpending': 'RLIMIT_SIGPENDING', 'stack': 'RLIMIT_STACK'}
# D4: fixed values for the limits that D3 inherited from the launch context (they made the flags hash depend on it)
FIXED_RLIMITS = {'core': (1, 1), 'memlock': (65536, 65536), 'nofile': (1024, 524288), 'nproc': (476152, 476152),
                 'sigpending': (476152, 476152), 'stack': (8388608, -1), 'msgqueue': (819200, 819200),
                 'cpu': (-1, -1), 'data': (-1, -1), 'rss': (-1, -1), 'rttime': (-1, -1), 'nice': (0, 0), 'rtprio': (0, 0)}


def ulimits(fsize):
    """Fixed container rlimits (D3's values, which were the runner's own, now written out so the flags hash no longer
    depends on the launch context) with core = 1:1, fsize = <fsize>:<fsize> and memlock = 64 KiB (D3: 8 MiB)."""
    out = []
    for name in _RLIMITS:
        if name == 'fsize':
            v = (fsize, fsize)
        elif name in FIXED_RLIMITS:
            v = FIXED_RLIMITS[name]
        else:
            continue
        out.append(f'{name}={v[0]}:{v[1]}')
    return out


def tmpfs_flags(c, uid, skip=()):
    """Every writable path of the container: tmpfs with size= AND nr_inodes= (tmpfs pages also count against --memory)."""
    own = f'uid={uid},gid={uid},mode=0755'
    spec = [('/tmp', f'rw,exec,nosuid,nodev,size={c["tmp_size"]},nr_inodes={c["tmp_inodes"]},mode=1777'),
            ('/tmp/work', f'rw,exec,nosuid,nodev,size=16m,nr_inodes=4096,{own}'),
            ('/tmp/e6', f'rw,exec,nosuid,nodev,size=16m,nr_inodes=4096,{own}'),
            ('/kaggle/working', f'rw,exec,nosuid,nodev,size={c["out_size"]},nr_inodes={c["out_inodes"]},{own}'),
            ('/var/tmp', f'rw,exec,nosuid,nodev,size={c["vartmp_size"]},nr_inodes=16384,mode=1777'),
            ('/dev/shm', f'rw,noexec,nosuid,nodev,size={c["shm_size"]},nr_inodes=4096,mode=1777'),
            ('/run', 'rw,noexec,nosuid,nodev,size=16m,nr_inodes=1024,mode=0755')]
    out = []
    for path, opts in spec:
        if path not in skip:
            out += ['--tmpfs', f'{path}:{opts}']
    return out


def disk_of(paths):
    """The whole-disk block device (e.g. /dev/nvme0n1) behind every path (they must share one disk); io.max applies to
    whole disks. Returns (device path, 'MAJ:MIN')."""
    devs = set()
    for p in paths:
        d = os.stat(p).st_dev
        devs.add((os.major(d), os.minor(d)))
    if len(devs) != 1:
        raise ValueError(f'bind-mount sources span {len(devs)} devices: {sorted(devs)}')
    maj, mnr = devs.pop()
    sysp = os.path.realpath(f'/sys/dev/block/{maj}:{mnr}')
    if os.path.exists(os.path.join(sysp, 'partition')):
        sysp = os.path.dirname(sysp)
    name = os.path.basename(sysp)
    with open(os.path.join(sysp, 'dev')) as f:
        mm = f.read().strip()
    dev = f'/dev/{name}'
    st = os.stat(dev)
    if not stat.S_ISBLK(st.st_mode) or f'{os.major(st.st_rdev)}:{os.minor(st.st_rdev)}' != mm:
        raise ValueError(f'{dev} is not the block device {mm}')
    return dev, mm


def io_flags(c):
    """D4: block-device I/O caps of a job container on the disk behind the bind mounts (cgroup v2 io.max; the weight is
    io.weight / io.bfq.weight, which has no effect with the 'none' scheduler unless io.cost is set up: the caps are the
    effective control)."""
    if not c.get('io_device'):
        return []
    d = c['io_device']
    return ['--blkio-weight', str(c['io_weight']), '--device-read-bps', f'{d}:{c["read_bps"]}',
            '--device-read-iops', f'{d}:{c["read_iops"]}', '--device-write-bps', f'{d}:{c["write_bps"]}',
            '--device-write-iops', f'{d}:{c["write_iops"]}']


SECCOMP_FILE = 'seccomp_g4s_d41.json'      # tools/make_seccomp_d41.py: Docker 29.7.2's default minus Linux AIO, io_uring denied
SECCOMP_SHA256 = '739cc9049e6f9ce05793bc5210b33b81cf99dd0abeaf71b5337817a9db2b8619'


def seccomp_profile(lib_dir=None):
    """Absolute path of the D4.1 seccomp profile next to this file; refuses a symlink or any other content."""
    p = os.path.join(lib_dir or os.path.dirname(os.path.abspath(__file__)), SECCOMP_FILE)
    st = os.lstat(p)
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f'{p} is not a regular file')
    got = sha256_file(p)
    if got != SECCOMP_SHA256:
        raise ValueError(f'{p} sha256 {got} != pinned {SECCOMP_SHA256}')
    return p


def base_flags(c, uid, gid, skip=()):
    f = ['--rm', '--pull', 'never', '--network', 'none', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
         *(['--security-opt', f'seccomp={c["seccomp"]}'] if c.get('seccomp') else []),
         '--read-only', '--user', f'{uid}:{gid}', '--hostname', c['hostname'], '--log-driver', 'none',
         '--memory', c['memory'], '--memory-swap', c['memory'], '--pids-limit', str(c['pids']), '--cpus', str(c['cpus']),
         '--cpu-shares', str(c['cpu_shares']), '--oom-score-adj', '500', '--workdir', '/kaggle/working']
    f += io_flags(c)
    for u in ulimits(c['fsize']):
        f += ['--ulimit', u]
    return f + tmpfs_flags(c, uid, skip)


def ro(src, dst=None):
    src = str(src)
    return ['--mount', f'type=bind,src={src},dst={dst or src},readonly']


def worker_env(c, run_dir):
    """The worker's environment inside the container: the D2 runner's variables (minus host cache paths) and the D2 worker's
    inherited PATH/LANG/USER, with Kaggle's /tmp layout. Commands get the harness's sanitized subset of it."""
    return {'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin',
            'HOME': '/tmp/home', 'LANG': 'en_US.UTF-8', 'USER': 'sandbox', 'TMPDIR': '/tmp', 'TZ': 'UTC',
            'TRANSFORMERS_NO_TF': '1', 'TRANSFORMERS_NO_FLAX': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_OFFLINE': '1',
            'VLLM_NO_USAGE_STATS': '1', 'DO_NOT_TRACK': '1', 'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'LITELLM_TELEMETRY': 'False',
            'LITELLM_LOG': 'WARNING', 'OPENAI_API_KEY': 'EMPTY', 'LOCAL_API_KEY': 'EMPTY', 'OTEL_SDK_DISABLED': 'true',
            'OTEL_PYTHON_DISABLED': 'true', 'PIP_DISABLE_PIP_VERSION_CHECK': '1', 'BERTH_SHIMS': '/tmp/shims',
            'XDG_CACHE_HOME': '/tmp/home/.cache', 'XDG_CONFIG_HOME': '/tmp/home/.config', 'MPLCONFIGDIR': '/tmp/home/.mpl',
            'HF_HOME': '/tmp/home/.hf', 'GEMMA4_EVAL_RUN': str(run_dir)}


def deps_key(reqs):
    return hashlib.sha256('\n'.join(sorted(reqs)).encode()).hexdigest()[:16]


STATFS = ['/tmp', '/tmp/work', '/tmp/e6', '/kaggle/working', '/var/tmp', '/dev/shm', '/run']


def write_env_file(path, cfg):
    data = json.dumps(cfg, separators=(',', ':'))
    if '\n' in data or '\0' in data:
        raise ValueError('G4S_CFG must be one line')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write('G4S_CFG=' + data + '\n')


# ------------------------------------------------------------------------------------------------ one container
def run_container(env, argv, name, kill_after_s, cap_out, cap_err, log=print, on_start=None):
    """docker run (attached; stdout = the frame channel). Reads stdout up to cap_out bytes and stderr up to cap_err bytes
    (the rest is drained and discarded), docker-kills the container after kill_after_s, and makes sure it is gone."""
    t0 = time.time()
    p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                         start_new_session=True)
    if on_start is not None:
        on_start()
    bufs = {'out': bytearray(), 'err': bytearray()}
    tot = {'out': 0, 'err': 0}

    def rd(stream, key, cap):
        while True:
            try:
                b = stream.read1(1 << 20)
            except (OSError, ValueError):
                break
            if not b:
                break
            tot[key] += len(b)
            room = cap - len(bufs[key])
            if room > 0:
                bufs[key] += b[:room]
    ts = [threading.Thread(target=rd, args=(p.stdout, 'out', cap_out), daemon=True),
          threading.Thread(target=rd, args=(p.stderr, 'err', cap_err), daemon=True)]
    for t in ts:
        t.start()
    killed = False
    try:
        p.wait(timeout=kill_after_s)
    except subprocess.TimeoutExpired:
        killed = True
        log('job deadline: docker kill', name)
        with contextlib.suppress(Exception):
            docker(env, 'kill', '--signal', 'KILL', name, timeout=60)
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                os.killpg(p.pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                p.wait(timeout=10)
    for t in ts:
        t.join(timeout=30)
    gone = ensure_gone(env, name)
    return {'rc': p.returncode, 'stdout': bytes(bufs['out']), 'stderr': bytes(bufs['err']), 'stdout_total': tot['out'],
            'stderr_total': tot['err'], 'stdout_over_cap': tot['out'] > cap_out, 'killed_by_orchestrator': killed,
            'gone': gone, 'wall_s': round(time.time() - t0, 3)}


def parse_frame(stdout, nonce):
    """The LAST well-formed frame with this nonce (length and sha256 checked). Returns (payload bytes or None, error)."""
    rx = re.compile(FRAME_RE_TMPL % re.escape(nonce.encode()))
    ms = list(rx.finditer(stdout))
    for m in reversed(ms):
        n, digest = int(m.group(1)), m.group(2).decode()
        body = stdout[m.end():m.end() + n]
        trailer = b'\nG4S-FRAME-END ' + nonce.encode() + b'\n'
        if len(body) == n and stdout[m.end() + n:m.end() + n + len(trailer)] == trailer and hashlib.sha256(body).hexdigest() == digest:
            return body, None
    return None, ('no frame' if not ms else 'frame damaged')


def safe_rel(name):
    if not isinstance(name, str) or name.startswith('/') or '\\' in name or len(name) > 600:
        return None
    parts = name.split('/')
    if len(parts) > 10 or any(p in ('', '.', '..') or not SAFE_PART.match(p) for p in parts):
        return None
    return '/'.join(parts)


def tar_members(payload, max_file, max_total, max_members=20000):
    """[(rel, bytes)] of the regular files in an in-memory tar; anything else (links, devices, odd names) is skipped."""
    out, skipped, total = [], [], 0
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:') as t:
        for i, m in enumerate(t):
            if i >= max_members:
                skipped.append('too many members')
                break
            rel = safe_rel(m.name)
            if rel is None or not m.isreg() or m.size > max_file or total + m.size > max_total:
                skipped.append((m.name[:120], 'rejected'))
                continue
            data = t.extractfile(m).read(max_file + 1)
            if len(data) != m.size:
                skipped.append((rel, 'size mismatch'))
                continue
            total += len(data)
            out.append((rel, data))
    return out, skipped


def write_new(root, rel, data):
    """Write data to root/rel: parents created inside root (a directory only this orchestrator writes), O_EXCL|O_NOFOLLOW."""
    root = os.path.realpath(root)
    path = os.path.join(root, rel)
    if os.path.commonpath([root, os.path.realpath(os.path.dirname(path))]) != root:
        raise ValueError(f'path escapes root: {rel}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
    return path


# ------------------------------------------------------------------------------------------------ strict ingestion (D4)
HOOK_ROW_KEYS = ('e8_hook_bound', 'e8_hook_rebound', 'e8_hook_calls', 'e8_hook_calls_failed')   # D4.2: E8 worker (6.2(g))
WORKER_ROW_KEYS = frozenset({'arm', 'task', 'seed', 'repeat_id', 'worker', 'resolved', 'tool_calls', 'patch_chars', 'error',
                             'duration', 'tokens', 'test_exit', 'test_summary', 'exc', 'tb', 'setup_fallback', 't_phase1',
                             't_agent', 't_setup', 't_verify', 'sampling_sent', 'wall', 'summaries', 'summary_seconds',
                             'summary_chars_in', 'llm_calls', 'pad_s', 'over_target', 'tok_out', 'tok_in', 'max_prompt',
                             *HOOK_ROW_KEYS})
MAX_LINE = 64 << 10           # a result or started line longer than this is refused before json.loads
MAX_RESULTS_FILE = 1 << 20    # results.jsonl / started.jsonl members larger than this are refused unread
MAX_INIT_JSON = 4 << 20       # PID 1's init.json (events, tar info, cgroup stats)
MAX_ARM_STATS = 256 << 10     # out/arm_stats.json (counts only; D4 real maximum 848 bytes)
# D4.1: value types of a worker row (the e6 worker's types over 3,740 real rows of D3/D4/pool runs); anything else is
# refused as tamper evidence (it would also break the runner's summary and the analysis)
_T_INT, _T_NUM = ('int',), ('int', 'float')
ROW_TYPES = {'arm': ('str',), 'task': ('str',), 'seed': _T_INT, 'repeat_id': _T_INT, 'worker': _T_INT, 'resolved': ('bool',),
             **{k: _T_INT + ('none',) for k in ('tool_calls', 'patch_chars', 'tokens', 'test_exit', 'setup_fallback', 'summaries',
                                                'summary_chars_in', 'llm_calls', 'over_target', 'tok_out', 'tok_in', 'max_prompt')},
             **{k: _T_NUM + ('none',) for k in ('duration', 't_phase1', 't_agent', 't_setup', 't_verify', 'wall', 'summary_seconds',
                                                'pad_s')},
             **{k: ('str', 'none') for k in ('error', 'test_summary', 'exc', 'tb')}, 'sampling_sent': ('dict', 'none'),
             'e8_hook_bound': ('bool',), 'e8_hook_rebound': ('list',), 'e8_hook_calls': _T_INT, 'e8_hook_calls_failed': _T_INT}
assert set(ROW_TYPES) == set(WORKER_ROW_KEYS)
# D4.2: the length caps of the worker's own string fields (worker_berth.py: error[:200], test_summary[:160], repr(e)[:300],
# traceback[-600:]; the E8 hook's exc [:300]; arm and task as the runner's job check), and the hook's name list
ROW_STR_CAPS = {'arm': 120, 'task': 120, 'error': 200, 'test_summary': 160, 'exc': 300, 'tb': 600}
MAX_REBOUND_NAMES, MAX_REBOUND_NAME = 64, 200
MAX_ROW_INT = 2 ** 53
MAX_SAMPLING_SENT_BYTES = 4096


def _jtype(v):
    return ('none' if v is None else 'bool' if isinstance(v, bool) else 'int' if isinstance(v, int) else 'float' if isinstance(v, float)
            else 'str' if isinstance(v, str) else 'dict' if isinstance(v, dict) else 'list' if isinstance(v, list) else 'other')


def row_type_problems(row):
    """Keys of a worker row whose value type is not the e6 worker's (ROW_TYPES); sampling_sent nests at most 4 levels.
    D4.2: string fields within the worker's length caps; numbers finite and within +-2**53; the E8 hook's name list a
    short list of short strings; sampling_sent at most 4 KiB as JSON."""
    import math
    bad = sorted(k for k, v in row.items() if k in ROW_TYPES and _jtype(v) not in ROW_TYPES[k])
    if isinstance(row.get('sampling_sent'), dict):
        if json_depth(row['sampling_sent'], 4) > 4:
            bad.append('sampling_sent depth')
        elif len(json.dumps(row['sampling_sent'], default=str)) > MAX_SAMPLING_SENT_BYTES:
            bad.append('sampling_sent size')
    for k, n in ROW_STR_CAPS.items():
        if isinstance(row.get(k), str) and len(row[k]) > n:
            bad.append(f'{k} length')
    for k, v in row.items():
        if k in ROW_TYPES and isinstance(v, float) and not math.isfinite(v):
            bad.append(f'{k} not finite')
        elif k in ROW_TYPES and isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v) > MAX_ROW_INT:
            bad.append(f'{k} out of range')
    rb = row.get('e8_hook_rebound')
    if isinstance(rb, list) and (len(rb) > MAX_REBOUND_NAMES or not all(isinstance(x, str) and len(x) <= MAX_REBOUND_NAME for x in rb)):
        bad.append('e8_hook_rebound items')
    for k in ('e8_hook_calls', 'e8_hook_calls_failed'):
        if isinstance(row.get(k), int) and not isinstance(row.get(k), bool) and row[k] < 0:
            bad.append(f'{k} negative')
    return bad


def job_artifacts(arm, task, seed, mode):
    """The out/ paths one job (arm, task, seed) may return (D4: anything else is refused and recorded as tamper evidence)."""
    s = f's{seed}'
    allowed = {f'llm/{arm}/{task}_{s}.jsonl.gz', f'patches/{arm}/{s}/{task}.diff', f'tests/{arm}/{s}/{task}.txt',
               f'{arm}/{s}/traces/trace_{task}.json', f'{arm}/{s}/logs/{task}.log'}
    return allowed


def split_lines(data, max_line=MAX_LINE):
    """Non-empty lines of a member, each capped BEFORE parsing: returns (lines <= max_line, n_over, over_sha256s)."""
    ok, over = [], []
    for l in data.split(b'\n'):
        if not l.strip():
            continue
        if len(l) > max_line:
            over.append((len(l), sha256_bytes(l)))
        else:
            ok.append(l)
    return ok, over


# ------------------------------------------------------------------------------------------------ owner names, hashed (D4.1)
OWNER_HASH_SCHEME = 'g4s-owner-hash-v1: sha256(salt bytes + pattern UTF-8); exact case'


class OwnerPatterns:
    """Owner names kept ONLY as salted sha256 values (local/owner_patterns_hashed.json, 0600 in a 0700 directory; the
    plain-text pattern file was converted once on the host and deleted). A text is matched by hashing, at EVERY character
    position, the substring of each stored pattern length and comparing it with the stored values: exact case, the same
    occurrences as D4's plain-text substring scan (non-overlapping, the longest pattern first at a position).
    `generic` marks patterns that are also ordinary software words (they occur in the Berth task records or as installed
    package names), which redaction leaves alone."""

    def __init__(self, path):
        d = json.load(open(path))
        if d.get('scheme') != OWNER_HASH_SCHEME:
            raise ValueError('unknown owner pattern hash scheme')
        self.salt = bytes.fromhex(d['salt'])
        self.patterns = d['patterns']
        self.by_len = collections.defaultdict(dict)
        for p in self.patterns:
            self.by_len[int(p['len'])][p['sha256']] = p
        self.lens = sorted(self.by_len, reverse=True)
        self.sha256 = sha256_file(path)

    def spans(self, text, cats=None, generic=True):
        """[(start, end, pattern)] of non-overlapping matches, left to right, longest first."""
        out, i, n = [], 0, len(text)
        base = hashlib.sha256(self.salt)
        lo = min(self.lens) if self.lens else n + 1
        while i <= n - lo:
            hit = None
            for L in self.lens:
                if i + L > n:
                    continue
                h = base.copy()
                h.update(text[i:i + L].encode('utf-8', 'surrogatepass'))
                p = self.by_len[L].get(h.hexdigest())
                if p and (cats is None or p['cat'] in cats) and (generic or not p.get('generic')):
                    hit = (i, i + L, p)
                    break
            if hit:
                out.append(hit)
                i = hit[1]
            else:
                i += 1
        return out

    def count(self, text, cats=None, generic=True):
        return collections.Counter(p['id'] for _, _, p in self.spans(text, cats, generic))

    def redact(self, text, cats=None, generic=False):
        """(text with every match replaced by '<owner-name:<12 hex of its salted sha256>>', number replaced)."""
        sp = self.spans(text, cats, generic)
        if not sp:
            return text, 0
        parts, j = [], 0
        for i, e, p in sp:
            parts += [text[j:i], f'<owner-name:{p["sha256"][:12]}>']
            j = e
        parts.append(text[j:])
        return ''.join(parts), len(sp)


# ------------------------------------------------------------------------------------------------ nvidia-smi (privacy)
def redact_smi(text):
    """Keep the GPU summary; replace EVERY row of the Processes table (any type: G, C, C+G, M, ...) with a sha256 marker."""
    out, in_procs, body, n = [], False, False, 0
    for line in text.splitlines():
        s = line.strip()
        if 'Processes:' in line:
            in_procs, body = True, False
            out.append(line)
            continue
        if in_procs:
            if s.startswith('|='):
                body = True
                out.append(line)
                continue
            if s.startswith('+'):
                out.append(line)
                if body:
                    in_procs = False
                continue
            if body and s and 'No running processes found' not in s:
                out.append(marker(s))
                n += 1
                continue
        out.append(line)
    return '\n'.join(out) + ('\n' if text.endswith('\n') else ''), n


def redact_apps_csv(text):
    """compute-apps CSV: keep pid and memory, replace the process name by a marker."""
    out, n = [], 0
    for i, r in enumerate(text.splitlines()):
        parts = r.split(', ')
        if i > 0 and len(parts) >= 3 and parts[0].strip().isdigit():
            out.append(', '.join([parts[0], marker(', '.join(parts[1:-1])), parts[-1]]))
            n += 1
        elif i > 0 and r.strip():
            out.append(marker(r))
            n += 1
        else:
            out.append(r)
    return '\n'.join(out), n


def smi_snapshot(path):
    p = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=60)
    q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'],
                       capture_output=True, text=True, timeout=60)
    a, n1 = redact_smi(p.stdout)
    b, n2 = redact_apps_csv(q.stdout)
    with open(path, 'w') as f:
        f.write(a + '\n' + b + '\n')
    return n1 + n2


# ------------------------------------------------------------------------------------------------ memory ceiling (D4.2)
RUNNER_MEMORY_MAX = 3 << 30           # MemoryMax of the runner's user scope (launch.sh, the canaries); the runner's own
RUNNER_MEMORY_MAX_LIMIT = 4 << 30     # process stays near 200 MB, docker CLIs and job frames add to its cgroup; the runner
                                      # refuses to start in a cgroup without memory.max <= this limit or with swap


def cgroup_memory(pid='self'):
    """{'cgroup', 'memory.max', 'memory.swap.max', 'memory.current', 'memory.peak', 'memory.events'} of a process's cgroup
    (cgroup v2); integers where the file holds a number, 'max' or None otherwise."""
    out = {}
    try:
        cg = open(f'/proc/{pid}/cgroup').read().strip().split('::', 1)[1]
    except (OSError, IndexError):
        return {'cgroup': None}
    out['cgroup'] = cg
    base = '/sys/fs/cgroup' + cg
    for f in ('memory.max', 'memory.swap.max', 'memory.current', 'memory.peak', 'memory.high'):
        try:
            v = open(os.path.join(base, f)).read().strip()
            out[f] = int(v) if v.isdigit() else v
        except OSError:
            out[f] = None
    try:
        out['memory.events'] = dict(l.split() for l in open(os.path.join(base, 'memory.events')).read().split('\n') if l.strip())
    except (OSError, ValueError):
        out['memory.events'] = None
    return out


def memory_ceiling_problem(info):
    """None if the cgroup bounds this process's memory (memory.max <= RUNNER_MEMORY_MAX_LIMIT, swap 0), else a reason."""
    mx, sw = info.get('memory.max'), info.get('memory.swap.max')
    if not isinstance(mx, int) or mx > RUNNER_MEMORY_MAX_LIMIT:
        return f'memory.max {mx!r} of cgroup {info.get("cgroup")} is not a ceiling <= {RUNNER_MEMORY_MAX_LIMIT} bytes'
    if sw != 0:
        return f'memory.swap.max {sw!r} is not 0'
    return None


def _user_env():
    return dict(os.environ, XDG_RUNTIME_DIR=os.environ.get('XDG_RUNTIME_DIR') or f'/run/user/{os.getuid()}')


def scope_argv(unit, memory_max=RUNNER_MEMORY_MAX, extra=()):
    """argv prefix that runs a command in a transient user-level systemd scope (no privileges: the user manager delegates
    the memory controller) with MemoryMax and no swap; systemd-run --scope execs the command in place (same PID)."""
    if not re.fullmatch(r'[A-Za-z0-9_.@-]{1,120}', unit):
        raise ValueError(f'bad unit name {unit!r}')
    return ['systemd-run', '--user', '--scope', '--quiet', '--collect', '--unit', unit,
            '-p', f'MemoryMax={int(memory_max)}', '-p', 'MemorySwapMax=0', *extra, '--']


def user_scope_env():
    """Environment for systemd-run --user (XDG_RUNTIME_DIR is unset in non-login shells)."""
    return _user_env()


# ------------------------------------------------------------------------------------------------ vLLM parent-death signal
PDEATH_WRAP = ('import ctypes, os, signal, sys\n'
               'ppid = int(sys.argv[1])\n'
               'ctypes.CDLL(None).prctl(1, int(signal.SIGTERM), 0, 0, 0)  # PR_SET_PDEATHSIG\n'
               'if os.getppid() != ppid:\n'
               '    sys.exit(97)\n'
               'os.execvp(sys.argv[2], sys.argv[2:])\n')


def pdeath_argv(python, cmd):
    """Run cmd so that it receives SIGTERM when this (orchestrator) process dies, even by SIGKILL (survives exec)."""
    return [python, '-c', PDEATH_WRAP, str(os.getpid()), *cmd]


def vllm_scope_argv(tag, python, cmd):
    """D4.2: vLLM in its own user scope (no MemoryMax: the runner's ceiling must not count the model server), still a
    child of the runner with the parent-death signal (systemd-run --scope execs the wrapper in place)."""
    unit = f'g4s-vllm-{tag}-{secrets.token_hex(3)}'[:120]
    return ['systemd-run', '--user', '--scope', '--quiet', '--collect', '--unit', unit, '-p', 'MemorySwapMax=0', '--',
            '/usr/bin/env', '-u', 'MALLOC_ARENA_MAX', *pdeath_argv(python, cmd)]     # vLLM keeps glibc's default arenas
