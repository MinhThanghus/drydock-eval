<!--
Copyright 2026 The drydock-eval Authors
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-->

# drydock-eval contain: operator guide

`drydock_eval/contain/reference/` is the contained job runner described in section 3 of the write-up, released as a
**reference implementation**. It is the code the authors verified, with a licence header added to every file, the
host-specific constants (paths, account name, lock-file location, port, bundle labels) replaced, and in two canary
files (`tools/canary_d4.py`, `tools/d4canary/canary_inner.py`) one check rewritten: the home-directory listing and the
write targets (section 7). The released tree has not been run in a container. It is not a turnkey
product: it expects a competition harness, tasks, a model and an agent bundle that you supply, and it was written for
one evaluation stack (the Berth task set, Gemma 4 E4B served by vLLM 0.19.1, google-adk 1.36.1).

This guide, the code it describes, its tests and the reviews of them were produced by AI coding agents working for the
authors; each check is stated with the command that reproduces it, and none replaces your own review. A human owner
approved the use of AI agents for this work and is responsible for the release.

Read the section "Status of this release" before you rely on anything here.

## 1. Status of this release

* **The released tree has not been run in Docker by the authors of the package.** The canary results of the write-up
  (Table 1: 200 of 200 checks for the final version) were obtained with the original files on the original host.
  Adding a licence header changes every file hash, so the image, runner and library hashes of this tree are new and no
  gate file exists for it. You create your own gate by passing the canaries on your host (section 6).
* What was checked for this release without Docker: the test suite of the kit (`pytest tests/`), which loads the
  released library and exercises the request-body scanner, the forwarder against a stub upstream, the ingestion
  helpers, the gate rule and the job-list rules; and the tree's own scanner check (`drydock-eval contain scan-check`).
* `SOURCES.json` lists every file of the tree with its sha256, whether it is verbatim or changed, and the sha256 of the
  verified source it was built from. `drydock-eval contain sources` compares a tree against it.
* The names in the code are the authors' internal ones. A short glossary is in section 9.

## 2. What you need

| You need | Used for | Shipped here |
|---|---|---|
| Linux with cgroup v2, a user-level systemd manager, `flock`, `setsid`, `nice` | the runner's own memory ceiling (`tools/scoped.sh`), launch scripts | no |
| Docker (`/usr/bin/docker`), permission to run containers with `--network none`, seccomp and block-I/O limits | one container per job | build files only (`docker/`) |
| A vLLM server environment with a GPU (`<deploy>/env-serve`) | agent mode only; the runner starts and stops vLLM itself | no |
| Your own copy of the competition harness packages, installed in a worker environment (`<deploy>/eval/hostenv2`) | the worker inside each container imports them | **no** (not ours to distribute) |
| The task set and its wheelhouse (`<deploy>/eval/data/...`, `<deploy>/wheelhouse`) | job inputs, mounted read-only | **no** |
| A served model directory with its tokenizer and chat template (`<deploy>/models/served_model`) | agent mode | **no** |
| An agent bundle (`evald/bundles/<name>/agent.yaml`) and, for per-request templates, `evald/templates/` | what is being evaluated | **no** |

Canary mode, gold mode and empty mode need Docker but **no GPU and no model**. Agent mode needs all of the above.

Files the code reads that are not in this package (you supply them or the corresponding check does not apply):
`jobs/drop_empty.json` and the E8 arm file for the E8 worker canaries; a chat template and `tokenizer.json` for the
canaries that hash them; `tools/seccomp/moby_profiles_seccomp_v0.2.3_default.json` (the Moby default profile, input of
`tools/make_seccomp_d41.py`, sha256 `536529b665dd0972c37bfb569f5d4ac8a53592e7b00752bc39ff063ca9864c74`);
`local/owner_patterns_hashed.json` (optional salted hashes of your own process names, so the canaries can tell that your
other processes were not touched); and the records of earlier generations that some canary checks and the gate writer
compare against (section 7).

## 3. Deployment layout

The scripts locate everything relative to their own directory. Nothing is configured by editing the code.

```
<deploy>/
  evald/              a copy of reference/ (make it with: drydock-eval contain init <deploy>/evald)
    run_berth.py launch.sh stop.sh status.sh lib/ docker/ tools/
    bundles/ templates/ jobs/ local/      yours
    runs/ cache/ canary_passed.json       created at run time
  env-serve/          Python environment with vLLM and the harness's serving dependencies
  eval/hostenv2/      worker environment (the competition harness packages)
  eval/data/          tasks (run_berth.py --comp, --pyds, --shims)
  models/served_model the model directory (--model-dir)
  wheelhouse/         (--wheelhouse)
  gpu.lock            flock file that serialises GPU use (agent mode holds it for the whole run)
```

Every location above is also a `run_berth.py` option (`--comp`, `--pyds`, `--shims`, `--wheelhouse`, `--worker-env`,
`--base-env`, `--model-dir`, `--cache`, `--port`, `--served-model-name`, `--model-name`, `--image`). The one
environment variable is `DRYDOCK_CONTAIN_HOST_LABEL` (default `host`): the text written to the `host` field of a run's
`summary.json`. The gate binds the options (`options_sha256`), so changing an option after the canaries means passing
them again.

Controls that are deliberately **not** configurable: the container account is `sandbox` with uid/gid 2000 (the image
build asserts that no account has uid 1000), the runner refuses to start below `nice 10` or without a cgroup memory
ceiling, at most 8 workers, GPU memory fraction at most 0.55, served model names must not contain `gemma4` (except the
one logged E8 exception), and there is no default mode.

## 4. What a job gets

One fresh container per job, started by image ID: `--network none`, `--cap-drop ALL`, `no-new-privileges`, the pinned
seccomp profile `lib/seccomp_g4s_d41.json` (Docker's default without Linux AIO; io_uring is denied), read-only root,
uid 2000, 2 CPUs, 10 GiB memory with no extra swap, 4,096 processes, tmpfs-only writable paths capped in bytes and
inodes, read/write rate caps on the disk behind the read-only bind mounts, no Docker socket, no GPU. The exact flag
list is in every run's `config.json`.

The only channel out is a per-job Unix socket to the forwarder in the runner (`HttpForwarder` in `lib/g4s_d4.py`). It
accepts only `POST /v1/chat/completions` with a JSON body that passes, in this order: the linear shape scan (depth 64,
20,000 containers, 50,000 values, 70,001 strings), UTF-8 JSON parsing, the depth check, and the field allow-list
(`check_chat_body`: allow-listed fields, text-only content, an approved chat-template hash when a template is sent).
Default caps per job: 2 concurrent connections, 120 new connections per minute, 60 requests per minute, 2,000 requests
in total, 1 MiB body, 64 MiB up, 64 MiB down. Headers are not forwarded except content type and length.
`drydock-eval contain check-body <file>` applies the body checks to a file, without a server.

One observed property you should know: the header cap (32 KiB) is checked while the end of the header block has not
yet arrived. A header block that arrives complete in a single read (at most 64 KiB) is not refused for its size. It is
still parsed, stripped and never forwarded; the bound on buffered header bytes is therefore cap plus one read, not cap.

Results come back only as PID 1's nonce-framed tar on the container's stdout; every member is size- and depth-capped
before parsing, and a job that was started always gets exactly one row (its own, a tamper row, or the runner's row).

## 5. The gate file

Agent mode (model-issued commands) is refused unless `evald/canary_passed.json` binds the configuration that is about
to run. The runner computes eight values and compares each with the file:

| Key | What it is |
|---|---|
| `image_id` | Docker image ID of the job image |
| `runner_sha256` | sha256 of `run_berth.py` |
| `lib_sha256` | sha256 of `lib/g4s_d4.py` |
| `flags_sha256` | hash of the image ID and the container flags of a job |
| `options_sha256` | hash of twelve runner options (deadlines, frame cap, model names, context length, GPU fraction, relay port, capture, ...) |
| `harness_sha256` | hash of the map of harness, worker and classifier file hashes |
| `model_sha256` | hash of the served model directory |
| `vllm_args_sha256` | hash of the vLLM safety arguments, parsers, dtype, context length and GPU fraction |

The file also carries a `scope` list; the run's family (`E8` or `E7/P1`) must be in it. The runner refuses when the file is absent, when any of the eight values is empty or differs (it names the keys), or
when the scope does not include the family. `--gate precanary:<tag>` is a narrower gate for validation blocks only
(`SMOKE_*`, `CAPTURE_*`, `CANARY_*`, `E8smoke`, `E8capture`): it needs a complete canary record `runs/<tag>_canary_d4.json`
with every part passed, bound to the image, runner, library and harness hashes.

The gate is a statement about hashes. It does not bind the host kernel, the Docker daemon and its configuration, the
vLLM package, or the contents of the task set. Whoever can write `canary_passed.json` can open the gate: protect the
deployment directory accordingly.

Read-only helpers (the enforcing check is in `run_berth.py`; these only report):

```
drydock-eval contain gate show   --root <deploy>/evald
drydock-eval contain gate verify --root <deploy>/evald --config <deploy>/evald/runs/<tag>/config.json
```

`gate show` prints what the gate file binds and the runner and library hashes of the tree. `gate verify` takes the
`config.json` a run wrote and says whether the gate file binds it (exit 0), with the refusal the runner would print
otherwise (exit 1).

## 6. Running

Checks that need neither Docker nor a GPU:

```
pip install -e ".[test]" && python -m pytest tests/          # the kit's tests (Docker tests skip themselves)
drydock-eval contain sources                                  # the packaged tree equals SOURCES.json
drydock-eval contain scan-check --quick                       # body scanner: linear time to the 1 MiB cap, exact on valid JSON (idle machine)
drydock-eval contain check-body request.json                  # would the forwarder accept this body?
```

The scanner check has a timing part (time at the 1 MiB cap and the ratios when the body size doubles), so it needs an
idle machine: the command reports the strict verdict of one run, and a busy machine can fail the timing part (12 ns per
byte was seen under load against 3.4 idle) while the exactness part still passes. Rerun it on an idle machine before
reading a failure as a defect. The kit's own test of it (`tests/test_contain_gate_cli.py`) repeats a timing-only failure
twice, skips with the numbers if the machine was loaded, and fails if it was not; a failure of the exactness part always
fails it.

Canaries (Docker, **no GPU, no model**; they start real job containers through the real runner in canary mode):

```
drydock-eval contain init <deploy>/evald
cd <deploy>/evald && docker/build.sh v8                       # builds g4s-sandbox:v8, records docker/image_v8.txt
drydock-eval contain canaries --root <deploy>/evald --tag c1  # = nice -n 10 /usr/bin/python3 -I -B tools/canary_d4.py --tag c1
```

The result is `runs/c1_canary_d4.json`. `--parts` selects groups (`battery,d4,d41,d42,d43,kills,backstop,sigkill,pdeath,gates,reaper,static`).
The live canary that sends attack requests to the real model (`--canary-vllm`) needs a GPU and an existing gate file.

A job list is a JSON array of jobs with `arm` and `task` (the bundle defaults to the arm name; agent mode also needs
the drop-list hash `a0_drop_sha256`; the runner fills in block, repeat, attempt and job id and refuses what does not fit):

```
drydock-eval contain run --root <deploy>/evald --tag g1 --jobs jobs.json --mode gold -- --block S --workers 4
```

This calls `launch.sh`, which starts the runner detached at nice 10 inside a memory-capped systemd scope; the log is
`runs/g1.launch.log`, the run directory `runs/g1/`. `stop.sh <tag>` ends a run; `status.sh <tag>` shows progress.
Every wrapper accepts `--dry-run` and prints the command it would run.

Before you run anything on a shared host:

* `docker/build.sh <version>` re-points the tag `g4s-sandbox:<version>`. The runner resolves the tag to an image ID at
  start and the gate binds the ID, so re-pointing a tag that another deployment uses makes that deployment's gate refuse.
  Use a version name of your own.
* The reaper removes every container, network and volume on the Docker daemon that carries the runner's label and a `g4s-`
  name and whose creating process is dead, whichever deployment created it. The canaries also load the host on purpose
  (CPU, memory, disk, process table). Do not run canaries next to a running evaluation.
* The GPU lock is a file inside one deployment. Two deployments on one host do not exclude each other.
* The runner refuses to start with less than 150 GiB free under the run directory (`--min-free-gib`).
* The Docker tests of the kit (`tests/test_contain_docker.py`) are skipped unless `DRYDOCK_CONTAIN_DEPLOY` names a
  deployment; they default to the image version `v8kit04`.

## 7. Known constraints of the reference tree

* Some canary checks compare against records of the previous generations of the runner ("prior" parity checks), and
  `tools/write_canary_passed_d43.py`, which wrote the authors' gate file, reads the validation history of the original
  deployment. Neither record is shipped (they are run outputs). On a fresh host those checks fail and the gate writer
  does not run as it is. We kept the checks rather than weaken them; use the writer as the reference for what a gate
  must contain and write your own after your canaries pass.
* The canary that lists `/home` inside the container (`home_visibility`) was written for a host whose operator account
  had the same name as the image's account. In this tree the expected listings are derived from the deployment path:
  `/home` may hold the image account `sandbox` and the first directory of a deployment that lives under `/home`, and
  each of these directories may hold only the mount stubs of the two worker environments. When the deployment lives
  in the home directory of an account named `sandbox`, this is the original check. The logic is tested without Docker
  (`tests/test_contain_gate_cli.py`); the canary itself has not been run in a container in this form. It assumes that
  the runner mounts the worker environments from their default places (`<deploy>/eval/hostenv2`, `<deploy>/env-serve`).
* `docker/build.sh` pins the sha256 of the two worker files (`want_w`, `want_s`). In this tree the pins are those of the
  released files, which differ from the verified worker files only by the licence header (their syntax trees are
  identical). The pin therefore protects the released bytes, not the verified ones; the link to the verified bytes is
  `SOURCES.json` (`verified_source_sha256`).
* The labels of the screened variants and of the adapter arms in `tools/analyze_e7.py` are neutral names (`bundle0_variant_1`
  to `_5`, `bundle0_adapter_A` and `_B`). They replace the authors' names for them in every place they were used; the
  analysis behaves as before when the arms are renamed the same way (checked on synthetic rows). The defaults of the
  command line use these labels (the default arm list of the screening, and which arms count as adapter arms); to use
  other names, give `--arms` and, from Python, `adapter_arms=`.
* The row classifier `tools/analyze_e7.py` embeds error tables whose `source` strings were neutralised, so the table
  hashes in records written by this tree differ from the ones in the authors' records. Classification of rows was
  compared on synthetic rows only.
* The adapters import the competition harness by its package names. You need the same harness version, or the
  `harness_sha256` map will lack files and the adapters may not load.
* `docker/build.sh` pins the sha256 of the two baseline worker files of this tree; it refuses to build if they change.

## 8. What remains

Stated plainly, as in section 3 of the write-up:

* **Containers share the host kernel.** A kernel exploit is outside what this design defends against.
* **User namespaces are not remapped.** uid 2000 in the container is uid 2000 on the host; isolation rests on the
  other controls (no capabilities, seccomp, read-only root, no-new-privileges).
* **The request parser runs in the trusted runner.** Every byte a job sends to the model endpoint is parsed by the
  runner's Python process. The scanner bounds its time and memory; a parser bug would be a bug in the trusted part.
* **A job can slow other jobs through the shared model server**, within its request and byte caps.
* **The claim is conditional**: against the stated adversary (the code the agent writes) and with the gate file
  verified, no tested attack succeeded. Passing canaries shows the absence of the tested failures, not of all failures.
* For this package in addition: the released tree itself is untested in Docker (section 1), and the gate does not cover
  the kernel, the Docker daemon or the vLLM package (section 5).

## 9. Glossary of internal names

| Name | Meaning |
|---|---|
| `D3`, `D4`, `D4.1`, `D4.2`, `D4.3` | successive hardening rounds of the runner; the released code is the last one. Comments cite the round in which a control was added. |
| `g4s` | prefix of the image (`g4s-sandbox`), library (`g4s_d4.py`) and in-image paths |
| `evald` | the deployment directory of the runner |
| `Berth` | the task set the runner was built for |
| `e6` worker, `E8` worker, hook | the baseline worker script, and the experiment-E8 worker that adds per-arm request settings |
| `E7/P1`, `E8` | experiment families; the gate's `scope` lists the families it covers |
| `A0` drop list | list of tasks excluded before a run; jobs carry its sha256 |
| `I09`, `I12`, ... | row classes of the classifier (infrastructure and tamper rows) |
| `owner` | the operator's account on the host, whose files and processes the canaries check are untouched |
| `gemma4swe=eval` | the Docker label on every object the runner creates, used by the reaper |
| `README.md` in code comments | this guide |

## 10. Third-party material

`lib/seccomp_g4s_d41.json` is derived from the default seccomp profile of the Moby project (github.com/moby/profiles,
seccomp/default.json, module version v0.2.3, Apache-2.0) by removing the Linux AIO system calls and adding one rule that
returns EPERM for them and for the io_uring system calls (`tools/make_seccomp_d41.py` derives it from the pinned upstream
file). It ships byte-for-byte as verified because the library pins its sha256, so the profile file itself carries neither
a licence field nor a statement of the change. The attribution, the Moby notice and the statement of the change are in
the `NOTICE` file at the top of the package (Apache-2.0 section 4); keep it with any copy of the profile. No harness code, task content, model file, chat template,
tokenizer or run output is part of this package.
