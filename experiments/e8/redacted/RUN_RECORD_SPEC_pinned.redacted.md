# E7 run record for rules v2.2 (what the HOSTA runner must write)

Written 2026-11-09, before any E7 run, for `analyze_e7.py` rules v2.2 (the default; PROTOCOL_AMENDMENTS.md A0-A9). It
replaces the v2.1 version of this file (A0-A8); every change is listed in section 6. It says what every result row, every
started record, every job file and the block manifest must carry, how re-runs, the extra batch and a void block are
handled, and what to check before the D block. It changes no HOSTA code (another track owns the runner); the field names of
the v2.1 version are kept, and A9 adds only the manifest fields of section 3.3 (`max_model_len`, `enable_lora`,
`max_lora_rank`, `served_adapter_names` in every run entry; `replaces` for a re-run block; the optional `smoke_block`).

**Record problems void the block (A9).** The analysis runs its record checks first and blind, before any outcome is
computed. A missing required field, a missing or duplicated started record, a run directory without `started.jsonl`, an
attempt gap, a job id outside its batch's job list, a `jobs.json` whose sha256 differs from the manifest, a provenance
or server-setting problem, or an unreadable record is a **record problem**: the block is **VOID**. Its rows are archived
unopened, the deviation is logged, and the block is re-run under a new block id (section 3.6). Re-runs inside the block
cannot repair a record problem, so `reruns` makes no batch for a void block. **Input problems** (the analysis was given
the wrong run directories, manifest, drop file, arm or block label) never void or fail a block: fix the inputs and run
the analysis again. **The smoke block of section 3.8 is mandatory before the first E7 job of each runner version (A9).**

## 1. One row per finished job (`out/results.jsonl`)

Required: `yes` = present and non-empty on every result row; `key` = the key is present on every row (the value may be
null or empty unless stated); `adapter_arm_A/adapter_arm_B rows` = present and non-empty on the rows of the adapter arm. This applies also to
the row the runner itself writes for a job that died without a results row (section 5.4).

| field | type | required | how to obtain it |
|---|---|---|---|
| `job_id` | string | yes | from the job entry: `<block>:<arm>:<task>:r<repeat>:a<attempt>` (make_jobs.py, `analyze_e7.job_id_v2`); must match the row's block, arm, task, repeat and attempt |
| `block` | string | yes | from the job entry: `S`, `C` or `D`, or `<S|C|D>_v<n>` for a block re-run after a void (section 3.6) |
| `arm` | string | yes (*existing*) | bundle name: `BUNDLE_A`, `BUNDLE_A_var1`, ..., or `adapter_arm_A` (adapter adapter_name_A) / `adapter_arm_B` (adapter adapter_name_B) (A7) |
| `task` | string | yes (*existing*) | instance id |
| `seed`, `repeat_id` | int | yes (*existing*) | the repeat label 1..R (`seed` names the output folder `s<seed>`) |
| `attempt` | int 1..5 | yes | from the job entry: 1 for the first run of a slot, 2-3 for re-runs, 4-5 for the extra batch (section 3.5) |
| `worker` | int | yes (*existing*) | worker index |
| `t_start_utc`, `t_end_utc` | ISO 8601 UTC with `Z` | yes | when the job was claimed / when its row was written |
| `resolved` | bool | yes (*existing*) | `TaskResult.resolved` (a non-boolean value is an infrastructure row, I10) |
| `error_class` | `agent` or `infra` | yes | `analyze_e7.classify_row_v22(row)['klass']` on the finished row (copy analyze_e7.py next to the worker, read-only) |
| `error_class_id` | table id | yes | `classify_row_v22(row)['id']` (`A00`..`A18`, `I01`..`I14`); the analysis recomputes it; a blind report shows only mismatches that involve an infrastructure id (A9) |
| `error_table_sha256` | hex | yes | `analyze_e7.ERROR_TABLE_SHA256_V22` (2ec64ba7a81041bf... for the v2.2 table); another value is noted, not decisive |
| `docker_image_id` | `sha256:...` | yes | `docker image inspect <image> --format {{.Id}}` once at runner start (run_berth.py already puts it in `summary.json`) |
| `harness_sha256` | object path -> sha256 | yes | section 2; computed once per worker at start |
| `vllm_version` | string | yes | `GET http://127.0.0.1:<port>/version` once after the server is up (or `vllm.__version__` in the serving env) |
| `served_model_name` | string | yes | the server's `--served-model-name` (block-wide: the base name also in the D block) |
| `request_model` | string | yes | the model string the worker sends: `openai/<served name>`, or `openai/<adapter name>` for adapter_arm_A/adapter_arm_B; one value per arm |
| `bundle_sha256` | hex | yes | tree hash of `bundles/<arm>` (section 2) |
| `a0_drop_sha256` | hex | yes | copied from the job entry (make_jobs.py writes it into every `.A0.json` job; section 3.1). A value that differs from the drop file given to the analysis STOPS it |
| `adapter_sha256` | hex | adapter_arm_A/adapter_arm_B rows | tree hash of the adapter directory the request used; `null` for other arms |
| `adapter_name` | string | adapter_arm_A/adapter_arm_B rows | `adapter_name_A` or `adapter_name_B` (never containing "gemma4"); one value per arm; must be one of the manifest's `served_adapter_names` (A9) |
| `error` | string | key (rows without `exc`) (*existing*) | `TaskResult.error[:200]` (`''` when none) |
| `test_exit`, `test_summary` | int, string | key (rows without `exc`) (*existing*) | pytest exit code and the last line of the test output |
| `t_agent`, `sampling_sent` | float s, object | key; non-empty on agent-outcome rows (*existing*) | agent-loop clock; sampling parameters of the job's first LLM request, as the worker records them now |
| `t_setup`, `t_verify`, `t_phase1` | float s | key (*existing*) | phase clocks of the worker (null when the phase did not run) |
| `exc`, `tb` | string | on exception (*existing*) | `repr(e)[:300]`, traceback tail; for a job that died without a results row, the runner's `g4s-d<N> infra: <why>` (section 5.4) |
| `tool_calls`, `llm_calls`, `tok_in`, `tok_out`, `patch_chars`, ... | | no (*existing*) | unchanged |
| `agent_error_phase1` | string | recommended | the `agent_error` that `run_agent_sandbox` returned, kept even when the task resolved (see 5.2) |
| `docker_infra_events` | int | recommended | this job's g4s events of kind `refused`, `popen_error`, `exec_error` or `no_status` (see 5.3) |

Block-wide fields (`docker_image_id`, `harness_sha256`, `vllm_version`, `served_model_name`) must not change during the
block, including its re-runs and the extra batch; the arm-specific ones (`bundle_sha256`, `adapter_sha256`,
`request_model`, `adapter_name`, and `sampling_sent` on the rows used) must not change within an arm (A3, A8).
Otherwise the block is VOID (A9).

## 2. Hash definitions

* **Tree hash** (bundles and adapters): `sha256("".join(f"{sha256(file)}  {path}\n" for path in sorted(paths)))` over
  every regular file below the directory (paths relative to it, `/`-separated, `__pycache__` excluded), hex digests.
* **`harness_sha256`**: an object with these keys (paths as written; values = sha256 of the file the worker actually
  imports or runs, found via the module's `__file__`):
  `lib/worker_berth.py`, `lib/sandbox_berth.py`, `lib/g4s_docker.py`, `run_berth.py`, `docker/run.py`,
  `docker/Dockerfile`, `swegemma/evaluate.py`, `swegemma/harness/agent_runner.py`, `swegemma/harness/verification.py`,
  `swegemma/harness/container_setup.py`, `swegemma/sandbox/subprocess.py`, `swegemma/sandbox/base.py`,
  `swegemma/tools/execution.py`, `swegemma/tools/workspace.py`, `swegemma-0.2.11.dist-info/RECORD`, and
  `analyze_e7.py` (the copy used for `error_class`). A runner version that uses other files (the per-job container
  runner uses `lib/g4s_d3.py` and its own `docker/init.py`) lists each file it actually uses under its path relative to
  the runner root; the analysis requires one value of the whole object per block, so any change of a file shows.
* **`a0_drop_sha256`**: sha256 of the bytes of the A0 drop file (the file `make_jobs.py --drop-file` read). The
  analysis is given the same file (`--drop-file`) and compares it with every row, every started record and every job
  entry of every batch's `jobs.json` (A8, A9).
* **`jobs_sha256`** (manifest): sha256 of the bytes of `runs/<tag>/jobs.json` as run_berth.py wrote it.
* **`void_report_sha256`** (manifest `replaces`, A9): sha256 of the bytes of the VOID report JSON of the replaced block.

Reference values of the local copies on 2026-11-09 (the HOSTA files may change before E7 while the Docker track works):
`lib/worker_berth.py` 263197d1573ae221..., `lib/sandbox_berth.py` f486c1d56791d262..., `lib/g4s_docker.py`
a0d3ed2060b043d2..., `run_berth.py` 8863c1e16be2788c... (host runner; the per-job container runner
`FOLDER_RUNNER_SRC/run_berth.py` is 06cd11fcc54ac846... with `lib/g4s_d3.py` e32de8a6b815d38f...), `docker/run.py`
f447c03178d74b11..., `swegemma/evaluate.py` 96a46f49267ffcf8..., `swegemma/harness/agent_runner.py` 4d7eccabd203c26f...,
`swegemma/harness/verification.py` 99e2a5fdab2be2b5..., `swegemma/harness/container_setup.py` 3192acdce35f2d0b...,
`swegemma/sandbox/subprocess.py` a26e9265d93f0387..., `swegemma/sandbox/base.py` aab491aabc841059.... The local
swegemma 0.2.11 matches its wheel RECORD in all 34 .py files. If the worker or swegemma change before E7, the error table
(ANALYSIS_VALIDATION.md section 3) must be re-checked; error text it does not know is I12 (infrastructure, penalised by
d_pen, listed).

## 3. Job files, started records, the block manifest, re-runs, the extra batch and a void block

### 3.1 Job files (A0, A8, A9)
After the final Docker revalidation (A0) the lead freezes the drop list (a JSON list of task ids, `{"dropped": [...]}`,
or one id per line; `[]` when nothing is dropped) and runs `make_jobs.py --drop-file <list>` once, before the first job
of any block. It checks every panel (S, C, D) first and refuses, writing no file, if any loses more than 10% of its
tasks; otherwise it writes `<job file>.A0.json` for the screening file and every block file, without the dropped tasks
and with `a0_drop_sha256` in every job. It also refuses, writing nothing, if any job of a `.A0.json` file would lack
`a0_drop_sha256`, `block`, `attempt`, `repeat_id` or `job_id` (A9). **Run only `.A0.json` files** (and the
`<out>.jobs.json` files that `analyze_e7.py reruns` writes, which carry the same fields). The block files without `.A0`
are templates (counterbalanced order, A8) and carry no drop hash.

**The runner refuses a job without `a0_drop_sha256` (A9):** before it starts any job it checks the entry, for example
with `analyze_e7.job_refusal_v22(job)` (returns the reason to refuse, or `None`; the value must be a 64-character
lower-case hex sha256), and refuses the whole job file if any entry fails. The analysis STOPS (no decision) when any job
entry of a batch's `jobs.json` (entries of the analysed block or without a block label) carries no `a0_drop_sha256`
(a template file was run) or another one than the drop file given, when a started record or row carries another one,
or when a row or started record belongs to a dropped task. The drop list may not change after a block's first job, and
a block re-run after a void uses the same drop file (section 3.6).

### 3.2 started.jsonl
**started.jsonl.** Before a job runs, the process that starts it appends exactly one line with `job_id`, `block`, `arm`, `task`, `seed`, `repeat_id`, `attempt`, `worker`, `t_start_utc`, `docker_image_id`, `harness_sha256`, `vllm_version`, `served_model_name`, `bundle_sha256`, `a0_drop_sha256`, and `adapter_sha256` for adapter_arm_A/adapter_arm_B (all required). For the per-job container runner that process is the host runner, before it starts the job's container; the worker inside the container must not add a second record for the same `job_id` (or the runner must not copy it out).

The analysis matches started records to rows by `job_id` (A8). Every started job without a row (worker killed, runner
deadline, crash) becomes an infrastructure-error row (I09) of its batch: the slot is missing, it is re-run, the row
counts in (4b) and is penalised in d_pen. A row without a started record, a run directory without `started.jsonl`, a
started record without a required field, or two started records with one `job_id` in one run directory are record
problems (A9: the block is VOID). A started record whose `a0_drop_sha256` differs from the drop file STOPS the analysis.

### 3.3 The block manifest (A8, A9)
The runner (or its launch script) keeps one manifest file per block, for example `runs/e7_<block>_manifest.json`, and
appends an entry when it starts each run of the block (the first run, every re-run batch and every pass of the extra
batch). Each entry records the vLLM server that run was served by (A9: read from the configuration the runner started
the server with, not typed by hand):

```json
{"block": "D",
 "runs": [{"run_tag": "e7_D_adapter_arm_A_a1", "jobs_sha256": "<sha256 of runs/e7_D_adapter_arm_A_a1/jobs.json>", "deadline_s": 50400,
           "max_model_len": 32768, "enable_lora": true, "max_lora_rank": [rank], "served_adapter_names": ["adapter_name_A"],
           "job_file": "jobs_D_block_adapter_arm_A.A0.json", "n_jobs": [count], "t_start_utc": "2026-11-12T01:00:00Z"},
          {"run_tag": "e7_D_adapter_arm_A_r1", "jobs_sha256": "...", "deadline_s": 7200,
           "max_model_len": 32768, "enable_lora": true, "max_lora_rank": [rank], "served_adapter_names": ["adapter_name_A"],
           "job_file": "reports/D_next_1.jobs.json"}],
 "smoke_block": {"block": "SMOKE_d4_1", "records_report_sha256": "<sha256 of its records report JSON>"}}
```

| field | where | required | meaning |
|---|---|---|---|
| `block` | top level | yes | the block id (`S`, `C`, `D`, or `<panel>_v<n>` after a void); must equal the analysed block |
| `runs` | top level | yes | one entry per run of the block, in start order |
| `replaces` | top level | re-run blocks | only in the manifest of a block `<panel>_v<n>` (section 3.6): `{"block": <id of the voided block>, "void_report_sha256": <sha256 of its VOID report>}`; in any other manifest it is an input problem |
| `smoke_block` | top level | no (noted when absent) | the smoke block run for this runner version (section 3.8): `{"block": ..., "records_report_sha256": ...}` |
| `run_tag` | run entry | yes | the run directory's basename (`runs/<tag>`) |
| `jobs_sha256` | run entry | yes | sha256 of `runs/<tag>/jobs.json` (section 2) |
| `deadline_s` | run entry | yes | the run's `--deadline-s` (section 3.4) |
| `max_model_len` | run entry | yes | the server's `--max-model-len` (a positive integer; A9) |
| `enable_lora` | run entry | yes | `true` when the server was started with `--enable-lora`, else `false` (A9) |
| `max_lora_rank` | run entry | yes | the server's `--max-lora-rank` (a positive integer) when LoRA is enabled; `null` (or the integer) without LoRA (A9) |
| `served_adapter_names` | run entry | yes | the adapter names the server serves (the names of `--lora-modules`); `[]` without LoRA (A9) |
| `job_file`, `n_jobs`, `t_start_utc` | run entry | no | notes |

Rules (A3, A8, A9): the four server settings must take one value across all runs of the block. When the analysed block
has an adapter arm (adapter_arm_A/adapter_arm_B), LoRA must be enabled in every run and the arm's `adapter_name` must be among the served
names, **also in a re-run batch or extra-batch pass that holds only BUNDLE_A jobs**: the per-job container runner enables
LoRA only when an arm of the batch's own jobs has adapters (`FOLDER_RUNNER_SRC/run_berth.py:439-458`), so a base-only re-run
batch would start a server without LoRA and void the D block. A missing or malformed setting, a value that differs
between runs, or an adapter arm without its adapter is a record problem (VOID). A `max_model_len` other than 32768 (the
`adk_submission` VllmConfig default, `server.py:152`) is noted: A18 counts context-length errors as agent outcomes "as on
Kaggle".

The analysis gets the manifest (`--manifest`) and every run directory of the block (`--runs`, any order); the run
directory's basename is its run tag (`runs/<tag>`), and its `jobs.json` (written by run_berth.py) is the batch's job
list: (4b) tests each batch against the candidate's share of its jobs, and every row's `job_id` must be in its batch's
job list. A listed tag whose directory is not given, a given directory the manifest does not list, a manifest of another
block, or a missing manifest is an input problem: the gate is INCONCLUSIVE until the analysis is run with the right
inputs. A manifest entry without a required field, or a `jobs.json` whose sha256 differs from the entry, is a record
problem (VOID).

### 3.4 Deadlines (A8)
Set `--deadline-s` above the run's expected duration: at about 200 runs/h (E6 Docker measurement, 8 workers) the C block
(576 jobs) takes about 3 h and the screen (576 jobs) about 3 h: use at least 6 h; the D block [redacted: 5 words, adapter detail]
**at least 12 h (43,200 s)**; a re-run batch or extra-batch pass at least 1 h or twice its expected duration. The
analysis notes a deadline below the expected duration, and a first D-block run below 12 h; a deadline that cuts a run
only creates I09 rows and missing slots, which the re-run procedure handles.

### 3.5 Re-run procedure and the extra batch (A1, A2, A8, A9)
1. Run the block's `.A0.json` job file to the end under a new run tag (every job at attempt 1); add the tag to the
   manifest.
2. Make the next batch with the analysis, never by hand (blind: it shows slots only):
   `analyze_e7.py reruns --runs <every run dir of the block> --panel panel_<C|D>.json --arms BUNDLE_A <arm>
   --drop-file <A0 list> --manifest <block manifest> --out reports/<block>_next_<k>`. It writes
   `<out>.jobs.json`, in block job order, with the same `arm`, `task`, `seed`/`repeat_id`, `block`, the next
   `attempt`, a new `job_id`, `a0_drop_sha256` and `batch_kind`:
   * **re-run batch** (`batch_kind` `rerun`): while any slot without a valid row has fewer than 3 attempts, the batch
     holds exactly those slots (attempts 2 and 3; A1). Slots that already used 3 attempts wait.
   * **extra batch** (`batch_kind` `extra`, A8): when no such slot is left, every slot of both arms still without a
     valid row gets its next attempt, 4 and then 5: the batch runs as at most two passes (attempt 4; then attempt 5 for
     the slots whose attempt 4 was an infrastructure error), each made by `reruns` without looking at any outcome.
   `reruns` refuses (exit 2) while an analysis input is incomplete, and makes **no batch for a VOID block** (decision
   `VOID`, empty job list; section 3.6). It lists slots with five failed attempts as exhausted.
3. Run that job file under a new run tag (`run_berth.py` refuses an existing tag, so no file of an earlier attempt is
   overwritten), on the same server configuration (section 3.3), image, harness files, bundles and adapter; add the tag
   to the manifest.
4. Repeat 2-3 until `reruns` writes no job, then open the gate (`analyze_e7.py gate --runs <every run dir of the
   block> --panel ... --arm-b <arm> --drop-file ... --manifest ...`). The gate may be run at any time: an INCONCLUSIVE,
   VOID or STOPPED gate report is blind (slots, provenance, error counts, record and load problems only).
5. The gate decides PASS or FAIL by its criteria once no reason is left. An input problem keeps it INCONCLUSIVE (fix
   the inputs). A record problem makes it **VOID** (section 3.6), never a FAIL (A9). **A gate whose only remaining
   reasons are slots without a valid row after attempt 5 is a FAIL: BUNDLE_A stays (A8).**
6. Never delete or edit a row, never re-run a slot that has a valid row, never run an out-of-plan repeat or attempt.
   Two rows that are not infrastructure errors in one slot, two rows with one attempt index, or a non-positive attempt
   STOP the analysis (A2, A8): the lead resolves them from `job_id` and logs the deviation.
7. If infrastructure errors look systematic (one task in both arms, one worker, a vLLM restart), stop the block and find
   the cause first (PROTOCOL stop rules) instead of spending attempts.
8. The screening (block `S`, `jobs_S_screen.A0.json`) follows the same procedure before `analyze_e7.py screen`
   (default: the six frozen arms; `--manifest` recommended): a screen with an input problem is INCONCLUSIVE and one
   with a record problem is VOID, both blind (no gain, ranking or finalist; A9); while a slot lacks a valid row or an
   arm has no rows, its finalist is PROVISIONAL.

### 3.6 A void block (A9)
A report with decision `VOID` (from `gate`, `screen`, `records` or `reruns`) means the block has record problems and no
outcome was computed. Then:
1. Keep that report unchanged (its JSON is the void record; note its sha256). Do not run `secondary` or any other
   outcome-reading tool on the block.
2. Archive the block's run directories unopened (move `runs/<tag>` of every run of the block, with the manifest, to an
   archive location; read no `results.jsonl`).
3. Log the deviation (block, run tags, the record problems listed in the report, the report's sha256).
4. Fix the runner. A changed runner is a new runner version: run the smoke block (section 3.8) again first.
5. Make the new block's job files: `make_jobs.py --drop-file <the same A0 list> --new-block-id <panel>_v<n>` (n = 2 for
   the first re-run, then 3, ...). It writes only that panel's `.A0.json` files with the new block id, e.g.
   `jobs_D_block_adapter_arm_A.D_v2.A0.json` (block `D_v2`, job ids `D_v2:<arm>:<task>:r<k>:a1`).
6. Run the new block from attempt 1 with a new manifest (`"block": "D_v2"`) that carries
   `"replaces": {"block": "D", "void_report_sha256": "<sha256 of the VOID report>"}`.
7. Analyse it with `--block D_v2 --void-report <the VOID report>` in addition to the usual arguments. The analysis checks
   the chain (block id `<panel>_v<n>`, `replaces` naming an earlier block of the panel, the report's sha256, decision
   VOID, block, panel and arms); any mismatch is an input problem. It STOPS if the voided block was analysed with
   another drop file. Rows of the voided block are ignored as another block's rows; never give its run directories to
   the new block's analysis.

### 3.7 Before the D block (A8, A9)
* 








[redacted: 164 words, adapter detail]

### 3.8 Smoke block (mandatory, A9)
Before the first E7 job of each runner version (any change of a `harness_sha256` value, the Docker image or the
runner's settings), run a smoke block through the runner and the analysis:
* a panel file of 2 tasks of the real block's panel with its own name (e.g. `{"panel": "SMOKE_d4_1", "tasks": [...]}`),
  a job file with both arms of the real block, 1 repeat, block id = that name, job ids `<name>:<arm>:<task>:r1:a1` and
  the sha256 of its own drop file (`[]`), its own manifest with every field of section 3.3, run on the server
  configuration of the real block (LoRA enabled with the adapter for a D block);
* recommended: one more job of the same arms killed on purpose by a short `--job-deadline-s`, to exercise the runner's
  row for a job that died without a results row (it must be I09 and carry every required field and its started record);
* then `analyze_e7.py records --runs <its run dir> --panel <its panel file> --arms BUNDLE_A <arm> --repeats 1
  --drop-file <its drop file> --manifest <its manifest> --out reports/smoke_<name>`. It must say
  **RECORDS COMPLETE** (exit 0; exit 3 otherwise) with no input or record problem (NON-FROZEN repeat count and open slots
  for killed jobs are expected). Keep the report; its sha256 goes into `smoke_block` of the E7 block's manifest.

## 4. What the analysis reads, and what makes it stop, void or wait

* STOP (exit 2, no decision, blind report): a row or started record of the block belonging to a dropped task; an
  `a0_drop_sha256` in a row or started record that differs from the drop file given; a job entry of a batch's
  `jobs.json` without `a0_drop_sha256` or with another one (A9); more than 10% of any panel (S, C or D) dropped; a
  re-run block whose voided block was analysed with another drop file (A9); two non-infrastructure rows in one slot;
  two rows with one attempt index; a non-positive or non-integer attempt.
* INCONCLUSIVE, input problem (blind; fix the inputs): no or wrong manifest, a listed run directory not given or a given
  one not listed, an analysed arm or block label without jobs, no drop file, a re-run block without a valid
  `replaces`/`--void-report` chain (A9).
* VOID, record problem (blind; section 3.6): a missing required field or started record, a duplicated started record,
  a run directory without `started.jsonl`, an attempt gap, a job id not in its batch's job list, a `jobs.json` that
  is missing, unreadable or differs from the manifest, a malformed manifest entry, provenance or server settings missing
  or disagreeing (A3, A9), an unreadable record.
* INCONCLUSIVE, slots (blind): a slot without a valid row that can still get an attempt (re-run batch or extra batch);
  FAIL when no slot can (A8).
* Ignored and listed: rows of another block, another arm, a task outside the panel, or a repeat outside 1..R.

## 5. Notes for the runner (from the harness source)

1. **Error strings.** The table of every row error the harness can write, with its class and file:line, is
   `analyze_e7.py error-table` (ANALYSIS_VALIDATION.md section 3). A8 changed four classes ('Failed to apply
   test_patch' A17 and context-length HTTP 400s A18 are agent outcomes; 'Failed to apply agent patch' with a Docker/g4s
   error I13 and 'Agent patch file missing' I14 are infrastructure). A9: I13 counts only when the patch detail *starts*
   with a Docker/g4s form (`docker: `, `g4s: `, `g4s run.py: `, `[Errno N] `, `Error response from daemon`,
   `Cannot connect to the Docker daemon`); patch-tool output that merely contains such text is A07.
2. **Infrastructure errors that resolve.** When an LLM request fails mid-run (`agent_runner.py:1039-1043`), the
   harness still evaluates any edits the agent left (`agent_runner.py:1054-1078`); if they pass, the row says
   resolved with no error (`verification.py:697`). Such a run counts as a solve. `agent_error_phase1` makes these rows
   visible; the analysis does not use it.
3. **Silent verification failures.** If docker cannot start the verification container after its three tries
   (`g4s_docker.py:321-334`) while running the tests, pytest never runs and the row is an ordinary unsolved run (exit
   code 125 or -1, empty error). The analysis lists used unsolved rows whose pytest exit code is not 0-5 or 124 as a
   diagnostic; `docker_infra_events` lets the lead see the cause.
4. **Jobs that die without a results row in the per-job container runner (A9).** The runner writes the row itself
   (`FOLDER_RUNNER_SRC/run_berth.py:290-294`). Its `exc` must start with `g4s-d<N> infra: ` followed by the reason (`job
   deadline`, `no results row` or a frame error; the D3 runner writes `g4s-d3 infra: ...`, a D4 runner
   `g4s-d4 infra: ...`): the analysis classifies it as I09 (re-run, counted in (4b), penalised by d_pen), as a started
   job without a row on the host runner. That row must carry every required field of section 1 (copied from the job
   entry and the runner's provenance; `error`, `test_exit`, `test_summary` may be absent because `exc` is set, the
   phase-clock keys and `sampling_sent` present with null). The D3 runner's row today carries only `arm`, `task`,
   `seed`, `repeat_id`, `worker`, `resolved`, `exc`, `tb`, `setup_fallback`, `wall` and `d3`, which would void the
   block (VERIFIED by `test_runner_row_exactly_as_d3_writes_it_voids_the_block`).
5. **Started records in the per-job container runner.** Today the worker inside the container writes the started
   record (`worker_berth.py:188-189`, fields `arm`, `task`, `repeat_id`, `worker`, `t`) and the runner copies it out
   after the job (`run_berth.py:299-302`): a container killed before the copy leaves no record, and the record lacks
   most required fields. Write it on the host before starting the container (section 3.2), once per `job_id`.

## 6. Changes from the v2.1 version of this file (A9)

1. Rules v2.2 replace v2.1 as the reference: `error_class` and `error_class_id` from `classify_row_v22`,
   `error_table_sha256` = `ERROR_TABLE_SHA256_V22` (2ec64ba7...). Field names unchanged.
2. Record problems void the block (no longer a FAIL); new section 3.6 (void procedure, `make_jobs.py --new-block-id`,
   block ids `<panel>_v<n>`, `--void-report`); section 4 split into STOP / input / VOID / slots; 3.5 steps 2, 4, 5 and 8
   updated; the `block` field allows `<panel>_v<n>`.
3. Manifest run entries: new required fields `max_model_len`, `enable_lora`, `max_lora_rank`, `served_adapter_names`
   (section 3.3 table and rules), including the warning about base-only re-run batches of the D block.
4. Manifest top level: new `replaces` (required only in the manifest of a re-run block) and optional `smoke_block`.
5. Job files: make_jobs.py refuses to write a runnable job without `a0_drop_sha256` (and the other identity fields);
   the runner must refuse a job without `a0_drop_sha256` (`analyze_e7.job_refusal_v22`); the analysis STOPS on a job
   entry of a batch's `jobs.json` without the drop hash or with another one, and on a started record with another one.
6. Started records: exactly one per `job_id` and run directory, written by the process that starts the job (host side
   for the per-job container runner); a duplicate is a record problem (v2.1: it became a synthetic row and a STOP).
7. Rows the runner writes for a job that died without a results row: `exc` starts `g4s-d<N> infra: ` and the row
   carries every required field; class I09 (section 5.4).
8. The smoke block is mandatory for each runner version and is checked with the new `analyze_e7.py records`
   subcommand (section 3.8, which replaces 3.7 of v2.1; the checks before the D block move to 3.7).
9. [redacted: 10 words, adapter detail]
10. `harness_sha256`: a runner version that uses other files lists them under their own paths (section 2); new
    reference hashes of the per-job container runner.
11. Notes 5.1 (anchored I13), 5.4 and 5.5 are new.
