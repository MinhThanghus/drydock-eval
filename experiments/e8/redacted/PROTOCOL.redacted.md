# E8: how much each scorer-path divergence changes Gemma 4 E4B agent outcomes (pre-registration)

**Status: FROZEN 2026-11-10, about 03:40 UTC, by the lead (14 item 4; LEAD item 26). The lead froze this file** 



































[redacted: 603 words, internal process history]
**No E8 run exists.** Nothing for this file ran on the GPU host, on a GPU or in Docker; all work was local CPU under `[cpu-priority]`.

**Labels.** **V** (VERIFIED): a file:line, a command output or a fetched page is named next to the claim. **I**
(INFERRED): reasoning, arithmetic or a simulation assumption. Paths: `R/` is `[repo-root]/`; `E8/` is
`R/experiments/FOLDER_E8/`.

The id "E8" names this experiment; `R/FOLDER_PRIZE_NOTES/landscape.md` uses "E8" for an unrelated catalogue entry.

## 0. The proposal in one screen

**Question.** A local evaluator can diverge from the scorer's serving path. For four such divergences, how much does
each change E4B's solve rate on Berth tasks, with everything else held at the scorer path?

**Arms** (LEAD_DECISIONS.md item 2). F is the scorer path. Each trap arm changes one factor against F:
**R** tool-result role `tool_responses` (request model string containing `gemma4`), **C** no compaction, **H** history
thoughts absent (the prompt of a client without adk_submission's reasoning mirror), **T** the Hugging Face (HF)
canonical chat template. google-adk 1.36.1, the BUNDLE_A bundle and the scorer sampling in every arm.

**Estimand (the same statement in sections 1, 8.1 and 12).**
> For each trap arm X in {R, C, H, T}: d_X = (1/N) Σ_i [ȳ(X, i) − ȳ(F, i)], the task-weighted solve-rate difference
> between X and F on the N tasks of the analysis set that have all four valid slots in both arms, where ȳ(a, i) is the
> mean of `solved` over arm a's four valid runs of task i. The tasks are fixed; the runs are the random part. The
> primary test is the exact two-sided task-level sign-flip test of the per-task differences, Holm-adjusted across the
> four trap arms at 0.05. Claims are worded "on these N tasks". The repository-cluster 95% interval is a reported
> secondary for "repositories like these", not a claim rule.

**Panel and stages** (item 3). Nested stages, all arms interleaved by task with the A8 counterbalanced order, 4
repeats per arm: stage 1 = panel B (118 tasks), stage 2 = A∖B (37), stage 3 = C∖A (76). The analysis set is every
task whose stage, and every earlier stage, was fully dispatched before **2026-11-11 06:00 UTC**. The only stop rules
are that cut-off, GPU availability and the A8 infrastructure rules.

**Power** (item 7; section 9). For an even 5-point change on the analysis set, the probability that Holm declares
the affected arm (the other three null) is 61-98% with stages 1-2 (155 tasks), 44-90% with stage 1 only (118) and
82-100% with all three (231), over repeat correlation 0.5-0.85, base rate 0.15-0.25 and two models of task
difficulty (V, simulation, 2,000 panels per cell). The 80%-power change under Holm is about 3.4 to 6.4 points
(155 tasks; drops and gains, section 9).

**GPU time** (section 10). 20 runs per task: stage 1 = 2,360 runs (7.4 h at D3's measured 318.6 runs/h; 9.8 h at a
conservative 240), stages 1-2 = 3,100 (9.7-12.9 h), all stages = 4,620 (14.5-19.3 h).

**Order before the main block (LEAD items 16 and 18).** (1) The shared runner and image (item 10) pass their canaries;
CPU gold/empty controls on the panel and the six smoke candidates (A0). (2) The E8 worker (Track D4.2, built on the GPU
host after D4.1) passes items 1-9 of the verifier checklist of 6.2, and the manipulation checker `manip_check_e8.py`
passes its independent verification (6.4). (3) The freeze (14 item 4): the checker's sha256 is filled in as
`FROZEN_CHECKER_SHA256`, `analyze_e8.py`'s sha256 is printed in section 15, and this file, `make_jobs_e8.py`,
`analyze_e8.py`, `manip_check_e8.py`, `analyze_e7_pinned.py`, `RUN_RECORD_SPEC_pinned.md` and the worker are frozen.
(4) After the freeze and before the first main-block job, checklist items 10-12 pass: the smoke block, the token-level
capture block and the A18 positive control (section 7). If any of them fails, the recovery rule applies (fix, rebuild,
new block id; 11.4).

## 1. Question, unit and estimand

**Question.** Each arm reproduces something a local evaluator can get wrong relative to the scorer path (Kaggle model
string, Kaggle-v2 template, scorer compaction, the reasoning mirror, scorer sampling, vLLM 0.19.1). How much does each
change Gemma 4 E4B's agent outcomes on Berth tasks with the BUNDLE_A bundle? E8 measures **divergence from the scorer
path**, in both directions; it does not assume that a divergence costs solves (section 2).

**Unit.** A task. `solved` = an agent-outcome row with `resolved` true (analyze_e7 rules v2.2; a context-length run
whose working-tree changes pass the tests is solved, section 3.1 C).

**Estimand.**
> For each trap arm X in {R, C, H, T}: d_X = (1/N) Σ_i [ȳ(X, i) − ȳ(F, i)], the task-weighted solve-rate difference
> between X and F on the N tasks of the analysis set that have all four valid slots in both arms, where ȳ(a, i) is the
> mean of `solved` over arm a's four valid runs of task i. The tasks are fixed; the runs are the random part. The
> primary test is the exact two-sided task-level sign-flip test of the per-task differences, Holm-adjusted across the
> four trap arms at 0.05. Claims are worded "on these N tasks". The repository-cluster 95% interval is a reported
> secondary for "repositories like these", not a claim rule.

**What the test tests (I).** Under the null that X and F runs of the same task are exchangeable (X changes no task's
outcome distribution), each per-task difference is symmetric about 0, so the sign-flip distribution is exact. A
rejection says X changes outcomes on these tasks; the sign of d_X gives the direction. In simulation, rejections with
the wrong sign were at most 0.05% of the panels of any cell (600 cells with an effect, 2,000 panels each; V, section 9).

**Secondary estimands** (exploratory, section 8.5): paired X − F differences in thought rate, failed edit calls,
tool-call parse failures, prompt tokens, compactions per run, agent time, context-length rows and other descriptors.

**Scope (I).** Effects of the evaluator configuration on this harness, bundle, the GPU host's single-GPU serving and these
Berth tasks. They are not hosted-score effects (section 13).

## 2. Background: what is public and what E8 adds

The divergences are public upstream; E8 cites them and claims no discovery.

**Role.** google-adk 1.36.1 switches the tool-result role to `tool_responses` whenever the request model string
contains `gemma4`, and says why in a source comment (`R/[harness-src]/google_adk/google/adk/models/lite_llm.py:810-814`,
V). Issue #7977 and PR #7978 (merged 2026-11-05) make the role depend on the provider and keep `tool_responses` for
Ollama and hosted_vllm (V, fetched 2026-11-09). **Direction is open:** issue #5650 reports that role `tool` makes small
Gemma models "misinterpret the tool result as a new turn", and #6334 asks for `tool_responses` on any backend (V as
quoted in `FOLDER_KIT_PAPER/AUDIT.md` item 3). R could raise or lower solves; the tests are two-sided.

**Thoughts.** Google's Gemma 4 prompt-formatting guide and HF google/gemma-4-E2B-it PR #44 state that thoughts are kept
only after the last user message (V as quoted in `R/FOLDER_PRIZE_NOTES/critic.md:159-165`). A forum thread and two public
notebooks report that an earlier scorer build dropped history thoughts through a `reasoning` / `reasoning_content`
mismatch (third-party, untrusted; text V in `R/FOLDER_NOTEBOOKS/…`).

**Template revisions.** google/gemma-4-E4B-it commit 6ea8ce9d and google/gemma-4-E4B-it-qat-w4a16-ct commit eb46a52a
(2026-07-15) changed the HF templates; the Kaggle-v2 template (sha256 33204f1a…) is the QAT repository's release
template from before that date (V, draft section 3.1).

**Related public measurements to cite** (not read in detail, I): ThoughtKeep (31B), "Sorted Keys, Broken Edits",
gemma4-swe-kit (including its history-rendering measurement, 6/13 → 3/35 coder-loop tasks, per AUDIT.md), the
serving checks that name the reasoning-field mismatch.

**What E8 adds.** (a) A baseline that reproduces the scorer's prompts token for token: an independent rendering
reproduces vLLM's prompt token counts on J6 (396/396) and D3 (360/360) and the ids on J5's captured request (V,
`R/FOLDER_WRITEUP_LOG/EXPERIMENTS.md` J5, J6; `R/experiments/FOLDER_RUNNER_EVAL/README_docker.md` section 8). (b) Measured E4B solve-rate
differences for four single-factor divergences, pre-registered, with a per-row check that each row received exactly
its arm's prompt. (c) Exposure: how often each divergence touches a real E4B request. The J2 erratum states that the
role switch's effect "is unknown" (V, `EXPERIMENTS.md:1663` ff.); E8 updates it.

**Coverage (AUDIT.md item 4).** E8 isolates three divergences (role, compaction, history thoughts) and swaps one
template as a whole (T). It does not measure "each documented trap": the history-thought newline and the
keep-all-thoughts variant get no arm (section 3.4).

## 3. Arms

### 3.1 Arm table

Shared by every arm: one vLLM server process (section 4); the BUNDLE_A bundle; google-adk 1.36.1 (named in every row);
temperature 0.2, top_p 0.95, max_completion_tokens 8,192, `enable_thinking` true, thinking budget 4,096, vLLM
generation_config top_k 64 (V, `R/experiments/FOLDER_RUNNER_EVAL/b2_report.md` B(2)); the 28-call budget, the 5-minute agent cap
and the 120 s command timeout. The frozen arm file is `E8/e8_arms.json` (sha256 f0ca2eb8…), written by
`make_jobs_e8.py`.

**F (scorer path).** Request model `openai//kaggle/input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2`, so
ADK sends role `tool`; the Kaggle-v2 template from the model directory (sha256 33204f1a…, 16,804 bytes);
`EventsCompactionConfig(compaction_interval 15, overlap_size 2, token_threshold 14336, event_retention_size 5)`;
history thoughts mirrored into `reasoning` and `reasoning_content` by adk_submission's patch. The GPU-host replica
reproduces the hosted scorer's prompts token for token (J5, J6, D3).

**R (tool-result role).**
- *Differs from F:* the request model string is `openai/google/gemma4-e4b-it` (LEAD item 2). It contains `gemma4`, so
  ADK 1.36.1 sends tool results with role `tool_responses` (`lite_llm.py:814, 961`; the role follows the request model
  string, `:2230`, V). The provider stays `openai`.
- *How:* one server serves two names, the Kaggle path first and `google/gemma4-e4b-it` second
  (`--served-model-name` takes a list, `engine/arg_utils.py:214`; the server answers to every name, V via the review).
  Only R jobs use the alias.
- *Rendering (V, review finding 9; lines re-read for this file):* vLLM copies `tool_call_id` only for role `tool`
  (`entrypoints/chat_utils.py:1542-1545`). The Kaggle-v2 template treats a `tool_responses` message as an ordinary
  turn: the assistant turn ends with a bare `<|tool_response>` marker (template lines 339-340) and the generation prompt
  opens a model turn (346-350). No `tool_responses` rendering matches any of D3's 320 multi-turn requests (V,
  README_docker.md section 8). The 7.2 renderer implements this.
- *Who gets it:* under ADK 1.36.1, any served name or local path containing `gemma4`. The kit's ADK role table is
  reported with E8: 1.34-1.39 substring rule; 2.5-2.11 a regex for any provider; 2.12 and later provider exemptions
  (LEAD item 2[redacted: 7 words, internal process history]).
- *Exposure (V):* every request after the first tool call; 335/381 requests end with a tool result in J5, 314/360 in
  D3.

**C (no compaction).**
- *Differs from F:* `events_compaction_config` unset, swegemma's default (`swegemma/config.py:118`;
  `harness/agent_runner.py:694-695` passes the config only when set; the official `swegemma/cli.py:245-265` builds the
  EvalConfig without it; V via the review).
- *Expected consequence:* prompts grow until vLLM rejects a prompt above 32,768 − 8,192 = 24,576 tokens with an HTTP
  400 (`renderers/params.py:405-425`; `max_completion_tokens` is the output allowance,
  `chat_completion/protocol.py:382`; V via the review). litellm 1.82.4 maps the message to
  `ContextWindowExceededError`, which matches A18 (V via the review). **A18 is an agent outcome** (LEAD item 4); the
  harness still grades the unsubmitted working-tree changes after the error (`agent_runner.py:1044-1081`), so such a
  run is unsolved unless its edits pass.
- *Exposure (V):* 11/24 runs compacted at least once in both J5 and D3; on the same trajectories 2/24 (J5) and 3/24 (D3)
  runs would pass 24,576 tokens without compaction (I, review estimate).

**H (history thoughts absent).** Corrected mechanism (review finding 2; lines re-read for this file, V):
- vLLM 0.19.1 hands the template only `message["reasoning"]` (`entrypoints/chat_utils.py:1512, 1537-1541`).
- google-adk 1.36.1 sets only `reasoning_content`. The scorer's harness adk_submission 0.2.13 copies `reasoning_content`
  into `reasoning` (`adk_submission/resolvers/generation.py:65-77`), through patched LiteLLM conversion functions,
  installed when the module is imported (`:356`). This "reasoning mirror" is what keeps history thoughts on the scorer
  path. LiteLLM does not add `reasoning`.
- *H is the prompt a client lacking the mirror produces* (google-adk plus LiteLLM without adk_submission, or the
  earlier scorer build). It is implemented by stripping both `reasoning` and `reasoning_content` from every assistant
  message at `litellm.acompletion`, in the outermost wrapper (after the patch), for every call including the compaction
  summarizer's. vLLM reads only `reasoning`, so history thoughts vanish server-side exactly as for a mirror-less client.
  It is described as such, not as a LiteLLM behaviour.
- *What H keeps (V):* thoughts stay in the ADK session and reach the compaction summarizer as plain text, because
  ADK's summarizer formats every text part, thought parts included (`google/adk/apps/llm_event_summarizer.py:73-81`); a
  mirror-less client does the same. `include_thoughts` false (`generation.py:80-90`) is a different, stronger
  manipulation that E8 does not test.
- *Exposure (V):* nearly every history assistant message carries a thought: J5 2,081 of 2,094 occurrences, D3 1,835 of
  1,835. The exposure measure counts the messages whose thought F would render (after the last user message, with tool
  calls).

**T (HF canonical template).**
- *Differs from F:* the template is HF's current canonical E4B template (blob fbe3b59b…, sha256 0a2c8073…, 18,569
  bytes; main of google/gemma-4-E4B-it and of the QAT repository, V via the draft).
- *How:* the worker sends the template in the request's `extra_body.chat_template`, merged into the existing
  `extra_body` (which holds `chat_template_kwargs` and `thinking_token_budget`), loaded from a path given at runner
  start and checked against sha256 0a2c8073…. The server runs with `--trust-request-chat-template` (E8 block only;
  without it vLLM refuses the request, `cli_args.py:92`). **Rows and dumps record only the template's sha256**: the
  worker's dump and sampling recorders replace the text by `sha256:<hex>`.
- *What it changes per request (V, draft section 3.2, J5 381 and D3 360 requests rendered):* a forced thought opener in
  the generation prompt after a tool result (J5 335/381, D3 314/360); turn closing after assistant text (J5 16/381, D3
  0/360). It does not change which history thoughts are kept or the thought newline in this traffic. The HF template's
  only `raise_exception` (string tool arguments) cannot fire through vLLM, which decodes arguments first (V via the
  review). 370/381 (J5) and 348/360 (D3) responses already carry a thought, so T's expected effect is small (I).
- *Residual risk (I):* with `--trust-request-chat-template`, code the model runs inside a job container could send its
  own template through the relay. vLLM renders templates in a jinja2 sandbox (`renderers/hf.py:15`, V via the review),
  so this allows denial of service, not code execution. The lead accepted T on these terms (LEAD item 2).

### 3.2 One factor per arm (I, from the code paths)

| Arm | Model string | Tool role | Compaction | History thoughts reach the template | Template |
|---|---|---|---|---|---|
| F | Kaggle path | `tool` | scorer config | yes (mirror) | Kaggle-v2 (server) |
| R | `google/gemma4-e4b-it` | `tool_responses` | scorer config | yes | Kaggle-v2 |
| C | Kaggle path | `tool` | none | yes | Kaggle-v2 |
| H | Kaggle path | `tool` | scorer config | no (both keys stripped) | Kaggle-v2 |
| T | Kaggle path | `tool` | scorer config | yes | HF canonical (per request) |

The per-row manipulation check (6.4) verifies the factor and that nothing else differs.

### 3.3 Exposure is measured, not assumed

For every row the checker counts the returned requests whose prompt F's rule would change (`n_requests_changed`,
section 6.4): R and T, each request re-rendered under F's rule (role `tool`, Kaggle-v2) and compared with its rendering
under the arm's rule; H, the requests holding an assistant message with tool calls after the last user message, whose
thought F would render (I: it assumes that message carried a thought, as 2,081 of 2,094 did in J5); C, the agent
requests after the first point where F's compaction rule would have fired; F, 0. The report gives the share per arm
after unblinding.

### 3.4 Considered and not run

A newline arm (identical under both templates with vLLM 0.19.1), keep-all thoughts (`preserve_thinking`, no public
Gemma 4 agent notebook uses it; it would change 15/381 and 25/360 requests), `include_thoughts` false, and combined
divergences (interactions are out of scope) (V/I, draft section 3).

## 4. Fixed elements: runner, server, model and bundle

**Runner** (LEAD item 10). 
[redacted: 29 words, other workloads] The image contains the E8 worker
(`worker_berth_e8.py`: the e6 worker plus the arm hook of 6.2; for jobs of other blocks of the same project the hook is inert), built on the GPU
host as Track D4.2 after D4.1 and verified against items 1-9 of the 6.2 checklist before the freeze and items 10-12
after it, before the first main-block job (LEAD items 16 and 18), and the runner's
guarded alias exception (the alias only as the second served name, only with the frozen arm file, only for R jobs).
Canaries, `canary_passed.json` and the D3 isolation design apply unchanged (README_docker.md sections 0-6). D3 runs
with `emulate_l4` 0 (`run_berth.py:239`, V via the review). E8 blocks run on the Track D4.3 runner (`run_berth.py`
9c05d6db…, `lib/g4s_d4.py` bb5d0377…; LEAD item 26), which changed only the forwarder's body scanner and the
canary-mode gate; image v8 fa686273…, container flags 307a018e…, options, vLLM args and model are D4.2's, and its
row-writing code (`tamper_row`, `died_row`, `_ingest`, `fallback_row`, `record_base`, `job_cfg`, `normalise`,
`run_job`) behaves exactly as D4.2's (V, `R/experiments/FOLDER_RUNNER_EVAL/verifier_report_d43.json`: static closure and the
d43h/d43c1 rows). Every agent-mode GPU run uses the default gate (`canary_passed.json` binding all 8 keys), never
`--gate precanary`.

**Server.** The project's server command through adk_submission's VllmServer (TP1, GPU utilisation 0.55, `max_model_len` 32,768, gemma4
tool and reasoning parsers, auto tool choice) plus `--served-model-name <Kaggle path> google/gemma4-e4b-it` and, for
the E8 block, `--trust-request-chat-template`. Neither changes F requests (I); the capture (7.2) verifies this at token
level. 
[redacted: 22 words, adapter detail]

**Model.** Kaggle's v2 weights (`model.safetensors` sha256 19d89d2a…); the runner records the sha256 of every file in
the model directory at start.

**Bundle** (LEAD item 9). BUNDLE_A (zip sha256 9a7da767…; tree hash as `RUN_RECORD_SPEC_pinned.md` section 2) in every arm. The arms
differ only in the arm file, never in the bundle.

## 5. Panel and stages

### 5.1 Draw rule (no outcome is read)

`make_jobs_e8.py --draw` implements it and reproduces the frozen `panel_E8.json` byte for byte (V, re-run for this
file: sha256 f03c0b93…).
1. **Pool:** the `validated` list of `R/experiments/FOLDER_E6/pool_validated.json` (579 tasks; sha256 ef6b3325…).
2. **Exclusions** (V): in E7 panel S or D, 248; reserved repositories tox-dev/platformdirs and encode/uvicorn, 76;
   gold verification over 300 s, 2; whitespace-normalised problem statement equal to an earlier kept or S/D task, 22.
   Result: **231 eligible tasks in 15 repositories.**
3. **Order:** ascending sha256(`"g4s-e8-panel-v1:" + instance_id`).
4. **Cumulative panels:** walk the order, take a task while its repository has fewer than `cap` tasks: B cap 16 (118
   tasks), A cap 24 (155), C no cap (231). B ⊂ A ⊂ C. Hashes are sha256 of `json.dumps(<ordered id list>)` with default
   separators: B f7f9b9d4…, A 8fefb3b0…, C 2c1887c5….

No E4B outcome exists for any eligible task (V, draft section 5.1). Berth is a separate Apache-2.0 dataset
(`R/[task-data]/README.md`), not competition data.

### 5.2 Counts per repository (V, `panel_E8.json`)

| Repository | B (stage 1) | A (stages 1-2) | C (stages 1-3) |
|---|---|---|---|
| pypa/packaging | 16 | 24 | 65 |
| pydantic/pydantic-settings | 16 | 24 | 48 |
| python-poetry/tomlkit | 16 | 24 | 35 |
| more-itertools/more-itertools | 16 | 23 | 23 |
| pallets/click | 16 | 22 | 22 |
| pallets/werkzeug | 9 | 9 | 9 |
| mahmoud/boltons | 7 | 7 | 7 |
| jd/tenacity, pytest-dev/pluggy | 6 each | 6 each | 6 each |
| python-humanize/humanize | 3 | 3 | 3 |
| hynek/structlog, python-attrs/attrs | 2 each | 2 each | 2 each |
| python-attrs/cattrs, jazzband/prettytable, marshmallow-code/marshmallow | 1 each | 1 each | 1 each |
| **total** | **118** | **155** | **231** |

Task sources (V): stage 1 has 55 core, 61 extended and 2 overlap tasks; stage 2 18 / 19 / 0; stage 3 37 / 39 / 0.

### 5.3 Nested stages and the cut-off (LEAD item 3)

- **Order.** The main job file runs stage 1, then stage 2, then stage 3. Within a stage, tasks are in draw order and
  all five arms of a task are adjacent, in the A8 counterbalanced order (6.1).
- **Cut-off: 2026-11-11 06:00 UTC.** Stage s is in the analysis set when every planned attempt-1 job of stages 1..s
  (after A0 drops) has a started record (or a row) with `t_start_utc` before the cut-off. The analysis set is the
  largest such nested set: B, A or C. The rule reads job ids and start times only (outcome-blind);
  `analyze_e8.dispatch_status` implements it.
- **Stop rules:** the cut-off, GPU availability and the A8 infrastructure rules (section 11). No other reason stops or
  extends the block. The runner's run deadline is set to end the main run at the cut-off or later; jobs killed by the
  deadline become I09 rows and are re-run if their stage is in the analysis set. A runner stop before the cut-off that
  needs no code, image or setting change is continued by the pre-registered blind resume (11.4), never by a hand-made
  job list.
- **Rows of stages outside the analysis set** are archived unopened and listed (counts only). Neither they nor any
  other E8 per-task outcome may be used for prompt or adapter development.
- Before the cut-off (by the wall clock, 11.6) the analysis set is provisional and every report is blind (unless all
  three stages are dispatched).

### 5.4 A0 controls (CPU only, before any agent job)

1. Run gold and empty controls of every panel and smoke task on the final runner.
2. Drop a task whose gold fails or whose empty patch passes, without replacement; freeze the drop file; its sha256 goes
   into every job (`make_jobs_e8.py --drop-file`, which writes `jobs_e8.A0.json`, `jobs_e8smoke.A0.json` and
   `jobs_e8capture.A0.json`; **run only `.A0.json` files**) and every row.
3. If more than 10% of panel B, A or C would be dropped, `make_jobs_e8.py` refuses and writes nothing; E8 then does not
   start (no amendment path; confirmed, LEAD item 20).

## 6. Runs

### 6.1 Repeats, job file and order

- 4 repeats per arm and task, independent vLLM samples (no request seed; the repeat id is a label); 8 workers.
- `make_jobs_e8.py` (deterministic; reads no outcome) writes `jobs_e8.json` (4,620 jobs: stage 1 2,360, stage 2 740,
  stage 3 1,520), `jobs_e8smoke.json` and `jobs_e8capture.json` (10 jobs each) and `jobs_e8.sha256`. Regenerating them
  reproduces the files byte for byte (V, `test_analyze_e8.py`).
- **Order (A8).** For each task and repeat r: odd r, the arms in sha256(`g4s-armorder:{task}:{r}:{arm}`) order; even r,
  the reverse of r − 1. Each arm's mean position within every task is 2.0 (V, test). The order equals
  `analyze_e7.interleaved_slots_v21` stage by stage (V, test).
- **Job ids:** `<block>:<arm>:<task>:r<repeat>:a<attempt>` (`analyze_e7.job_id_v2`). **Block id:** `E8`; after an abort
  or a void (11.4, 11.6) `E8b`, `E8c`, …. A resume (11.4) keeps the block id and the job ids.
- **Every job carries** the `RUN_RECORD_SPEC_pinned.md` (v2.2) job fields (`arm`, `task`, `seed` = repeat label, `repeat_id`,
  `attempt`, `block`, `job_id`, and in `.A0.json` files `a0_drop_sha256`) plus the E8 fields `stage`, `bundle`,
  `e8_arms_sha256`, `e8_arm_config_sha256`, `request_model`, `compaction_config`, `history_reasoning` and
  `request_chat_template_sha256` (the arm's entry, so each row can be checked on its own).
- **Request dump** (`--dump 1`; the worker dump of 6.2(f), which `llm_dump_sha256` names, LEAD item 23) for every
  job: the per-row manipulation check needs it. J5's dumps took 2.8 MB for 24
  jobs (V), so about 0.5 GB for 4,620 jobs (I).

### 6.2 The worker hook (`worker_berth_e8.py`, Track D4.2)

Built on the GPU host as Track D4.2, after D4.1 (the shared runner), and independently verified against the checklist
below (LEAD item 16) in the order of LEAD item 18: items 1-9 before the freeze, so before any E8 job; items 10-12 after
the freeze and before the first main-block job. On top of the e6 worker:
1. The model registry uses the arm's `request_model`.
2. EvalConfig gets the arm's compaction config (None for C).
3. **One wrapper W, installed last, bound to every name through which ADK reaches LiteLLM.** google-adk 1.36.1 does not
   call `litellm.acompletion` through the `litellm` module: `google/adk/models/lite_llm.py` imports LiteLLM lazily
   (`_ensure_litellm_imported`, lines 172-196, which copies `litellm.acompletion` and `litellm.completion` into the
   module globals `acompletion` and `completion` once, at the first call, guarded by `_LITELLM_IMPORTED`) and
   `LiteLLMClient.acompletion` calls that module global (lines 495-518; streaming uses the same call, 2357; the
   summarizer reaches it through `LiteLlm.generate_content_async`, V, read for this file). The e6 worker's dump and
   sampling wrappers rebind both names, but only when ADK's copy is the one they replace
   (`R/experiments/FOLDER_E6/lib/worker_berth.py:69-85, 112-120`, V); E8 binds them unconditionally.
   The E8 hook therefore:
   - **(a)** calls `google.adk.models.lite_llm._ensure_litellm_imported()` first, so the lazy copy cannot later
     overwrite W;
   - **(b)** takes the chain C = {the original `litellm.acompletion`, the e6 dump wrapper, the e6 sampling wrapper} and
     binds W to **`litellm.acompletion`**, **`google.adk.models.lite_llm.acompletion`** and every other module-level
     name that holds a member of C. "Holds" means a loaded module's `__dict__` maps a name (`acompletion` or an alias,
     for example after `from litellm import acompletion as call`) to that object. Two kinds of module are excluded: the
     `litellm` package itself (`litellm.*`, whose internal references stay) and the worker's own module (`__main__`
     when it runs as a script), whose chain globals (the e6 worker's `_orig_ac`, `_ac`, `_orig_ac2`, `_ac2`, or the E8
     equivalents) must keep pointing down the chain: bound to W they would make W call itself;
   - **(c)** scans `sys.modules` again with the same rule: if any module outside those two exclusions still maps a name
     to a member of C, the worker refuses to start (exit with an infrastructure error, before any job, no row);
   - **(d)** at the start and the end of every job asserts that both names are W (else the job ends as an
     infrastructure error, I08) and wraps the synchronous `litellm.completion` and
     `google.adk.models.lite_llm.completion` with a counter: any call fails the job (I08);
   - **(e)** W is the outermost `acompletion` wrapper (installed after adk_submission's patch, which changes ADK's
     message conversion, not `acompletion`, and after the e6 dump and sampling wrappers, which W calls, so the dump and
     `sampling_sent` record what was sent): H deep-copies the messages and deletes `reasoning` and
     `reasoning_content` from every assistant message, in every call (summarizer included); T merges `chat_template`
     into `extra_body` (template text from the start-time path, sha256-checked); F, R and C pass the call unchanged.
     W counts per job what was sent (tool-result roles, thought characters sent, messages stripped) and **its calls by
     result (LEAD item 12): a call that returned a response adds 1 to `e8_hook_calls`; a call that raised (any
     exception, `asyncio.CancelledError` included: a context-length 400, a transient error that ADK's retry plugin then
     retries, a time-cap cancellation) adds 1 to `e8_hook_calls_failed` and is re-raised unchanged**;
   - **(f)** **dumps a call that raised.** The e6 dump wrapper writes a record only after a response returns
     (`worker_berth.py:71-80`, V), so a failed call would leave no trace. When an exception reaches W, W appends to the
     job's dump the request as sent (after its own change; template text replaced as in item 4), `error` =
     {`type`: the exception's class name, `message`: its first 2,000 characters} and `response` and `usage` null. A
     returned call is dumped as before (request, response, usage; `error` absent or null). **This worker dump is the
     job's request dump (LEAD item 23):** the job's file under the worker's `out/llm/`, written under W, which is the
     outermost wrapper above every recorder. It holds one record per call through W, in call order: the e6 dump wrapper
     directly below W writes a returned call, and W itself writes a call that raised. So its records with a response
     equal `e8_hook_calls`, and its records with an error equal `e8_hook_calls_failed`, by construction (checklist
     item 7(e)). Its sha256 goes into the row's `llm_dump_sha256` (6.3). The D4 runner's forwarder capture keeps its
     own `capture_sha256` and is secondary evidence only (7.2): it logs one record per server request (a call retried
     inside LiteLLM gives several, a call that never got a status gives none), and a call cancelled by the time cap
     whose response the forwarder still logged as 200 would make its returned records exceed `e8_hook_calls`;
   - **(g)** writes into every row the worker writes (its exception rows included) `e8_hook_bound` (true when both
     names were W at the start and the end of the job), `e8_hook_rebound` (the qualified names W was bound to, at least
     the two above), `e8_hook_calls` and `e8_hook_calls_failed` (integers ≥ 0). For jobs of other blocks of the same project W passes every call
     through unchanged.
4. The dump and sampling recorders write `sha256:<hex>` instead of a request template's text.

Nothing else changes. **Verifier checklist for `worker_berth_e8.py`** (Track D4.2; an independent verifier, not the
builder; items 1-9 before the freeze, items 10-12 after it and before the first main-block job, LEAD item 18; if any of
items 10-12 fails, the recovery rule applies: fix, rebuild, new block id, 11.4). The verifier records the worker's sha256, its diff against the
e6 worker it extends (`R/experiments/FOLDER_E6/lib/worker_berth.py`, sha256 8f06cc04… at this revision, or the version
in the D4.1 image), every command and every output, and passes the worker only if every item passes. Items 1-9 run in
the D4.2 image on CPU, with LiteLLM pointed at a stub OpenAI-compatible server that returns canned responses or errors
(no GPU); items 10-12 use the smoke block, the capture block and the A18 positive control (section 7).
1. **Diff.** The diff contains only items 1-4 above (registry model, compaction config, W and its installation, the
   failed-call dump, the hook row fields, template redaction): no change to the agent loop, tools, sampling, grading or
   the fields of the e6 dump record.
2. **Both names, both orders.** `litellm.acompletion is W` and `google.adk.models.lite_llm.acompletion is W` after
   installation in two fresh processes: one installing before ADK's first model call (`_LITELLM_IMPORTED` false at
   installation), one after a completed call. Calling `_ensure_litellm_imported()` again afterwards leaves both W. Both
   are still W at the end of one job of each kind: a job of another block of the same project and one job per arm F, R, C, H, T.
3. **Other references.** A planted module imported before installation that ran `from litellm import acompletion` and
   `from litellm import acompletion as other` ends with both names bound to W, and `e8_hook_rebound` lists both. The
   worker's chain globals are not rebound: one call through W reaches the stub exactly once (no recursion, no double
   call). After installation, `gc.get_referrers(<the original litellm.acompletion>)` shows no module `__dict__` outside
   `litellm.*` and the worker's module. A planted holder that cannot be rebound (a `types.ModuleType` subclass whose
   `__setattr__` raises, holding the original) makes the worker exit with an infrastructure error before any job,
   writing no row.
4. **Per-job assertion and the synchronous path.** Rebinding `google.adk.models.lite_llm.acompletion` (then, in a
   second job, `litellm.acompletion`) to the original inside a job ends that job as I08 with `e8_hook_bound` false;
   `analyze_e8.py check` on that row reports a hook problem (VOID). A call to `litellm.completion` or
   `google.adk.models.lite_llm.completion` inside a job ends it as I08.
5. **Outermost, and what each arm sends.** With probes in the chain, a call from ADK passes W, then the sampling
   wrapper, then the dump wrapper, then LiteLLM. F, R and C: the keyword arguments W passes on equal those it received
   (deep equality). No arm mutates the caller's message list (H deep-copies). H: every request reaching the stub,
   summarizer calls included, has no assistant message with `reasoning` or `reasoning_content`, while the ADK session
   still holds the thought parts. T: the request reaching the stub carries the full template text in
   `extra_body.chat_template` beside the unchanged `chat_template_kwargs` and `thinking_token_budget`, while the dump
   and `sampling_sent` carry `sha256:0a2c8073…`; a template file with another sha256 makes the worker refuse to start.
6. **Summarizer.** A job forced past `token_threshold`: the summarizer's call passes W, is counted and is dumped; in
   arm C no summarizer call occurs.
7. **Call counting and the failed-call dump (LEAD item 12).** Against the stub: (a) k returned calls give
   `e8_hook_calls` = k, `e8_hook_calls_failed` = 0 and k dump records with a response; (b) a context-length 400 (the
   stub answers with vLLM 0.19.1's 400 body for a prompt above 24,576 tokens) gives `e8_hook_calls_failed` = 1 and one
   dump record holding the request, `error.type` `ContextWindowExceededError` and null `response` and `usage`, and
   the row maps to A18 under `classify_row_v22`; (c) a transient error that ADK's retry plugin retries
   (`R/[harness-src]/adk_eval_core/adk_eval_core/plugins/retry_plugin.py`). LiteLLM's OpenAI client retries a 503 inside
   one `acompletion` call (litellm 1.89.7 in [tools-env-name]: a single 503 then a 200 gives 2 server requests, 1 returned call and
   0 failed calls; three 503s give 3 server requests and 1 failed call, V, the [internal-id-1] re-check's probe), so a single 503
   never reaches W. The stub therefore answers 503 to every attempt LiteLLM makes within one call (3 with litellm
   1.89.7's default `max_retries` 2; the verifier records the image's LiteLLM version and the attempts the stub saw), so
   the call raises and ADK's plugin retries it, then a success: `e8_hook_calls_failed` 1, `e8_hook_calls` 1 more, two
   dump records (the failed call and the success); (d) a cancellation during a call (the time cap) is counted in
   `e8_hook_calls_failed`, dumped with `error.type` `CancelledError` and re-raised (the job still ends as the
   harness's time-cap row); (e) in every case `e8_hook_calls + e8_hook_calls_failed` equals the job's dump records,
   and the row's `llm_dump_sha256` is the sha256 of that worker dump (LEAD item 23), never of the forwarder capture.
8. **Row fields.** Every row the worker writes, exception rows included, carries `e8_hook_bound` (bool),
   `e8_hook_rebound` (a list of strings with both names), `e8_hook_calls` and `e8_hook_calls_failed` (integers ≥ 0).
   Rows of other blocks of the same project from the same image differ from the e6 worker's rows only by these fields (E7's analysis ignores
   them).
9. **The runner's row.** For a job killed before the worker writes a row, the D4 runner writes its `g4s-d4 infra: `
   row (`RUN_RECORD_SPEC_pinned.md` section 5.4) with every E8 field of 6.3 copied from the job entry and the runner's
   provenance, and these seven keys present with null (LEAD item 19): the four hook keys `e8_hook_bound`,
   `e8_hook_rebound`, `e8_hook_calls`, `e8_hook_calls_failed`, and `llm_dump_sha256`, `summaries` and `max_prompt`
   (without them the row is a required-field problem and the block is VOID, V, [internal-id-1] re-check); `analyze_e8.py check`
   gives that row I09 and no record problem.
10. **Smoke block.** `analyze_e8.py check --smoke` reports CHECK PASSED (so no row names a dump other than its worker
    dump, 6.3); on every smoke row `e8_hook_calls` equals the records with a response, and `e8_hook_calls_failed` the
    records with an error, of the worker dump that `llm_dump_sha256` names (LEAD item 23). The forwarder capture is
    not held to these counts.
11. **Capture block (7.2).** Every request vLLM logged for a job matches one record of the job's worker dump, and every
    worker-dump record with a response matches one logged request (a failed call may lack a logged request when it
    never reached the server, for example a refused connection; several logged requests may match one dump record
    when LiteLLM retried inside one call, item 7(c)). The forwarder capture (`capture_sha256`) is checked as secondary
    evidence with `manip_check_e8.py --dump-key capture_sha256` (7.2).
12. **A18 positive control (7.1).** One over-length request through the E8 worker stack on the real server: the error
    text maps to A18 under `classify_row_v22`, the call counts in `e8_hook_calls_failed` (not in `e8_hook_calls`), the
    dump holds the request with its error, and the checker's record for that row passes (failed calls are not
    checked, 6.4).

`analyze_e8` makes a worker row whose `e8_hook_bound` is not true, whose `e8_hook_rebound` lacks either name, or whose
call counts are not integers ≥ 0 a record problem (the block is VOID, 11.6; tests
`test_hook_not_bound_to_both_names_is_void`, `test_hook_counts_must_be_integers_on_worker_rows`), and skips the hook
check on the runner's own rows: its I09 row (6.3; `test_runner_row_for_a_dead_job_is_i09_not_a_hook_problem`) and its
tamper row (I12; 6.3, LEAD item 25; `test_runner_tamper_row_is_i12_never_a_hook_or_record_problem`). The checker fails
a row whose `e8_hook_calls` differs from its dump's records with a response (M01, 6.4); with the worker dump named, the
two agree by construction (`test_cli_worker_dumps_are_consistent_with_the_hook_counts`). A row whose `llm_dump_sha256`
names any other dump (the forwarder capture, or a dump mixing formats) gets no manipulation record and is a record
problem (`dump_e8`; the block is VOID, 11.6; `test_cli_row_naming_the_capture_is_a_record_problem`; LEAD item 23).

### 6.3 Run records

E8 rows follow **`E8/RUN_RECORD_SPEC_pinned.md`** (sha256 258c9246…; LEAD item 13): a read-only byte copy of E7's
`RUN_RECORD_SPEC.md` as it stood for rules v2.2, the version that matches the pinned analyze_e7. 

[redacted: 30 words, other experiment state] **E8 never reads the E7 spec in place**: whoever builds the D4 runner's E8 records or
the E8 worker reads the pinned copy (`test_run_record_spec_is_vendored` checks its hash). The rules apply with block
`E8`, including started records, the block manifest, re-runs, the extra batch and resume batches, with these additions
(enforced by `analyze_e8.py`):

| Field | Required | Scope | Content |
|---|---|---|---|
| `bundle`, `bundle_sha256` | yes | block-wide in E8 | `BUNDLE_A` and its tree hash, the same in every arm |
| `served_model_name` | yes | block | the list `[Kaggle path, "google/gemma4-e4b-it"]` |
| `request_model` | yes | arm | as in the arm file |
| `e8_arms_sha256` | yes | block | sha256 of `e8_arms.json` (f0ca2eb8…) |
| `e8_arm_config_sha256` | yes | arm | sha256 of the arm's canonical JSON entry |
| `compaction_config`, `request_chat_template_sha256` | key | arm | as in the arm file (null for C / non-T) |
| `history_reasoning` | yes | arm | `echo` or `strip` |
| `server_chat_template_sha256`, `trust_request_chat_template`, `adk_version` | yes | block | 33204f1a…, true, 1.36.1 |
| `stage` | yes | row | 1, 2 or 3 (0 in the smoke and capture blocks) |
| `llm_dump_sha256` | key | row | sha256 of the job's request dump: the worker dump written under W (6.2(f); LEAD item 23), never the forwarder capture (`capture_sha256`, secondary evidence) |
| `summaries`, `max_prompt` | key | row | existing worker fields |
| `e8_hook_bound` | yes (*) | row | true: both acompletion names were W at the job's start and end (6.2) |
| `e8_hook_rebound` | key (*) | row | the names W was bound to (at least the two of 6.2) |
| `e8_hook_calls`, `e8_hook_calls_failed` | key (*) | row | integers ≥ 0: calls through W that returned a response; calls through W that raised (6.2(e), LEAD item 12) |
| `max_model_len`, `enable_lora`, `max_lora_rank`, `served_adapter_names` | yes | every run entry of the block manifest | 32768, false, null, [] (E7 A9) |

(*) On every row the worker writes, `e8_hook_bound` must be true, `e8_hook_rebound` must hold both names and both
counts must be integers ≥ 0, or the row is a hook record problem (6.2). **The runner's own row** for a job that died
without a results row (`exc` starting `g4s-d<N> infra: `, I09; pinned spec section 5.4), for example a job killed by
the run deadline at the cut-off (5.3), cannot know these facts: it carries the four hook keys and `llm_dump_sha256`,
`summaries` and `max_prompt` with null (LEAD item 19; every other field of this table is copied from the job entry and
the runner's provenance), the hook check skips it, and it gets no manipulation record (6.4). Before this revision such a row would have voided the block (V: the probe of
workflow [internal-id-1] on the earlier code gave VOID with `required_fields_e8` and `hook_e8`).

**The runner's tamper row (LEAD item 25).** When a job writes anything that is not exactly what the worker and PID 1
write (an unexpected `out/` member, a result row of another job or with unexpected keys or value types, more than one
row, an `init.json` written in the job, …), the D4.2 runner (`run_berth.py` fd80b1ea, `tamper_row()` and `_ingest()`; unchanged in the D4.3 runner
9c05d6db…, LEAD item 26)
refuses it, records it in the run directory (`tamper.jsonl`, `refused/`) and writes its own row in place of the job's:
`resolved` false and `error` `g4s-d<N> tamper: <the kinds refused>` with no `exc`, which `classify_row_v22` makes I12
(unknown error text); the four hook keys, `summaries`, `max_prompt`, the phase clocks and `sampling_sent` null; every
other field of this table from the job entry and the runner's provenance; for an E8 job `llm_dump_sha256` still names
the job's worker dump when the job wrote one (V, the code). Anything running in the job, the model's tool calls
included, can cause one (I, from the runner code). 


[redacted: 63 words, run history] Such a row is **treated like the runner's I09 row**: the four hook
keys need only be present (null allowed; a missing key is a required-field problem, VOID); no hook check and no
manipulation check apply (it gets no record, even when it names a dump, and a record for it is ignored, so it stays I12
and is never M01); it is an infrastructure row whose slot is re-run (A8) and which d_pen penalises like any I12 row; and
it never voids the block by itself. Only an I12 row whose `error` starts with that text is a tamper row: a row the
worker wrote with the same text (an exception row, a resolved row) keeps the hook check. **Its tamper evidence is
reported after unblinding** (11.6): blind outputs count it as an I12 row of its arm, attempt and batch and show only the
`g4s-d<N> tamper: ` prefix of its error text; the unblinded report lists every tamper row of the block
(`results.tamper_rows`: per arm the count; per row the job, attempt and batch, whether it lies in the analysis set and
whether its slot got a valid row, the refusal kinds, and the runner's refused-item count and untrusted-row sha256),
descriptively. 







[redacted: 121 words, run history]

**Which dump the row names (LEAD item 23).** On a row the worker writes, `llm_dump_sha256` is the sha256 of the job's
worker dump (6.2(f)). In D4's rows it was the forwarder capture's (V: on the D4 smoke rows `llm_dump_sha256` equals
`capture_sha256`, and the worker's dump is `worker_dump_sha256`), so the D4.2 runner must write the worker dump's sha256
there. `analyze_e8` judges the format of the dump the row names: a record computed from a capture, or from a dump mixing
the two formats, is not attached, and the row is a record problem (`dump_e8`; the block is VOID, 11.6). V: the real D4
smoke rows with the D4.2 hook fields added give VOID with only `dump_e8` (10 of 10 rows in each of `e8smoke_f1` and
`e8smoke_f2`); the same rows renamed to their worker dumps give 8 of 10 records passing in each run, the 2 T rows M01
because D4's arm shim added the template below the recorders (6.4 Validation).

Started records also carry `stage`, `e8_arms_sha256`, `e8_arm_config_sha256` and `request_model`. A row whose arm
fields differ from the arm file, or a `gemma4` model string outside R, is an arm-configuration problem (11.7).
`sampling_sent` must agree across arms once the request template is removed. **Rules v2.2 (E7 A9, as implemented by
the pinned analyze_e7):** the server settings are read from every run entry of the block manifest
(`check_server_settings_v22`; missing, malformed, different between runs or other than E8's values above is a record
problem); every job of every batch's `jobs.json` and every started record must carry the drop file's sha256 (any other
value STOPS the analysis, `a0_jobfile_stops_v22`); a started record repeating a job_id of its run directory is a record
problem; the per-job container runner's row for a job that died without a results row (`g4s-d<N> infra: …`) is I09;
its tamper row (`g4s-d<N> tamper: …`, above) is I12; I13 is anchored to the Docker/g4s forms. **Record problems make
the block VOID** (11.6).

**E7 A10's other items are accepted as pinned (LEAD item 11).** The pinned analyze_e7 (e86ccd81…, rules v2.2 as
verified) is E8's rule set, including the record edge cases E7 fixed later under A10. Measured on the pinned code by the
protocol re-check ([internal-id-2], V; the last item as the [internal-id-2] builder reported it from the code, I): a row whose task or repeat
label lies outside the plan while its job_id names a planned job is ignored, so its slot is re-run; a started record whose labels disagree with its job_id is ignored; a row without an
`attempt` field gives VOID (through the required fields, not a STOP); a container's copy of a started record without
`job_id` STOPS the analysis (A2); `max_model_len` 32768.0 (a float) is a server-setting problem (VOID); re-runs take the
next attempt from rows only. **The D4 runner therefore must, for E8:** write `job_id` and `attempt` on every row and
every started record (LEAD item 11); write started records on the host only, never a container copy without `job_id`;
write the four server settings with JSON types exactly as in the table (`max_model_len` the integer 32768); and keep
every row's task, repeat and attempt labels equal to its job_id's.

### 6.4 Per-row blind manipulation check (LEAD items 5, 12, 14, 17, 22, 23, 24 and 25)

**Every row of every arm is checked on CPU from its dump, blind, inside the frozen analysis** (LEAD item 17). The
checker is `E8/manip_check_e8.py` (Apache-2.0, self-contained: the Python standard library, jinja2 and tokenizers; no
other E8, E7 or harness module). `analyze_e8.py` executes exactly the checker bytes it hashed (compile + exec, as for the
analyze_e7 pin, 8.6) and, at every run of `check`, `report` and `reruns`, recomputes every row's record from the row's
raw request dump; **no stored manipulation record is read or trusted** (there is no `--manip` input). A row's dump is
found by its `llm_dump_sha256` among the files under the run directories given (the worker's `out/llm/` and the D4
runner's `capture/`, regular files, content-addressed); a row whose dump is not found gets no record (an input problem:
the report stays blind until the run directory with the dump is given). **The named dump must be the worker dump**
(LEAD item 23; 6.2(f), 6.3): a record computed from a capture-format or mixed dump is not attached, and the row is a
record problem (`dump_e8`). The checker reads the row's `job_id`, `arm` and
`e8_hook_calls`, the dump, the arm file and three local files pinned by sha256 inside the checker (V, the copies the D4
verifiers use): the Kaggle-v2 template `R/models/[model-dir]/chat_template.jinja` (33204f1a…; byte-identical to
`R/experiments/FOLDER_RUNNER_EVAL/FOLDER_RUNNER_SRC/tools/kaggle_v2_chat_template.jinja`), the HF canonical template
`R/experiments/FOLDER_RUNNER_EVAL/FOLDER_RUNNER_SRC/templates/hf_canonical_e4b.jinja` (0a2c8073…) and the tokenizer
`R/models/[model-dir]/tokenizer.json` (cc8d3a0c…); it never reads `resolved`, errors or tests and never ships template
text. It reads both dump formats: the worker's dump (6.2(f); returned = no `error` and a `response`; only
`usage.prompt_tokens` to compare), the one `analyze_e8` uses, and the D4 runner's forwarder capture (the HTTP body as
sent, with vLLM's `prompt_token_ids`, `completion_token_ids` and `usage`; returned = status 200), which the checker's
own CLI reads with `--dump-key capture_sha256` in the smoke and capture blocks and for the D4.2 verifier (7.2;
secondary evidence, LEAD item 23). **The checks apply to the calls that returned a response** (LEAD
item 12): a dump record of a call that raised has no `usage` to reproduce; it is counted, never checked, so a
context-length run of arm C stays an agent outcome (A18), not M01, and so does a row whose transient error ADK retried.
1. Every dumped request that returned a response (summarizer calls included), re-rendered under the arm's rule with
   vLLM 0.19.1's preprocessing (text content parts, JSON-decoded tool-call arguments, the thought only under
   `reasoning`, `tool_call_id` only for role `tool`, 3.1; the renderer settings of the D4 verifiers'
   `tools/render_check.py` and `tools/capture_check_d4.py`; Kaggle-v2 for F, R, C, H; HF canonical for T), must
   reproduce vLLM's `usage.prompt_tokens`, and the token ids where vLLM logged them.
2. The arm's factor (the arm-file dimensions in which the arm differs from F) must be present in those requests: R, the
   request model string `google/gemma4-e4b-it` and every tool result with role `tool_responses`; every other arm, role
   `tool`; C, no returned summarizer call and no compaction summary (a user message whose first part is ADK's
   `For context:`) in any request; non-C arms, summarizer calls exactly where ADK 1.36.1's token-threshold rule puts
   them, as far as the dump shows it (below); H, no assistant message carries `reasoning` or `reasoning_content`;
   non-H arms, every history thought carries both keys with equal text, and **every history assistant turn whose own
   returned response had a thought carries it** under both keys with equal text, in every returned request that
   holds the turn (presence, not only equality; LEAD item 22; a turn is named by its position, never by its text,
   LEAD item 24; below); T, every request carries the template hash
   0a2c8073…, other arms none.
3. Nothing else may differ in those requests: each of those dimensions where the arm equals F, the sampling (agent
   calls: temperature 0.2, top_p 0.95, max_completion_tokens 8,192, `chat_template_kwargs` {enable_thinking: true},
   thinking_token_budget 4,096; summarizer calls: only the last two; 3.1), the tool list (BUNDLE_A's, one canonical
   sha256, the same in all 344 agent requests of the D3 parity dumps over 24 tasks and in all 442 agent requests of the
   D4 smoke captures and worker dumps, V) and the message shape (roles
   and keys).
4. The hook was not bypassed: the row's `e8_hook_calls` equals the number of dump records with a response (6.2; with
   the worker dump this holds by construction unless the hook was bypassed or the dump is incomplete). The
   failed calls (`e8_hook_calls_failed`; dump records with an error) never fail the check; a row whose failed-call
   count differs from its dump is described after unblinding (8.5), not M01.

*Summarizer placement (I, from ADK 1.36.1 `apps/compaction.py` and `flows/llm_flows/compaction.py`, implemented in
`compaction_findings`).* A returned summarizer call is accepted only right after a returned call whose
`usage.prompt_tokens` reached `token_threshold` (failed calls in between are transparent; a summarizer right after
another one is accepted when the first or the agent call before it reached the threshold). A missing summarizer call is
flagged only when a returned agent call reached the threshold, the next returned call is an agent call (failed calls
are skipped and their content is never judged), no summarizer call was attempted in between (a failed summarizer
attempt never counts against the row), and that call's request shows events to compact under ADK's rule (more than
`event_retention_size` candidate events after the last compaction and a non-zero safe split that keeps each tool
response with its call). Where the dump cannot show ADK's
state, the rule does not flag (it errs towards passing, never towards M01). On the D4 smoke captures every one of the
11 summarizer calls sits right after a request of 14,433 to 15,951 prompt tokens (V).

*History-thought presence (LEAD items 22 and 24; `response_thought`, `align_turns`, `turn_links` and `check_dump` in the
checker).*
- **A turn is named by position, never by its text (LEAD item 24).** A turn is the response of one returned agent call
  and is named by that call. Each returned agent request's assistant messages are mapped by position onto the turns of
  the previous returned agent request; failed calls in between are skipped.
  - Without a returned summarizer call in between, the history only grows: message n is the previous request's message
    n, and one further message is a new turn.
  - After a returned summarizer call, compaction may have removed the oldest turns: the messages before the newest one
    are the previous request's last ones.
- **Tying a new turn to its response.** A new turn is the response of the previous returned agent call. In the worker
  dump the response's tool-call ids must also equal the message's, so a turn with tool calls is the response with
  those ids. A text-only turn has no ids and is named by its position alone, as in the capture, which logs no response
  ids.
- **Identical texts.** A message's text without its two thought keys only confirms a mapping; it never names a turn.
  Two turns with identical text are distinct turns: for example, the same final answer given again after the harness
  nudge. Each such turn carries the requirement of its own response only.
- **When no mapping fits.** If no mapping fits, or more than one does (possible only with identical texts after a
  compaction), the request's turns are left unnamed and counted in `n_history_requests_unaligned`.
- **Following the turn.** A named turn is checked in every later returned request that holds it, until compaction
  removes it.
- **Whether the response had a thought.** In the worker dump: its `reasoning` (the e6 recorder's copy of LiteLLM's
  `reasoning_content` or `reasoning`) is non-blank. In the capture: vLLM's completion token ids hold the thought
  channel's closing token (id 101 of the pinned tokenizer) after a non-blank thought. T's HF generation prompt already
  opens the channel, so T completions hold no opening token.
- **Erring towards passing.** A turn the dump cannot name or tie to a response, or whose own response had no thought
  (or a blank one), is never required to carry one. A per-message rule would not err this way: in J5, 13 of 2,094
  history assistant messages carry no thought.

V (workflow [internal-id-3] scratch scripts, CPU; the item-24 checker gives the same counts, [internal-id-4] (f) below):
- (a) On the D4 smoke runs, a completion holds the closing token exactly when the worker dump's response has a thought:
  431 of 431 agent calls with one, 11 of 11 without.
- (b) Both formats tie the same 422 turns, 415 of them with a thought. Each of the 315 echo-arm turns whose response had
  a thought carries it wherever it appears, so all 20 capture rows and the 16 non-T worker rows still pass.
- (c) On the BUNDLE_A parity dumps of the scorer path, checked as arm F, no turn is flagged:
  - J5 (`FOLDER_OUT_DOCKER/parity_run_A`): 342 turns tied, 341 with a thought. Its 13 thoughtless history messages
    are all the one turn whose response had no thought.
  - D3: 320 tied, 320 with a thought.
  - D4 `parity_run_B`: 374 and 371; `parity_run_C`: 332 and 331; worker dumps and captures alike.
- (d) These all fail, in both formats: the real H dump labelled F; an echo-arm row (F, R, C, T) with the newest history
  thought of every request removed, or with every history thought removed (prompts re-rendered so that only this rule
  can fail); a single stripped turn. The v1 checker passed the first two ([internal-id-5] `corrupt.py` C3c and C6).

V (workflow [internal-id-4], CPU; LEAD item 24; the item-24 checker b075ed73… against the v2 checker efa623eb…):
- (e) **The re-check's reproduction** ([internal-id-3] `collision22.py`). A text-only answer is given with a thought, then the
  harness nudge, then the same answer again, whose own response had no thought.
  - The v2 checker named turns by their text: it failed this case in both formats ("no history thought although the
    response of call 0 had one").
  - The item-24 checker passes all four thought orders in both formats and ties both answers (2 turns, not 1).
- (f) **Every real dump, old against new.** 240 dumps, 4,214 returned requests:
  - the 20 E8 smoke rows under their own arm, in both formats; D4's T worker dumps were also checked with the template
    hash added, as D4.2's W records it;
  - J5, D3, D4 parity, D4 smoke, D4 canaries [redacted: 6 words, adapter detail], all checked as F.

  Results:
  - No presence flag anywhere.
  - No change of `pass` or of any record field except the descriptive counts.
  - `n_history_requests_unaligned` is 0 everywhere.
  - 
[redacted: 29 words, adapter detail]
  - The E8 smoke rows give the same results as before: 20 of 20 captures pass, 16 of 20 worker dumps pass (D4's T
    dumps lack the hash), and 4 of 4 T worker dumps pass once the hash is added.
- (g) **The re-check's 564 history-thought attacks** ([internal-id-3] `attack22.py`) give identical outcomes, reasons and counts
  under three checkers: v2, the re-check's own 5-line fix, and the item-24 checker.

A row without a returned call has nothing to check: its record has `n_requests` 0 and `factor_present` true, and it
passes when its `e8_hook_calls` is 0. The runner's own I09 row (6.3) has no dump and gets no record; `analyze_e8`
ignores a record for it, so it stays I09. The runner's tamper row (6.3; LEAD item 25) gets no record either, even when
its `llm_dump_sha256` names the job's worker dump; a record for it is ignored, so it stays I12 and is never M01.

**Record** (one per row, recomputed at every run): `job_id`, `arm`, `pass` (bool), `checker_sha256` (set by
`analyze_e8` to the sha256 of the checker bytes it executed, never taken from the checker), `dump_sha256` (must equal the
row's `llm_dump_sha256`), `dump_readable`, `dump_format` (`worker` on every record `analyze_e8` attaches, LEAD item
23), `n_requests` (dump records with a response: the requests checked), `n_requests_failed` (dump records with an
error: not checked), `n_prompt_tokens_reproduced`, `n_ids_checked`, `n_ids_reproduced`, `factor_present`,
`factor_checks`, `other_differences` (list), `hook_calls_match`, `n_requests_changed` (exposure, 3.3),
`n_history_turns_linked` and `n_history_thought_turns` (descriptive: the history turns tied to their response, and
those whose response had a thought, LEAD item 22), `n_history_requests_unaligned` (descriptive: returned agent
requests holding assistant messages that no unique positional mapping names, LEAD item 24; 0 on every real dump
checked) and `reasons`. **pass = `dump_readable` and every returned request reproduced
(tokens and ids) and `factor_present` and no other difference and `hook_calls_match`**: a record fails only for what its
returned calls, its dump or its hook count show, so a row whose returned calls pass is never M01. `analyze_e8.py`
recomputes this formula from the record's fields and rejects a record that disagrees, and rejects pass true with an
`n_requests` other than the row's `e8_hook_calls` (`test_record_pass_formula_guard`,
`test_pass_record_must_cover_exactly_the_returned_calls`).

**Validation (V, `test_manip_check_e8.py`, CPU).** On the D4 E8 smoke captures (runs `e8smoke_f1` and `e8smoke_f2`, 2
tasks × 5 arms each, Track D4 infrastructure data; `e8_hook_calls` taken from the runner's own forwarder count of 200
responses, as these rows predate the D4.2 fields) all 20 rows pass on the forwarder capture (the dump the D4 rows name;
D4.2 rows name the worker dump, LEAD item 23): 453 of 453 returned requests reproduced, token ids included, each arm's
factor present, no other difference, history thoughts present on all 422 tied turns. On the worker's own dumps of the
same rows the 16 non-T rows pass (counts only); the 4 T rows fail with the factor absent,
because D4's arm shim added the template below the e6 recorders, so neither the worker dump nor `sampling_sent` carries
the hash; 6.2(e) and checklist item 5 require D4.2's W to sit above them. The checker's CLI on those rows rewritten
as D4.2 rows (`llm_dump_sha256` = the worker dump) passes 8 of 10 by default and 10 of 10 with `--dump-key
capture_sha256`. Synthetic corruptions all fail with the
targeted reason: R's model string or one role changed; `tool_responses` outside R; T's template hash removed, a T row
given an F dump, a template hash outside T; an H request carrying `reasoning_content`; a non-H thought without the
mirror into `reasoning`; a history thought missing in F, R, C or T (the newest turn of every request, every turn, one
turn, or the H dump labelled F); a summarizer call or a compaction summary in C; F without its summarizer call where
the rule fires, or with it moved below the threshold; sampling, tool list and message-shape changes; one token id or
the server's count changed; a request edited; a hook count off by one or not an integer; a truncated dump. Failed calls
(a context-length 400, two 503s before a success, a failed summarizer call, an uncompacted request that failed) leave
the record passing, and so does a turn without a thought whose own response had none (or a blank one), or that the
worker dump cannot tie to a response.

*Identical-text turns (LEAD item 24).* These tests use synthetic trajectories built from real smoke records, with
every prompt re-rendered under the arm's rule:
- The same text-only answer given twice after the harness nudge passes in every thought order (the re-check's
  reproduction), in both formats and in every echo arm, with both answers tied.
- A lost thought among identical answers fails and names its own call. The answer whose own response had none is never
  flagged. This also holds with a failed call between the answers and across a compaction.
- Where identical texts after a compaction allow two mappings, that request's turns are left unnamed, and the row
  passes.

**Consequences.** A row fails its check only for what its returned calls show; a failed check marks the row as
infrastructure class **M01** (re-run; reported); M01 is in the d_pen classes. (V: in
`test_failed_calls_are_counted_not_checked`, 8 context-length rows of arm C with one failed call each stay A18 and the
block unblinds.) A row whose dump is not found keeps every report blind until its dump is given. Records of another
checker version or for another dump are input problems.

**The checker and its frozen hash (LEAD items 14, 17 and 20).** The checker is built and independently verified before
the freeze (LEAD item 18); it judges the key the server reads (`reasoning`)
[redacted: 16 words, internal process history]. `checker_sha256` covers exactly the bytes of `manip_check_e8.py`: it imports no
other project code, and the templates and tokenizer it reads are pinned by sha256 constants inside those bytes, so the
hash freezes them too; the jinja2 and tokenizers versions are recorded in every output (3.1.6 and 0.23.3 in [tools-env-name], V),
not frozen by hash. At freeze its sha256 is filled in as `FROZEN_CHECKER_SHA256` in `analyze_e8.py` (the only change to
that file at freeze; whoever verifies the freeze checks that one-line diff) and printed here:

> `FROZEN_CHECKER_SHA256` = **b075ed73acea92e95ef2ac91d9bf1d00d115fd957dae7e0baaa5e065651d0c16**

From then on every subcommand refuses (exit 3, only `<out>.refused.json`) when `manip_check_e8.py` is missing, a
symbolic link or hashes to anything else (`test_checker_pin_refusals_in_a_fresh_process`); records carry the hash of
the bytes that ran, and a record of any other version is never attached (an input problem: `report` stays blind,
`reruns` refuses). `report` and `reruns` refuse to run at all while the value is not filled in, and unless the command
line gives `--checker-sha256` equal to it and `--analyze-e8-sha256` equal to the sha256 of the `analyze_e8.py` that
runs, both copied from the frozen protocol (this section and section 15); `check` verifies either flag when given. This
closes the protocol re-check's route ([internal-id-2] `checker_swap.py`, V there: after unblinding, a "new checker version" failing
exactly the 423 unsolved H rows made `reruns` issue 423 H re-runs) and the [internal-id-1] re-check's residual (forged records
carrying the frozen hash): there is no record input to forge. `test_frozen_checker_hash_refuses_other_checker_records`,
`test_frozen_hash_flags_on_the_command_line`, `test_freeze_fill_in_in_a_fresh_process` and
`test_cli_recomputes_records_from_dumps` cover these. **A checker change after the freeze aborts the block** (recovery
rule, 11.4; LEAD item 20).

**Cost (V/I).** On the D4 smoke runs (20 rows, 453 returned requests) `analyze_e8`'s recomputation took 41.3 s with one
process and 22.6 s with `--workers 4` [redacted: 4 words, load detail], with identical records (V). At about 23 requests per row the
4,620-job block holds about 105,000 requests: of the order of 1-3 hours of CPU per run, so the protocol's commands use
`--workers 8` or more (I).

## 7. Checks before the main block

### 7.1 Smoke block (records)

- Block `E8smoke`: the 2 smoke tasks × 5 arms × 1 repeat (`jobs_e8smoke.A0.json`).
- **Smoke and capture tasks** are outside the eligible pool: among the 24 BUNDLE_A parity tasks, those that compacted at
  least once in both captured parity runs (J5 and D3: 6 tasks, V), in sha256(`g4s-e8-smoke-v1:` + id) order, the first
  two from different repositories: **marshmallow_2937** and **more-itertools_1253** (V, recomputed for this file). Both
  are E7 panel S tasks; E7's analysis ignores rows of another block.
- **Replacement rule (A0).** The six candidates in that order are marshmallow_2937, more-itertools_1253,
  more-itertools_1248, boltons_440, tenacity_640, structlog_739 (the membership as listed by the protocol review; the
  order and repositories V, `test_smoke_replacement_rule`; all six are E7 panel S tasks, outside the E8 pool). A0
  controls run on all six. If A0 drops a smoke task, the smoke and capture blocks take, after A0 drops, the first
  candidate and then the next candidate from another repository (`make_jobs_e8.smoke_tasks_for`; for example, without
  marshmallow_2937: more-itertools_1253 and boltons_440). `make_jobs_e8.py --drop-file` writes the smoke and capture
  `.A0.json` files with the replacements and `analyze_e8 check --smoke` analyses the same two tasks; if no pair from
  two repositories remains, make_jobs_e8 refuses and the lead decides before any agent job.
- After the freeze (LEAD item 18; 6.2 checklist item 10), `analyze_e8.py check --smoke --block E8smoke` must report
  CHECK PASSED: no required field, started record, manifest, server-setting, job-list, provenance, hook, dump (every
  row names its worker dump, LEAD item 23) or arm-configuration problem and every manipulation check passed,
  recomputed by the frozen checker from the worker dumps (6.4). **Smoke and capture outcomes are never opened**
  (`report --smoke` is refused).
- **A18 positive control:** one over-length request through the E8 worker stack must yield an error text that
  `classify_row_v22` maps to A18, counted in `e8_hook_calls_failed` and dumped with its error, with a passing
  manipulation record (6.2 checklist item 12).

### 7.2 Capture block (prompts at token level)

Block `E8capture` (the same 10 jobs), 1 worker, `--dump 1 --vllm-log-requests 1 --vllm-debug 1`. For every request,
vLLM's prompt token ids must equal an independent rendering under the arm's rule; `GET /v1/models` lists both names; F
requests on the E8 server are token-identical to a Kaggle-v2 rendering; every request vLLM logged matches one record
of its job's worker dump and every worker-dump record with a response matches one logged request (6.2 checklist item
11). **The forwarder capture is secondary evidence here** (LEAD item 23): `manip_check_e8.py --runs <capture run>
--arms-file e8_arms.json ASSETS --dump-key capture_sha256 --out …` checks every row's captured requests at token level,
ids included, with the same rules as 6.4 (the history-presence rule reads each response from vLLM's completion ids).
A hook-count difference there is read against the capture's statuses (several server requests for a call LiteLLM
retried, none for a call that never got a status), not as a bypass. Any difference stops E8 until explained.
Cost: about 20 jobs plus server starts (I).

## 8. Analysis (`analyze_e8.py`)

### 8.1 Primary

> For each trap arm X in {R, C, H, T}: d_X = (1/N) Σ_i [ȳ(X, i) − ȳ(F, i)], the task-weighted solve-rate difference
> between X and F on the N tasks of the analysis set that have all four valid slots in both arms, where ȳ(a, i) is the
> mean of `solved` over arm a's four valid runs of task i. The tasks are fixed; the runs are the random part. The
> primary test is the exact two-sided task-level sign-flip test of the per-task differences, Holm-adjusted across the
> four trap arms at 0.05. Claims are worded "on these N tasks". The repository-cluster 95% interval is a reported
> secondary for "repositories like these", not a claim rule.

- **Test.** With K_i = 4·D_i (integers in −4..4), p = #{ε ∈ {±1}^n : |Σ ε_i |K_i|| ≥ |Σ K_i|} / 2^n over the n tasks
  with K_i ≠ 0, counted exactly by subset sums (`signflip_exact`; equal to brute-force enumeration on 300 random cases,
  V). Enumeration is always feasible, so the 99,999-draw Monte Carlo version (seed 20261109) is reported as a check.
- **Holm** over R, C, H, T at family-wise 0.05 (`holm_adjust`, exact fractions). A comparison that cannot be computed
  enters Holm with p = 1 (the family stays at four).
- **Claim rule.** "X raises/lowers E4B's solve rate against F on these N tasks" iff X's Holm-adjusted p ≤ 0.05; the
  direction is the sign of d_X.
- **Intervals** (descriptive): the 95% task-level interval inverting the sign-flip test under a constant shift (the same
  99,999 frozen draws for every shift; grid 0.01 refined to 10⁻⁴); the simultaneous 98.75% (Bonferroni, 1 − 0.05/4)
  interval; the t interval with tasks as clusters (CR1 by task, `cluster_mean_ci`; validated by the review: size 4.75%,
  coverage 95.3-99.7%). Claims follow Holm, so a claimed arm's Bonferroni interval may touch 0.

### 8.2 Analysis set and missing slots (LEAD items 3-4)

- Analysis set: section 5.3, after A0 drops.
- **Primary set per comparison:** tasks with all 4 valid slots in both X and F.
- If any slot of X or F is missing once no slot can get another attempt (or the re-run cut-off has passed, 11.3), the
  comparison is labelled **INCOMPLETE** and also reported with: the **all-rows sensitivity** (task means over the valid
  rows; tasks with at least one valid row in each arm) and **worst-case bounds** over every analysis-set task, the
  missing slots of both arms imputed in opposite directions (upper: missing X slots solved and missing F slots
  unsolved; lower: the reverse; denominators 4). These are reported, not claim rules.
- **Time-cap rows (A05)**, if any occur in the primary set, get a sensitivity treating them as missing.
- **Context-length 400s (A18)** are agent outcomes in every arm.

### 8.3 Secondary: repositories like these

Per comparison, on the primary set: analyze_e7's null-imposed Rademacher wild cluster bootstrap-t over repositories,
exact by enumeration (`exact_rademacher_p`); two-sided p = min(1, 2·min(p_upper, p_lower)); the 95% interval is every
δ that neither one-sided exact test rejects at 0.025 (grid 0.01 over [−1, 1], refined to 0.001). Checks: the 9,999-draw
Monte Carlo version (BOOT_SEED 20261109) and Webb weights, both two-sided; the CR1 interval with t(G − 1). Descriptives
from `gate_core`: the per-repository table, the repository-equal estimate and the leave-one-repository-out range. Five
repositories hold 117 of panel A's 155 tasks; with G = 15 the smallest attainable two-sided p is 2·2⁻¹⁵ (I).

### 8.4 Sensitivities (reported, not decisive)

- **d_pen_X:** every X slot of the primary set with a row of class I01, I02, I03, I09, I12, I13, I14 or M01 (any
  attempt) scored unsolved; **d_pen_F:** the same for F (`analyze_e7.d_pen_task_diffs`, E8 class list). If X's Holm
  decision changes when d is replaced by d_pen_X or d_pen_F (other arms unchanged), the result is labelled "sensitive
  to infrastructure errors".
- All-rows, worst-case bounds and time-cap-as-missing (8.2).
- Optional descriptive split by task source (core vs extended).

### 8.5 Secondary and descriptive outcomes (exploratory; no multiplicity claim)

Paired task differences X − F with CR1 repository-cluster 95% intervals: from the rows, compactions per run
(`summaries`), share of runs with a compaction, `max_prompt`, **unsolved** context-length rows (A18: a run whose working
tree passes after the HTTP 400 is resolved and carries no error, swegemma `verification.py:697`, so A18 counts only
unsolved context-length runs; the field is `context_length_unsolved`), time-cap (A05), tool-budget (A02) and no-submit
(A06) rows, agent time (mean, median, p90), LLM and tool calls, failed LLM calls (`e8_hook_calls_failed`: calls that
raised, LEAD item 12; `llm_calls_failed`), tokens; from the traces (`analyze_e7.secondary`), thought rate (all and
non-terminal), failed edit and write calls, malformed calls; from the manipulation records, exposure (share of
requests the manipulation changed), failed calls dumped and the rows whose dumped failed calls differ from their
`e8_hook_calls_failed` (a failed call the dump missed; no rule). Failed-call counts appear only after unblinding: per
arm they would reveal context-length runs. Absolute solve rates per arm with CR1 intervals, and solved counts per arm
and repeat.

### 8.6 Code and pinning

`analyze_e8.py` (Apache-2.0) reuses, never copies, analyze_e7 **rules v2.2** (E7 A0-A9): `classify_row_v22`,
`load_runs_v21` (with `dedupe_started`, as v2.2 calls it), `assemble_v21`, `rerun_jobs_v21`, `check_manifest_v21`,
`batch_checks_v21`, `required_fields_v21`, `provenance_check_v21`, `check_server_settings_v22`,
`a0_jobfile_stops_v22`, `blind_class_mismatches_v22`, `job_refusal_v22`, `d_pen_task_diffs`, `harness_4b_stratified`,
`exact_rademacher_p`, `wild_cluster_bootstrap_t`, `cluster_mean_ci`, `gate_core`, `secondary`, and v2.2's reason
categories (input, record, slots). M01 enters every E7 rule as `classify_row_e8` (M01 on top of `classify_row_v22`),
passed as the classifier wherever an analyze_e7 function takes one and substituted for `classify_row_v21` and
`classify_row_v22` by a context manager while analyze_e8 runs.

**Pin (E7 A10, last item; accepted with its record edge cases by LEAD item 11, 6.3).** `analyze_e7_pinned.py` is a
byte copy of E7's `analyze_e7.py` at sha256 **e86ccd81…**
(the version verified for rules v2.2), 


[redacted: 57 words, other experiment state]. `analyze_e8` reads only that copy, never the E7 folder: the copy must be a regular file (a symbolic link is
refused) whose bytes hash to the pinned value, and analyze_e8 executes exactly those bytes (`compile` + `exec`, no
import system, so no `__pycache__` file can substitute other code; `make_jobs_e8.py` is loaded the same way and its
sha256 recorded). Otherwise every subcommand refuses (exit 3) and writes only `<out>.refused.json`.
`--allow-unpinned-e7` runs `check` and `reruns` on the copy present, with every output marked non-frozen; **`report`
refuses the flag**, pinned or not, so no unblinding runs on unpinned rules.

Subcommands: `report` (blind while any reason is open, 11.6), `check` (always blind; smoke, capture and monitoring),
`reruns` (the next blind batch with the E8 job fields; wall clock only, 11.3). Each recomputes every row's manipulation
record from the dumps with the vendored checker (6.4, LEAD item 17); an unblinded report also writes the records
(`<out>.manip.jsonl`), blind outputs carry only their sha256. **Frozen hashes (LEAD items 14 and 17; 6.4):**
`report` and `reruns` refuse (exit 3) until `FROZEN_CHECKER_SHA256` is filled in, and then run only with
`--checker-sha256` equal to it and `--analyze-e8-sha256` equal to the sha256 of the `analyze_e8.py` that runs, both
copied from the frozen protocol; records of another checker version are never used. Every output records the
wall-clock time, the time the report was made for, `code_sha256` (analyze_e8, make_jobs_e8, the pinned copy,
manip_check_e8 and the frozen checker hash), the checker hash enforced, the template and tokenizer hashes and the
jinja2 and tokenizers versions the checker used, the flags given and `runtime` (Python and numpy versions, the
interpreter, whether it ran isolated with `-I`). Tests: `test_analyze_e8.py`, 66 tests on synthetic rows with
hand-computed answers, including every as-of and pin trick of the reviews, the hook-call rule, the runner's I09 row,
the runner's tamper row (LEAD item 25; one test on a copy of the D4.2 CPU E8 smoke runs, the verifier's `v1tamper`),
the frozen-hash flags, the one-line freeze in a fresh process, the recomputation from dumps, the checker pin, the
pass-formula guard and the worker-dump rule (LEAD item 23); `test_manip_check_e8.py`, 56 tests of the checker on the
D4 smoke dumps (capture and worker) and synthetic corruptions, history-thought presence and positional turn identity
included (6.4; LEAD items 22 and 24) (V: 66 and 56 passed, [tools-env-name] `python -I -B -m pytest`, CPU, workflow [internal-id-6]). A realistic-size synthetic block (231 tasks × 5 arms × 4
repeats, random solves) runs through `report` with the frozen settings in 212 s (the re-check [internal-id-7]) to 264 s
(the [internal-id-2] revision, real CLI and system clock, [redacted: 4 words, load detail]) wall on a loaded CPU (V; the earlier "about
51 s" was not reproduced; not re-measured for the [internal-id-1] revision, whose additions are per-row field checks, I). An
independent verifier re-implements the primary estimate before any E8 data exists (done twice on the statistics,
which neither revision changes).

## 9. Sample size and power (LEAD item 7)

**Method** (V; `E8/power/sim_e8_task_power.py`, sha256 a19e5fbe…, outputs `power/sim_main.json` and
`power/sim_mixed.json`). 2,000 simulated panels per cell (Monte Carlo SE ≤ 1.1 points). Per panel: one F arm and four
trap arms sharing F's outcomes, 4 Bernoulli repeats each, the real repository sizes of B, A and C. Test: the exact
two-sided sign-flip test of this protocol; Holm over the four trap arms. Exactness of the p-value code checked against
brute-force enumeration (`power/check_exact_p.py`, maximum difference 0, V).
- **Difficulty models** (I, assumptions), calibrated to population mean m and repeat correlation ρ = Var(p_i)/(m(1−m)):
  logit-normal with a repository share of 0.25 of the latent variance (the E7 generator's form; m 0.19, ρ 0.7 gives
  μ −5.16, s 5.59, as in the draft); a two-level beta (repository mean ~ Beta, task ~ Beta around it; 25% of Var(p_i)
  between repositories; the reviewer's form).
- **Grid:** ρ 0.5 / 0.7 / 0.85 × m 0.15 / 0.19 / 0.25 × both models × stage sets B / A / C. 

[redacted: 25 words, bundle measurement]
- **Scenarios:** "one" = one affected arm, three null arms (Holm power for the affected arm); "all" = all four arms
  affected (per-arm Holm power). Effect patterns: even logit shift of every task (5 points ≈ −1.19 logits at the
  central cell); breaks (a random share of tasks loses 6 logits; 31% for −5 points at the central cell); zero-mean
  heterogeneity (20% of tasks lose 6 logits, the rest gain so that the population mean change is 0).

**Size** (V): per-arm rejection under no effect 1.8-4.5%, Holm family-wise error 1.2-4.2%, over all 54 null cells.

**Power for an even 5-point change**, % (V; range over the 18 cells of ρ × m × model; central cell ρ 0.7, m 0.19,
logit-normal / beta in brackets):

| Analysis set | Unadjusted (drop) | Holm, one arm affected (drop) | Holm, all arms affected (drop, per arm) | Holm, one arm (gain) |
|---|---|---|---|---|
| B: stage 1, 118 tasks (2,360 runs) | 66-97 [91.0 / 88.9] | 44-90 [77.3 / 75.5] | 53-94 [85.7 / 83.4] | 41-90 |
| A: stages 1-2, 155 tasks (3,100) | 79-100 [96.7 / 95.7] | 61-98 [89.2 / 88.9] | 69-99 [94.8 / 92.9] | 56-97 |
| C: stages 1-3, 231 tasks (4,620) | 93-100 [99.4 / 98.8] | 82-100 [98.3 / 96.6] | 89-100 [99.2 / 98.2] | 78-100 |

Power is lowest at ρ 0.5 with base 0.25 and highest at ρ 0.85 (I: more variance sits between tasks, so a shift moves
fewer, larger task differences).

**Change detected with 80% power** (points; linear interpolation between simulated changes of 3, 4, 5, 6 and 8
points for drops; 3, 4, 5, 6, 7 and 8 for gains; V/I). The main grid stops at a gain of 6; gains of 7 and 8 come from
`power/sim_gain_supplement.py` (the same `run_cell` code, sha256 a19e5fbe… as run, the same 18 cells per set, 2,000
panels each, seeds 10001+; output `power/sim_gain_supplement.json`, V, run for this revision):

| Analysis set | Unadjusted, drop | Holm one arm, drop | Holm one arm, gain | Central cell, Holm one arm, drop |
|---|---|---|---|---|
| B | 3.3-5.9 | 4.3-7.1 | 4.3-7.4 | 5.2 / 5.3 |
| A | below 3 to 5.1 | 3.4-6.0 | 3.5-6.4 | 4.5 / 4.6 |
| C | below 3 to 4.2 | below 3 to 4.9 | below 3 to 5.2 | 3.6 / 3.8 |

The worst B gain cell is ρ 0.5, m 0.25 (logit-normal): 75.3% at +7 and 85.9% at +8 points (the protocol review's own
generator gave 76.45% and 87.15%, about 7.3 points). **These ranges, drops and gains together, are frozen as the M of
null statements** (`analyze_e8.MDE_HOLM_ONE`: B "about 4.3 to 7.4", A "about 3.4 to 6.4", C "below 3 to about 5.2").

**Other patterns** (V): a 5-point drop confined to a random share of tasks (breaks): unadjusted A 62-91%, Holm one arm
41-78%; B 50-80 / 28-61; C 78-98 / 61-93. An 8-point break drop: A 93-100 / 83-99. Zero-mean heterogeneity: rejection
4.0-5.5% unadjusted, 1.0-1.8% under Holm, with the panel's own effect varying by 1.1-1.5 points (sd); wrong-sign
rejections ≤ 0.05%.

**Consistency checks** (V): at the central cell on A, this test gives 96.7% for a 5-point drop; the review's task-level
CR1 test gave 97.15% and its repository-cluster test 86.45% (Holm 67.65%), which is why the lead chose the task-level
primary.

## 10. GPU time

20 runs per task (5 arms × 4 repeats). Rates: **318.6 runs/h**, measured by D3 on a 24-job BUNDLE_A list with 8 workers
(V, README_docker.md:222), and a conservative **240 runs/h** (the review estimated 287-422 runs/h from per-job wall
times, I).

| Analysis set | Runs | Hours at 318.6 / h | Hours at 240 / h |
|---|---|---|---|
| Stage 1 (B) | 2,360 | 7.4 | 9.8 |
| Stages 1-2 (A) | 3,100 | 9.7 | 12.9 |
| Stages 1-3 (C) | 4,620 | 14.5 | 19.3 |

Add: smoke and capture (20 jobs, the capture with 1 worker) plus server starts, under 0.5 h; re-runs and the extra
batch, about 0.5 h at a 3% infrastructure-error rate (I). To dispatch all three stages before the cut-off at 240 runs/h,
the main block must start by about 2026-11-10 10:45 UTC (I, arithmetic). C's long prompts can lower throughput for every
arm (I; load monitor, 11.5). E8 shares the GPU host with [redacted: 6 words, other workloads]; the lead sets the order.

## 11. Infrastructure errors, recovery, re-runs, load and blind reporting

### 11.1 Error classes

analyze_e7's ERROR_TABLE_V22 unchanged (rules v2.2, sha256 2ec64ba7…; it is ERROR_TABLE_V21, 53adfa32…, with I09 also
covering the per-job container runner's `g4s-d<N> infra: …` rows and I13 anchored, E7 A9), plus **M01** (manipulation
check failed, infrastructure). d_pen classes: I01, I02, I03, I09, I12, I13, I14 and M01. The runner's tamper row
(`g4s-d<N> tamper: …`, 6.3; LEAD item 25) is I12: infrastructure, re-run, penalised in d_pen.
A18 (context length), A05 (time cap), A02 (tool budget) and A06 (no submit) are agent outcomes. A non-context HTTP 400
stays I01 (infrastructure, re-run) under the frozen table; if an arm's manipulation causes it systematically, its slots
end missing and the comparison is INCOMPLETE with worst-case bounds (8.2). No row is reclassified by hand.

### 11.2 Manipulation failures

A row whose returned calls fail the check (6.4; failed calls are never checked, LEAD item 12) is M01: its slot is
missing and gets the next attempt. **Stop rule:** while an M01 slot is
open the block pauses and the lead classifies the cause blind (arm, the check's reasons, error texts). A cause that
needs a code or image fix triggers the recovery rule (11.4); a transient cause is re-run.

### 11.3 Re-runs and the re-run cut-off

A1/A8 unchanged: the analysis's `reruns` command, never a person, builds each batch (attempts 2-3; then one blind extra
batch, attempts 4-5, all arms), only for analysis-set slots, refusing while manipulation records are missing or an
analysis input is incomplete, and making no batch for a VOID block (E7 A9). It is run only after the previous run of
the block has ended (a job still running looks like a started job without a row, I09). Never delete or edit a row;
never re-run a slot that has a valid row. **Re-run cut-off: 2026-11-12 00:00 UTC** (confirmed, LEAD item 20). `reruns` decides by the
wall clock only: it takes no as-of (an `--as-of` given is ignored and recorded), so no backdated or unreadable time can
reopen the window. After the cut-off, or when no slot can get another attempt, every slot still without a valid row is
missing and the missing-slot rule (8.2) applies.

### 11.4 Recovery rule (LEAD item 6)

Any fix that changes code or the image aborts the block: rebuild, re-run the canaries, and restart all arms and slots
under a new block id (`E8b`, …). Aborted rows are archived unopened and listed as a deviation. The cut-off does not
move. A restarted block (after an abort or a VOID, 11.6) names the block it replaces in its manifest: `"replaces":
{"block": <an earlier E8 block id>, "reason": "VOID" | "abort", "record_sha256": <sha256 of that block's VOID report or
of the lead's abort record>}`; the first block names none. A missing or malformed entry is an input problem
(`analyze_e8.restart_chain`; E8 keeps its E8b, E8c, … ids instead of E7's `<panel>_v<n>`).

**Blind resume (pre-registered; LEAD item 15).** A runner stop before the cut-off is a **resume**, not an abort,
when the lead classifies it blind (from runner, vLLM and system logs; no outcome) as needing no change to code, image, model, arm
file, job file or server settings (for example a vLLM or worker crash, a reboot, [redacted: 6 words, owner workload]). Then:
1. Wait until no job of the block is running.
2. Run `make_jobs_e8.py --resume <every run directory of the block so far> --drop-file <the block's A0 list>`. It
   rebuilds the frozen `jobs_e8.A0.json` from the panel and the drop file, reads only job ids, blocks, attempts and
   drop-file hashes from the run directories (`jobs.json`, `out/started.jsonl`, `out/results.jsonl`), and writes
   `jobs_e8.resume<k>.A0.json`: the frozen file's **attempt-1 jobs that never started** (no started record and no row),
   **in file order, entries unchanged (same block id, same job ids)**, plus `jobs_e8.resume<k>.deviation.json` (the run
   directories with their file hashes, counts, the first never-started position and the output's sha256). It refuses,
   writing nothing, at or after the cut-off (wall clock), when not exactly one run directory holds the frozen main job
   file, and on any job entry, started record or row outside the frozen plan (or a re-run of a frozen slot), with
   another block or drop-file sha256, or an attempt-1 job started in two run directories.
3. Run the resume file under a new run tag of the same block with the same runner, image and server settings, and add
   the tag to the block manifest.
4. Log the resume as a deviation with the deviation record's sha256.
Jobs that were running at the stop (a started record, no row) are I09 and get their next attempts through `reruns`
(11.3); the analysis needs nothing else (V: `test_resumed_block_is_analysed`; the resume files equal the protocol
re-check's reference outputs byte for byte: 3,420 jobs from position 1,200, sha256 c66fd797…, and 3,418 jobs with two
later jobs already started, 3260c181…). A stop that needs any change, or one after which the runner's records cannot
be read, is an abort.

### 11.5 Load monitor (LEAD item 8; blind, pooled across arms)

`analyze_e8` reports, never per arm: the share of rows with `t_agent` above 240 s (overall and per batch; a batch above
5% is flagged; a batch is shown on its own only when every arm has the same number of rows in it, because a re-run
batch of one arm would show that arm's time-cap rows; other batches enter only the overall share), and vLLM
pre-emptions (overall, and per shown batch; `vllm:num_preemptions`, a counter in vLLM 0.19.1
`v1/metrics/loggers.py:583`, V; the runner samples `/metrics` into JSON lines with `t_utc` and
`vllm:num_preemptions_total`; a decrease is a server restart). Agent-time secondaries are stated as measured under
mixed load. No rule depends on the monitor.

### 11.6 Blind reporting and unblinding

Until unblinding, every report carries only: analysis-set status, slots per arm, infrastructure rows by class, arm and
batch, the infrastructure-share diagnostics, manipulation-check results, provenance, record and manifest problems, and
the load monitor; no solve count, d, p, interval, repository table or secondary metric (`analyze_e8.py check`, and
`report` while any reason is open). M01 details never name the row's own class. The runner's tamper rows (6.3) count
as I12 rows like any other, with only the `g4s-d<N> tamper: ` prefix of their error text shown; their evidence (the
refusal kinds, the untrusted row's sha256) is reported after unblinding only (`results.tamper_rows`; LEAD item 25).

`report` unblinds **once, for all arms together**, when no reason is open: the analysis set is final (cut-off passed, or
all three stages dispatched); no slot can get another attempt before the re-run cut-off; every agent-outcome row's dump
was found and its manipulation record recomputed by the frozen checker in this run (6.4); records, manifest, server
settings, provenance, the hook fields and arm configuration are clean; analyze_e7 is the pinned version; and `report`
runs as the frozen `analyze_e8.py` with the frozen checker (6.4; the command refuses otherwise). Blind reports show only
class mismatches that involve an infrastructure id, agent ids masked (E7 A9).

**Time.** Every decision uses the **wall clock** (UTC, timezone-aware; a `TZ` setting does not move it). `report` and
`check` accept `--as-of` only as `YYYY-MM-DDTHH:MM:SS[.ffffff]` followed by `Z` or `+00:00`, **not later than the wall
clock**; anything else (a naive time, another offset, a date, an epoch number, `now`, a lowercase `z`, a blank, a
future time) is refused: exit 3, no report, only `<out>.refused.json`. An earlier `--as-of` can only keep a report
blind (the provisional analysis set and the re-run window both close later); it never opens one. (It reads the data
present at the wall clock, so it is not a reconstruction of an earlier report.) `reruns` ignores `--as-of` (11.3).
Every output records the wall-clock time, the time it was made for, the code hashes and the interpreter;
`protocol_settings` says that the settings, cut-offs, pinned analyze_e7, frozen checker hash and system clock are the
protocol's. That the code is the frozen one is enforced on the command line (`--analyze-e8-sha256`, 6.4) and recorded
in `code_sha256`.

**Decisions** (E7 A9 order): an input problem (clearable) gives INCONCLUSIVE until the inputs are fixed; otherwise a
record problem gives **VOID** (blind; `reruns` makes no batch): the block's run directories are archived unopened, the
deviation is logged, the cause is fixed (a code or image fix follows 11.4) and, if time remains before the cut-off, all
arms and slots restart under a new block id whose manifest names this block (11.4); otherwise E8 is INCONCLUSIVE and
makes no claim. An arm-configuration problem is a record problem and also triggers the stop rule (11.7).

### 11.7 Stop rules

E7's ([redacted: 5 words, owner workload], any isolation canary failure, provenance problems); a capture or smoke
failure (7); an arm-configuration mismatch (6.3) or any `tool_responses` role or `gemma4` model string outside R; an
open M01 slot (11.2); systematic infrastructure errors (one arm, one worker, a vLLM restart): find the cause before
spending attempts.

## 12. Reported regardless of outcome, and publication

**Estimand (as in sections 0, 1 and 8.1).**
> For each trap arm X in {R, C, H, T}: d_X = (1/N) Σ_i [ȳ(X, i) − ȳ(F, i)], the task-weighted solve-rate difference
> between X and F on the N tasks of the analysis set that have all four valid slots in both arms, where ȳ(a, i) is the
> mean of `solved` over arm a's four valid runs of task i. The tasks are fixed; the runs are the random part. The
> primary test is the exact two-sided task-level sign-flip test of the per-task differences, Holm-adjusted across the
> four trap arms at 0.05. Claims are worded "on these N tasks". The repository-cluster 95% interval is a reported
> secondary for "repositories like these", not a claim rule.

**Reporting list.**
1. The analysis set (stages, N, cut-off status, A0 drops) and, per trap arm: d, exact p, Holm-adjusted p, claim, the
   Monte Carlo p, the 95% and simultaneous intervals, the task-level t interval; the repository-cluster p, interval,
   Monte Carlo and Webb checks, CR1, repository-equal estimate, per-repository table and leave-one-out range.
2. Sensitivities: INCOMPLETE label, all-rows, worst-case bounds, time-cap-as-missing, d_pen_X, d_pen_F and the
   "sensitive to infrastructure errors" label.
3. Absolute solve rates per arm and solved counts per arm and repeat (not hosted scores).
4. All secondary and exposure measures of 8.5, per arm and as paired differences.
5. **Null results** are worded "no difference detected on these N tasks; the 95% interval is [lo, hi]; the design
   detects an even change of M points with 80% power", with the frozen M of section 9; never "no effect".
6. Checks: the capture's token-level results per arm, the smoke check, the A18 positive control, the A0 drop list,
   manipulation-check results per arm (M01 rows and their causes).
7. Infrastructure and process: attempts, batches, exhausted slots, infrastructure rows by class, arm and batch, the
   infrastructure-share diagnostics, the load monitor, provenance, every deviation and abort.
8. An update to the J2 erratum with R's measured difference.
9. **The Writeup and the kit README report every arm that ran**, null results included, with intervals and the frozen
   detectable-change range, worded as divergences from the scorer path in either direction.
10. Files and sha256 values: job files, manifest, arm file, drop file, the manipulation records the unblinded report
    wrote (`<out>.manip.jsonl`), analysis outputs, this protocol, `make_jobs_e8.py`, `analyze_e8.py`, the pinned
    analyze_e7, `RUN_RECORD_SPEC_pinned.md`, the worker, the checker `manip_check_e8.py`, the templates and the
    tokenizer.

**Bundle and publication rule** (LEAD item 9; [redacted: 11 words, owner approval]).
- **Before 2026-12-03:** aggregates, per-task and per-arm outcome tables (task, arm, repeat, attempt, solved, error
  class, agent time; no paths, host names or container details), analysis code and hashes. Request dumps may be
  published with the system prompt replaced by its sha256, after the redaction tool removes sandbox paths and host
  names.
- **On 2026-12-03:** the full BUNDLE_A bundle and the raw dumps.
- Templates are cited by hash and difference, never reproduced; the kit loads them from a user path and checks the
  sha256.
- 




[redacted: 99 words, redaction plan]

## 13. Limits of interpretation (I)

- **Setting.** The GPU host serves at TP1 on one RTX 5090; Kaggle uses TP4 on 4×L4. Bare `pytest` is missing on the GPU host's PATH (D3);
  D3 caps each job at 2 CPUs and 10 GiB. These hit every arm equally: they limit transfer, not the comparison.
- **Panel.** 15 repositories, five dominant; claims are about these tasks. The repository-cluster interval addresses
  "repositories like these" only loosely (G = 15).
- **Model and prompt.** E4B with BUNDLE_A only; nothing about 31B or other prompts.
- **Arms.** One realistic configuration per divergence; R follows ADK 1.36.1's name rule (newer ADK selects by
  provider); T swaps the whole template; H keeps thoughts in compaction summaries. In H the summarizer's own requests
  also lack history thoughts (LEAD item 2), so d_H includes any effect that passes through the summaries.
- **Interference.** C's longer prompts load the shared server; agent time can rise in every arm (11.5). Server load,
  prefix caching and block order cannot plausibly bias solves (the 300 s cap is far from observed run times; the cache
  shares only identical token prefixes; arm positions are balanced; V/I via the review).

## 14. Open items for the lead before freezing




















[redacted: 363 words, internal process history] Still open:
1. 












[redacted: 207 words, internal process history]
2. *Decided (LEAD item 23):* which dump a D4.2 row's `llm_dump_sha256` names. **It names the worker dump** written
   under W (6.2(f)), whose records with a response equal `e8_hook_calls` and whose records with an error equal
   `e8_hook_calls_failed` by construction, so checklist item 10's two equalities apply to it.
   - Why not the capture: the D4 runner named the forwarder capture (V: on the D4 smoke rows `llm_dump_sha256` equals
     `capture_sha256`; the worker's dump is `worker_dump_sha256`). The capture holds no record of a call that never got
     a status (V, the [internal-id-5] re-check: two D4 H rows, both time-capped at t_agent 300 s, whose forwarder saw 26 and 43
     requests but logged 25 and 42 statuses), and it holds one record per server request when LiteLLM retries inside
     a call (7(c)). A call cancelled by the time cap whose response the forwarder still logged as 200 would make
     `n_requests` exceed `e8_hook_calls`: M01 on an agent outcome (I).
   - The capture stays secondary evidence. The capture block checks it at token level with `manip_check_e8.py
     --dump-key capture_sha256` (7.2). It is not held to the hook counts.
   - Enforcement: `analyze_e8` makes a row naming a capture-format or mixed dump a record problem (`dump_e8`, 6.3,
     6.4).
   - The D4.2 runner writes the worker dump's sha256 into `llm_dump_sha256`. W sits above the recorders, so T's worker
     dumps carry `sha256:0a2c8073…` (6.2(e), checklist item 5); D4's arm shim did not, which is why D4's T worker dumps
     fail (6.4 Validation).
3. Confirm that the BUNDLE_A bundle contributes no request text outside the system prompt; otherwise the pre-12-03 dump
   redaction must hash every bundle-derived string.
4. **Freeze steps** (LEAD items 14, 17 and 18), in this order, on the files as independently verified:
   1. Check that `manip_check_e8.py`, `analyze_e8.py`, `test_analyze_e8.py`, `test_manip_check_e8.py`,
      `make_jobs_e8.py`, `analyze_e7_pinned.py`, `RUN_RECORD_SPEC_pinned.md`, `e8_arms.json`, the job files and this
      file still have the sha256 values the verification recorded (section 15 lists this revision's), and that
      `E8/__pycache__` does not exist.
   2. `C` = sha256 of `manip_check_e8.py`. At this revision ([internal-id-6], LEAD item 25) C is
      `b075ed73acea92e95ef2ac91d9bf1d00d115fd957dae7e0baaa5e065651d0c16` (V), unchanged since [internal-id-4]: the checker did
      not change in [internal-id-6] [redacted: 12 words, internal process history].
   3. In `analyze_e8.py`, replace the single line `FROZEN_CHECKER_SHA256 = None` by `FROZEN_CHECKER_SHA256 = '<C>'`
      (the only change to that file: `diff` shows one line; at this revision it is line 124).
   4. `A` = sha256 of `analyze_e8.py` after step 3. At this revision A is
      `16a80a2a92ff9916fd061b0e10e73c4df80c7ccb62c0241d01ca74de9931f2f7` (V: computed in a scratch frozen mirror by
      steps 2-6, all tests passing there). 
[redacted: 26 words, internal process history] If the files of step 1
      differ from section 15's values, these two values do not apply and the freeze stops.
   5. In this file, replace the three placeholders: the checker value in 6.4 and in section 15 by `C`, and the frozen
      `analyze_e8.py` value in section 15 by `A` (each placeholder is the bold text "filled in at freeze"; `C` and `A`
      go in bold the same way). Optionally change the status line to FROZEN with the time.
   6. Run, from `E8/`: `PY -I -B -m pytest -q -p no:cacheprovider test_analyze_e8.py test_manip_check_e8.py`. Every
      test must pass; `test_protocol_states_the_frozen_hashes_consistently` then checks both printed values against
      the code. Delete any `__pycache__` the run left (none is expected with `-B`).
   7. `P` = sha256 of this file after step 5. The lead records `P`, `C` and `A` (and the step-1 hashes) in
      `LEAD_DECISIONS.md` or the deviations log. A second person checks the one-line diff of `analyze_e8.py` and the
      three-placeholder diff of this file.
   8. From then on `report` and `reruns` run only with `--checker-sha256 <C> --analyze-e8-sha256 <A>`; any change to
      `manip_check_e8.py` or `analyze_e8.py` after the freeze aborts the block (11.4; LEAD item 20).
5. Then, before the first main-block job: checklist items 10-12 in the smoke block, the capture block and the A18
   positive control (7; LEAD item 18).
6. LOW items accepted as documented (LEAD item 21), none blocking: `report` does not refuse a non-isolated interpreter
   (the runtime record shows it); the load monitor's overall counts of two successive checks differ by a hidden
   one-arm batch's figures; the resume tool accepts a hand-made reordered earlier resume directory and cannot see a run
   directory not passed to it (no `--manifest`); `restart_chain` accepts a sha256 followed by a newline and the block
   id `E8a`; a panel or tasks-file problem exits 1 without a refusal record; an interrupted re-run batch is re-issued
   with the same attempt-2 job ids, which the runner must accept; the year 1 is written as `1-01-01` in an as-of; the
   tests import `analyze_e8` through the import system (the CLI runs the source); the full-size report's time depends
   on the CPU load ([redacted: 10 words, load detail]).

## 15. Files (sha256 prefixes on 2026-11-10, about 00:45 UTC, workflow [internal-id-6])

- **Frozen hashes** (filled in at freeze, 14 item 4): `FROZEN_CHECKER_SHA256` = **b075ed73acea92e95ef2ac91d9bf1d00d115fd957dae7e0baaa5e065651d0c16** (6.4);
  frozen `analyze_e8.py` sha256 = **16a80a2a92ff9916fd061b0e10e73c4df80c7ccb62c0241d01ca74de9931f2f7** (`report` and `reruns` require both on the command line).
- This experiment: `LEAD_DECISIONS.md` 4a652573… (items 1-26
[redacted: 20 words, internal process history]);
  `PROTOCOL_DRAFT.md` fcf9773b…; `panel_E8.json` f03c0b93…; `e8_arms.json` f0ca2eb8…; `jobs_e8.json` 4fe9fb3a…;
  `jobs_e8smoke.json` 394cacc9…; `jobs_e8capture.json` a7abd121…; `jobs_e8.sha256` 7b319ba4…; `analyze_e7_pinned.py`
  e86ccd81… (byte copy of E7's `analyze_e7.py`, read-only); `RUN_RECORD_SPEC_pinned.md` 258c9246… (byte copy of E7's
  `RUN_RECORD_SPEC.md` for rules v2.2, read-only, LEAD item 13); `make_jobs_e8.py` b101bc6b… (unchanged in this
  revision); this revision ([internal-id-6]), before the freeze's one-line fill-in: `manip_check_e8.py` b075ed73… (the checker
  v3: LEAD item 24; unchanged in [internal-id-6][redacted: 10 words, internal process history]), `analyze_e8.py`
  b679f289… (LEAD item 25; [redacted: 4 words, internal process history]), `test_analyze_e8.py` b2c0aa1d… (LEAD item 25;
  [redacted: 4 words, internal process history]), `test_manip_check_e8.py` 8a553996… (unchanged since [internal-id-4]; 
[redacted: 4 words, internal process history]); `power/`:
  `sim_e8_task_power.py` a19e5fbe… as run (recorded in `sim_main.json` 2cc04952…, `sim_mixed.json` 09bab8bb… and
  `sim_gain_supplement.json` 4d4d23c2…), ef2d96cd… with the Apache-2.0 header and a docstring fix only (the supplement
  re-run with it gives identical cells, V); `sim_gain_supplement.py` 3e82f35a…; `check_exact_p.py` f7612bc8…;
  `summarise_sim.py` e220ad1a…; `summarise_mde.py` 6e19e7e1…. No `__pycache__` folder.
- 


[redacted: 47 words, other experiment state]
- Checker inputs (local copies, read from the paths given and checked against the hashes inside `manip_check_e8.py`;
  never copied into E8): Kaggle-v2 template 33204f1a… (`R/models/[model-dir]/chat_template.jinja`, V; the same bytes as
  `R/experiments/FOLDER_RUNNER_EVAL/FOLDER_RUNNER_SRC/tools/kaggle_v2_chat_template.jinja`); HF canonical template 0a2c8073…
  (`R/experiments/FOLDER_RUNNER_EVAL/FOLDER_RUNNER_SRC/templates/hf_canonical_e4b.jinja`, V); tokenizer cc8d3a0c…
  (`R/models/[model-dir]/tokenizer.json`, V); jinja2 3.1.6 and tokenizers 0.23.3 in [tools-env-name] (V). The tests also read the
  D4 smoke runs `R/experiments/FOLDER_E6/FOLDER_OUT_D4/runs/e8smoke_f1` and `e8smoke_f2`, and ([internal-id-6], LEAD item 25) the
  D4.2 CPU E8 smoke runs `R/experiments/FOLDER_E6/FOLDER_OUT_D42/runs/d42c2_w1` and `d42c2_w2` (results.jsonl e0f166b7…
  and 36b094c2…, jobs.json da95c5ce… and d0a4d207…) with their manifest `manifests/E8smoke_d42c2_manifest.json`
  (212e3ef4…) and the drop file `R/experiments/FOLDER_RUNNER_EVAL/FOLDER_RUNNER_SRC/jobs/drop_empty.json` (37517e5f…), copied
  into the test's temporary folder before any change (environment variables `E8_TOKENIZER`, `E8_TEMPLATE_KAGGLE`,
  `E8_TEMPLATE_T`, `E8_D4_SMOKE_RUNS`, `E8_D42_SMOKE_RUNS`, `E8_D42_DROP_FILE` override the paths).
- Commands (CPU, `[cpu-priority]`, run from `E8/`; `PY` = `R/[tools-env]/bin/python`, 
[redacted: 12 words, interpreter detail]; `ASSETS` = `--tokenizer R/models/[model-dir]/tokenizer.json --template-kaggle
  R/models/[model-dir]/chat_template.jinja --template-t
  R/experiments/FOLDER_RUNNER_EVAL/FOLDER_RUNNER_SRC/templates/hf_canonical_e4b.jinja`): `PY -I -B make_jobs_e8.py [--draw]
  [--drop-file A0] [--out-dir D]`; `PY -I -B make_jobs_e8.py --resume RUN_DIR [RUN_DIR ...] --drop-file A0 [--out-dir
  D]`; `PY -I -B -m pytest -q -p no:cacheprovider test_analyze_e8.py test_manip_check_e8.py`; `PY -I
  power/sim_e8_task_power.py OUT --n 2000 --workers 6 --grid main|mixed`; `PY -I -B power/sim_gain_supplement.py OUT
  --n 2000 --workers 4`; `PY -I -B analyze_e8.py report|reruns --runs … --manifest … --drop-file … ASSETS --workers 8
  --checker-sha256 <frozen> --analyze-e8-sha256 <frozen> --out … [--as-of YYYY-MM-DDTHH:MM:SSZ]` (`reruns` ignores
  `--as-of`); `PY -I -B analyze_e8.py check …` (the same inputs; the two hash flags optional); the checker on its own
  (D4.2 verifier, smoke and capture blocks): `PY -I -B manip_check_e8.py --runs RUN_DIR [...] --arms-file e8_arms.json
  ASSETS [--dump-key capture_sha256] --out PREFIX` (default `--dump-key llm_dump_sha256`, the worker dump; the
  capture as secondary evidence, 7.2).
