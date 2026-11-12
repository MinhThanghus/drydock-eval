<!--
Copyright 2026 The drydock-eval Authors
SPDX-License-Identifier: Apache-2.0
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# drydock-eval: a safety and fidelity kit for evaluating Gemma 4 coding agents locally

`drydock-eval` (import name `drydock_eval`) helps you evaluate a Gemma 4 coding agent on your own hardware: run
untrusted agent code without giving it your machine (`contain`), check that the numbers transfer to the Kaggle scorer of
the Gemma 4 agent competition (`conform`), and decide between two agent configurations without fooling yourself
(`compare`). Apache-2.0 (see `LICENSE`; `NOTICE` carries the attribution of the Moby seccomp profile from which the
profile of `contain` is derived). It ships no competition data: no harness wheels or harness source, no tasks, graphs,
embeddings, model files, tokenizer or chat template, and no agent bundle, prompt, request dump or run output. Anything
competition-specific (the chat template, the tokenizer, a capture of your own harness, the harness packages
themselves) is supplied by you and, where it matters, checked by sha256.

**How this was made.** The code, the tests and the reviews named in this README were produced by AI coding agents
working for the authors. Each verification step is stated with the command that reproduces it, and none of them replaces
review by a human: run the commands, and read the code you intend to rely on. A human owner approved the use of AI
agents for this work and is responsible for the release. Where this README says "independent AI review", it means a
review by a separate AI agent, not by the one that wrote the code.

Modules:

| Module | Command | What it does | What it needs |
|---|---|---|---|
| `drydock_eval.contain` | `drydock-eval contain ...` | Reference implementation of a contained runner: one container per job with no network, a request forwarder with an allow-list and caps, a gate file that binds eight hashes, and the canary suite. The request filter and the gate rule can be exercised without Docker | Docker for anything that starts a container; a vLLM server, a GPU and your own harness copy for agent runs |
| `drydock_eval.conform` | `drydock-eval conform ...` | Checks that what your local stack sends to the model equals what the Kaggle scorer sends (`check`), and that a task's code-graph and embedding files make the harness's graph tools behave as on the scorer (`graph`) | jinja2 |
| `drydock_eval.compare` | `drydock-eval compare ...` | Decides between agent configurations from a results table: the frozen five-criterion pass rule (`gate`), Holm-corrected task-level sign-flip tests (`signflip`), and the simulators behind the published size and power numbers (`simulate`) | numpy |

`experiments/e8/` holds the E8 pre-registration package: the task panel and redacted copies of the documents and code (see
[E8 pre-registration package](#e8-pre-registration-package)).

Install (Python 3.10+):

```bash
pip install -e .              # core: conform (recorder, checks, reports) and contain; needs only jinja2 (tomli on 3.10)
pip install -e '.[compare]'   # + numpy, for drydock-eval compare
pip install -e '.[hf]'        # + tokenizers, to load a tokenizer.json for the template check
pip install -e '.[adk]'       # + google-adk 1.36.1, litellm 1.82.4, openai 2.32.0 for the live demo and mutation suite
pip install -e '.[test]'      # + pytest and numpy
```

## Scope and limits, in one place

- **`contain` is a reference implementation, not a product.** It is the authors' final runner with host-specific
  constants replaced, and one canary check (the home-directory listing and the write targets) rewritten so that it no
  longer assumes the operator's account name; this rewrite has not run in a container. It keeps the authors' internal names (explained in the operator guide's glossary). The
  released tree has **not** been run in Docker: the canary counts in the write-up were measured on the original files,
  and every released file has a new hash because a licence header was added. `drydock_eval/contain/SOURCES.json` lists
  each released file with the sha256 of the verified file it was built from. You must pass the canaries on your own
  host before you trust it, and what a pass does not show is listed in the guide under "What remains": containers share
  the host kernel, user namespaces are not remapped, the request parser runs in the trusted runner, and a job can slow
  others through the shared model server within its caps.
- **`conform` checks one scorer path** (google-adk 1.36.1, vLLM 0.19.1, the Gemma 4 chat template you supply). Its
  limits are in [Limits](#limits-read-before-relying-on-a-pass).
- **`compare` holds its level only under its assumptions.** The simulated false-pass rate is at most 1.9% when tasks
  are exchangeable, about 8% when a candidate breaks a fifth of the repositories with no mean effect, and 30 to 40%
  if tasks are dropped after outcomes are seen. After release 0.4.0 was built, the rule was applied to three real
  candidates; none passed (the supplement on the branch `paper` gives every quantity).
- **Nothing here was measured on your stack.** Every number in this README was measured by the authors on their own
  hardware and task set; where the kit can reproduce it, the command is given, and where it cannot, the table says `none`.

## Paper claims and the commands that reproduce them

"Docker" marks commands that start containers; they were not run on the released tree (see above). Everything else was
run for this release on a CPU-only machine.

| Claim in the write-up | Command | Notes |
|---|---|---|
| Section 3: what one job gets (flags, forwarder allow-list, caps) | `drydock-eval contain guide` (section 4); `drydock-eval contain check-body body.json` | the second prints the forwarder's decision on a request body of yours; no Docker |
| Section 3: the request-body scanner is exact and linear in the body size up to the 1 MiB cap | `drydock-eval contain scan-check` (`--quick` for a short run) | no Docker; also `pytest tests/test_contain_lib.py tests/test_contain_gate_cli.py` |
| Section 3: agent runs start only if eight recorded hashes still match | `drydock-eval contain gate show --root <deploy>/evald`; `drydock-eval contain gate verify --config runs/<tag>/config.json --root <deploy>/evald` | reads a gate file your own canary run wrote; the refusal cases are tested without Docker |
| Table 1, final runner: 200 of 200 canary checks | `drydock-eval contain canaries --root <deploy>/evald --tag c1` | Docker. Some checks compare against records of earlier runner generations and against template and tokenizer files that are not shipped; on a fresh host those checks fail or do not apply (guide, section 7). Rows 1 to 6 of Table 1 concern earlier runners, which are not released |
| Section 3: reference patch solves, empty patch does not, on every task | `drydock-eval contain run --root <deploy>/evald --tag g1 --jobs jobs.json --mode gold -- --block S` (and `--mode empty`) | Docker, your harness copy and the task set; no GPU |
| Section 3: container gone after a killed runner; runs per hour | none | needs a GPU, the model and an agent bundle; the measurement is not reproducible from this package |
| Section 5: the five-criterion pass rule | `drydock-eval compare gate table.csv --baseline A --candidate B` | exit code 0 PASS, 1 FAIL, 3 INCONCLUSIVE |
| Table 3: simulated pass rates | `drydock-eval compare simulate table3` (`--quick` takes seconds to a minute, depending on the number of workers) | rows 1, 2 and 5 reproduce the published cells to their printed precision at the default 3,000 panels; row 4 (even gain) gives 61.2% and 91.1% against the published about 60% and about 90%, within Monte Carlo error; the dropping row (3) is a re-implementation and agrees within Monte Carlo error |
| Section 5: independent re-implementations agreed with the rule code | none | the re-implementations are not shipped; `tests/test_compare_crosscheck.py` compares the packaged rule code with values frozen from the original script |
| Section 6: Holm-corrected task-level sign-flip tests; power of the design | `drydock-eval compare signflip table.csv --reference F`; `drydock-eval compare simulate power` | the E8 outcome rows are not part of this package |
| Sections 4 and Table 2: the eight conformance checks and their measured detection | `drydock-eval conform check ...`; `drydock-eval conform mutate ...` | see the next section |

## Conformance checks: `drydock-eval conform`

A local evaluation of a Gemma 4 agent is only comparable with the Kaggle scorer if two things hold: the model sees the
same prompts, and it samples with the same settings. Several small differences change the prompt without any error.
Four of them are known upstream (cited below). This module turns them, plus your submission's sampling, tool and
compaction settings, into executable checks over captured requests. It also measures how reliably each check detects
each drift.

### How it works

1. **Record.** Put the logging proxy between your agent and your model server, as the last hop before the server.
   Every chat-completions request body and reply (including `usage`) is stored in a capture file before the reply
   goes back to the client. Request headers are never written. Use `--request-token-ids`: the server then returns
   the prompt token ids (vLLM's `return_token_ids`), and the template check compares ids instead of lengths.
2. **Make a reference from your own harness.** Run the competition's harness with your submission's config through
   the recorder, then `ref`. The reference holds your tool list, every request field with its value, and the message
   shape: fields per role, which thought keys travel together, tool_call_id, tools on every call, the role skeleton
   (how many system messages and where, and which roles your harness itself puts directly after a tool result), the
   sha256 of the follow-up user texts your harness sends (its nudges), the system prompt's sha256 after the chat
   template's own trim, and how tool results are encoded (JSON or text, wrapper, serialisation). It stays on your
   machine. Build it from several tasks, so that per-task differences are not mistaken for drift.
3. **Check.** Run the eight checks below over the capture of your local stack. You get one PASS / FAIL / WARN / SKIP
   line per check and an optional JSON report. The exit code is 1 when any check fails.

```bash
# 1. record: point the agent at http://127.0.0.1:8099/v1 (or /s/<label>/v1 to label a session)
drydock-eval conform record --upstream http://127.0.0.1:8000 --out local.jsonl --request-token-ids

# 2. a reference from a capture of YOUR OWN harness (kept on your machine)
drydock-eval conform ref harness_capture.jsonl --out ref.json

# 3. check (the template and tokenizer are yours; the template's sha256 is verified)
drydock-eval conform check local.jsonl --ref ref.json \
    --template chat_template.jinja --template-sha256 <sha256> \
    --tokenizer tokenizer.json --special-tokens tokenizer_config.json --json report.json
```

`check` also imports client-side LiteLLM call dumps (JSONL with `model`, `messages`, `tools`, sampling kwargs,
`extra_body`, `response`, `usage`). For a one-worker run whose vLLM server logged requests, `--vllm-log vllm.log`
attaches the server's prompt token ids by order. It refuses a capture spread over several dump files without time
fields, because their order is unknown. Failed exchanges (non-2xx, for example an HTTP 500 that the client retried)
are left out and counted in the report header.

### Where the expectations come from

- **Scorer-path properties** (built-in profile `kaggle-e4b-2026`, the default). These hold for every submission: the
  tool-result role, google-adk's compaction config, how vLLM 0.19.1 hands messages to the template (content format,
  and which thought key it reads) and the template hash.
- **Sampling and request fields** are set by each submission's generation config, not by the scorer.
  `kaggle-e4b-2026` therefore has no sampling expectation, and the sampling check reports SKIP until you pass `--ref`.
  A hand-written TOML profile also works (`--profile my.toml`; an unknown profile name or base is an error):

  ```toml
  base = "kaggle-e4b-2026"
  [agent_sampling]                 # fields every agent request must send, with these values (an illustration)
  temperature = 0.7
  max_completion_tokens = 4096
  [agent_sampling.chat_template_kwargs]
  enable_thinking = true
  [agent_sampling_forbidden]       # fields that must be absent
  keys = ["top_k"]
  ```

  `[summarizer_sampling]` and `[summarizer_sampling_forbidden]` do the same for the compaction-summarizer calls. The
  package ships no submission's sampling settings: the live demo and the test fixtures send the competition harness's
  own defaults for an agent that sets no generation config (see the live demo below).
- **Server version.** `--server-thought-keys reasoning,reasoning_content` switches the thought rule to vLLM >= 0.22
  (see below). The `thoughts`, `newline` and `shape` checks all follow it.
- **System prompt.** The text is compared as the Gemma 4 template renders it: each content part trimmed, and the
  parts joined by one space (a string is one part). Whitespace around the text therefore passes, because the server's
  prompt does not change; any other edit fails. `--system-prompt auto` (default) FAILs a system prompt your reference
  never sent when the reference has a single system text, and WARNs when the reference's system text varies (for
  example per task). `same` always FAILs a new text. `any` skips the text, for a deliberate change of agent
  configuration; the number and position of system messages are still checked.

### The eight checks

| Check | Expectation | Why it matters |
|---|---|---|
| `role` | every occurrence of every tool result, in every request, has role `tool`. Every assistant tool call has a tool-role result carrying its id in the same request (without ids: at least as many tool results as calls), and each result directly follows the assistant message holding its call (only other results in between). No user or system message carries a `tool_call_id`. No tool result leaves the history between two consecutive requests unless a compaction happened in between (a result that comes back with the same call and content under a new id is not lost). Every tool call a reply made comes back as an assistant tool call in a later request of its trajectory (matched by id, or by tool name and arguments), under the name the reply gave it, unless a compaction happened right after that reply. When the replies keep calling tools, at least one request carries a tool-role result | google-adk switches to `tool_responses` when the LiteLLM model string contains `gemma4`; the Gemma 4 template then renders a separate turn instead of an inline tool response. A proxy that rewrites only older (history) results, or only the newest, as `user` text, flattens history calls into text, or merges parallel results is caught too. So is a full switch to text-mode tool calling (calls as assistant text, results as user text, tools still sent) and a client that drops `tool_calls`. The Gemma 4 template renders a call's results only from the tool messages right after it, so a message between a call and its results, or results moved before their call, removes them from the prompt. None of this needs `usage` in the capture |
| `sampling` | every request field equals your reference (or profile). A field your reference never sends FAILs when it changes sampling, the prompt or the parsed answer on vLLM 0.19.1 (for example `top_k`, `min_p`, `stop`, `logit_bias`, `tool_choice`, `include_reasoning`, chat-template arguments). It WARNs when it has no effect there (`seed`, `stream`, `user`, a value equal to vLLM's request default, a field vLLM ignores) | sampling and thinking change behaviour |
| `tools` | every agent request carries the reference tool list (canonical-JSON sha256 per tool and for the ordered list; field-level diff when the reference keeps schemas). A request without tools is a mismatch | a changed docstring, parameter or tool order, or a dropped tool list, changes the prompt |
| `shape` | message fields per role, tool_call_id on tool results and tools on every call match the reference. **Role skeleton:** the number and position of system messages, the first role, and every role the reference sends in all requests; a `developer` message at position 0 where the reference sends `system` WARNs (the Gemma 4 template renders both as the system turn). **After a tool result:** google-adk sends the model's next turn there, so a user, system or developer message there is judged unless your reference's own harness puts that role there. A user message there is expected when the reply before it was empty or thought-only (google-adk leaves such a reply out of the history, and the harness's continuation nudge then follows the tool result), or, when the reply before it is not in the capture, when its text is one of the follow-up texts your reference sends. Other such messages, whatever their text, FAIL when they were inserted two or more times after a reply the history should show, and WARN when inserted once or when the reply before them is not in the capture (a compaction summary does not count). **System text:** the system prompt's sha256 after the template's trim of each part (see `--system-prompt`). **Tool-result encoding:** JSON or text, the JSON wrapper (google-adk's `{"result": ...}` or the tool's own object), and the serialisation (separators, indent, `ensure_ascii`, key order) that reproduces the text. **Thought keys:** a combination the reference never sends FAILs when the server reads none of the keys sent (vLLM 0.19.1 reads only `reasoning`) or when the two keys carry different text; any other new combination leaves the prompt unchanged and WARNs. Content-form differences (string vs parts) WARN | the template check re-renders the request the server received, so it cannot see a client that drops, renames, folds or re-serialises content before the server: a system prompt folded into the first user message, an injected system message, a "Continue." nudge after the tool results in every request, or re-serialised tool results all change the prompt and still render consistently |
| `template` | the server's prompt token ids (or, without ids, `prompt_tokens`) equal a rendering of the same request through vLLM 0.19.1's message parse path with the reference chat template and tokenizer. On mismatch it reports the first differing token, the text around it, and which known variant reproduces the server | catches a different or patched chat template on your server |
| `thoughts` | history thoughts reach the server under the key it reads, for exactly the tool-calling assistant turns after the last user message (the Gemma 4 rule). A thought sent only under a key the server ignores, or two thought keys that disagree, FAIL. Thoughts copied into content are flagged. With thinking on, at least one reply must carry a thought field (else the server has no reasoning parser). With server ids, the thoughts the server actually rendered are checked too | the most common local stack loses every history thought (next section) |
| `newline` | each history thought the server renders equals the reply that produced it, byte for byte, so its close form (`\n<channel|>` or `\n\n<channel|>`) is the scorer's | a client that trims or pads thoughts changes one token per thought |
| `compaction` | google-adk's token-threshold compaction replayed with the observed `usage.prompt_tokens`: threshold 14,336, retention 5. Summaries must appear before exactly the predicted calls, and each request must keep exactly the predicted raw call/result pairs. On mismatch: the threshold interval and retention that the observations imply. A trajectory that cannot be replayed to the end gives WARN | a local run without the scorer's compaction config (or with other values) sees longer or shorter histories |

Pairing a history message with the reply that produced it uses the turn's content (tool names, arguments, answer
text), aligned from the newest message backwards. Repeated or missing tool-call ids therefore do not mis-pair turns.

### History thoughts: the key vLLM reads (upstream sources)

**VERIFIED by reading the public vLLM 0.19.1 wheel** (`vllm/entrypoints/chat_utils.py`, equal to the v0.19.1 tag).
- vLLM 0.19.1 reads a history thought only from the `reasoning` field of an assistant message (chat_utils.py:1512).
  It copies that text into both `reasoning` and `reasoning_content` for the template (:1537-1541).
- A `reasoning_content` sent by the client is never read.
- Its other preprocessing, which the kit's rendering and mock server follow:
  - `tool_call_id` is kept only for role `tool` (:1542-1545);
  - content `None` becomes no parts (:1514-1515);
  - the "string" content format joins text parts with `\n` (:1412);
  - empty `tool_calls` are dropped (:1567-1570);
  - tool-call arguments are JSON-decoded (:1572-1578).

**Clients.** google-adk 1.36.1's LiteLLM wrapper writes a history thought only as `reasoning_content` (lite_llm.py:934-940).
A plain google-adk + LiteLLM stack in front of vLLM 0.19.1 therefore loses every history thought, without any error.
The scorer's harness copies the thought into `reasoning` as well. In our scorer-path dumps, 2,081 of 2,081 assistant
history messages with a thought (counted per request) carry both keys with equal text. The kit's live demo does the same (`RunConfig.mirror_reasoning`, on by default).
The mutation `thought-key-reasoning-content-only` is the plain client.

**Later vLLM.** vLLM issue #38488 ("`reasoning_content` silently dropped on incoming assistant messages") reported the
behaviour. vLLM PR #42664 ("Normalize reasoning_content to reasoning for client compatibility") renames an incoming
`reasoning_content` to `reasoning` when `reasoning` is absent; `reasoning` wins when both are present. Its code is in
`protocol.py` at tag v0.22.0 and not at v0.21.0 (VERIFIED by reading both tags). For such a server, pass
`--server-thought-keys reasoning,reasoning_content`.

### The role switch, by google-adk version

What google-adk's LiteLLM wrapper sends as the tool-result role. **VERIFIED** by reading `src/google/adk/models/lite_llm.py` at every
release tag from 1.30.0 to 2.14.0, the 1.34.3 wheel (no tag), and the main branch on 2026-11-09. The installed 1.36.1
was also probed live (`drydock-eval conform adk-roles`). Only 1.36.1 was executed; the other rows come from static reading.

| google-adk | Rule | `openai//kaggle/input/models/google/gemma-4/...` | `hosted_vllm/google/gemma-4-e4b-it` | `openai/gemma4-e4b-it` |
|---|---|---|---|---|
| 1.30.0 - 1.33.0 | no switch | tool | tool | tool |
| 1.34.0 - 1.39.1, 2.0.0 - 2.4.0 | `"gemma4" in model.lower()` (lite_llm.py:814 in 1.34.0-1.39.1) | tool | tool | **tool_responses** |
| 2.5.0 - 2.11.0 | regex `gemma-?4`, any provider | **tool_responses** | **tool_responses** | **tool_responses** |
| 2.12.0 - 2.14.0 | regex, except providers openai, azure, lm_studio | tool | **tool_responses** | tool |
| main (commit b33b43e5) | as 2.12, plus vertex_ai | tool | **tool_responses** | tool |

The competition harness requires `google-adk>=1.34,<2.0` and ships 1.36.1. On the scorer path the model string has
the form `openai//kaggle/input/models/google/gemma-4/...`, so tool results go out as `tool`. Our own captures confirm
this. A local stack that serves the model under a path or name containing `gemma4` sends `tool_responses` (live demo
below). So does any 2.5-2.11 install, and 2.12+ with a non-OpenAI provider prefix, even with Kaggle's path.

**Upstream sources (cite; the behaviour is not our discovery).**
- google/adk-python PR #5655, "fix: use tool_responses role for gemma4 models in LiteLLM integration", closed
  2026-05-18. 1.34.0, released the same day, is the first version with the switch (the link is INFERRED from title
  and timing).
- Issue #6482 reported that it breaks LM Studio (closed 2026-10-07).
- Issue #7977 and PR #7978 cover Vertex AI Model Garden MaaS (both closed 2026-11-05). The change landed on main as
  b33b43e5 and is not in the v2.14.0 tag.

### History thoughts and the newline before `<channel|>` (upstream sources)

- **Thought retention.** The Gemma 4 prompt-formatting guide (ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)
  says thoughts from previous turns are stripped, but must not be removed between the function calls of one model
  turn. The Gemma 4 chat templates render a history thought only for tool-calling assistant turns after the last user
  message; Hugging Face google/gemma-4-E2B-it PR #44 discusses the consequence.
- **Newline.** Hugging Face google/gemma-4-31B-it discussion #135 reports an extra newline when thinking is rendered.
- **Our refinement (VERIFIED on our scorer-path captures).** vLLM keeps the model's own trailing newline in many
  thoughts, so the scorer renders `\n\n<channel|>` for those and `\n<channel|>` for the rest. On 24 captured
  trajectories, 1,186 of 1,985 visible history thoughts end in `\n\n<channel|>` (both counts are per request). The
  two forms tokenize to the same length with the Gemma tokenizer and differ in one token id. A token-count comparison
  therefore cannot see this drift. The `newline` check (client side) and an id comparison (server side) can.

### What a PASS does not cover

- **Server-side settings.** The kit sees the wire, not what the server applies by itself. In vLLM 0.19.1 these
  settings are invisible on the wire:
  - `--generation-config` and `--override-generation-config`: sampling defaults for every field a request leaves out
    (`config/model.py:1391-1411`);
  - `--default-chat-template-kwargs`;
  - `--chat-template` and `--chat-template-content-format`;
  - the reasoning and tool-call parsers.

  The template check sees the chat-template part through the prompt ids. Nothing on the wire shows sampling defaults.
- **A proxy after the recorder.** Anything between the recorder and the server is invisible. Record at the last hop.

### Live CPU demo (real google-adk, no GPU, no model)

```bash
drydock-eval conform demo            # needs the [adk] extra; --out-dir keeps the captures
```

A real `LlmAgent` with `LiteLlm` (google-adk 1.36.1, litellm 1.82.4) talks HTTP through the recorder to a scripted
mock server.
- **The mock.** It plays the model's moves on an invented toy repository. It follows vLLM 0.19.1's message parse path
  and reports honest `prompt_tokens` by rendering each request with a toy template.
- **The client.** The agent config puts the competition harness's own defaults for an agent that sets no generation
  config on the wire: `max_completion_tokens` 16,384, `chat_template_kwargs.enable_thinking` true and
  `thinking_token_budget` 4,096, with no temperature, top_p or top_k (VERIFIED by running the harness wheel's
  generation-config code locally). They are not a recommendation; set `RunConfig` to your own agent's values. A small
  LiteLLM client wrapper copies each history thought into `reasoning`, as the scorer's harness does.

- **Role.** The same run with a Kaggle-style input path sends 6 of 6 tool results as `tool`. With `openai/gemma4-e4b-it`
  it sends 6 of 6 as `tool_responses`, rendered as a separate `<|turn>tool_responses` turn (VERIFIED, test
  `test_role_switch_end_to_end`). With your own template and tokenizer (`--template ... --tokenizer ...`), the demo
  also prints real token counts for both roles. On one of our scorer-path requests, the Gemma 4 E4B template gives
  3,508 tokens with `tool` and 3,506 with `tool_responses`.
- **Compaction.** With the scorer's config, the first summary call comes before the call whose previous prompt is the
  first at or above 14,336 tokens, and that request keeps 3 call/result pairs: PASS. Without compaction the check
  reports the call where a summary was expected: FAIL. With threshold 8,192 it reports the call where a summary
  appeared although the rule does not fire, plus the implied threshold interval: FAIL.

### Measured detection: `drydock-eval conform mutate`

The mutation suite injects each drift into **live** google-adk runs, at the layer where it happens in practice:
- **config:** the agent's model string, compaction config, sampling, or a plain client without the reasoning mirror;
- **proxy:** a request edit between client and server, as a different client version or a router would make;
- **server:** a documented variant of the toy chat template, with and without the server returning prompt token ids,
  or a server that returns thoughts inside the answer (no reasoning parser).

How the suite runs:
- Every mutated run is a fresh run, so the client reacts to the usage it receives. No capture is edited afterwards.
- The reference for the tools, sampling and shape checks is derived from the scenario's baseline capture, as a user
  derives it from their own harness.
- "No effect" means that the requests, the usage and the prompts the server rendered are identical to the baseline.

Four synthetic scenarios cover different patterns:
- `short`: no compaction;
- `long`: compaction during the run;
- `nudge`: a text answer, a second user message, and compaction at the end of an invocation;
- `parallel`: two tool calls per turn.

Results, VERIFIED (`drydock-eval conform mutate --jobs 4`, 416 live runs: 4 baselines and 103 mutations x 4
scenarios; 85 mutations are drifts and 18 are benign controls). The suite includes, under their own names, the 26
drifts an independent AI review of version 0.2.0 built (`tk-*`, `sh-*`, `ro-*`, `sv-*`), and the drifts added with the
fixes for them. It also includes the drifts and controls from the first review of 0.3.0 (text-mode tool calling, a
user message after the tool results, whitespace around the system text that the template trims) and from the second
(the harness's own nudge after an empty or thought-only reply, sparse and one-off insertions after the tool results,
message-order edits that make the Gemma 4 template drop tool results, renamed history calls, and tool-call ids
regenerated in every request).

| Drift class | Mutations | Applicable runs | Detected by the intended check | No effect |
|---|---|---|---|---|
| role | model string contains `gemma4`; only older (history) tool results renamed; history results sent with legacy role `function` | 12 | 12 | 0 |
| role pairing | history tool results sent as user text, or as `user` with the id kept; history calls flattened into text; only the newest results sent as `user`; parallel results merged into one message | 17 | 17 | 3 |
| template, server returns ids | 6 template variants; the server runs with `--chat-template-content-format string` | 24 | 24 | 4 |
| template, counts only | the same 6 variants | 21 | 13 | 3 |
| thought key | plain google-adk + LiteLLM (thoughts only as `reasoning_content`); the two keys disagree; LiteLLM `thinking_blocks`; a `thinking` content part; `reasoning` empty or null next to `reasoning_content`; only the newest turn mirrored; a plain client on a vLLM 0.22 server, judged by the scorer's 0.19.1 rule | 32 | 32 | 0 |
| thoughts | strip all, keep latest only, move into content, server without reasoning parser, the first 25 characters of each thought copied into content | 20 | 20 | 0 |
| newline | trim, append, both keys cut to 20 characters | 12 | 12 | 0 |
| compaction | off, threshold 8,192 / 20,000, retention 3 / 7, a client-side window that drops the oldest call/result pair | 19 | 19 | 5 |
| sampling | temperature 1.0 / 0.6 or top_p 1.0 set (the baseline sends neither), max tokens 4,096 (baseline 16,384), thinking off, budget 2,048 / none, added `top_k` 64, `min_p` 0.05, `repetition_penalty` 1.1, a stop sequence, `tool_choice` "required"; the client omits `chat_template_kwargs` and the server defaults thinking off | 52 | 52 | 0 |
| tools | description edited, parameter added, tool removed, order swapped, no tools after the first request | 20 | 20 | 0 |
| role skeleton | a proxy injects a second system message; the system message folded into the first user message, sent as `developer` (WARN, see below), or moved behind the first user message; a "Continue." user message after the newest tool results of every request, after every group of results, or in every third request; a system message after the newest results; one "Continue." in one request only (WARN by design: a single insertion) | 36 | 36 | 0 |
| role order | a "Continue." user message or an empty assistant message between the newest call and its results; the newest results moved before their call; history calls renamed `functions.<name>` | 16 | 16 | 0 |
| text-mode tool calling | every call sent as assistant text and every result as user text, tools still sent; assistant turns lose `tool_calls` while the results stay | 8 | 8 | 0 |
| system text | a sentence appended to the system prompt by a proxy; the agent's instruction reworded | 8 | 8 | 0 |
| tool-result encoding | JSON re-serialised (indent 1 and `ensure_ascii`; compact separators); `{"result": x}` unwrapped to text or re-wrapped as `{"output": x}` | 16 | 16 | 0 |
| message fields | tool results carry a `name`; a tool result split into two text parts (WARN); content `""` instead of none on tool-calling turns (WARN) | 12 | 12 | 0 |
| **total** | 85 drift mutations | **325** | **317** | 15 |

- **The 8 misses** are the two same-length server-template changes, seen through token counts only: a trimmed
  history-thought newline and a doubled newline at the end of the system turn. This is the blind spot described
  above. With ids, all 8 are detected. 265 applicable runs changed the prompt the toy server rendered; 257 were
  detected by the intended check. Twelve runs changed the prompt without any FAIL: these 8, and the 4 runs of the
  system message sent as `developer`, which `shape` reports as a WARN. The Gemma 4 template renders a `developer`
  message at position 0 as the system turn (0 of 183 real prompts change), while the toy template does not, so the
  suite counts that WARN as detected.
- **Without `usage`.** The text-mode and message-order drifts fail `role`, and the repeated "Continue." and system
  messages fail `shape`, without any usage data. In the live suite some also fail `compaction`, because the mock
  reports usage; the unit tests and the real-data probes below remove usage and still get the FAIL. A single
  inserted message gives only a `shape` WARN without usage.
- **The review's 26 drifts.** Against 0.2.0, 80 of their runs changed the server prompt and 14 of those were missed
  by every check (system prompt folded into the user message, tool-result JSON re-serialised or unwrapped, an
  injected system message). Now all 80 are detected by the intended check. Four of the 26 change the request but not
  the toy prompt (`sh-tool-name-field`, `sh-tool-content-two-parts`, `sh-assistant-content-empty`,
  `sv-vllm022-plain-client`). They are detected too: the split into two text parts and the empty content by a shape
  WARN (as the suite expects for changes that render the same), the other two by a FAIL.
- **No-effect cases.** These are drifts that cannot change anything in that scenario: compaction variants in `short`;
  "keep all thoughts" where no thought is outside the rule; merging parallel results where every turn has one call;
  the `string` content format in `short`, which has no message with two or more text parts.
- **Also flagged by a second check (104 runs).**
  - Role drifts also fail `shape`; partial switches to `user`, text-mode tool calling, a user message between a call
    and its results, and the "Continue." messages also fail `compaction`, whose replay then sees other histories.
  - An injected system message also reaches google-adk's summarizer call, which is then counted as an agent call: in
    the three scenarios with compaction, `role`, `sampling`, `tools` and `compaction` fail too. `shape` names the
    cause.
  - Template variants that show or hide thoughts also fail `thoughts`, which sees them in the server's ids.
  - Thought-key drifts also fail `newline` and `shape`.
  - Dropping the tool list also fails `shape`; a client-side window also fails `role` (tool results leave the history
    without a compaction).
- **False alarms: 0 FAIL.**
  - No FAIL appears in the 32 baseline check results (4 scenarios x 8 checks).
  - None of the 72 benign control runs raises one. The controls: reversed JSON key order; `max_tokens` alias (WARN);
    `stream: false` (WARN); an OpenAI `user` field (WARN); a task text that starts with "For context:"; an HTTP 500
    that the client retries; tool-call ids repeated on every turn; no tool-call ids from the server;
    `event_retention_size` 0, checked with a profile that says so; a client that sends only `reasoning`, the key vLLM
    0.19.1 reads (shape WARN); a plain client on a vLLM 0.22 server, checked with
    `--server-thought-keys reasoning,reasoning_content` (shape WARN); the agent's instruction reworded, checked with
    `--system-prompt any`; a newline after the system text; a space before it; the system text as one text part with
    whitespace around it (shape WARN for the content form); the model's turn-ending answer arriving as a thought
    only, or empty, so that google-adk drops it and the harness's nudge follows a tool result; tool-call ids
    regenerated in every request, with the arguments sent compact. The three system-text controls leave the
    server's prompt unchanged, because the template trims the system text. The two turn-end controls change the run
    only in `nudge`, the one fixture with a turn-ending answer; there `shape` explains the nudge by the dropped reply.
  - 24 of the 72 control runs carry a WARN, all of them expected: the three sampling controls, the two thought-key
    controls and the one-part system text, in every scenario.
  - "Clean" means no FAIL, not an unchanged prompt. Reversed key order changes the toy prompt, because the toy
    template serialises tools in insertion order; the real Gemma 4 template sorts keys (`dictsort`).
- **Determinism.** Two full runs gave identical outcomes, failing checks, warnings and effect flags in all 412 mutated
  runs. They took 31 s and 93 s of suite time on 16 cores under a load average of about 10 and 17-21 (other work was
  running).
- **Earlier versions.** Version 0.1.0 had 33 drift mutations and 4 controls: 117 of 121 detected, 0 false alarms. It
  had no thought-key drift, because its mock read either thought key. Version 0.2.0 had 44 drift mutations and 9
  controls: 157 of 165 detected, 0 false alarms, but it had none of the review's content-level drifts. The first
  0.3.0 build had 74 drift mutations and 12 controls: 273 of 281 detected, 0 false alarms among its own controls. An
  independent AI review then found two problems in the checks (and a wrong claim about the graph bands, corrected
  below). The build FAILed whitespace around the system text that the template trims (8 of 8 live runs; on real data
  0 of 183 prompts changed). Text-mode tool calling and a "Continue." message after the tool results failed only
  `compaction`, so a capture without usage passed. Both are fixed, and the review's cases are now in the suite. The
  344 mutated runs the earlier build had give the same results as before. Version 0.3.0 had 78 drift mutations and
  15 controls: 289 of 297 detected, 0 false alarms among its own controls. A second independent AI review then ran the
  competition harness's own continuation nudge end to end with google-adk: when the model's turn-ending answer is
  empty or thought-only, google-adk drops it, the nudge follows a tool result, and 0.3.0's rule FAILed `shape`. That
  is fixed in 0.3.1 and in the suite as two controls. Of the 372 mutated runs 0.3.0 had, 368 give the same results;
  the 4 runs of the system message sent as `developer` now WARN instead of FAIL. A third review, of the first 0.3.1
  build, found that `shape` accepted your reference's follow-up text after a tool result even when the reply before
  it was visible: a client that drops the model's visible answers, or a proxy that sends the harness's own nudge text
  after the results, passed without usage (on real data, below). A follow-up text now explains such a message only
  where the reply before it is not in the capture. This build also switched the demo's request fields to the
  harness's defaults and gave the toy tokenizer ids that any instance can decode. Each of these changes left all 412
  mutated runs with the same results as the first 0.3.1 build. The toy tokenizer is now version 3
  (`drydock-eval-toy-tokenizer-v3`): a word's id holds the word's own bytes, so a check without usage no longer
  reports hidden thoughts because a fresh tokenizer could not decode the server's ids. Toy captures saved by earlier
  versions carry version 2 ids and FAIL `template` and `thoughts` now; run the demo again to compare.

### Validation on scorer-path captures (not distributed)

We ran `drydock-eval conform check` on our own captures of the scorer path. These were Kaggle-identical E4B prompts
from a scorer replica with google-adk 1.36.1 and vLLM 0.19.1. They contain task text, so they stay private. All
results below are VERIFIED.
- **24 trajectories, 381 requests (client-call dumps), with a TOML sampling profile of the submission that produced
  them (not shipped).**
  - role: PASS, 2,076 tool-result occurrences (335 distinct), all `tool`.
  - template: 381 of 381 server prompt counts equal the hash-verified Gemma 4 E4B template rendering. These are
    counts only; the report says so.
  - thoughts: PASS, 334 distinct thoughts, 1,985 sent across requests, all under `reasoning`.
  - newline: PASS, all echoed verbatim.
  - compaction: 24 of 24 trajectories follow the rule. 11 compact, and the 15 summary calls fall exactly where the
    replay predicts. The implied threshold interval is (14,289, 14,425], which contains 14,336.
  - sampling: WARN only, because a client-call dump does not record `top_p` and records `max_tokens` rather than the
    wire's `max_completion_tokens`.
- **Held-out reference (false alarms on real data).** We made a reference from 12 of the 24 trajectories and checked
  the other 12 (193 requests): all eight checks PASS, in both directions, with the role skeleton, system text and
  tool-result encoding compared. The 12 trajectories are 12 different tasks, and their reference holds one system
  text, one skeleton (one system message, first) and 1,098 tool results, all JSON objects in one serialisation (no
  `{"result": ...}` wrapper on this path).
  - Re-checked with the current version, which also compares the system text after the template's trim, what follows
    a tool result, and whether every call a reply made comes back as a tool call. It holds for that split and for a
    second split into alternate files, in both directions: 8 of 8 PASS each time, on 178-188 agent requests. Between
    156 and 164 reply tool calls were followed into later requests each time, and none failed to come back.
  - With usage removed, `template` and `compaction` SKIP and the other six PASS. No reference of either split has a
    user message directly after a tool result.
- **1 trajectory with vLLM-logged prompt ids (11 requests), with that reference.**
  - All eight checks PASS, and 11 of 11 prompts are equal by token ids.
  - For all 55 checked thought occurrences, the server prompt shows a thought exactly where the rule expects one.
- **Probes on those 11 real requests.**
  - Dropping `reasoning` from every history thought (what plain google-adk + LiteLLM sends) gives FAIL from
    thoughts, newline, shape and template: 55 of 55 thoughts were sent under a key vLLM 0.19.1 does not read.
  - Making the two keys disagree gives FAIL from thoughts, newline and shape. Template still PASSes, because the
    server reads the unchanged key.
  - Trimming the history-thought newline leaves all 11 counts unchanged, but changes the ids in 8 of 11 prompts.
  - Renaming the tool role changes 10 of 11 prompts, by 2 tokens per tool result.
- **Content-level probes on those 11 requests.** Each edit is applied to the requests, and the server's prompt is
  emulated by rendering the edited request with the real template and tokenizer (vLLM 0.19.1 parse path). Reference:
  the 12-trajectory one above.

  | Edit | Prompts changed | Token change | Checks that FAIL |
  |---|---|---|---|
  | system prompt folded into the first user message | 11/11 | 0 | shape |
  | a second, injected system message | 11/11 | +11 | shape |
  | a sentence appended to the system prompt | 11/11 | +3 | shape |
  | tool-result JSON with compact separators | 10/11 | down to -89 | shape |
  | tool-result JSON with indent 2 | 10/11 | up to +130 | shape |
  | tool results wrapped as `{"result": ...}` | 10/11 | up to +30 | shape |
  | tool-result JSON with `ensure_ascii` | 0/11 (all ASCII) | 0 | none (no effect) |
  | history tool results sent as user text | 9/11 | down to -1,770 | role, compaction |
  | only the newest tool results sent as user text | 10/11 | down to -1,964 | role, compaction |
  | history calls flattened into text | 9/11 | down to -1,785 | role, compaction |
  | control: only `reasoning` sent | 0/11 | 0 | none (shape WARN) |
  | control: plain client on vLLM 0.22, declared | 0/11 | 0 | none (shape WARN) |

  Version 0.2.0, run on the same edits, passed all eight checks on the first six, gave only a compaction FAIL on the
  three role drifts, and FAILed shape on both controls (false alarms). On the 12 held-out trajectories, where some tool results contain non-ASCII text, `ensure_ascii` changes 36
  tool-result occurrences and shape FAILs.
- **Probes for the review of 0.3.0, on the alternate split.** The reference comes from 12 trajectories, and each
  edit is applied to the 183 requests of the other 12. The server is emulated as above (real template and tokenizer,
  vLLM 0.19.1 parse path). "Without usage" removes `usage` from every reply.

  | Edit | Prompts changed | Token change | Checks that FAIL | Without usage |
  |---|---|---|---|---|
  | control: a newline after the system text | 0/183 | 0 | none (first 0.3.0 build: shape) | not run |
  | control: a space before the system text | 0/183 | 0 | none (first 0.3.0 build: shape) | not run |
  | control: the system text as one part, whitespace around it | 0/183 | 0 | none (shape WARN) | not run |
  | two spaces for one inside the system text | 178/183 | 0 to +1 | shape | not run |
  | text-mode tool calling (calls as assistant text, results as user text) | 166/183 | down to -5,876 | role, compaction | role |
  | assistant turns lose `tool_calls`, results stay | 166/183 | down to -11,814 | role, compaction | role |
  | "Continue." after the newest tool results | 161/183 | down to -5,862 | shape, compaction | shape |
  | "Continue." after every group of tool results | 166/183 | -5,742 to +80 | shape, compaction | shape |
  | no edit | 0/183 | 0 | none | none |

  The first 0.3.0 build gave `role` SKIP and `shape` PASS on the text-mode and "Continue." edits, and an overall
  PASS once usage was removed. The prompt counts include the 5 summarizer requests, whose prompts these edits leave
  unchanged.
- **Probes for the review of 0.3.1, on the same alternate split and setup.**

  | Edit | Prompts changed | Token change | Checks that FAIL | Without usage |
  |---|---|---|---|---|
  | control: tool-call ids regenerated in every request, arguments sent compact | 0/183 | 0 | none (0.3.0: role, compaction) | not run |
  | control: the system message sent as `developer` | 0/183 | 0 | none (shape WARN) | not run |
  | control: `reasoning_content` edited, `reasoning` verbatim | 0/183 | 0 | thoughts, newline, shape (by design: the two keys disagree) | not run |
  | a "Continue." user message between the newest call and its results | 166/183 | down to -5,995 | role, compaction | role |
  | an empty assistant message between the newest call and its results | 166/183 | down to -2,244 | role, compaction (shape WARN) | role |
  | the newest results moved before their call | 14/183; the real template raises on 152 more | down to -1,005 | role; template (it cannot render those requests) | role |
  | history calls renamed `functions.<name>` | 166/183 | 0 to +64 | role | role |
  | a system message after the newest results | 161/183 | 0 to +10 | shape | shape |
  | "Continue." after the newest results in every third request | 49/183 | down to -5,618 | shape, compaction | shape |
  | one "Continue." in the whole capture | 1/183 | -466 | compaction (shape WARN) | none (shape WARN) |

  The held-out baselines still give 8 of 8 PASS on both splits in both directions (with usage; without usage
  `template` and `compaction` SKIP and the rest PASS). Three more captured runs of the same 24 tasks (360, 420 and 377
  calls), checked against a reference from all 24 trajectories of the first run, give no FAIL and no WARN.
  None of these 96 trajectories has a user message directly after a tool result. The harness's nudge after an
  empty or thought-only reply was therefore tested end to end against the mock (google-adk 1.36.1, the `nudge`
  scenario): with a reference from that scenario, from all four, or from `short` alone (which holds no nudge text),
  `shape` explains all 5 such messages by the dropped reply, with and without usage, and no check FAILs except
  `sampling` against the `short` reference, which has no compaction-summarizer calls to compare with; 0.3.0 FAILed
  `shape` there.
- **Probes for the review of the first 0.3.1 build, on the same alternate split and setup.** The reference half holds
  one follow-up text of the harness (its nudge after a text answer); the second and third edits insert that text.

  | Edit | Prompts changed | Token change | Checks that FAIL | Without usage |
  |---|---|---|---|---|
  | a client drops the model's visible text-only answers (the harness's nudge then follows a tool result) | 11/183 | -381 to -84 | shape, compaction (first 0.3.1 build: compaction) | shape (first 0.3.1 build: none) |
  | the harness's own follow-up text after the newest results of every request | 161/183 | -5,841 to -20 | shape, compaction (first 0.3.1 build: compaction) | shape (first 0.3.1 build: none) |
  | the same text after the newest results, once per trajectory | 12/183 | -674 to -89 | shape, compaction (first 0.3.1 build: compaction) | shape (first 0.3.1 build: none) |

  With the reference from the other half, all three also FAIL `shape` with and without usage (the first 0.3.1 build
  passed the dropped answers there even with usage). The baselines give no FAIL and no WARN in either direction, with
  and without usage. On the mock's `nudge` scenario, google-adk compacts the history between the model's visible
  answer and the nudge; a client that drops that answer still passes there, because the compaction hides the reply
  from the check and the nudge's text is the reference's follow-up text (see Limits).

The compaction replay also matches google-adk 1.36.1's own token-threshold function at 2,908 of 2,908 random
decision points, 971 of them compactions (test `test_compaction_replay_matches_google_adk`). The cases use retention
0-7, 0-3 calls per turn, text-only turns, nudges, and summaries whose own usage exceeds the threshold. One ADK 1.36.1
detail matters here (VERIFIED by code reading and a live run). The end-of-invocation compaction check runs even after
a compaction inside the invocation, because the flag that should skip it is set on a copied invocation context. The
replay models that (`--post-invocation always`, the default).

### Tests

Release 0.4.0 collects 442 tests. On CPU, without Docker and without a GPU, `pytest` gave 435 passed and 7 skipped.
The skips are the five Docker tests of the contained runner (`tests/test_contain_docker.py`, skipped unless a
deployment is named), the opt-in live test below, and the opt-in comparison against the authors' original analysis
files (`tests/test_compare_crosscheck.py`, which passes when the two files are supplied). Per module:
- conform and the graph check: 158 tests in eight files, described below;
- the release check: 72 tests (`tests/test_release_check.py`);
- contain: 148 tests without Docker (`tests/test_contain_lib.py` 86, `tests/test_contain_gate_cli.py` 62) and 5
  with Docker;
- compare: 59 tests (`tests/test_compare_units.py` 33, `tests/test_compare_crosscheck.py` 26).

The whole suite took about 95 s on an idle 16-core workstation. An sdist and a wheel of 0.4.0 built offline pass the release
check (the sdist is written with empty owner and group names and a fixed time, and the release check refuses account names in tar members). With the wheel installed offline into a fresh Python 3.12 environment that sees numpy, jinja2, pytest and
google-adk, the unpacked sdist's tests give the same 435 passed and 7 skipped, and `drydock-eval contain sources`
reports the packaged reference tree equal to its manifest. The Docker tests have not been run for this release
candidate; see the contained-runner section for what they cover and how to run them.

The rest of this subsection describes the conform tests as measured for release 0.3.1, which had 186 tests on CPU
(plus one opt-in):
- 181 unit tests that need no google-adk:
  - 21 regression tests for the first independent AI review's findings (`tests/test_conform_review_fixes.py`);
  - 25 for the second review's: role skeleton, system text, tool-result encoding, partial role switches, thought keys
    judged by the server's key, partial thought leaks, server settings in the mock
    (`tests/test_conform_content_drifts.py`);
  - 22 for the third review's: whitespace around the system text passes while real edits fail, text-mode tool
    calling and dropped `tool_calls` fail `role` without usage, a user message after the tool results fails `shape`
    without usage (`tests/test_conform_trim_and_text_mode.py`);
  - 20 for the fourth review's (0.3.1): the harness's nudge after an empty or thought-only reply passes (with and
    without usage, also after a compaction), a follow-up text from the reference explains a message whose reply is
    not captured but not one after a visible reply (also when a client drops the model's visible answer), one
    insertion WARNs and two FAIL, results separated from their call and renamed calls fail `role`,
    tool-call ids regenerated per request pass `role` and `compaction`, a `developer` system message WARNs, and the
    `newline` wording when only the thought keys disagree (`tests/test_conform_after_tool_and_order.py`);
  - 24 tests of the graph check on synthetic graphs and arrays, including the two-sided bands
    (`tests/test_graph_check.py`);
  - 29 tests of the release check, including an unpacked sdist and the 0.3.1 rules (`tests/test_release_check.py`;
    72 in release 0.4.0);
- 1 oracle test against google-adk;
- 4 live tests: role switch, compaction demo, the full mutation set on the `nudge` fixture against its expected
  outcome table, and the plain google-adk + LiteLLM client end to end.

Under a load average of 16-17 on 16 cores (other work was running), the suite took 68 s. Without google-adk, the
oracle and live tests are skipped. An sdist built offline holds the package, the tests and `scripts/`. With the wheel
installed offline into a fresh Python 3.14 environment without google-adk, the unpacked sdist's tests give 181 passed
and 2 skipped (the oracle and live tests). A Python 3.12 environment with jinja2 3.1.2 passes a smoke test of the
0.3.1 behaviours. The release check finds nothing in the sdist, the wheel or the unpacked tree.
`DRYDOCK_SLOW=1 pytest` adds one test that checks the expected outcome of all 412 mutated runs on all four fixtures
(the five live tests and the oracle test took 126 s under a load average of 14-17). The full four-scenario suite is
`drydock-eval conform mutate --jobs 4`.

### Limits (read before relying on a PASS)

- **Counts versus ids.** A template PASS on lengths alone cannot rule out one-token substitutions or same-length
  whitespace changes; the report labels such a PASS "same LENGTH". The mutation suite shows the gap on the toy
  template. With the real E4B template and tokenizer, an independent AI review found that count-only comparison missed
  3 of 6 whitespace-only server-template changes that ids caught. Use `--request-token-ids`, which asks vLLM for
  `return_token_ids`. vLLM 0.19.1 then returns `prompt_token_ids` in the reply (`chat_completion/serving.py:1615-1617`,
  and in the first chunk when streaming), by static reading. It was tested here against the mock, not a real vLLM.
  The other route is `--vllm-log`.
- **Capture level.** At client-call level (imported dumps), fields added later in the HTTP stack cannot be judged
  absent; `sampling` reports them as WARN. A reference made from a client-call dump downgrades unknown-field FAILs to
  WARN for the same reason.
- **Thought key by server version.** The default rule is vLLM 0.19.1's (only `reasoning`). Other servers may read
  other keys; pass `--server-thought-keys` or `server_reasoning_keys` in a profile.
- **Content checks need a matching reference.** The role skeleton, system text and tool-result encoding are judged
  against your reference. A reference made from one task cannot tell a per-task system prompt from a drift (the
  report says so); build it from several tasks or pass `--system-prompt any`. A reference made by version 0.2.0 has
  none of these fields, and shape then says that they were not compared; rebuild it with `ref`. A reference made by
  the first 0.3.0 build hashes the raw system text (whitespace around it then counts) and does not record what
  follows a tool result. A reference made by 0.3.0 does not record your harness's follow-up texts, and it counts the
  harness's nudge after an empty or thought-only reply as a role your harness puts after a tool result (which then
  lets every such message pass). Shape says when a reference is older than the current rules; `ref` rebuilds it.
- **What no check compares.** The kit compares the form of what is sent, not its task-specific content: the task
  text and other user texts (a proxy that prefixes the task changes the prompt and passes), the values inside tool
  results (a proxy that truncates strings in the JSON but keeps its serialisation passes; with usage, `compaction`
  may notice the shorter prompts), and the order of parallel tool results (reversed results pass). An extra
  assistant message after the tool results (for example "Noted.") has no rule of its own either. On the Gemma 4
  template it inserts `<turn|>` and a newline after `<tool_response|>`, before the model's next call, so even an
  empty one (content `""` or none) adds 2 tokens, and "Noted." adds 5; only `compaction` may notice the longer
  prompts when usage is present. The mirror case passes too: a client that keeps the empty or thought-only reply that
  google-adk drops sends `<tool_response|><turn|>` and a newline before the nudge's user turn, where google-adk sends
  `<tool_response|><|turn>user`, which is 2 tokens more per occurrence. On the mock, with content none no check FAILs
  or WARNs, even with usage; with content `""`, `shape` gives only a content-form WARN.
- **System text follows the Gemma 4 template.** It is compared with each part trimmed and the parts joined by one
  space, which is how that template renders a system message at position 0. A template that joins parts without a
  space (the toy template does) renders some re-splits of the system text the same; shape still FAILs those. A
  system message at a later position is rendered inside the Gemma 4 template's message loop, which does not trim or
  join parts with a space, so for such a message the kit's comparison can differ from the prompt in both directions
  (a re-split with spaces passes although the prompt changes; a split at no space fails although it does not).
- **Serialisation can only be seen where the text shows it.** The encoding check tries `ensure_ascii`, default and
  compact separators, indent none/1/2/4 and sorted keys. ASCII-only results cannot reveal `ensure_ascii`, and flat
  objects whose keys are already sorted cannot reveal `sort_keys`; such re-serialisations leave the prompt unchanged
  too. A serialiser none of these settings reproduces FAILs when your reference has one known form.
- **Call/result pairing assumes google-adk's history.** google-adk sends every history call with its result right
  after it, sends each call back as a native tool call under its own name in the next request, and only compaction
  removes them. A client that trims old calls itself, renames them, or sends calls as text, fails `role`, as it
  should for the scorer path. The rule assumes one agent per trajectory, as in the competition harness (INFERRED: a
  multi-agent setup whose sub-agents see other agents' calls as text was not tested). Tool-call ids may change from
  request to request: `role` then matches results by call and content, and `compaction` matches the retained calls
  by the reply that made them (tool name, arguments, answer text). Results whose content repeats exactly within a
  trajectory can still defeat that fallback.
- **What may follow a tool result.** google-adk puts the model's next turn there. The competition harness nudges
  only when a google-adk invocation ends, that is on a reply without a tool call, and its call and time limits end
  the task without a nudge (by reading its code). So its nudge follows a tool result only when that reply was empty
  or thought-only: google-adk 1.36.1 leaves such a reply out of the history (VERIFIED end to end against the mock).
  Such a nudge passes when the capture holds the empty or thought-only reply. A role your own harness puts after a
  tool result in other cases passes when your reference has it there. Anything else after a reply the capture shows,
  including your reference's own follow-up text, WARNs once and FAILs from the second insertion on: a proxy that
  inserts one message in one request only gets a WARN. Where the reply before such a message is not in the capture
  (a capture that starts mid-conversation), it passes when its text is a follow-up text your reference sends and
  WARNs otherwise. A compaction between the reply and the message hides the reply in the same way: on the mock's
  `nudge` scenario, a client that drops the model's visible answer before a compaction passes, because the nudge
  carries the reference's follow-up text.
- **Grouping.** Sessions are separated by turn continuity or by `/s/<label>/` URL prefixes. Label concurrent
  sessions. On four of our real captures (12 trajectories each, interleaved request by request without labels), two
  were grouped correctly at 2, 4 and 12 sessions at a time; one was mis-grouped from 4 at a time and one already at
  2. `role`, `newline` and `compaction` then FAIL on a correct stack (first found by the independent AI review of
  0.3.0; 0.3.0 and 0.3.1 give the same verdicts). The request that follows an empty or thought-only reply cannot be
  linked to that reply by content, so without labels it joins the most recent session.
- **Scope of the compaction replay.** It models google-adk's token-threshold compaction (1.36.1 semantics,
  `post_invocation` configurable). It does not model the invocation-count sliding window, which needs 15 user
  invocations and so never fires within one agent run.
- **The mock is not a model.** The demo and mutation suite show what reaches the wire and how the checks react. They
  say nothing about solve rates. Measuring each drift's effect on E4B solve rates needs real model runs.
- **No reasoning parser.** The mock's "no reasoning parser" server puts `thought\n` + the thought into the answer. We
  INFER that from vLLM skipping special tokens by default; it was not observed on a real server.
- **ADK versions.** The table above comes from source reading. Only 1.36.1 was executed.

## Graph and embedding fidelity: `drydock-eval conform graph`

The competition's harness offers three graph tools: `search_similar_code`, `get_code_neighbors` and
`get_code_subgraph`. They read one code-graph JSON and one embedding NPZ per task. The scorer has official files for
every task. A local evaluator on new tasks must supply its own, and these are often stand-ins; Berth, for example,
ships Gaussian vectors. This check reads a task's files the way the harness does and reports where the tools would
behave unlike the scorer. It ships no graph or vector data. Its reference is a table of statistics measured on the
official files, which stay on your machine.

```bash
# one task, with the directories your evaluator uses
drydock-eval conform graph --graph-dir data/graphs --embeddings-dir data/embeddings \
    --repo owner/name --base-commit <sha> --tools search_similar_code,get_code_neighbors
# every task in a task file (repo, base_commit); --tools also accepts a reference made with 'conform ref'
drydock-eval conform graph --graph-dir ... --embeddings-dir ... --tasks tasks.jsonl --tools ref.json --jobs 6 --json graph.json
# two files directly; --against compares the neighbour lists with a second NPZ for the same graph
drydock-eval conform graph --graph g.json --embeddings standin.npz --against official.npz
# your own reference from your copy of the official files (statistics only; --per-file rows are for local use)
drydock-eval conform graph-ref --graph-dir .../graphs --embeddings-dir .../embeddings --out myref.json --jobs 6
```

The exit code is 1 when any check fails. The check needs only the standard library: NPZ files are read with
`zipfile` and `struct`.

### What the harness does with the files

**VERIFIED** by reading the competition's harness wheel 0.2.11 and by running its code next to this check (see
"Validation against the harness" below). The behaviour is re-implemented; no harness code is copied.
- **Prompt section** (`harness/agent_runner.py:336-337,416-466`). The first user message gets a "Code Intelligence
  Tools" section only when both of these hold:
  - at least one graph tool is declared, or the agent declares no tool list at all;
  - a graph file and an embedding file each exist under one of four names and are larger than 100 bytes. The names
    are `<repo>_<base_commit>`, `<owner>_<repo>_<base_commit>`, `<owner>_<repo>` and `<repo>`.
- **Files the tools load** (`graph/embedding_utils.py:53-90`). The tools take the first of six names that exists, with
  no size test. The six names add nested `<owner>/<repo>...` paths. So an empty file under one name can win over the
  valid file that earned the section.
- **One vector per node** (`graph/graph_utils.py:154-175,279-334`; `graph/embedding_utils.py:186-210,316-436`;
  `graph/retrieval_utils.py:89-124`).
  - An inline `embedding` in the graph JSON wins over the NPZ.
  - Otherwise the node id is looked up in the NPZ: the exact key, then the shortest key ending in `.<id>`, `;<id>` or
    `/<id>`, then a case-insensitive match. A node with no match gets a zero vector.
  - Vectors are padded or cut to the length of the first node's vector.
  - The NPZ is loaded in one go with `allow_pickle=False`, so a single object entry loses every vector.
- **`search_similar_code`** (`graph/retrieval_utils.py:771-912`; `tools/graph.py:198-288`). A query without whitespace
  is resolved to a node. Every other node is ranked by cosine, ties broken by name. The cosine is printed to the agent
  as `similarity`, to 4 decimals. A query with whitespace uses text matching and no vectors.

### The three checks

| Check | PASS | FAIL | WARN |
|---|---|---|---|
| `graph-section` | both files earn the section, and the tools load those same files | a graph tool is declared and the section is missing, or the tools would load an empty file instead | the declared tools are unknown (`--tools` not given) and the section is missing; or the tools read a different, non-empty file |
| `graph-align` | every node has a vector of its own: one length, finite, 1-D, and the NPZ loads with `allow_pickle=False` | unreadable files; an NPZ entry that loses the whole file; nodes that borrow another node's vector or get zeros; mixed lengths; NaN or inf | differences from the official format: unused or `<repo>;`-prefixed keys, a length other than 256, all-zero vectors, inline vectors, a `name` different from the id, duplicate or undeclared nodes |
| `graph-vectors` | each of five statistics lies inside its two-sided band around the official files | (never) | stand-in: any statistic outside its band. Bands from the official per-file range [min, max]: random-pair cosine mean, min - 0.10 to max + 0.10; random-pair cosine sd and the share of pairs at or above 0.99, min / 2 to max x 2; median top-10 similarity, distance from 1 at most 3 x the official maximum distance and at least a third of the minimum; edge AUC, min - 0.05 up to halfway between the official maximum and 1 |

`graph-section` gives SKIP when no graph tool is declared, because the scorer then adds no section either.

The statistics are deterministic (seed 0):
- vector norms;
- the cosine of 20,000 random node pairs: mean, sd, 5th/50th/95th percentiles and the share at or above 0.99;
- the **edge AUC**: the probability that a call-graph edge has a higher cosine than a random pair (up to 5,000
  distinct non-loop edges);
- for 200 sampled nodes, the top-10 list `search_similar_code` would return: its similarities, and the share of
  graph neighbours in it.

`--against` adds the overlap@10 of the two neighbour lists.

### Reference numbers (built-in `kaggle-dev-2026`)

Measured with `graph-ref` on the 127 official graph/embedding pairs of the public dev tasks, one pair per commit
(**VERIFIED**). Each cell gives the range over files.

| Repository | Files | Nodes | Random-pair cosine mean | Random-pair cosine sd | Pairs at or above 0.99 | Edge AUC | Median top-10 similarity |
|---|---|---|---|---|---|---|---|
| fastapi | 67 | 3,238-4,597 | 0.7630-0.7934 | 0.1941-0.2121 | 7.0-9.0% | 0.9240-0.9438 | 0.9991-0.9996 |
| httpx | 1 | 740 | 0.8707 | 0.1419 | 18.8% | 0.8527 | 0.9997 |
| requests | 12 | 711-759 | 0.8545-0.8638 | 0.1394-0.1426 | 10.6-12.8% | 0.8952-0.9051 | 0.9985-0.9992 |
| rich | 47 | 1,920-2,021 | 0.8975-0.9016 | 0.0951-0.0987 | 10.8-12.0% | 0.8434-0.8537 | 0.9992-0.9994 |
| all | 127 | 711-4,597 | 0.7630-0.9016 | 0.0951-0.2121 | 7.0-18.8% | 0.8434-0.9438 | 0.9985-0.9997 |

- Norms run from 3.65 to 9.02.
- The bands are therefore: random-pair cosine mean 0.663-1.0; cosine sd 0.0476-0.4242; pairs at or above 0.99
  3.52-37.7%; median top-10 similarity 0.9955-0.9999; edge AUC 0.7934-0.9719. Version 0.2.0 had only the first band
  and the lower edge-AUC limit.
- The margins were chosen so that, leaving one repository out, bands built from the other three still pass every file
  of the held-out repository (127 of 127 in total).
- The hidden test repositories are private. The band describes the pattern of the dev set and was not checked on the
  hidden set.

### What it says on the official files and on Berth's stand-ins

**VERIFIED**, local runs; neither set of files is shipped.

| Input | `graph-section` | `graph-align` | `graph-vectors` |
|---|---|---|---|
| Official: 129 tasks, all three graph tools declared | PASS 129/129 | PASS 127/127 (one pair per commit) | PASS 127/127 |
| Berth: 653 tasks, its graphs and Gaussian stand-in vectors | PASS 653/653 | PASS 653/653 | **WARN 653/653**, each with 4 of 5 statistics outside (all but the cosine sd) |

- **Berth statistics.** Random-pair cosine mean -0.001 to 0.001, edge AUC 0.472-0.538, norms 13.0-19.1. The median
  top-10 similarity is 0.10-0.18, where the scorer prints 0.9985-0.9997.
- **Two stand-ins on official graphs** (one commit per repository, compared with `--against` the official vectors):
  - Gaussian vectors give WARN. Their neighbour lists overlap the official ones at 0.002-0.011.
  - The official vectors shuffled among nodes keep the collapse (random-pair cosine 0.763-0.900, inside that band)
    but lose the structure (edge AUC 0.486-0.510). They get WARN from the edge-AUC band, with overlap 0.003-0.014.
- **Calibrated stand-ins** (built by the independent AI review of version 0.2.0 to imitate the official numbers, one
  official file per repository: Gaussian vectors smoothed over the call graph plus a shared offset, hashed names or
  texts plus an offset, official vectors plus noise). 9 of 16 passed version 0.2.0. Now all 16 give WARN, each with
  2-4 statistics outside; all 16 are outside the bands for pairs at or above 0.99 and for the median top-10
  similarity. Their overlap@10 with the official lists is 0.08-0.37.
- **Adversarial stand-ins** (ours, to probe the bands): the official vectors shuffled among nodes, then smoothed over
  the call graph with weight 0.25, 0.5, 1 or 2, one file per repository. 2 of 16 PASS (fastapi, weights 1 and 2) with
  overlap@10 0.027 and 0.035: statistics like the scorer's, neighbour lists almost disjoint from it. The other 14
  WARN on the edge AUC (2 also on the cosine sd).
- **Stand-ins tuned to the bands without any official vector** (built by the independent AI review of version 0.3.0,
  one official graph per repository). Each vector is a shared offset plus a hashed direction for the node's
  top-level package and one for its parent module, smoothed over the call graph for requests and rich. Four or five
  numbers were tuned with numpy against the band values that ship in the wheel. All 4 PASS `graph-vectors`, with
  overlap@10 against the official lists of 0.071 (fastapi), 0.306 (httpx), 0.286 (requests) and 0.179 (rich).
- **So the bands can be targeted directly.** Anyone with a graph and the shipped band numbers can tune stand-ins to a
  PASS; no official file is needed. The check catches only stand-ins whose statistics are far from the official
  ones, such as Berth's Gaussian vectors, shuffled official vectors and the calibrated stand-ins above. A PASS is no
  evidence that a stand-in is good.

With Berth-style vectors, `search_similar_code` prints similarities near 0.17 where the scorer prints about 1.0, and
its neighbour lists are unrelated to the scorer's. Whether this changes an agent's solve rate is not measured here.

### Validation against the harness (local, not shipped)

We ran the harness wheel's own functions next to the check (**VERIFIED**):
- **NPZ reader.** The kit's standard-library reader equals numpy's `np.load(allow_pickle=False)` on all 780 official
  and Berth NPZ files: 1,111,164 arrays and 284,457,984 values, bit-identical.
- **Per-node vectors.** The kit's vector matrix equals the matrix the harness ranks in 27 of 27 cases:
  - 15 adversarial fixtures: keys borrowed by suffix or by case, zero vectors, prefixed keys, inline vectors, mixed
    lengths (twice), an object entry, a `name` different from the id, a node text used as a key, undeclared nodes,
    `links` instead of `edges`, duplicate ids, an NPZ with no matching key, and a clean pair;
  - 6 official and 6 Berth files.
- **Section and lookup.** The section decision equals the harness's prompt builder in 32 of 32 cases (8 file layouts
  x 4 tool sets). The tool file lookup agrees in 16 of 16.
- **Ranking.** For 90 sampled `search_similar_code` queries on 6 files (3 official, all from rich, and 3 Berth), the
  printed similarities are identical in 90 of 90. The node names are identical in 83 of 90; all 7 others are official
  queries, and their names differ only among nodes with equal printed similarities (float32 in the harness against
  float64 here).

### Limits of the graph check

- **What a `graph-vectors` PASS means.** Five summary statistics of your vectors lie inside bands around the 127
  official dev files: `search_similar_code` prints similarities on the scorer's scale, unrelated nodes look about as
  alike as on the scorer, and call-graph neighbours are about as similar to each other as on the scorer. That is all.
  It says nothing about which nodes the tool returns. A WARN is the informative result: it shows that the tools
  would behave unlike the scorer. A PASS shows only that this check found no such sign.
- **What it does not show.**
  - That the neighbour lists equal the scorer's. The encoder is unpublished. Stand-ins built from node names and the
    call graph alone, tuned to the shipped band numbers, pass with overlap@10 of 0.07-0.31, and shuffled official
    vectors smoothed over the call graph pass with overlap@10 below 0.05. Do not treat a PASS as evidence that a
    stand-in is good. Use `--against` where official vectors exist.
  - That the vectors come from the same embedding model, or that the graph has the official node set (`graph-align`
    judges only the format).
  - That agents solve the same tasks. The effect of stand-in vectors on solve rates is not measured.
  - That the bands hold on the hidden test repositories. They were validated on the 4 public dev repositories only.

  The check prints this with every PASS and WARN.
- **Norms** do not affect a cosine ranking. They are reported only.
- **Cache size.** The harness keeps up to 50,000 keys in memory (node ids and node texts). Graphs above about 25,000
  nodes may see evictions, which the check does not model.
- **Graph content.** `get_code_neighbors` and `get_code_subgraph` read only the graph. How faithful your graph builder
  is to the official one (async functions, nested definitions, `self.` calls) is not judged here; the alignment check
  reads only the format.
- **Speed.** Pure Python: 4.5 s for a 4,287-node graph under a load average near 40 on 16 cores. Use `--jobs` for task
  files.

## Contained runner (reference implementation): `drydock-eval contain`

An agent under evaluation writes and runs code. `contain` is the authors' final runner for doing that on a shared
workstation, released as a reference implementation with its canary suite. The operator guide is the documentation:

```bash
drydock-eval contain guide                 # the operator guide (drydock_eval/contain/GUIDE.md)
drydock-eval contain path                  # where the packaged reference tree is
drydock-eval contain sources               # compare the tree with SOURCES.json (exit 1 on any difference)
drydock-eval contain init <deploy>/evald   # copy the tree to a deployment directory
```

### What it is

One container per job: `--network none`, all capabilities dropped, `no-new-privileges`, a seccomp profile, a read-only
root, an unprivileged account, 2 CPUs, 10 GiB of memory, 4,096 processes, temporary filesystems capped in bytes and
inodes, block-I/O limits. The only channel out is a per-job Unix socket to a forwarder in the runner. The forwarder
passes chat-completion requests only, with allow-listed fields and text-only content, under connection, rate and size
caps (request body 1 MiB), and refuses anything else with a recorded status. Agent runs start only if a gate file,
written after a full canary pass, still matches eight recomputed hashes: the image, the runner, the library, the job
flags, the options, the harness files, the model files and the model-server arguments.

### What you can do without Docker

```bash
drydock-eval contain check-body body.json                 # the forwarder's decision on one request body
drydock-eval contain scan-check --quick                   # the body scanner: exact, and linear up to the size cap
drydock-eval contain gate show --root <deploy>/evald      # the eight bound values of a gate file, next to the tree's
drydock-eval contain gate verify --config runs/<tag>/config.json --root <deploy>/evald   # exit 0 binds, 1 refused
```

### What needs Docker (and was not run on the released tree)

```bash
drydock-eval contain canaries --root <deploy>/evald --tag c1          # the canary suite: no GPU, no model
drydock-eval contain run --root <deploy>/evald --tag g1 --jobs jobs.json --mode gold -- --block S --workers 4
```

Read the guide's section 6 before either command on a shared host. Three things there can hurt another deployment on
the same Docker daemon: the build script re-points the image tag you give it, the reaper removes every labelled
container whose owner process is dead (whichever deployment started it), and the canaries deliberately load CPU,
memory, disk and the process table.

### Configuration

Nothing is configured by editing code. The scripts locate everything relative to the deployment directory
(`<deploy>/evald`, `<deploy>/env-serve`, `<deploy>/eval/...`, `<deploy>/models/served_model`, `<deploy>/gpu.lock`);
the model name, port, deadlines and worker count are options of the runner; the label written into records comes
from the environment variable `DRYDOCK_CONTAIN_HOST_LABEL`. Limits that are security controls (the container flags,
the forwarder's allow-list, the seccomp profile, the uid) are not options: change them and the gate refuses.

### Tests

`tests/test_contain_lib.py` and `tests/test_contain_gate_cli.py` load the released library and run without Docker
and without a GPU: the request-body scanner (accepted and refused forms, exactness against a JSON parse, time linear in
the body size up to the cap), the forwarder against a stub upstream (allow-list, body, header and response caps,
connection and rate limits, the order of refusals), the gate rule (each of the eight bound values and the scope,
refusal of a changed file, the pre-canary gate), record writing, job-list validation and the command line.
`tests/test_contain_docker.py` holds the tests that need Docker. They are marked `docker` and skipped unless
`DRYDOCK_CONTAIN_DEPLOY` names a deployment and a Docker daemon answers:

```bash
DRYDOCK_CONTAIN_DEPLOY=<deploy>/evald python -m pytest tests/test_contain_docker.py -m docker -v
```

### Limits

- Not run in Docker as released; no gate file is shipped. See [Scope and limits](#scope-and-limits-in-one-place).
- It needs your own copy of the competition harness packages, the task set, a model and an agent bundle. None is
  shipped, and the adapters name the harness packages they import.
- Some canary checks compare against records of the authors' earlier runner generations, and the gate writer requires
  them. On a fresh host those checks fail and the writer refuses, until you adapt the list (guide, section 7).
- The residual risks are stated in the guide under "What remains". A canary pass is evidence against the attacks that
  were tried, on the kernel and Docker version it ran on, and nothing more.

## Paired comparisons: `drydock-eval compare`

Tasks from one repository succeed and fail together, so a comparison that treats tasks as independent overstates its
evidence. `compare` works on a plain results table, one row per run:

```
arm,task,repository,repeat,solved,agent_time,row_class
A,pkg_101,org/pkg,1,1,412.5,agent
B,pkg_101,org/pkg,1,0,398.0,agent
```

`agent_time` (seconds) and `row_class` (`agent`, the default, or `infra` for an infrastructure error whose slot must be
filled by another row) are optional. CSV with a header line, or JSON Lines with the same keys. The loader reports every
problem at once and refuses the table; nothing is repaired silently.

```bash
drydock-eval compare gate table.csv --baseline A --candidate B       # exit 0 PASS, 1 FAIL, 3 INCONCLUSIVE
drydock-eval compare signflip table.csv --reference F                # every other arm against F, Holm-corrected
drydock-eval compare simulate table3 --quick                         # size and power of the rule (seconds to a minute, depending on the number of workers)
drydock-eval compare simulate power --quick                          # power of the sign-flip design
drydock-eval compare simulate table3 --sizes 12,7,7,5,3              # the rule on a panel shaped like yours
```

### The pass rule (`gate`)

A candidate passes only if all five criteria hold: (1) the task-weighted mean gain in solve rate is at least 2/58;
(2) one-sided p <= 0.025 under a null-imposed wild cluster bootstrap-t over repositories (Rademacher weights, 9,999
draws, fixed seed); (3) at least three repositories gain and every leave-one-repository-out estimate is positive;
(4) mean agent time is at most 1.2 times the baseline's, and the candidate's share of infrastructure-error rows is not
significantly above its share of planned slots; (5) no repository loses two task-equivalents or more. The decision is
INCONCLUSIVE while a planned slot has no counted run. The thresholds are constants and cannot be set from the command
line: a rule tuned after seeing outcomes is not this rule. The rule cannot pass with fewer than six repositories: the
smallest one-sided p that the exact sign-flip enumeration can reach with G repositories is 1/2^G, which is above 0.025
for G of 5 or fewer (a candidate that solves every task the baseline fails still FAILs on criterion 2 with five
repositories, and passes with six).

### Several arms against one reference (`signflip`)

Per arm: the task-weighted difference in solve rate against the reference, an exact two-sided task-level sign-flip
p-value (Monte Carlo with 99,999 frozen draws when the exact count is not feasible; the report says which), Holm's
correction over the arms at 0.05, the inverted 95% interval, and the repository-cluster analysis as the secondary.

### The simulators (`simulate`)

`table3` gives the pass rate of the rule under no effect, skewed effects, tasks dropped after seeing outcomes, an even
gain and a gain confined to four repositories, on the two panel shapes of the write-up or on your own (`--sizes`).
`power` gives power and family-wise error of the sign-flip design. `--quick` uses fewer simulated panels and prints the
Monte Carlo standard error beside each rate. Measured for this release at the defaults (3,000 panels per cell, 400 for
the dropping row; with `--workers 5` the run took 37 s): rows 1, 2 and 5 of Table 3 equal the published cells to their printed
precision, row 4 (even gain) gave 61.2% and 91.1% against the published about 60% and about 90% (standard error 0.9 and
0.5 points), and the dropping row gave 29.75% and 42.50% against the published 31.75% and 40.75%, within Monte Carlo error
(standard error about 2.4 points). That row is a re-implementation written for the kit.

### Where the code comes from, and how it was checked

The statistics are copied block for block from the authors' analysis and simulation scripts into
`drydock_eval/compare/_vendored_*.py`; each file names the source line ranges it copies. The table loader, the
composition (`gate.py`, `signflip.py`, `simulate.py`) and the command line were written for the kit.
`tests/test_compare_crosscheck.py` compares the packaged functions with **expected values frozen from the original
scripts** on synthetic data (`tests/data/compare_expected.json`, which records the sha256 of each original). The
originals are not imported by the default test run and are not shipped. If you have them, set
`DRYDOCK_ORIGINAL_ANALYZE_E7` (and optionally `DRYDOCK_ORIGINAL_ANALYZE_E8`) to their paths and one more test imports
them and compares on fresh random data.

### What was left out

- Validation of run records and their provenance fields, and the manipulation check of the E8 study.
- Classification of error text into agent and infrastructure classes: the table's `row_class` column states it.
- The limit of three attempts per slot, task drop lists, dispatch stages and cut-offs, blind reports and re-run
  handling.
- Stratification of the harness-error test by re-run batch, the sensitivity analyses for infrastructure errors and the
  descriptive secondary tables.
- The panel loader of the simulator (it reads the authors' panel and task files) and its harness-error scenarios;
  the multiprocessing driver and resume file of the power simulator.
- The panels themselves: `simulate` embeds, for the two panels of the write-up, only the position of each task's
  repository in a sorted list. No task identifier and no repository name is stored.

### Limits

- The rule's level holds for exchangeable tasks. It does not hold for skewed effects or for tasks dropped after
  unblinding (the simulator shows by how much). After release 0.4.0 was built, the rule was applied to three real
  candidates; none passed (the supplement on the branch `paper` gives every quantity).
- `gate` and `signflip` trust the table: they cannot tell whether a row was produced by the configuration its `arm`
  label names, or whether `row_class` was assigned before outcomes were seen.
- The thresholds (2/58, 0.025, 1.2, two task-equivalents) were chosen for one leaderboard and one task set.

## E8 pre-registration package

`experiments/e8/` holds the pre-registered E8 study as documents and code: the frozen task panel (`panel_E8.json`: task
ids of the public Berth dataset, repository names, counts and hashes) as a byte copy, and in `experiments/e8/redacted/`
**redacted copies** of the protocol, the freeze record, the decisions file, the run-record spec, the analysis, the
manipulation check, the pinned earlier analysis, the job-list builder, the arm file and the two test files, with the
script and the counts that describe the redaction. **The redacted files are not the frozen files:** the sha256 values
that the paper cites are those of the originals, which will be released byte for byte on 2026-12-03. The job list
(`jobs_e8.json`), the deviations log (a live log: a snapshot hash only) and `jobs_e8.sha256` are published by hash only.
Everything is listed with its sha256 in `experiments/e8/MANIFEST.json` and `experiments/e8/README.md`. The redacted code
does not run as released (the analysis pins the hashes of the frozen files); `experiments/e8/redacted/README.md` says how
to read and test the copies. No run output, request dump or result of E8 is part of this package. The E8 folder is in the
source distribution and the repository, not in the wheel.

## Release check (maintainers)

This tree holds only shippable files: the package, its tests, this README, the licence, build files,
`experiments/` and `scripts/release_check.py`. The authors' write-up and review records are kept elsewhere. Run the
check before pushing the repository or uploading a distribution; it exits 1 on any finding.

```bash
python scripts/release_check.py                                  # this tree
python scripts/release_check.py --deny-file ../private/denylist.txt   # plus your private names
python scripts/release_check.py --no-tree --archive dist/*.tar.gz --archive dist/*.whl
```

Built-in rules (standard library only):
- **layout:** only shippable top-level entries; cache and virtual-environment folders must be listed in `.gitignore`;
  no data files (captures, arrays, archives), no symbolic links and nothing above 512 KB; the authors' build records
  (`PROVENANCE.json`, `BUILD_REPORT.md`) are tolerated in a working tree only when `.gitignore` lists them, and never
  in an archive;
- **header:** every text file except LICENSE and NOTICE carries `SPDX-License-Identifier: Apache-2.0` (JSON files: a `_license`
  field); shell scripts and Dockerfiles included;
- **home-path:** absolute home or scratch paths, and paths that start with a tilde;
- **token:** Hugging Face, GitHub, OpenAI-style, AWS, Slack and Google API keys, private-key blocks, Kaggle keys,
  quoted values assigned to key, token, secret or password names, and unquoted values containing a digit assigned to
  names ending in token, key, secret or password (`NAME_TOKEN=...` in a shell or `.env` line, `password: ...` in
  YAML) (all printed masked);
- **address:** private network addresses (the shared 100.64/10 range, 10/8, 172.16/12, 192.168/16) and
  e-mail addresses (example.com, .org and .net are allowed);
- **remote:** remote-shell and copy commands, mesh-VPN product names and their host-name suffixes;
- **internal-id:** identifiers of workflow, agent and tool runs of internal tooling, and UUIDs;
- **gpu-row:** rows of a GPU process table (memory-in-use columns, process rows);
- **task-id:** `<name>_<40-hex commit>` file stems, `owner__repo-<number>` instance ids, base-commit values
  (`base_commit = ...`, "base commit <sha>") and bare 40-hex hashes (`.../commit/<sha>` URLs and "commit <sha>"
  citations are allowed);
- **harness-code:** the competition harness's module names in Python code, outside comments, docstrings and
  file:line citations (the README cites the harness by file and line), and any import of the `kaggle` package;
- **archive-meta:** the tar members of an sdist must not carry an account name as owner or group (empty or root only);
- **third-party:** by path: folders named for bundles, templates, adapters, models, tokenizers, wheelhouses, captures
  or run outputs; tokenizer, chat-template, agent-configuration, gate and result files; model, array and archive
  formats; any `.jinja` file other than the kit's own toy template;
- **manifest:** every file under `drydock_eval/contain/reference/` must be listed in
  `drydock_eval/contain/SOURCES.json` with its sha256, and every file listed in `experiments/e8/MANIFEST.json` must
  match its sha256. A file listed there as a verbatim copy is excused from the header rule only (a header would change
  the bytes); its content is scanned like any other file.

Scoped exceptions. The reference tree of `contain` needs a few strings the rules above refuse: the container's own
account folder in the Dockerfile and the inner canary script, in-container cache folders, the fixed well-known gateway
and resolver addresses the no-network canary tries to reach, and the harness module names in the five adapter files
that import the operator's harness. Each exception is tied to one file, one rule and an exact text; the check prints
how many matches used an exception, `--json` lists every one, and a test fails if an exception is used outside
`drydock_eval/contain/reference/`.

Machine names, account names and internal folder names are private, so they are not in the script. Keep them in a
deny file outside the tree, one `<any|code>: <regex>` rule per line; the check refuses a deny file inside the tree.
A deny file may also hold `allow: <path regex> :: <text regex>` lines. They excuse matches of deny-file rules only
(never of a built-in rule), only in paths the first pattern matches, and only when the whole matched text equals the
second pattern.
