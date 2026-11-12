<!--
Copyright 2026 The drydock-eval Authors
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# Redacted copies of the E8 pre-registration files

## Read this first

- **The redacted files are NOT the frozen files.** Each file here is a copy of a frozen file in which a short list of
  private strings (the name of the machine the experiment ran on as it appears in folder names, the internal label of the
  agent bundle under test, internal folder names, internal workflow ids, a scratch-folder name) was replaced by neutral
  placeholders, and in 5 files 47 passages of prose (about other users and workloads of the machine, the authors' process history,
  the state of another experiment, run history, load and interpreter details, bundle measurements and adapter design details) were replaced by visible markers
  `[redacted: N words, category]` (listed below, counts only).
- **The sha256 values that the paper cites are those of the originals**, for example the protocol's
  `070c9655d96372d054f430727259de7b25af2ca93c3ec4c9295dfdc37aeb8cf4`. The hashes of the redacted copies are listed below only so that these copies can be
  identified.
- **The originals will be released byte for byte on 2026-12-03.** A reader can then hash them against the
  cited values, and can re-apply the substitution table (below) to them to reproduce every redacted hash listed here.
- **Until then** a reader can check consistency only between the redacted protocol's stated values and the redacted code:
  for example that the hash of the checker and of the analysis that the protocol prints are the ones the code enforces
  (`test_protocol_states_the_frozen_hashes_consistently` passes on the redacted pair), and that the redacted
  `make_jobs_e8.py` regenerates the redacted arm file and job lists. A reader cannot check the cited hashes themselves,
  except one: `analyze_e8.redacted.py` has no substitution and is byte-identical to the frozen `analyze_e8.py`, so its
  sha256 is the frozen analysis hash stated in the protocol.
- **`PROTOCOL.redacted.md` is therefore shorter in words than the frozen protocol** (1,862 words were cut from the protocol and 2,337 over all files: PROTOCOL.md 1,862, analyze_e7_pinned.py 180, RUN_RECORD_SPEC_pinned.md 179, LEAD_DECISIONS.md 59, FREEZE_RECORD.md 57; the largest cut is the status
  paragraph's revision trail). Every statement that a rule or a hash depends on is intact; the values of both frozen
  hashes (checker and analysis) are printed in the sections that state them.
- **Frozen wording.** The protocol's sentence that the replica "reproduces the hosted scorer's prompts token for token" is the
  protocol's own frozen wording, written before the E8 block ran; the paper restates it with its conditions. Read it with
  those conditions, not as a stand-alone claim.
- **Throughput figure.** The 318.6 runs/h at `PROTOCOL.md` lines 1039-1041 (section 10, GPU time) was measured on an earlier
  runner version than the one behind the paper's final-runner figure; the two are not comparable.
- **The adapter and LoRA facts in the spec and the analysis are limited.** The documents still say that adapter arms and an adapter-serving
  server exist (labels neutralised). The adapter rank value, the job count of the adapter block, the adapter-versus-plain parity check
  with its numeric limit, the temperature-0 isolation check and the prose about the LoRA runs' dumps are cut.
- **`RUN_RECORD_SPEC_pinned.redacted.md` was written for the authors' own run records** (it says what the authors' runner
  must write); it is vendored here because the frozen analysis follows it, and it is not a general specification.
- **The redacted set does not run as released.** The analysis refuses (exit 3) any vendored file whose hash differs from
  the frozen value, and the redacted checker and pinned analysis are different bytes. See "Running the redacted copies".

## Files

| redacted file | what the original is | sha256 of the original (the cited value) | bytes | sha256 of this redacted copy | bytes | substituted occurrences |
|---|---|---|---:|---|---:|---|
| `PROTOCOL.redacted.md` | the pre-registered protocol | `070c9655d96372d054f430727259de7b25af2ca93c3ec4c9295dfdc37aeb8cf4` | 121,308 | `95dcffcbbd3c67d10b41dcdc0205b8ed68e46ef1dc9dd57be09055895855d8cd` | 110,590 | 92 |
| `FREEZE_RECORD.redacted.md` | when and how the package was frozen | `981445584ecaf16ac2a98439494c37cd8216afa547d912db5b22d9f93e4fcd54` | 2,514 | `fa0020f3b1b8279aaf5932340426f578e7c8e4ea207aef0f8f45cdf7b9a9d530` | 2,196 | 1 |
| `LEAD_DECISIONS.redacted.md` | the decisions the protocol cites by item number | `4a652573b046d78933b6e51bf1b8f0925242121538ef455672d92f9fddcd040b` | 9,962 | `0dd9d6e38314170d5be2e46567e2a6c2abe4406cff58e717001d20f5f261b553` | 9,747 | 6 |
| `RUN_RECORD_SPEC_pinned.redacted.md` | the vendored run-record spec (a byte copy of another experiment's spec) | `258c9246c8461aa57b2045ee6a95443eb4e9035b014f76391e8d405f5aa897f8` | 30,995 | `6a738c4d02ed604f700ef8850d81e3cd2eacea90be9c55c211e0ac6edaf671c5` | 30,268 | 40 |
| `analyze_e8.redacted.py` | the frozen analysis | `16a80a2a92ff9916fd061b0e10e73c4df80c7ccb62c0241d01ca74de9931f2f7` | 128,727 | `16a80a2a92ff9916fd061b0e10e73c4df80c7ccb62c0241d01ca74de9931f2f7` | 128,727 | 0 (none: byte-identical to the frozen file) |
| `manip_check_e8.redacted.py` | the frozen per-row manipulation check | `b075ed73acea92e95ef2ac91d9bf1d00d115fd957dae7e0baaa5e065651d0c16` | 46,059 | `c392e0d71580120da805b8d416188b84898cb5a04969a2f2ac6287b785c70baf` | 46,063 | 2 |
| `analyze_e7_pinned.redacted.py` | the pinned copy of another experiment's analysis that the E8 analysis reuses | `e86ccd811424e98462f9696d1e977d780108dc68b9dea6cea47d88053613b50c` | 293,278 | `17825c1b712a59624d940e555623d999d26547d4cc25a8b9322c8ff997fd50b0` | 292,831 | 55 |
| `make_jobs_e8.redacted.py` | builds the panel, arm file and job lists | `b101bc6b012c673f4a5078cb140a30a53720ffeb9d55d83dcd6bd4f1a7bcc83c` | 33,148 | `3c4fd1402ed14fbab4e4a42f9f33d2d84ec90d37e2e3e7923937c519aec75982` | 33,152 | 9 |
| `e8_arms.redacted.json` | the arm file (sha256 goes into every job) | `f0ca2eb83d4727942887e21ab68abfe03b885d2e6324da8918e6b33faf461afe` | 2,856 | `dea12b0c3559eef90defd1a5743b9f84dbcd45ee3512d6abef757f3ec286552e` | 2,858 | 1 |
| `test_analyze_e8.redacted.py` | tests of analyze_e8.py and make_jobs_e8.py | `b2c0aa1d628890a7b8de41b3c1d030e9a91887b200239ab0d680fac8f5084d05` | 104,883 | `2e0b802aa15dbe213f17f0b610d802a24d5ee37f52028d57bb9e627721329be8` | 104,970 | 20 |
| `test_manip_check_e8.redacted.py` | tests of manip_check_e8.py | `8a5539964bc0040d02cad2f25f88f32e56f91208539504cf099c1dcda1d1bf97` | 47,669 | `b9baa5dd1e6133edbea94faf7c555188d77adb3a1f71bc124dc922c21dd06b98` | 47,753 | 19 |

The redacted copies keep the line structure of the originals: no line was added or removed, so a line number in the
protocol or in a comment refers to the same line in the copy. Written by `redact_e8.py` (in this folder; it reads the
substitution pairs from a private file that is not part of the release).

## Files held back, published by hash only

| file | sha256 | bytes | why |
|---|---|---:|---|
| `jobs_e8.json` | `4fe9fb3a70c6415e2e49d9aa5e87d31ba8c94db9e8fa28f85d93c0d62e973bfc` | 2,854,928 | 2.8 MB, and every one of its 4,620 jobs carries the agent bundle's internal label. Regenerating it from the redacted `make_jobs_e8.py` and the frozen `panel_E8.json` (shipped in the parent folder) gives a list in which the label is `BUNDLE_A` and the arm-file hash is the redacted arm file's; that list has sha256 `51eb46d4bf48f4f3d8b74ed4f564fde48eaf6226b0e5679fa4039088ac6f01ce`. |
| `DEVIATIONS.md` | `453e76269f2e2ea12aa369d23fc8208bee7602e48e2f6e029f3510913d5fe31c` | 15,696 | the log of deviations from the protocol is still open (append-only; not a frozen file), so a snapshot hash only (snapshot taken 2026-11-10 15:53 UTC; the log is still open and gets appended to, so this value will be superseded). |
| `jobs_e8.sha256` | `7b319ba457ad4e03db4732efa7d474a68f113c2b55557f2d6e9fa60cdb19bb28` | 328 | four lines (below); the file type is not one the kit's release check accepts. |

The four lines of the frozen `jobs_e8.sha256`: `e8_arms.json` `f0ca2eb83d4727942887e21ab68abfe03b885d2e6324da8918e6b33faf461afe`, `jobs_e8.json`
`4fe9fb3a70c6415e2e49d9aa5e87d31ba8c94db9e8fa28f85d93c0d62e973bfc`, `jobs_e8smoke.json` `394cacc92a9e0781468804bf8e3c650cca82dbb80183d52fb09f75fde3db490c`,
`jobs_e8capture.json` `a7abd121922cd5eaebd16b9dd946964e9124c38ded101df2ae8019cbc4ff5d77`.

## The substitution table

One pass over each original, longest string first. A string is replaced only when it is a whole token: no ASCII letter
stands directly before or after it. Distinct private strings have distinct placeholders, and no placeholder occurs in any
original, so the table can be inverted given the original. The private side of the table is not published here; it will be
released together with the originals on 2026-12-03.

| placeholder | meaning | occurrences (all files) |
|---|---|---:|
| `FOLDER_RUNNER_SRC` | folder name: the source mirror of the container runner, as it appears in paths | 16 |
| `FOLDER_RUNNER_EVAL` | folder name: the runner's evaluation folder, as it appears in paths | 15 |
| `FOLDER_OUT_DOCKER` | folder name: output folder of the earlier Docker parity runs | 1 |
| `FOLDER_OUT_D42` | folder name: output folder of the D4.2 runner version | 2 |
| `FOLDER_OUT_D4` | folder name: output folder of the D4 runner version | 4 |
| `FOLDER_E7` | folder name: the E7 experiment's folder | 1 |
| `FOLDER_E6` | folder name: the E6 experiment's folder | 11 |
| `FOLDER_E8` | folder name: this experiment's folder | 2 |
| `HOSTA_RUNS_PER_HOUR` | constant name containing the host machine's name in capitals | 3 |
| `HOSTA` | the host machine's name, as a word or inside folder names | 10 |
| `BUNDLE_A_var1` | bundle label with a variant suffix (an arm label of the earlier bundle screen) | 2 |
| `BUNDLE_A_var2` | bundle label with a variant suffix (an arm label of the earlier bundle screen) | 1 |
| `BUNDLE_A_var3` | bundle label with a variant suffix (an arm label of the earlier bundle screen) | 1 |
| `BUNDLE_A_var4` | bundle label with a variant suffix (an arm label of the earlier bundle screen) | 1 |
| `BUNDLE_A_var5` | bundle label with a variant suffix (an arm label of the earlier bundle screen) | 1 |
| `BUNDLE_A` | the label of the authors' agent bundle | 37 |
| `bundle_a` | the bundle label in lower case (inside a function name) | 2 |
| `[scratch]` | a scratch folder of the authors' tooling | 2 |
| `jobs of other blocks of the same project` | neutral phrase for other blocks' jobs (was a pair of block labels) | 2 |
| `Rows of other blocks of the same project` | neutral phrase for other blocks' rows (was a pair of block labels) | 1 |
| `a job of another block of the same project` | neutral phrase for a job of another block (was two block labels) | 1 |
| `other blocks of the same project and E8` | neutral phrase for other blocks (was two block labels) | 1 |
| `The project's server command` | neutral phrase (was a block label) | 1 |
| `adapter_arm_B` | the label of the second adapter arm | 15 |
| `adapter_arm_A` | the label of the first adapter arm (not replaced where it starts the name of an internal document, which stays) | 22 |
| `adapter_name_A` | the name of the first adapter | 5 |
| `adapter_name_B` | the name of the second adapter | 3 |
| `parity_run_A` | folder name of an earlier parity run | 2 |
| `parity_run_B` | folder name of an earlier parity run | 1 |
| `parity_run_C` | folder name of an earlier parity run | 1 |
| `[repo-root]` | the root folder of the authors' research tree (only where followed by a slash) | 8 |
| `[tools-env]` | the interpreter environment folder | 4 |
| `[tools-env-name]` | the name of the interpreter environment | 4 |
| `[py-lib-dir]` | the interpreter's library folder name | 1 |
| `FOLDER_KIT_PAPER` | folder name: the kit's paper folder | 1 |
| `FOLDER_KIT` | folder name: the kit folder | 0 |
| `FOLDER_PRIZE_NOTES` | folder name: notes of a prize-research folder | 2 |
| `FOLDER_WRITEUP_LOG` | folder name: the authors' paper log | 1 |
| `FOLDER_NOTEBOOKS` | folder name: public notebooks | 1 |
| `[harness-src]` | the harness source folder | 2 |
| `[task-data]` | the task dataset folder | 3 |
| `[model-dir]` | the model folder name | 11 |
| `[cpu-priority]` | a shell priority setting | 4 |
| `the isolation condition stated in the protocol holds as deployed` | neutral wording of a verifier's statement | 1 |
| `"max_lora_rank": [rank]` | an adapter rank value in an example | 2 |
| `"n_jobs": [count]` | a job count in an example | 1 |
| `[internal-id-1]` | internal workflow id | 5 |
| `[internal-id-2]` | internal workflow id | 5 |
| `[internal-id-3]` | internal workflow id | 6 |
| `[internal-id-4]` | internal workflow id | 4 |
| `[internal-id-5]` | internal workflow id | 3 |
| `[internal-id-6]` | internal workflow id | 8 |
| `[internal-id-7]` | internal workflow id | 1 |

Per file (occurrences):

- `FREEZE_RECORD.redacted.md` (1): `[scratch]` 1; cut at line 25: `[redacted: 57 words, internal process history]`
- `LEAD_DECISIONS.redacted.md` (6): `FOLDER_RUNNER_EVAL` 1, `FOLDER_E8` 1, `HOSTA` 1, `BUNDLE_A` 1, `other blocks of the same project and E8` 1, `the isolation condition stated in the protocol holds as deployed` 1; cut at line 3: `[redacted: 19 words, internal process history]`, cut at line 51: `[redacted: 7 words, other experiment state]`, cut at line 42: `[redacted: 28 words, other workloads]`, cut at line 39: `[redacted: 5 words, owner bundle]`
- `PROTOCOL.redacted.md` (92): `FOLDER_RUNNER_SRC` 6, `FOLDER_RUNNER_EVAL` 9, `FOLDER_OUT_DOCKER` 1, `FOLDER_OUT_D42` 1, `FOLDER_OUT_D4` 1, `FOLDER_E6` 5, `FOLDER_E8` 1, `BUNDLE_A` 12, `jobs of other blocks of the same project` 2, `Rows of other blocks of the same project` 1, `a job of another block of the same project` 1, `The project's server command` 1, `parity_run_A` 1, `parity_run_B` 1, `parity_run_C` 1, `[repo-root]` 1, `[tools-env]` 1, `[tools-env-name]` 4, `FOLDER_KIT_PAPER` 1, `FOLDER_PRIZE_NOTES` 2, `FOLDER_WRITEUP_LOG` 1, `FOLDER_NOTEBOOKS` 1, `[harness-src]` 2, `[task-data]` 1, `[model-dir]` 6, `[cpu-priority]` 2, `[internal-id-1]` 5, `[internal-id-2]` 4, `[internal-id-3]` 3, `[internal-id-4]` 4, `[internal-id-5]` 2, `[internal-id-6]` 7, `[internal-id-7]` 1; cut at line 1095: `[redacted: 6 words, owner workload]`, cut at line 1160: `[redacted: 5 words, owner workload]`, cut at line 3: `[redacted: 603 words, internal process history]`, cut at line 1228: `[redacted: 363 words, internal process history]`, cut at line 1248: `[redacted: 207 words, internal process history]`, cut at line 184: `[redacted: 7 words, internal process history]`, cut at line 801: `[redacted: 16 words, internal process history]`, cut at line 1287: `[redacted: 12 words, internal process history]`, cut at line 1292: `[redacted: 26 words, internal process history]`, cut at line 1321: `[redacted: 20 words, internal process history]`, cut at line 1328: `[redacted: 10 words, internal process history]`, cut at line 1329: `[redacted: 4 words, internal process history]`, cut at line 1330: `[redacted: 4 words, internal process history]`, cut at line 1330: `[redacted: 4 words, internal process history]`, cut at line 522: `[redacted: 30 words, other experiment state]`, cut at line 947: `[redacted: 57 words, other experiment state]`, cut at line 1336: `[redacted: 47 words, other experiment state]`, cut at line 564: `[redacted: 63 words, run history]`, cut at line 577: `[redacted: 121 words, run history]`, cut at line 991: `[redacted: 25 words, bundle measurement]`, cut at line 1052: `[redacted: 6 words, other workloads]`, cut at line 266: `[redacted: 29 words, other workloads]`, cut at line 1197: `[redacted: 11 words, owner approval]`, cut at line 1205: `[redacted: 99 words, redaction plan]`, cut at line 284: `[redacted: 22 words, adapter detail]`, cut at line 731: `[redacted: 6 words, adapter detail]`, cut at line 737: `[redacted: 29 words, adapter detail]`, cut at line 824: `[redacted: 4 words, load detail]`, cut at line 975: `[redacted: 4 words, load detail]`, cut at line 1315: `[redacted: 10 words, load detail]`, cut at line 1351: `[redacted: 12 words, interpreter detail]`
- `RUN_RECORD_SPEC_pinned.redacted.md` (40): `FOLDER_RUNNER_SRC` 3, `HOSTA` 3, `BUNDLE_A_var1` 1, `BUNDLE_A` 5, `adapter_arm_B` 7, `adapter_arm_A` 12, `adapter_name_A` 4, `adapter_name_B` 2, `"max_lora_rank": [rank]` 2, `"n_jobs": [count]` 1; cut at line 232: `[redacted: 164 words, adapter detail]`, cut at line 172: `[redacted: 5 words, adapter detail]`, cut at line 324: `[redacted: 10 words, adapter detail]`
- `analyze_e7_pinned.redacted.py` (55): `FOLDER_RUNNER_SRC` 3, `FOLDER_RUNNER_EVAL` 1, `HOSTA_RUNS_PER_HOUR` 3, `HOSTA` 6, `BUNDLE_A_var1` 1, `BUNDLE_A_var2` 1, `BUNDLE_A_var3` 1, `BUNDLE_A_var4` 1, `BUNDLE_A_var5` 1, `BUNDLE_A` 12, `adapter_arm_B` 8, `adapter_arm_A` 10, `adapter_name_A` 1, `adapter_name_B` 1, `parity_run_A` 1, `[repo-root]` 2, `[tools-env]` 1, `[py-lib-dir]` 1; cut at line 1228: `[redacted: 11 words, internal process history]`, cut at line 2034: `[redacted: 11 words, internal process history]`, cut at line 3171: `[redacted: 11 words, internal process history]`, cut at line 2066: `[redacted: 13 words, adapter detail]`, cut at line 3155: `[redacted: 32 words, adapter detail]`, cut at line 3900: `[redacted: 81 words, adapter detail]`, cut at line 3194: `[redacted: 18 words, adapter detail]`, cut at line 3195: `[redacted: 3 words, adapter detail]`
- `e8_arms.redacted.json` (1): `BUNDLE_A` 1
- `make_jobs_e8.redacted.py` (9): `FOLDER_E7` 1, `FOLDER_E6` 2, `BUNDLE_A` 2, `[repo-root]` 2, `[task-data]` 2
- `manip_check_e8.redacted.py` (2): `BUNDLE_A` 2
- `test_analyze_e8.redacted.py` (20): `FOLDER_RUNNER_SRC` 2, `FOLDER_RUNNER_EVAL` 3, `FOLDER_OUT_D42` 1, `FOLDER_OUT_D4` 1, `FOLDER_E6` 2, `BUNDLE_A` 2, `bundle_a` 2, `[repo-root]` 1, `[tools-env]` 1, `[model-dir]` 2, `[cpu-priority]` 1, `[internal-id-2]` 1, `[internal-id-6]` 1
- `test_manip_check_e8.redacted.py` (19): `FOLDER_RUNNER_SRC` 2, `FOLDER_RUNNER_EVAL` 1, `FOLDER_OUT_D4` 2, `FOLDER_E6` 2, `[scratch]` 1, `[repo-root]` 2, `[tools-env]` 1, `[model-dir]` 3, `[cpu-priority]` 1, `[internal-id-3]` 3, `[internal-id-5]` 1

## Cuts

47 passages of private prose are replaced by a visible marker `[redacted: N words, category]`, N being the original
passage's word count. A passage that spans lines is replaced by blank lines and, on the last of them, the marker, so no line
number moves. The cut text is not published here. These are the only changes that are not table substitutions.

| file | lines | original words | category |
|---|---|---:|---|
| `FREEZE_RECORD.md` | 25-28 | 57 | internal process history |
| `LEAD_DECISIONS.md` | 3-4 | 19 | internal process history |
| `LEAD_DECISIONS.md` | 51 | 7 | other experiment state |
| `LEAD_DECISIONS.md` | 42-43 | 28 | other workloads |
| `LEAD_DECISIONS.md` | 39 | 5 | owner bundle |
| `PROTOCOL.md` | 1095 | 6 | owner workload |
| `PROTOCOL.md` | 1160 | 5 | owner workload |
| `PROTOCOL.md` | 3-39 | 603 | internal process history |
| `PROTOCOL.md` | 1228-1247 | 363 | internal process history |
| `PROTOCOL.md` | 1248-1261 | 207 | internal process history |
| `PROTOCOL.md` | 184 | 7 | internal process history |
| `PROTOCOL.md` | 801-802 | 16 | internal process history |
| `PROTOCOL.md` | 1287 | 12 | internal process history |
| `PROTOCOL.md` | 1292-1293 | 26 | internal process history |
| `PROTOCOL.md` | 1321-1322 | 20 | internal process history |
| `PROTOCOL.md` | 1328 | 10 | internal process history |
| `PROTOCOL.md` | 1329 | 4 | internal process history |
| `PROTOCOL.md` | 1330 | 4 | internal process history |
| `PROTOCOL.md` | 1330-1331 | 4 | internal process history |
| `PROTOCOL.md` | 522-524 | 30 | other experiment state |
| `PROTOCOL.md` | 947-950 | 57 | other experiment state |
| `PROTOCOL.md` | 1336-1339 | 47 | other experiment state |
| `PROTOCOL.md` | 564-567 | 63 | run history |
| `PROTOCOL.md` | 577-585 | 121 | run history |
| `PROTOCOL.md` | 991-993 | 25 | bundle measurement |
| `PROTOCOL.md` | 1052 | 6 | other workloads |
| `PROTOCOL.md` | 266-267 | 29 | other workloads |
| `PROTOCOL.md` | 1197 | 11 | owner approval |
| `PROTOCOL.md` | 1205-1210 | 99 | redaction plan |
| `PROTOCOL.md` | 284-285 | 22 | adapter detail |
| `PROTOCOL.md` | 731 | 6 | adapter detail |
| `PROTOCOL.md` | 737-738 | 29 | adapter detail |
| `PROTOCOL.md` | 824 | 4 | load detail |
| `PROTOCOL.md` | 975 | 4 | load detail |
| `PROTOCOL.md` | 1315 | 10 | load detail |
| `PROTOCOL.md` | 1351-1352 | 12 | interpreter detail |
| `RUN_RECORD_SPEC_pinned.md` | 232-241 | 164 | adapter detail |
| `RUN_RECORD_SPEC_pinned.md` | 172 | 5 | adapter detail |
| `RUN_RECORD_SPEC_pinned.md` | 324 | 10 | adapter detail |
| `analyze_e7_pinned.py` | 1228 | 11 | internal process history |
| `analyze_e7_pinned.py` | 2034 | 11 | internal process history |
| `analyze_e7_pinned.py` | 3171 | 11 | internal process history |
| `analyze_e7_pinned.py` | 2066 | 13 | adapter detail |
| `analyze_e7_pinned.py` | 3155-3156 | 32 | adapter detail |
| `analyze_e7_pinned.py` | 3900-3904 | 81 | adapter detail |
| `analyze_e7_pinned.py` | 3194 | 18 | adapter detail |
| `analyze_e7_pinned.py` | 3195 | 3 | adapter detail |

## Which substitutions fall in code

Code of `make_jobs_e8.py`, `analyze_e7_pinned.py` and the two test files contains substituted strings; `analyze_e8.py` and
`manip_check_e8.py` contain none in code (the checker has one in its docstring and one in a comment; the analysis has none).
Compared as syntax trees, the redacted code differs from the frozen code only in string constants and in one constant's
name; nothing else changed. What these strings do:

- `make_jobs_e8.redacted.py:75` sets `BUNDLE = 'BUNDLE_A'`. The value is written as `bundle` into the arm file and into every
  job, and `analyze_e8.py` requires every result row's `bundle` to equal the arm file's (`analyze_e8.py:953`; a mismatch is a
  record problem and voids the block). The redacted arm file also has a different sha256, which every job and every row
  carries as `e8_arms_sha256`. **Outcome rows that carry the original label and the original arm-file hash cannot be
  analysed with the redacted arm file; they need the same two substitutions (the label to `BUNDLE_A`, the arm-file hash
  to the redacted file's), or the original arm file, which will be released.**
- `make_jobs_e8.redacted.py:196` and `:206` are default folder names for `--draw` (the pool and the other experiment's
  panels); `--draw` needs the authors' local pool files, which are not released, so nothing is lost.
- `analyze_e7_pinned.redacted.py` lines 1309, 1312, 1325, 1328 and 3216 are `source` texts inside the error-class tables.
  The tables' ids, classes, kinds and patterns are unchanged, but the tables' sha256 values (computed from their text and
  written into every row as `error_table_sha256`) are different; a row carrying the original table hash gets a note in the
  report, nothing else. Lines 54-55 (arm labels of that experiment's screening), 1039, 1995, 2761, 3073, 3799 (decision text),
  4626 and 4712 (command-line help) are not used by the E8 analysis. Lines 2065, 2448 and 2451 rename one constant
  consistently.
- `analyze_e7_pinned.redacted.py:1242` (`ADAPTER_ARMS`, two string values: `adapter_arm_A`, `adapter_arm_B`), the notes at 1843, 2874
  and 3632 and the help texts at 4626 and 4712 carry the neutralised adapter-arm labels. The constant is compared with the arm
  label only in the earlier experiment's gate and provenance functions; the E8 analysis passes no adapter arms and its arms are
  `F`, `R`, `C`, `H`, `T`, so nothing changes for E8 (the constant is renamed consistently). The adapter names and the labels of
  earlier parity runs (`adapter_name_A`/`_B`, `parity_run_A` to `_C`) occur in comments and documents only. Three mentions of an internal
  document's file name (in `analyze_e7_pinned.redacted.py`, lines 2, 1128 and 1224) are not substituted: the substitution of the
  first adapter arm's label is skipped where it starts that file name.
- The test files' default folder names (lines 54-62; 39-43) point at folders that do not exist outside the authors'
  tree (the model folder name in `test_analyze_e8.redacted.py:54-55` and `test_manip_check_e8.redacted.py:39-40` is `[model-dir]`; these are
  default paths only and the environment variables override them): set the environment variables `E8_TOKENIZER`, `E8_TEMPLATE_KAGGLE`, `E8_TEMPLATE_T`, `E8_D4_SMOKE_RUNS`,
  `E8_D42_SMOKE_RUNS` and `E8_D42_DROP_FILE` as in the protocol (section 15).

## Running the redacted copies

The frozen analysis refuses to run unless three files hash to values held in `analyze_e8.py`: the pinned analysis
(`PINNED_ANALYZE_E7_SHA256`), the checker (`FROZEN_CHECKER_SHA256`, which `report` and `reruns` also require on the command
line as `--checker-sha256`, and the analysis's own hash as `--analyze-e8-sha256`) and, in the tests, the run-record spec.
The redacted vendored files are different bytes, so the redacted analysis refuses them; `--allow-unpinned-e7` lets only
`check` and `reruns` accept a different pinned analysis, and the checker has no override. **No command-line flag makes the
redacted pair run.** The released originals run as frozen.

To read or test the redacted copies you can replace three constants and re-run; this is a local change to a copy and
produces a file that is not the frozen analysis:

| file | line | constant | frozen value | value for the redacted copies |
|---|---:|---|---|---|
| `analyze_e8.py` | 115 | PINNED_ANALYZE_E7_SHA256 | `e86ccd811424e98462f9696d1e977d780108dc68b9dea6cea47d88053613b50c` | `17825c1b712a59624d940e555623d999d26547d4cc25a8b9322c8ff997fd50b0` |
| `analyze_e8.py` | 120 | PINNED_RUN_RECORD_SPEC_SHA256 | `258c9246c8461aa57b2045ee6a95443eb4e9035b014f76391e8d405f5aa897f8` | `6a738c4d02ed604f700ef8850d81e3cd2eacea90be9c55c211e0ac6edaf671c5` |
| `analyze_e8.py` | 124 | FROZEN_CHECKER_SHA256 | `b075ed73acea92e95ef2ac91d9bf1d00d115fd957dae7e0baaa5e065651d0c16` | `c392e0d71580120da805b8d416188b84898cb5a04969a2f2ac6287b785c70baf` |
| `test_analyze_e8.py` | 189 | literal in test_pinned_analyze_e7 (same as line 115) | `e86ccd811424e98462f9696d1e977d780108dc68b9dea6cea47d88053613b50c` | `17825c1b712a59624d940e555623d999d26547d4cc25a8b9322c8ff997fd50b0` |
| `test_analyze_e8.py` | 1789 | literal in test_run_record_spec_is_vendored (same as line 120) | `258c9246c8461aa57b2045ee6a95443eb4e9035b014f76391e8d405f5aa897f8` | `6a738c4d02ed604f700ef8850d81e3cd2eacea90be9c55c211e0ac6edaf671c5` |

A sixth edit concerns only the copy of the protocol that `test_protocol_states_the_frozen_hashes_consistently` reads (not the
published file): replace the printed checker hash (three places) by the redacted checker's hash and the printed analysis hash
(two places) by the sha256 of the rebound `analyze_e8.py` (`6947b30577ddec51136251927da6c2818034c2c84bdd9843b94759205f18b0e4`). Without this sixth edit that one test fails by design,
because the protocol prints the frozen values; with it the test checks the self-consistency of the rebound copies.

After the edits the redacted `analyze_e8.py` has its own sha256 (`6947b30577ddec51136251927da6c2818034c2c84bdd9843b94759205f18b0e4`), and that is the value to pass as `--analyze-e8-sha256`;
the redacted protocol still prints the frozen values (the frozen checker's and the frozen analysis's), which is what the
paper cites.

## Equivalence evidence

Run by the package builder on 2026-11-10, CPU only, with the run-folder variables pointed at an empty folder and the 64 tests that
read run folders deselected (their outcomes stay blind until the E8 cut-off; they were not run). **58 of the 122 tests read no run
folder.** Results:

- frozen originals: 58 of 58 pass, when the Berth task file and the three `power/*.json` simulation outputs, which are not part of
  this release, are supplied;
- redacted copies with the documented rebinding (the five edits of the table above): 57 of 58 pass; the one failure is
  `test_protocol_states_the_frozen_hashes_consistently`, which fails by design because the rebound code no longer equals the frozen
  hashes that the protocol prints; with the sixth edit (above) 58 of 58 pass;
- redacted copies as released: 21 of 58 pass (every pin and hash test fails, as described under "Running the redacted copies").

The independent verifier of the kit reproduced the originals (58 of 58), the five-edit result (57 of 58) and the as-released result
(21 of 58). The two code files `manip_check_e8.py` and `analyze_e8.py` have equal syntax trees in original and redacted form (the
redacted checker differs only in a docstring and a comment), `analyze_e8.redacted.py` is byte-identical to the frozen file, and the
redacted `make_jobs_e8.py` regenerates the arm file and the job lists to the hashes in `REDACTION_MANIFEST.json`.

## Glossary of internal labels that remain in the text

- `D3`, `D4`, `D4.1`, `D4.2`, `D4.3`: an internal runner version label each (versions of the authors' container runner, its
  image and the E8 worker). The documents do not map them to anything else, and this README does not either.
- `J5`, `J6`: parity runs whose request dumps are cited as evidence. `E6`, `E7`: other experiments or blocks of the same
  project (`E7` is also the source of the pinned analysis and the run-record spec).
- `A0` to `A18`, `I01` to `I14`, `M01`: the row-class ids of the pinned analysis (`A` agent outcome, `I` infrastructure,
  `M01` manipulation check failed); `A0`/`A8`/`A9`/`A10` also name amendments of another experiment's rules.
- `LEAD item N`: item N of `LEAD_DECISIONS.md`. `V` and `I` in the protocol: verified and inferred statements (explained
  in the protocol's second paragraph).
- `g4s`: an internal prefix of the runner's messages and of the seeds of some hashes (`g4s-e8-panel-v1:`); it is part of
  frozen values and is not expanded here.
- `[internal-id-N]`: an internal workflow identifier (`N` from 1 to 7, numbered in order of first appearance in the
  protocol). `FOLDER_*`: a folder name; `HOSTA`: the machine name; `BUNDLE_A`: the bundle label.

## Licence and marking

"The E8 authors" in the copyright line of three frozen code files (`analyze_e8.redacted.py`, `manip_check_e8.redacted.py`,
`make_jobs_e8.redacted.py`, and the two test files) are the same authors as "The drydock-eval Authors" of the rest of the kit; the
frozen text is kept as it is, under Apache-2.0 either way.

These files are licensed under Apache-2.0 like the rest of the kit. The redacted copies carry no licence header of their
own (some documents had none; adding one to any file would change the bytes whose hashes are listed). They are marked
instead in `../MANIFEST.json`: every one is listed with its sha256, with `"verbatim": true` (the kit's mechanism: the
file is exempt from the licence-header check only and is scanned by every other rule), `"redacted": true` and
`"original_sha256"` naming the frozen file it was made from. `REDACTION_MANIFEST.json` in this folder has the same facts
in full, with the per-file substitution counts.
