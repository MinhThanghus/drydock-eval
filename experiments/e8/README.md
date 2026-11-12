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
# E8 pre-registration package (redacted copies; documents and code only)

E8 is the pre-registered experiment of the write-up: how much each scorer-path divergence changes the outcomes of a
Gemma 4 E4B coding agent. One faithful arm (F) is compared with four arms that each switch one divergence back on
(R, C, H, T). The pre-registered estimate for each of the four is the task-weighted difference in solve rate against F,
tested with an exact two-sided task-level sign-flip test and Holm's correction over the four arms. The plan, the
analysis code and the job list were frozen, and their sha256 values recorded, before any agent run of the experiment.

The documents and code described here were produced by AI coding agents working for the authors (see "How this was made"
in the top-level README), and so was the redaction described below; the hashes can be rechecked by anyone with the
originals.

This folder holds no run outputs, no request dumps, no prompts and no results. It is a record of what was fixed in
advance, so that a reader can check the published analysis against it. The E8 folder is in the source distribution and
the repository, not in the wheel.

## Read this first: the redacted files are not the frozen files

- Most of the frozen files name details of the authors' environment, so this release carries **redacted copies**: private
  strings are replaced by neutral placeholders, and 47 passages of prose in five files by visible markers
  `[redacted: N words, category]`. The method, the substitution table (placeholders only), the cut list (counts only) and
  the glossary are in [`redacted/README.md`](redacted/README.md); `redacted/redact_e8.py` is the script, and
  `redacted/REDACTION_MANIFEST.json` has every count.
- **The sha256 values that the paper cites are those of the frozen originals** (the column "original" below), not those of
  the redacted copies. The redacted copies cannot be used to check a cited hash, except one: `analyze_e8.redacted.py` has no
  substitution and is byte-identical to the frozen analysis.
- **The originals will be released byte for byte on 2026-12-03.** Anyone can then hash them against the cited values and
  re-apply the substitution table to reproduce every redacted hash listed here.
- The redacted copies are listed in `MANIFEST.json` with `"verbatim": true` (the release check's mechanism: exactly these
  bytes, exempt from the licence-header check only, scanned by every other rule), `"redacted": true` and
  `"original_sha256"`. Here `verbatim` means "these exact bytes", not "verbatim of the frozen file". They are licensed
  under Apache-2.0 like the rest of the kit.

## What is in this release

| path | what it is | sha256 of the frozen original (cited) | sha256 of this copy |
|---|---|---|---|
| `panel_E8.json` | the frozen task panel: 231 task ids of the public Berth dataset from 15 repositories (three stages of 118, 37 and 76 tasks), exclusion counts, the repository of every task, two smoke-test tasks, the sha256 of each ordered task list; ids, repository names, counts and hashes only | `f03c0b93f529f6c781e26fec67abe30dcfbf49bcb031cdf5a273614eb7c90857` | the same: a byte copy |
| `redacted/FREEZE_RECORD.redacted.md` | when and how the package was frozen | `981445584ecaf16ac2a98439494c37cd8216afa547d912db5b22d9f93e4fcd54` | `fa0020f3b1b8279aaf5932340426f578e7c8e4ea207aef0f8f45cdf7b9a9d530` |
| `redacted/LEAD_DECISIONS.redacted.md` | the decisions the protocol cites by item number (marked cuts) | `4a652573b046d78933b6e51bf1b8f0925242121538ef455672d92f9fddcd040b` | `0dd9d6e38314170d5be2e46567e2a6c2abe4406cff58e717001d20f5f261b553` |
| `redacted/PROTOCOL.redacted.md` | the pre-registered protocol (marked cuts) | `070c9655d96372d054f430727259de7b25af2ca93c3ec4c9295dfdc37aeb8cf4` | `95dcffcbbd3c67d10b41dcdc0205b8ed68e46ef1dc9dd57be09055895855d8cd` |
| `redacted/RUN_RECORD_SPEC_pinned.redacted.md` | the vendored run-record spec (written for the authors' own run records) | `258c9246c8461aa57b2045ee6a95443eb4e9035b014f76391e8d405f5aa897f8` | `6a738c4d02ed604f700ef8850d81e3cd2eacea90be9c55c211e0ac6edaf671c5` |
| `redacted/analyze_e7_pinned.redacted.py` | the pinned copy of the earlier analysis that the E8 analysis reuses | `e86ccd811424e98462f9696d1e977d780108dc68b9dea6cea47d88053613b50c` | `17825c1b712a59624d940e555623d999d26547d4cc25a8b9322c8ff997fd50b0` |
| `redacted/analyze_e8.redacted.py` | the frozen analysis (**byte-identical to the frozen file**) | `16a80a2a92ff9916fd061b0e10e73c4df80c7ccb62c0241d01ca74de9931f2f7` | `16a80a2a92ff9916fd061b0e10e73c4df80c7ccb62c0241d01ca74de9931f2f7` |
| `redacted/e8_arms.redacted.json` | the arm file (its sha256 goes into every job) | `f0ca2eb83d4727942887e21ab68abfe03b885d2e6324da8918e6b33faf461afe` | `dea12b0c3559eef90defd1a5743b9f84dbcd45ee3512d6abef757f3ec286552e` |
| `redacted/make_jobs_e8.redacted.py` | builds the panel, arm file and job lists | `b101bc6b012c673f4a5078cb140a30a53720ffeb9d55d83dcd6bd4f1a7bcc83c` | `3c4fd1402ed14fbab4e4a42f9f33d2d84ec90d37e2e3e7923937c519aec75982` |
| `redacted/manip_check_e8.redacted.py` | the frozen manipulation check | `b075ed73acea92e95ef2ac91d9bf1d00d115fd957dae7e0baaa5e065651d0c16` | `c392e0d71580120da805b8d416188b84898cb5a04969a2f2ac6287b785c70baf` |
| `redacted/test_analyze_e8.redacted.py` | tests of the analysis and of make_jobs_e8 | `b2c0aa1d628890a7b8de41b3c1d030e9a91887b200239ab0d680fac8f5084d05` | `2e0b802aa15dbe213f17f0b610d802a24d5ee37f52028d57bb9e627721329be8` |
| `redacted/test_manip_check_e8.redacted.py` | tests of the manipulation check | `8a5539964bc0040d02cad2f25f88f32e56f91208539504cf099c1dcda1d1bf97` | `b9baa5dd1e6133edbea94faf7c555188d77adb3a1f71bc124dc922c21dd06b98` |

`MANIFEST.json` lists every copy. `redacted/README.md` explains the placeholders, the cuts and the glossary.

## Held back, published by hash only

| file | sha256 | bytes | why |
|---|---|---:|---|
| `jobs_e8.json` | `4fe9fb3a70c6415e2e49d9aa5e87d31ba8c94db9e8fa28f85d93c0d62e973bfc` | 2,854,928 | the frozen job list (4,620 jobs): 2.8 MB, and every job carries the internal label of the agent bundle, which the release names by sha256 only. The redacted `make_jobs_e8.redacted.py`, run with the frozen `panel_E8.json`, regenerates a redacted job list (see `redacted/README.md`) |
| `DEVIATIONS.md` | `453e76269f2e2ea12aa369d23fc8208bee7602e48e2f6e029f3510913d5fe31c` | 15,696 | the log of deviations from the protocol. **This hash is a snapshot** (taken 2026-11-10): the log is not a frozen file, it is appended to while the experiment runs, and its hash will change |
| `jobs_e8.sha256` | `7b319ba457ad4e03db4732efa7d474a68f113c2b55557f2d6e9fa60cdb19bb28` | 328 | four lines: the sha256 of `e8_arms.json`, `jobs_e8.json`, `jobs_e8smoke.json` and `jobs_e8capture.json` (listed in `redacted/README.md`; a file of this type is not accepted by the release check) |

## Running the redacted copies

They do not run as released: the frozen analysis refuses (exit 3) any vendored file whose hash differs from the frozen
value, and the redacted checker and pinned analysis are different bytes. To read or test the copies, make five edits
in copies (three constants in the analysis and two literals in its tests, set to the hashes of the redacted files), and a sixth
in the copy of the protocol that one test reads; the lines, the values and the consequence (the copy is then not the frozen analysis, and has its own sha256) are in
`redacted/README.md`, section "Running the redacted copies". The released originals run as frozen. Outcome rows that
carry the original bundle label and arm-file hash cannot be analysed with the redacted arm file; use the originals.

## What was checked

Of the 122 tests of the two test files, **58 read no run folder**; the other 64 read run folders (whose outcomes stay blind
until the E8 cut-off) and were not run. The package builder ran the 58 on CPU, and the kit's independent verifier reproduced
the first, second and fourth results below:

- frozen originals: 58 of 58 pass, when the Berth task file and three power-simulation JSON files, which are not shipped,
  are supplied;
- redacted copies with the five documented edits: 57 of 58 pass; the one failure is
  `test_protocol_states_the_frozen_hashes_consistently`, which fails by design because the rebound code no longer equals the
  frozen hashes that the protocol prints;
- with the sixth edit, 58 of 58 pass. That sixth edit makes this test check the **self-consistency of the rebound copies**,
  not the frozen values;
- redacted copies as released: 21 of 58 pass (every pin and hash test fails, as described above).

The two code files `manip_check_e8` and `analyze_e8` have equal syntax trees in original and redacted form (the checker
differs in a docstring and a comment only). **These tests were not re-run when this package was assembled**, and nothing in
the package was executed: the redacted code is shipped as documents.

## What you can do with the kit alone

The statistical methods of the frozen analysis are in the kit as a generic library with its own tests:

```
drydock-eval compare signflip --help     # exact task-level sign-flip test, Holm, inverted intervals
drydock-eval compare gate --help         # the frozen five-criterion pass rule: candidate against baseline
drydock-eval compare simulate --help     # size and power of the design, with a quick mode
```

See the "Paired comparisons" section of the top-level README for what was copied from the analysis files, how the
copies were checked against the originals, and what was left out.
