# Data of the extended version

Run-level results of the paper "Contain, Conform, Compare (Extended Version)". Appendix D of the paper describes
every column. The files hold no prompt, no model output, no patch, no test output, no error text and no machine
detail. The agent bundles under test and their prompts are withheld while the competition runs, so arms carry neutral
labels (`arms.csv`), and the tasks of the competition's development set carry opaque codes (`dev-001` ...), because
the competition data may not be redistributed.

| File | Records | Content |
|---|---|---|
| `runs.csv` | 5024 | one line per agent run of Sections 7, 8 and 10 (31B and E4B; development tasks and Berth tasks) |
| `experiments.csv` | 31 | one line per block: model, machine, task source, tasks, repositories, arms, repeats, runs, solved |
| `arms.csv` | 37 | one line per arm: model, family, runs, one-sentence description of the change |
| `controls.csv` | 2494 | grading controls: each task graded with its reference patch (must pass) and with no patch (must fail) |
| `serving_bench.csv` | 72 | serving measurements on four L4 GPUs: prefill, time to first token, tokens per second by concurrency |
| `serving_servers.csv` | 7 | start-up time and cache size of each measured server |
| `submissions.csv` | 5 | our hosted submissions: date, arm, model, public score as displayed |
| `pass_rule_applications.json` | 3 | the three applications of the pass rule: inputs, every criterion, decision, per-repository totals |
| `e8_runs.csv` | 4636 | case study (Section 9): one line per written result row |
| `e8_compare_table.csv` | 4636 | the same rows in the input format of `drydock-eval compare` |
| `e8_tasks.csv` | 231 | case study, per task: repository, stage, solved repeats in each arm |
| `e8_repositories.csv` | 15 | case study, per repository: tasks, solved runs per arm, summed change against arm F |
| `e8_summary.json` | 1 | case study: the registered analysis results |

## Coded values in `runs.csv`

| Column | Values |
|---|---|
| `experiment` | `pub31` (31B, development tasks), `l4check` (31B on four L4 GPUs), `pubE4` (E4B, development tasks), `berth31`, `berthE4` (Berth tasks on hosted notebooks), `parity` (workstation against hosted notebook), `cand1`, `cand2`, `cand3` (the three programmes of Section 10) |
| `machine` | `colab_emulated_l4` (one GPU with the latency of four L4 GPUs emulated), `kaggle_4xl4` (hosted notebook, four L4 GPUs), `workstation` (one RTX 5090) |
| `task_source` | `competition_dev` (development set of the competition, opaque task codes), `berth` (Berth task set) |
| `condition` | `standard`, `masked_test_runners` (Section 7.4) |
| `resolved` | 1 if the hidden tests of the task pass after the agent's patch, else 0 |
| `end_reason` | `none` (the session ended by itself), `session_timeout`, `tool_budget`, `no_submit` |
| `outcome_id` | workstation runs only: `A00` solved; `A01` unsolved, run completed; `A02` tool-call limit reached; `A05` unsolved at the time limit; `A06` no submission. Empty for runs on other machines |
| `test_exit` | exit code of the test command; 0 = tests passed; -1 = no test was run |

Times are in seconds. `wall_s` is the whole job, `t_agent_s` the agent phase, `t_setup_s` and `t_verify_s` the
set-up and the grading. `emulation_pad_s` is the waiting time added by the latency emulation. Empty cells mean that
the quantity was not recorded for that block. Blocks on hosted notebooks have no start and end times.

In `controls.csv`, `block` is `pool` (the whole task pool on the workstation, without a container),
`runner-final-D`, `runner-final-S`, `runner-final-E` (the final runner on the panels C and H, on the screen set S, and
on the case-study panel with two tasks of S), or `kaggle-cpu-1`, `kaggle-cpu-2` (hosted notebooks).

`SHA256SUMS` lists the sha256 of every file in this directory.
