# Data of the case study and of the pass-rule applications

Outcome rows and analysis results of the paper "Contain, Conform, Compare" (Section 6: five arms, 231 tasks in 15 repositories, 4 repeats; Section 5: the applications of the pass rule). Written by `supp/build_supp.py`, which first recounts the solved runs, the task-level differences, the per-repository sums and the request counts from the result rows and stops if any of them differs from the frozen analysis report (sha256 `1c13f21eac77efead5520f9c6bfe30318f0523f63e68dfcd245cc3465aacf6df`).

The files hold no prompt, request, patch, test output, error text or machine detail. The agent bundle under test is withheld until the competition ends; the arms of the pass-rule applications are therefore named `baseline` and `candidate`.

| File | Records | Content |
|---|---|---|
| `e8_runs.csv` | 4636 | one line per written result row: 4620 analysed runs and 16 rows of interrupted jobs (8 further interrupted jobs wrote no row) |
| `e8_compare_table.csv` | 4636 | the same rows in the input format of `drydock-eval compare` |
| `e8_tasks.csv` | 231 | per task, in the order of the panel file: repository, stage, and solved repeats (0 to 4) in each arm |
| `e8_repositories.csv` | 15 | per repository: tasks, solved runs per arm, summed change against F in task-equivalents (solved runs / 4) |
| `e8_summary.json` | | the analysis results, copied field by field from the frozen report; estimates are fractions of 1, not percentage points |
| `pass_rule_applications.json` | 3 | one record per application of the pass rule, from the frozen gate reports: counts, gain, test, conditions, decision, and `repository_totals` (per repository: tasks and summed gain in task-equivalents) |

## Columns of `e8_runs.csv`

| Column | Content |
|---|---|
| `batch` | 1 to 5: the batch of the block in which the job ran; batch 5 holds the re-runs |
| `stage` | 1, 2 or 3: the nested stage of the job list that holds the task (118, 37 and 76 tasks) |
| `arm` | F, R, C, H or T |
| `task` | task identifier in the public task collection the panel was drawn from |
| `repository` | repository of the task (owner/name); the unit of clustering |
| `repeat` | 1 to 4 |
| `attempt` | 1; 2 for the re-run of an interrupted job |
| `row_class` | agent: an analysed run. infrastructure: an interrupted job, not analysed |
| `outcome_id` | A00 solved; A01 unsolved, run completed; A02 tool-call budget reached; A05 unsolved at the five-minute limit; A06 no submission; A18 unsolved, over-long prompt rejected; I09 interrupted job |
| `resolved` | 1 if the run is solved (the tests of the task pass on the edits the run left), else 0 |
| `t_agent_s` | seconds of the agent phase of the job |
| `t_setup_s` | seconds of the setup phase, before the agent starts |
| `t_verify_s` | seconds of the grading phase; empty if no test was run |
| `model_calls` | chat requests of the run that returned a response |
| `model_calls_failed` | chat requests of the run that raised an error instead |
| `tool_calls` | tool calls made by the agent |
| `tokens_in` | prompt tokens, summed over the requests of the run |
| `tokens_out` | completion tokens, summed over the requests of the run |
| `max_prompt_tokens` | largest prompt of the run, in tokens |
| `summaries` | history summaries (compactions) made during the run |
| `patch_chars` | characters of the diff the run left; 0 if none |
| `test_exit` | exit status of the test command; 0 = tests passed; -1 = no test was run |
| `start_utc` | start of the job (UTC) |
| `end_utc` | end of the job (UTC) |
| `requests_checked` | returned requests of the run that the frozen checker re-rendered |
| `requests_changed` | of them, requests whose prompt the reference rule of arm F would render differently (the exposure count; 0 in F) |
| `check_passed` | 1 if the run passed the prompt check |

Rows of interrupted jobs (`row_class` = `infrastructure`) hold the first ten columns and the two times of day; their other fields are empty.

## Columns of `e8_compare_table.csv`

`arm`, `task`, `repository`, `repeat`, `solved` (0 or 1), `agent_time` (seconds), `row_class` (`agent` or `infra`).

## Reproducing the analysis

```
git clone https://github.com/MinhThanghus/drydock-eval
pip install "./drydock-eval[compare]"
drydock-eval compare signflip e8_compare_table.csv --reference F --repeats 4
```

This returns the estimates, exact p-values, Holm-adjusted p-values and repository-level results of the paper. The 95% intervals of the task-level test are found with frozen Monte Carlo draws and depend on the order of the tasks: the command sorts task identifiers, the registered analysis used the order of the panel file (the order of `e8_tasks.csv`). `supp/reproduce.py` runs both and compares them with `e8_summary.json`.

## Checksums

`SHA256SUMS` lists the sha256 of every file of this directory except itself.
