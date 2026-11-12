# E8: lead decisions on the draft (2026-11-09 ~11:00 UTC, before any E8 run)


[redacted: 19 words, internal process history] No E8 run exists.

1. **Estimand.** Primary: the fixed panel. d_X = task-weighted solve-rate difference X - F on the panel's tasks, tested
   with a task-level exact (or 99,999-draw Monte Carlo if enumeration is infeasible) sign-flip test of per-task arm-mean
   differences, two-sided, Holm-adjusted across the trap arms; claims are worded "on these N tasks". Secondary
   (reported, not a claim rule): the repository-cluster 95% interval (inverted null-imposed Rademacher wild cluster
   bootstrap-t, exact over repositories) for "repositories like these".
2. **Arms.** F, R, C, H, T (five arms), each one factor against F; ADK 1.36.1 named in every row.
   - R: a second vLLM served name containing 'gemma4' (alias string `google/gemma4-e4b-it`), so ADK 1.36.1 sends
     'tool_responses'. The real ADK version table (1.34-1.39 substring; 2.5-2.11 regex for any provider; 2.12+ provider
     exemptions) is reported.
   - C: events_compaction_config unset (swegemma's default when not given).
   - H: the prompt a client lacking adk_submission's reasoning mirror produces: strip both `reasoning` and
     `reasoning_content` from history assistant messages at litellm.acompletion (outermost wrapper, after the patch),
     for every call including the summarizer; vLLM 0.19.1 reads only `reasoning`, so history thoughts vanish server-side.
     Described as such (not as a LiteLLM behaviour).
   - T: the Hugging Face canonical E4B chat template (sha256 0a2c8073...), sent per request
     (--trust-request-chat-template, E8 block only); rows and dumps record only its hash.
3. **Panel and stages.** Nested stages, all arms interleaved by task with the A8 counterbalanced order: stage 1 = panel
   B (118 tasks), stage 2 = A minus B (37), stage 3 = C minus A (76); 4 repeats per arm. The analysis set is every task
   whose stage was fully dispatched before the pre-registered cut-off 2026-11-11 06:00 UTC (outcome-blind); the only
   stop rules are that cut-off, GPU availability and the A8 infrastructure rules.
4. **Primary analysis set:** tasks with all 4 valid slots in both arms of the comparison; an all-rows sensitivity is
   reported if any slot is missing. Context-length 400s (A18) are agent outcomes; time-cap rows (A05) get a sensitivity
   treating them as missing.
5. **Manipulation checks.** Every row of every arm is checked blind on CPU from its dumps: every dumped request,
   re-rendered under the arm's rule, must reproduce vLLM's prompt tokens (ids where logged); the arm's factor must be
   present (role, compaction events, thoughts absent, template hash) and nothing else may differ. A failed check marks
   the row as an infrastructure error (re-run) and is reported.
6. **Recovery rule.** Any fix that changes code or the image aborts the block: rebuild, re-run canaries, restart all
   arms and slots under a new block id; aborted rows are archived unopened and listed as a deviation.
7. **Power statement** as a range over repeat correlation (0.5-0.85), base rate (0.15-0.25) and difficulty model
   (logit-normal, two-level beta), exact-test Holm figures, both one-arm-affected and all-arms-affected scenarios.
8. **Load monitor** (blind, pooled across arms): vLLM pre-emptions and the share of rows with t_agent > 240 s; agent-time
   secondaries are stated as measured under mixed load.
9. **Bundle.** BUNDLE_A [redacted: 5 words, owner bundle]. Before 2026-12-03, publish aggregates, per-task/arm
   outcome tables, analysis code and hashes; request dumps are published with the system prompt replaced by its sha256.
   The full bundle and raw dumps are published on 2026-12-03.
10. **Runner.** 
[redacted: 28 words, other workloads]

## Round 2 decisions (2026-11-09 ~16:45 UTC, still before any E8 run)
11. The pinned analyze_e7 (e86ccd81, v2.2 as verified) is accepted as E8's pinned rules, including the record edge cases
    E7 later fixed under A10; E8's own runner records (D4) carry job_id and attempt on every row.
12. Hook calls: e8_hook_calls counts calls that returned a response; e8_hook_calls_failed counts calls that raised
    (context-length 400s, retried transient errors); failed requests are dumped with their error; manipulation checks
    apply to returned responses only, so arm C's context-length runs stay agent outcomes (A18), not M01.
13. Vendor RUN_RECORD_SPEC.md as it stood for v2.2 (sha256 258c9246...[redacted: 7 words, other experiment state]) as
    FOLDER_E8/RUN_RECORD_SPEC_pinned.md and cite it; E8 never reads the E7 spec in place.
14. The checker (analyze_e8.py) sha256 is frozen in PROTOCOL.md at freeze time; report and reruns refuse records or
    invocations from any other checker version.
15. Resume vs abort after a runner stop is decided blind by the lead from infrastructure facts only; the restart-chain
    manifest entry, the smoke replacement rule and the null-statement detectable-change wording in the current draft
    are approved.
16. The E8 worker (hook fields e8_hook_bound, e8_hook_rebound, e8_hook_calls, e8_hook_calls_failed; both acompletion
    names bound) is built on HOSTA as Track D4.2 after D4.1 and verified against the PROTOCOL 6.2 checklist before any E8 job.

## Round 3 decisions (2026-11-09 ~18:25 UTC, still before any E8 run) - converge to freeze
17. Manipulation checks are computed inside the frozen analysis: analyze_e8 loads the checker from vendored bytes whose
    sha256 is frozen, recomputes every row's manipulation result from the raw dumps at check/report time, and never
    trusts a stored manipulation record; a row whose returned calls pass is never M01.
18. Checklist order: worker checklist items 1-9 and the checker's verification pass before the freeze; items 10-12
    (smoke block, capture block, A18 positive control) pass after the freeze and before the first main-block job; if
    any fails, the recovery rule applies (fix, rebuild, new block id).
19. Item 7(c) is reworded for LiteLLM's internal retries; the runner's own I09 row carries the named keys with null.
20. Re-run cut-off 2026-11-12 00:00 UTC and the A0 rule (tasks failing D4.1 controls are dropped before any run) are
    confirmed. A checker change after the freeze aborts the block (recovery rule).
21. Remaining LOW items are accepted as documented; no further protocol rounds unless a verifier finds HIGH/MEDIUM.

## Round 4 decisions (2026-11-09 ~20:40 UTC, before any E8 run)
22. For the echo arms (F, R, C, T) the checker requires that every history assistant turn that had a thought in its
    returned response carries that thought under both keys with equal text (presence, not only equality); a missing
    history thought in a non-H arm fails the manipulation check.
23. A D4.2 row's llm_dump_sha256 names the worker dump written by W (the outermost wrapper, above every recorder), which
    is consistent with e8_hook_calls by construction; the forwarder capture is kept as secondary evidence only.
24. Turn identity in the presence rule is positional, not textual: a history turn is identified by its tool-call ids
    (worker dump) or by its ordinal position among the trajectory's assistant turns (capture); two turns with identical
    text are distinct turns, and a turn whose own response had no thought is never required to carry one.
25. A runner tamper row (I12, error 'g4s-d<N> tamper: ...') in an E8 block is treated like the runner's own I09 row: the
    hook keys need only be present (null allowed), no hook or manipulation check applies, it is an infrastructure row
    (re-run per A8), and it never voids the block by itself. Tamper evidence is reported after unblinding.

## Round 5 decision (2026-11-10 ~03:30 UTC, before any E8 run)
26. After the D4.3 build and its independent verification (`R/experiments/FOLDER_RUNNER_EVAL/verifier_report_d43.json`: no
    HIGH/MEDIUM; the isolation condition stated in the protocol holds as deployed) and a second independent verification of item 25
    (`verification/verifier_report_e8_i25_rerun.json`: no HIGH/MEDIUM):
    - E8 blocks run on the D4.3 runner (`run_berth.py` 9c05d6db..., `lib/g4s_d4.py` bb5d0377...; image v8 fa686273...,
      flags 307a018e..., options, vLLM args and model unchanged). Its row-writing code behaves exactly as D4.2's
      (verified statically and on the d43h and d43c1 rows), so the analysis validated on D4.2 rows applies. PROTOCOL
      4 and 6.3 name it, and 6.3's tamper-row count is updated.
    - The gate file afa3e50f (scope other blocks of the same project and E8) is enabled only after the E8 freeze; no E8 main-block job starts
      before checklist items 10-12 pass.
    - Every agent-mode GPU run from now on (validation, smoke, capture, the A18 control, the main block) uses the
      default gate; `--gate precanary` (it binds only image, runner, library and harness) is not used.
    - The runner's I09 row naming a truncated worker dump after a PID-1 deadline kill needs no rule: runner I09 rows
      get no manipulation record (6.3, 6.4).
    - Accepted as documented residuals (operator level, LOW/INFO): a hard link to a block's manifest, a canary run
      creating another block's not-yet-existing manifest, the shared pty pool, result frames up to about 1.3 GB under
      the 3 GiB ceiling, the GPU lock checked by inode, the upstream-port check-then-use gap. Blind masking stays as
      implemented (per-arm I12 counts and the tamper prefix are visible blind; refusal kinds only after unblinding).
    - PROTOCOL 13 gets the H summarizer note. All of this is text: `analyze_e8.py` and `manip_check_e8.py` are
      unchanged, so C and A are unchanged; P is recomputed at the freeze.
