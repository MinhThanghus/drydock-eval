# E8 freeze record (PROTOCOL 14 item 4, step 7)

Frozen by the lead on 2026-11-10 at about 03:40 UTC, before any E8 agent run. LEAD item 26 (`LEAD_DECISIONS.md`
4a652573..., items 1-26) made text-only changes to PROTOCOL.md just before the freeze (the D4.3 runner named, the
tamper count updated, the H summarizer note, the verifications recorded). Neither code file changed in that step.

- **C** (`manip_check_e8.py`) = `b075ed73acea92e95ef2ac91d9bf1d00d115fd957dae7e0baaa5e065651d0c16`
- **A** (`analyze_e8.py` after the one-line fill-in at line 124) = `16a80a2a92ff9916fd061b0e10e73c4df80c7ccb62c0241d01ca74de9931f2f7`
  (equals the value both item-25 verifiers computed in their own frozen mirrors)
- **P** (`PROTOCOL.md` after the three placeholders and the status line) = `070c9655d96372d054f430727259de7b25af2ca93c3ec4c9295dfdc37aeb8cf4`
  (pre-fill text 11901d70... + LEAD item 26 text = 0b280209...)
- Step 1 hashes (section 15 values, all matched; no `__pycache__`): manip_check_e8 b075ed73, analyze_e8 b679f289 (before
  fill-in), test_analyze_e8 b2c0aa1d, test_manip_check_e8 8a553996, make_jobs_e8 b101bc6b, analyze_e7_pinned e86ccd81,
  RUN_RECORD_SPEC_pinned 258c9246, e8_arms f0ca2eb8, jobs_e8 4fe9fb3a, jobs_e8smoke 394cacc9, jobs_e8capture a7abd121,
  jobs_e8.sha256 7b319ba4, panel_E8 f03c0b93, PROTOCOL_DRAFT fcf9773b, LEAD_DECISIONS 4a652573.
- Step 6: `PY -I -B -m pytest -q -p no:cacheprovider test_analyze_e8.py test_manip_check_e8.py`: 122 passed (101 s); no
  `__pycache__`.
- Diffs for the second-person check (step 7): [scratch] `freeze_e8/analyze_e8.freeze.diff` (1 line) and
  `freeze_e8/PROTOCOL.freeze.diff` (3 placeholders + status line).
- A0 job files (empty drop list `drop_empty.json` 37517e5f, from the D4.3 controls: gold 481/481, empty 0/481):
  `jobs_e8.A0.json` 0c155041..., `jobs_e8smoke.A0.json` bfd48cbd..., `jobs_e8capture.A0.json` f526eea0...
  (`make_jobs_e8.py --drop-file`, written outside E8/).
- From now on: `analyze_e8.py report|reruns ... --checker-sha256 <C> --analyze-e8-sha256 <A>`; any change to either
  code file aborts the block (11.4; LEAD item 20). Code files are set read-only.
- 


[redacted: 57 words, internal process history]
