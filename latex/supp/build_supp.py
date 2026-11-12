"""Build the supplement's generated tables (gen/s_*.tex, gen/supp_macros.tex) and the public data files from the
analysis outputs of the case study (Section 6 of the paper) and of the pass-rule applications (Section 5).

usage (python -I -B):
  build_supp.py --e8-report e8_report.json --e8-manip e8_report.manip.jsonl --e8-runs <dir with one folder per batch: <name>_r<batch>/out/results.jsonl>
                --panel panel_E8.json --gate "<label>::<task set, in words>::<gate report>" [--gate ...]
                --data <out dir> --gen <out dir>

No table cell of the supplement is typed by hand: this script writes them. Before it writes anything it recounts, from
the result rows, the solved rows per arm, the task-level differences, the per-repository sums and the request counts,
and stops if any of them differs from the frozen analysis report.

The data files hold outcome rows and analysis results only: no prompt, no request, no patch, no test output, no
error text, no path, no machine detail. Fields are copied by name from a fixed list (COLUMNS, and the keys named in
summary()); nothing else of a row or of a report is written.
"""
import argparse, collections, csv, hashlib, io, json, pathlib, re, sys
from fractions import Fraction

ARMS = ['F', 'R', 'C', 'H', 'T']
TRAPS = ARMS[1:]
REPEATS = 4
OUTCOMES = [('A00', 'solved'), ('A01', 'unsolved, run completed'), ('A05', 'five-minute limit'),
            ('A18', 'over-long prompt rejected'), ('A02', 'tool-call budget (28)'), ('A06', 'no submission')]
INFRA_TEXT = {'g4s-d4 infra: no frame': 'job container ended without a result frame',
              'job in started.jsonl without a results row': 'job started, no result row'}
MEASURES = [('summaries', 'history summaries per run', 2), ('any_compaction', 'runs with a summary (share)', 3),
            ('max_prompt', 'largest prompt (tokens)', 0), ('context_length_unsolved', 'over-long prompt rejected (share)', 3),
            ('time_cap', 'five-minute limit (share)', 3), ('tool_budget', 'tool-call budget reached (share)', 3),
            ('no_submit', 'no submission (share)', 3), ('t_agent', 'agent time (s)', 1), ('llm_calls', 'model calls', 2),
            ('llm_calls_failed', 'failed model calls', 3), ('tool_calls', 'tool calls', 2),
            ('tok_in', 'prompt tokens, summed', 0), ('tok_out', 'completion tokens', 0)]
COLUMNS = ['batch', 'stage', 'arm', 'task', 'repository', 'repeat', 'attempt', 'row_class', 'outcome_id', 'resolved',
           't_agent_s', 't_setup_s', 't_verify_s', 'model_calls', 'model_calls_failed', 'tool_calls', 'tokens_in',
           'tokens_out', 'max_prompt_tokens', 'summaries', 'patch_chars', 'test_exit', 'start_utc', 'end_utc',
           'requests_checked', 'requests_changed', 'check_passed']
COLDOC = [
    ('batch', '1 to 5: the batch of the block in which the job ran; batch 5 holds the re-runs'),
    ('stage', '1, 2 or 3: the nested stage of the job list that holds the task (118, 37 and 76 tasks)'),
    ('arm', 'F, R, C, H or T'),
    ('task', 'task identifier in the public task collection the panel was drawn from'),
    ('repository', 'repository of the task (owner/name); the unit of clustering'),
    ('repeat', '1 to 4'),
    ('attempt', '1; 2 for the re-run of an interrupted job'),
    ('row_class', 'agent: an analysed run. infrastructure: an interrupted job, not analysed'),
    ('outcome_id', 'A00 solved; A01 unsolved, run completed; A02 tool-call budget reached; A05 unsolved at the five-minute limit; '
                   'A06 no submission; A18 unsolved, over-long prompt rejected; I09 interrupted job'),
    ('resolved', '1 if the run is solved (the tests of the task pass on the edits the run left), else 0'),
    ('t_agent_s', 'seconds of the agent phase of the job'),
    ('t_setup_s', 'seconds of the setup phase, before the agent starts'),
    ('t_verify_s', 'seconds of the grading phase; empty if no test was run'),
    ('model_calls', 'chat requests of the run that returned a response'),
    ('model_calls_failed', 'chat requests of the run that raised an error instead'),
    ('tool_calls', 'tool calls made by the agent'),
    ('tokens_in', 'prompt tokens, summed over the requests of the run'),
    ('tokens_out', 'completion tokens, summed over the requests of the run'),
    ('max_prompt_tokens', 'largest prompt of the run, in tokens'),
    ('summaries', 'history summaries (compactions) made during the run'),
    ('patch_chars', 'characters of the diff the run left; 0 if none'),
    ('test_exit', 'exit status of the test command; 0 = tests passed; -1 = no test was run'),
    ('start_utc', 'start of the job (UTC)'),
    ('end_utc', 'end of the job (UTC)'),
    ('requests_checked', 'returned requests of the run that the frozen checker re-rendered'),
    ('requests_changed', 'of them, requests whose prompt the reference rule of arm F would render differently (the exposure count; 0 in F)'),
    ('check_passed', '1 if the run passed the prompt check'),
]
assert [c for c, _ in COLDOC] == COLUMNS


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def die(msg):
    sys.exit('build_supp: ' + msg)


def check(cond, msg):
    if not cond: die('check failed: ' + msg)


# ---------------------------------------------------------------------------------------------------------- loading
def load_rows(runs, panel, manip):
    rows = []
    for b in sorted(p for p in pathlib.Path(runs).iterdir() if p.is_dir() and (p / 'out/results.jsonl').exists()):
        for ln in open(b / 'out/results.jsonl'):
            r = json.loads(ln)
            check(r['run_tag'] == b.name and r['block'] == 'E8', 'run tag / block of a row in %s' % b.name)
            rows.append(r)
    out = []
    for r in rows:
        m = manip.get(r['job_id']) if r['error_class'] == 'agent' else None
        check(r['error_class'] != 'agent' or m is not None, 'agent row without a prompt-check record: ' + r['job_id'])
        g = lambda k: '' if r.get(k) is None else r[k]
        out.append(dict(
            batch=int(re.fullmatch(r'\w+_r(\d)', r['run_tag']).group(1)), stage=r['stage'], arm=r['arm'], task=r['task'],
            repository=panel['task_repo'][r['task']], repeat=r['repeat_id'], attempt=r['attempt'],
            row_class={'agent': 'agent', 'infra': 'infrastructure'}[r['error_class']], outcome_id=r['error_class_id'],
            resolved=int(bool(r['resolved'])), t_agent_s=g('t_agent'), t_setup_s=g('t_setup'), t_verify_s=g('t_verify'),
            model_calls=g('e8_hook_calls'), model_calls_failed=g('e8_hook_calls_failed'), tool_calls=g('tool_calls'),
            tokens_in=g('tok_in'), tokens_out=g('tok_out'), max_prompt_tokens=g('max_prompt'), summaries=g('summaries'),
            patch_chars=g('patch_chars'), test_exit=g('test_exit'), start_utc=g('t_start_utc'), end_utc=g('t_end_utc'),
            requests_checked='' if m is None else m['n_requests'], requests_changed='' if m is None else m['n_requests_changed'],
            check_passed='' if m is None else int(bool(m['pass']))))
    key = lambda x: (x['batch'], x['start_utc'], x['arm'], x['task'], x['repeat'], x['attempt'])
    return sorted(out, key=key)


# ------------------------------------------------------------------------------------------------- recount and check
def recount(rows, panel, rep):
    res = rep['results']
    ag = [r for r in rows if r['row_class'] == 'agent']
    slots = collections.Counter((r['arm'], r['task'], r['repeat']) for r in ag)
    check(len(slots) == len(ARMS) * len(panel['tasks']) * REPEATS and max(slots.values()) == 1, 'one agent row per slot')
    check(set(r['task'] for r in ag) == set(panel['tasks']), 'tasks of the rows = tasks of the panel')
    solved = collections.Counter()                     # (arm, task) -> solved repeats
    for r in ag: solved[(r['arm'], r['task'])] += r['resolved']
    for a in ARMS:
        n = sum(solved[(a, t)] for t in panel['tasks'])
        check(n == res['solve_rates'][a]['solved_rows'], 'solved rows of arm %s: %d' % (a, n))
        for k, v in res['solve_rates'][a]['solved_per_repeat'].items():
            check(v == sum(r['resolved'] for r in ag if r['arm'] == a and r['repeat'] == int(k)), 'solved per repeat')
        check(sum(r['requests_checked'] for r in ag if r['arm'] == a) == res['exposure'][a]['requests'], 'requests checked')
        check(sum(r['requests_changed'] for r in ag if r['arm'] == a) == res['exposure'][a]['requests_changed'], 'requests changed')
    check(all(r['check_passed'] == 1 for r in ag), 'every agent row passed the prompt check')
    COL = dict(t_agent='t_agent_s', llm_calls='model_calls', llm_calls_failed='model_calls_failed', tool_calls='tool_calls',
               tok_in='tokens_in', tok_out='tokens_out', summaries='summaries', max_prompt='max_prompt_tokens')
    for a in ARMS:                                     # the exported columns give the report's per-arm means
        m = res['comparisons'][a if a != 'F' else 'R']['row_secondary']['arms'][a]
        x = [r for r in ag if r['arm'] == a]
        for k, col in COL.items():
            check(m[k]['n'] == len(x) and abs(sum(r[col] for r in x) / len(x) - m[k]['mean']) < 1e-6, 'mean of %s in arm %s' % (col, a))
        check(abs(sum(r['outcome_id'] == 'A05' for r in x) / len(x) - m['time_cap']['mean']) < 1e-9, 'time-limit share, arm ' + a)
        check(abs(sum(r['outcome_id'] == 'A18' for r in x) / len(x) - m['context_length_unsolved']['mean']) < 1e-9, 'rejected-prompt share, arm ' + a)
        check(abs(sum(r['summaries'] > 0 for r in x) / len(x) - m['any_compaction']['mean']) < 1e-9, 'compaction share, arm ' + a)
    repos = sorted(set(panel['task_repo'].values()))
    for a in TRAPS:
        c = res['comparisons'][a]['primary']
        K = {t: solved[(a, t)] - solved[('F', t)] for t in panel['tasks']}
        check(sum(K.values()) == c['sum_K'] and Fraction(sum(K.values()), REPEATS * len(K)) == Fraction(c['d_exact']), 'd of arm ' + a)
        check(sum(1 for v in K.values() if v) == c['n_nonzero'], 'tasks with a non-zero difference, arm ' + a)
        for g in repos:
            s = Fraction(sum(K[t] for t in K if panel['task_repo'][t] == g), REPEATS)
            check(s == Fraction(c['repository']['repo_table'][g]['sum_gain_exact']), 'repository sum %s %s' % (a, g))
    return solved, repos


# ----------------------------------------------------------------------------------------------------- data products
def write_csv(path, header, lines):
    buf = io.StringIO(newline='')
    w = csv.writer(buf, lineterminator='\n'); w.writerow(header); w.writerows(lines)
    pathlib.Path(path).write_text(buf.getvalue())


def summary(rep, rep_sha, panel, rows):
    res, bs = rep['results'], rep['blind_sections']
    pick = lambda d, keys: {k: d[k] for k in keys}
    comp = {}
    for a in TRAPS:
        c = res['comparisons'][a]; p = c['primary']; g = p['repository']
        loo = [v['loo'] for v in g['repo_table'].values()]
        comp[a] = dict(
            primary=dict(pick(p, ['n_tasks', 'd', 'd_exact', 'p_exact', 'p_exact_fraction', 'n_nonzero', 'sum_K', 'p_mc', 'mc_draws', 'mc_seed']),
                         interval95=pick(p['interval95'], ['lo', 'hi', 'level', 'method']),
                         interval_simultaneous=pick(p['interval_simultaneous'], ['lo', 'hi', 'level', 'method']),
                         t_interval95=pick(p['t_interval95'], ['lo', 'hi'])),
            holm_adjusted_p=c['holm_adjusted_p'], claim=c['claim'],
            repository=dict(pick(g, ['n_repos', 'n_clusters_nonzero', 'p_two_exact', 'min_attainable_p_two', 'p_two_mc', 'p_two_webb', 'repo_equal']),
                            cr1=pick(g['cr1'], ['lo', 'hi']), interval95=pick(g['interval95'], ['lo', 'hi', 'method']),
                            loo_min=min(loo), loo_max=max(loo),
                            repo_table={k: pick(v, ['n_tasks', 'sum_gain_exact', 'mean_gain', 'loo']) for k, v in g['repo_table'].items()}),
            sensitivity={k: {kk: vv for kk, vv in v.items() if kk in ('n_tasks', 'd', 'd_exact', 'p_exact', 'n_nonzero', 'sum_K', 'n_time_cap_rows', 'n_penalised')}
                         for k, v in c['sensitivity'].items() if k in ('time_cap_as_missing', 'd_pen_x', 'd_pen_f')},
            paired_run_measures={k: pick(c['row_secondary']['paired_differences'][k], ['n', 'G', 'mean', 'lo', 'hi']) for k, _, _ in MEASURES},
            arm_run_measures={k: pick(c['row_secondary']['arms'][a][k], ['n', 'mean', 'median', 'p95']) for k, _, _ in MEASURES})
    f_meas = {k: pick(res['comparisons']['R']['row_secondary']['arms']['F'][k], ['n', 'mean', 'median', 'p95']) for k, _, _ in MEASURES}
    for a in TRAPS:
        check(all(res['comparisons'][a]['row_secondary']['arms']['F'][k]['mean'] == f_meas[k]['mean'] for k, _, _ in MEASURES), 'F measures are the same in every comparison')
    infra = {a: dict(rows=v['rows'], by_batch={k[-1]: n for k, n in v['by_batch'].items()},
                     kinds={INFRA_TEXT[k]: n for k, n in v['details'].items()}) for a, v in bs['infrastructure'].items()}
    lm, mc, sl = bs['load_monitor'], bs['manipulation_checks'], bs['slots']
    return dict(
        about='Results of the pre-registered five-arm experiment of Section 6 of the paper, copied field by field from the frozen '
              'analysis report (sha256 below). Estimates d are differences in solve rate (arm minus F) as fractions of 1, not percentage points.',
        analysis_report_sha256=rep_sha, status=rep['status'], report_as_of_utc=rep['as_of'], cutoff_utc=rep['cutoff'],
        design=dict(tasks=len(panel['tasks']), repositories=len(set(panel['task_repo'].values())), repeats=rep['repeats'], arms=ARMS,
                    family=rep['family'], alpha=rep['alpha'], monte_carlo_draws=rep['n_mc'], monte_carlo_seed=rep['mc_seed']),
        frozen_code_sha256=pick(rep['code_sha256'], ['analyze_e8', 'make_jobs_e8', 'analyze_e7_pinned', 'manip_check_e8']),
        arms_file_sha256=rep['e8_arms_sha256'],
        solve_rates={a: dict(rate=v['task_weighted']['mean'], cr1_lo=v['task_weighted']['lo'], cr1_hi=v['task_weighted']['hi'],
                             solved_rows=v['solved_rows'], solved_per_repeat=v['solved_per_repeat'], used_rows=v['used_rows'])
                     for a, v in res['solve_rates'].items()},
        holm=pick(res['holm'], ['adjusted', 'reject', 'alpha', 'family']), comparisons=comp, f_run_measures=f_meas,
        exposure={a: pick(v, ['rows', 'requests', 'requests_changed', 'share_changed', 'failed_calls_dumped', 'rows_failed_calls_not_matching_dump'])
                  for a, v in res['exposure'].items()},
        slots=dict(planned=sl['n_planned'], with_used_row=sl['n_with_used_row'], attempts=sl['attempts_hist'], per_arm=sl['per_arm']),
        infrastructure_rows=infra, tamper_rows=res['tamper_rows']['n_rows'],
        prompt_checks=dict(rows_checked=mc['rows_checked'], passed=mc['passed'], failed=len(mc['failed'])),
        load=dict(threshold_s=lm['threshold_s'], rows_above=lm['rows_above'], rows=lm['rows_with_t_agent'], share_above=lm['share_above'],
                  flag_share=lm['flag_share'], preemptions=lm['preemptions']))


def gate_record(label, task_set, path):
    """One application of the pass rule: the fields the paper and the supplement use, under neutral arm names."""
    d = json.load(open(path)); out = dict(application=label, task_set=task_set, gate_report_sha256=sha(path), decision=d['decision'], repeats=None)
    if d.get('kind') == 'gate':                                    # the first application's report layout
        g, b = d['gain'], d['gain']['bootstrap']; A, B = d['arm_a'], d['arm_b']
        arms = d['secondary']['arms']
        sol = {k: round(arms[k]['solved']['mean'] * arms[k]['solved']['n']) for k in (A, B)}
        out.update(tasks=d['n_tasks_used'], repositories=g['n_repos'], repeats=d['repeats'], slots=d['slots']['n_planned'],
                   slots_with_row=d['slots']['n_with_used_row'], runs_per_arm=arms[A]['solved']['n'],
                   solved=dict(baseline=sol[A], candidate=sol[B]), gain_exact=g['d_exact'], gain=g['d'],
                   repositories_positive=g['n_repos_positive'], repositories_negative=g['n_repos_negative'], loo_min=g['loo_min'],
                   time=dict(ratio=d['time']['ratio'], mean_baseline_s=d['time']['mean_a'], mean_candidate_s=d['time']['mean_b']),
                   infrastructure_rows=dict(baseline=d['harness']['by_arm'][A]['rows'], candidate=d['harness']['by_arm'][B]['rows']),
                   worst_repository_total=d['guard_h']['min_total'],
                   criteria={k: d['criteria'][k] for k in ('gain', 'p', 'pos_repos', 'loo', 'time', 'harness', 'guard_h')},
                   thresholds={k: d['thresholds'][k] for k in ('gain_min', 'alpha', 'min_pos_repos', 'time_ratio_max', 'guard_h_limit', 'n_boot', 'seed')},
                   panel_sha256=d['panel_sha256'])
    else:                                                          # the later applications' report layout
        g, b = d['core'], d['core']['bootstrap']; A, B = d['baseline'], d['candidate']; s = d['arm_summary']
        out.update(tasks=d['n_tasks_used'], repositories=g['n_repos'], repeats=d['provenance']['repeats'],
                   slots=s[A]['planned'] + s[B]['planned'], slots_with_row=s[A]['rows'] + s[B]['rows'], runs_per_arm=s[A]['planned'],
                   solved=dict(baseline=s[A]['solved'], candidate=s[B]['solved']), gain_exact=d['gain_exact'], gain=d['gain'],
                   repositories_positive=g['n_repos_positive'], repositories_negative=g['n_repos_negative'], loo_min=g['loo_min'],
                   time=dict(ratio=d['time']['ratio'], mean_baseline_s=d['time']['mean_baseline'], mean_candidate_s=d['time']['mean_candidate']),
                   infrastructure_rows=dict(baseline=s[A]['infra_rows'], candidate=s[B]['infra_rows']),
                   worst_repository_total=float(min(Fraction(v['sum_gain_exact']) for v in g['repos'].values())),
                   criteria={k: d['criteria'][k] for k in ('gain', 'p', 'pos_repos', 'loo', 'time', 'harness')},
                   thresholds={k: d['thresholds'][k] for k in ('gain_min', 'alpha', 'min_pos_repos', 'time_ratio_max', 'n_boot', 'seed')},
                   panel_sha256=d['provenance']['panel_sha256'])
    check(Fraction(out['gain_exact']) == Fraction(out['solved']['candidate'] - out['solved']['baseline'], out['runs_per_arm']), 'gain = solved difference / runs (%s)' % label)
    out['bootstrap'] = {k: b[k] for k in ('t', 'p', 'count', 'n_boot', 'seed', 'n_clusters', 'n_clusters_nonzero', 'p_exact_enumeration')}
    # per repository: tasks and the summed gain in task-equivalents (sum over its tasks of candidate minus baseline solved repeats, / repeats)
    tot = sorted(((Fraction(v['sum_gain_exact']), k, v['n_tasks']) for k, v in g['repos'].items()))
    out['repository_totals'] = [dict(repository=k, tasks=n, sum_gain_exact=str(f), sum_gain=float(f)) for f, k, n in tot]
    check(len(tot) == out['repositories'] and sum(n for _, _, n in tot) == out['tasks'], 'repository totals cover every task (%s)' % label)
    check(sum(f for f, _, _ in tot) / out['tasks'] == Fraction(out['gain_exact']), 'mean gain = sum of the repository totals / tasks (%s)' % label)
    check((sum(f > 0 for f, _, _ in tot), sum(f < 0 for f, _, _ in tot)) == (out['repositories_positive'], out['repositories_negative']),
          'repositories gaining and losing (%s)' % label)
    check(float(tot[0][0]) == out['worst_repository_total'], 'worst repository total (%s)' % label)
    return out


# -------------------------------------------------------------------------------------------------------- LaTeX
def num(x, nd=0):
    s = ('%%.%df' % nd) % x
    if nd == 0: s = '{:,}'.format(int(s)).replace(',', '{,}')
    return s


def sgn(x, nd):
    """A signed number in math mode."""
    s = ('%%+.%df' % nd) % x
    if float(s) == 0: s = s.replace('-', '+')
    if nd == 0: s = '{:+,}'.format(int(s)).replace(',', '{,}')
    return '$' + s.replace('+', '{+}').replace('-', '{-}') + '$'


def iv(lo, hi, nd, scale=1.0):
    return '[%s, %s]' % (sgn(lo * scale, nd), sgn(hi * scale, nd))


def intc(n):
    return '{:,}'.format(int(n)).replace(',', '{,}')


def table(gen, name, caption, spec, head, lines, size='\\footnotesize', note=None, place='!htb'):
    body = ' \\\\\n'.join(' & '.join(str(c) for c in ln) for ln in lines)
    tex = ('\\begin{table}[%s]\n\\centering\n\\caption{%s}\\label{tab:s-%s}\n%s\n\\begin{tabular}{%s}\n\\toprule\n%s \\\\\n\\midrule\n%s \\\\\n\\bottomrule\n\\end{tabular}\n'
           % (place, caption, name, size, spec, ' & '.join(head), body))
    if note: tex += '\n\\par\\smallskip\\parbox{0.96\\linewidth}{\\footnotesize %s}\n' % note
    (gen / ('s_%s.tex' % name)).write_text(tex + '\\end{table}\n')


def esc(s):
    return s.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')


def tables(gen, S, rows, panel, solved, repos, gates, files):
    ag = [r for r in rows if r['row_class'] == 'agent']
    C = S['comparisons']
    # outcomes per arm
    cnt = collections.Counter((r['arm'], r['outcome_id']) for r in ag)
    full = [r for r in ag if r['outcome_id'] == 'A00' and r['t_agent_s'] >= 300.0]          # solved, at the five-minute limit
    check(set(k for _, k in cnt) <= {k for k, _ in OUTCOMES}, 'an outcome class that the table does not list')
    check(cnt[('F', 'A18')] == 0, 'no F run ended on a rejected prompt')
    table(gen, 'outcomes', 'Outcome of every analysed run, by arm (one row per task, arm and repeat).', 'l' + 'r' * (len(OUTCOMES) + 1),
          ['Arm'] + ['\\parbox[b]{1.75cm}{\\raggedleft %s}' % lab.capitalize() for _, lab in OUTCOMES] + ['Runs'],
          [[a] + [intc(cnt[(a, k)]) for k, _ in OUTCOMES] + [intc(sum(cnt[(a, k)] for k, _ in OUTCOMES))] for a in ARMS],
          note='Only the runs of the first column are solved; the other columns split the unsolved runs by how the run ended. A run that '
               'ended at a limit was still graded on the edits it had made: %d solved runs lasted the full five minutes (all in arm %s), and '
               '%d of the %d runs unsolved at that limit had left edits that were tested.'
               % (len(full), '/'.join(sorted(set(r['arm'] for r in full))), sum(1 for r in ag if r['outcome_id'] == 'A05' and r['patch_chars'] > 0),
                  sum(1 for r in ag if r['outcome_id'] == 'A05')))
    # solve rates
    sr = S['solve_rates']
    table(gen, 'rates', 'Solve rate per arm: task-weighted mean with a 95\\% interval from a repository-clustered (CR1) standard error; solved runs per repeat.',
          'lrrrrrrrr', ['Arm', 'Solve rate (\\%)', '95\\% interval', 'Solved runs', 'Repeat 1', '2', '3', '4', 'Runs'],
          [[a, num(100 * sr[a]['rate'], 2), '[%s, %s]' % (num(100 * sr[a]['cr1_lo'], 1), num(100 * sr[a]['cr1_hi'], 1)), sr[a]['solved_rows']]
           + [sr[a]['solved_per_repeat'][str(k)] for k in range(1, 5)] + [sr[a]['used_rows']] for a in ARMS])
    # tasks by solved repeats
    never = sum(1 for t in panel['tasks'] if all(solved[(a, t)] == 0 for a in ARMS))
    always = sum(1 for t in panel['tasks'] if all(solved[(a, t)] == REPEATS for a in ARMS))
    hist = {a: collections.Counter(solved[(a, t)] for t in panel['tasks']) for a in ARMS}
    table(gen, 'taskhist', 'Tasks by the number of solved repeats (of four), per arm.', 'l' + 'r' * (REPEATS + 2),
          ['Arm'] + ['%d solved' % k for k in range(REPEATS + 1)] + ['Tasks'],
          [[a] + [hist[a][k] for k in range(REPEATS + 1)] + [sum(hist[a].values())] for a in ARMS],
          note='%d of the %d tasks were solved in no run of any arm (0 of %d runs) and %d in every run; the other %d tasks carry every difference between arms.'
               % (never, len(panel['tasks']), len(ARMS) * REPEATS, always, len(panel['tasks']) - never - always))
    # primary
    down = {a: sum(1 for t in panel['tasks'] if solved[(a, t)] < solved[('F', t)]) for a in TRAPS}
    up = {a: sum(1 for t in panel['tasks'] if solved[(a, t)] > solved[('F', t)]) for a in TRAPS}
    for a in TRAPS: check(down[a] + up[a] == C[a]['primary']['n_nonzero'], 'tasks lower + higher = tasks that differ, arm ' + a)
    table(gen, 'primary', 'Primary analysis: change in solve rate against F in percentage points, tasks with fewer and with more solved repeats than in F, exact two-sided task-level sign-flip test, Holm adjustment over the four arms, and intervals from inverting the test.',
          'lrrrrrrrr', ['Arm', '$d$', 'Exact $d$', '\\parbox[b]{1.5cm}{\\raggedleft Tasks lower; higher}', 'Exact $p$', 'Holm $p$',
                        '\\parbox[b]{1.7cm}{\\raggedleft Monte Carlo $p$}', '95\\% interval', '98.75\\% simultaneous'],
          [[a, sgn(100 * C[a]['primary']['d'], 2), '$%s$' % C[a]['primary']['d_exact'].replace('-', '{-}'), '%d; %d' % (down[a], up[a]),
            num(C[a]['primary']['p_exact'], 4), num(C[a]['holm_adjusted_p'], 3), num(C[a]['primary']['p_mc'], 4),
            iv(C[a]['primary']['interval95']['lo'], C[a]['primary']['interval95']['hi'], 2, 100),
            iv(C[a]['primary']['interval_simultaneous']['lo'], C[a]['primary']['interval_simultaneous']['hi'], 2, 100)] for a in TRAPS], size='\\footnotesize\\setlength{\\tabcolsep}{4pt}')
    # repository level
    table(gen, 'repolevel', 'Secondary analysis at the repository level (15 repositories): exact wild cluster bootstrap-$t$ with Rademacher weights, its inverted 95\\% interval, the six-point (Webb) variant, a CR1 interval, the repository-equal estimate and the range of leave-one-repository-out estimates. Estimates in percentage points.',
          'lrrrrrrr', ['Arm', '\\parbox[b]{1.9cm}{\\raggedleft Repositories that differ}', 'Exact $p$', 'Webb $p$', '95\\% interval', 'CR1 interval',
                       '\\parbox[b]{1.9cm}{\\raggedleft Repository-equal}', 'Leave-one-out range'],
          [[a, C[a]['repository']['n_clusters_nonzero'], num(C[a]['repository']['p_two_exact'], 4), num(C[a]['repository']['p_two_webb'], 4),
            iv(C[a]['repository']['interval95']['lo'], C[a]['repository']['interval95']['hi'], 2, 100),
            iv(C[a]['repository']['cr1']['lo'], C[a]['repository']['cr1']['hi'], 2, 100), sgn(100 * C[a]['repository']['repo_equal'], 2),
            iv(C[a]['repository']['loo_min'], C[a]['repository']['loo_max'], 2, 100)] for a in TRAPS], size='\\footnotesize\\setlength{\\tabcolsep}{4pt}')
    # per repository
    ntask = collections.Counter(panel['task_repo'].values())
    order = sorted(repos, key=lambda g: (-ntask[g], g))
    def tot(a, g): return sum(solved[(a, t)] for t in panel['tasks'] if panel['task_repo'][t] == g)
    def gain(a, g):
        f = Fraction(C[a]['repository']['repo_table'][g]['sum_gain_exact'])
        return sgn(float(f), 2) if f else '0'
    table(gen, 'repos', 'The 15 repositories of the panel: tasks, runs solved in arm F (of four per task) and, per arm, the summed change against F in task-equivalents (solved runs divided by four).',
          'lrrrrrr', ['Repository', 'Tasks', 'F solved runs', 'R', 'C', 'H', 'T'],
          [['\\texttt{%s}' % esc(g), ntask[g], '%d of %d' % (tot('F', g), REPEATS * ntask[g])] + [gain(a, g) for a in TRAPS] for g in order]
          + [['\\midrule\nAll', sum(ntask.values()), '%d of %d' % (sr['F']['solved_rows'], sr['F']['used_rows'])]
             + [sgn(C[a]['primary']['sum_K'] / REPEATS, 2) for a in TRAPS]])
    # sensitivities
    SENS = [('time_cap_as_missing', 'five-minute-limit runs of both arms treated as missing', 'n_time_cap_rows'),
            ('d_pen_x', 'slots of the arm with an infrastructure row scored unsolved', 'n_penalised'),
            ('d_pen_f', 'slots of F with an infrastructure row scored unsolved', 'n_penalised')]
    lines = []
    for a in TRAPS:
        for i, (k, lab, nk) in enumerate(SENS):
            s = C[a]['sensitivity'][k]
            lines.append([a if i == 0 else '', lab, s['n_tasks'], s[nk], sgn(100 * s['d'], 2), num(s['p_exact'], 4)])
    table(gen, 'sens', 'Pre-registered sensitivity analyses of the primary estimate (percentage points; exact two-sided sign-flip $p$, unadjusted).',
          'llrrrr', ['Arm', 'Analysis', 'Tasks', 'Runs affected', '$d$', 'Exact $p$'], lines)
    # exposure
    ex = S['exposure']
    failed = [r for r in ag if r['model_calls_failed'] > 0]
    fc = collections.Counter(r['outcome_id'] for r in failed)
    check(all(r['model_calls_failed'] == 1 for r in failed) and set(fc) == {'A00', 'A05', 'A18'}, 'failed calls: one per run, three outcome classes')
    for a in ARMS: check(sum(r['model_calls_failed'] for r in ag if r['arm'] == a) == ex[a]['failed_calls_dumped'], 'failed calls of arm ' + a)
    check(all(r['model_calls'] == r['requests_checked'] for r in ag), 'returned requests = requests checked, in every run')
    table(gen, 'exposure', 'Exposure: requests re-rendered by the frozen checker, and those whose rendered prompt the arm\'s divergence changed.',
          'lrrrrr', ['Arm', 'Runs', 'Requests checked', 'Changed by the divergence', 'Share (\\%)', 'Failed calls recorded'],
          [[a, intc(ex[a]['rows']), intc(ex[a]['requests']), intc(ex[a]['requests_changed']), num(100 * ex[a]['share_changed'], 1), ex[a]['failed_calls_dumped']] for a in ARMS],
          note='No run has more than one failed call. Of the %d failed calls, %d are in runs unsolved at the five-minute limit, %d in runs that '
               'ended on a rejected over-long prompt and %d in solved runs.' % (len(failed), fc['A05'], fc['A18'], fc['A00']))
    # run measures (exploratory)
    lines = []
    for k, lab, nd in MEASURES:
        lines.append([lab, num(S['f_run_measures'][k]['mean'], nd)] + [sgn(C[a]['paired_run_measures'][k]['mean'], nd) for a in TRAPS])
        lines.append(['', ''] + ['{\\scriptsize %s}' % iv(C[a]['paired_run_measures'][k]['lo'], C[a]['paired_run_measures'][k]['hi'], nd) for a in TRAPS])
        if k != MEASURES[-1][0]: lines[-1][-1] += ' \\\\[-2pt]\n\\addlinespace[2pt]%'
    table(gen, 'measures', 'Exploratory run measures: mean in arm F, and the paired difference of each arm against F; below it, its 95\\% interval (CR1, repository clusters). No multiplicity adjustment.',
          'lrrrrr', ['Measure', 'F mean', 'R $-$ F', 'C $-$ F', 'H $-$ F', 'T $-$ F'], lines, size='\\footnotesize\\setlength{\\tabcolsep}{3.5pt}')
    # execution record
    by = collections.Counter((r['batch'], r['row_class']) for r in rows)
    infra_all = collections.Counter()
    for a in ARMS:
        for b, n in S['infrastructure_rows'][a]['by_batch'].items(): infra_all[int(b)] += n
    ENDED = {1: 'runner reached its open-file limit', 2: 'external termination signal', 3: 'model server ended (out of memory)',
             4: 'job list completed', 5: 're-runs of the interrupted jobs completed'}
    lines, tot_started = [], 0
    batch_lines = lines
    for b in sorted(set(k for k, _ in by)):
        norow = infra_all[b] - by[(b, 'infrastructure')]
        tot_started += by[(b, 'agent')] + infra_all[b]
        lines.append([b, intc(by[(b, 'agent')] + infra_all[b]), intc(by[(b, 'agent')]), by[(b, 'infrastructure')], norow, ENDED[b]])
    table(gen, 'batches', 'Execution record: the block ran as four batches and one batch of re-runs. An interrupted job is an infrastructure row, with or without a written result row; its slot was re-run once.',
          'rrrrrl', ['Batch', '\\parbox[b]{1.2cm}{\\raggedleft Jobs started}', '\\parbox[b]{1.4cm}{\\raggedleft Analysed runs}',
                     '\\parbox[b]{1.9cm}{\\raggedleft Interrupted, row written}',
                     '\\parbox[b]{1.9cm}{\\raggedleft Interrupted, no row}', 'How the batch ended'], lines)
    # pass-rule applications
    def yn(v): return 'met' if v else 'not met'
    head = ['', *['Application %s' % g['application'] for g in gates]]
    G = gates
    lines = [['Task set'] + [g['task_set'] for g in G],
             ['Tasks; repositories; repeats'] + ['%d; %d; %d' % (g['tasks'], g['repositories'], g['repeats']) for g in G],
             ['Runs per arm (all with a result row)'] + [intc(g['runs_per_arm']) for g in G],
             ['Solved runs: baseline, candidate'] + ['%d, %d' % (g['solved']['baseline'], g['solved']['candidate']) for g in G],
             ['Mean gain (points; exact fraction)'] + ['%s ($%s$)' % (sgn(100 * g['gain'], 2), g['gain_exact'].replace('-', '{-}')) for g in G],
             ['\\quad (1) gain at least $2/58$'] + [yn(g['criteria']['gain']) for g in G],
             ['Bootstrap-$t$; one-sided $p$ (9{,}999 draws)'] + ['%s; %s' % (sgn(g['bootstrap']['t'], 3), num(g['bootstrap']['p'], 4)) for g in G],
             ['One-sided $p$ by exact enumeration'] + [num(g['bootstrap']['p_exact_enumeration'], 4) for g in G],
             ['\\quad (2) $p \\le 0.025$'] + [yn(g['criteria']['p']) for g in G],
             ['Repositories gaining; losing'] + ['%d; %d' % (g['repositories_positive'], g['repositories_negative']) for g in G],
             ['\\quad (3a) at least three repositories gain'] + [yn(g['criteria']['pos_repos']) for g in G],
             ['Lowest leave-one-repository-out gain (points)'] + [sgn(100 * g['loo_min'], 2) for g in G],
             ['\\quad (3b) every leave-one-out gain positive'] + [yn(g['criteria']['loo']) for g in G],
             ['Agent time, candidate / baseline (s)'] + ['%s / %s' % (num(g['time']['mean_candidate_s'], 1), num(g['time']['mean_baseline_s'], 1)) for g in G],
             ['\\quad (4a) ratio at most 1.2'] + ['%s: %s' % (num(g['time']['ratio'], 3), yn(g['criteria']['time'])) for g in G],
             ['Infrastructure rows: baseline, candidate'] + ['%d, %d' % (g['infrastructure_rows']['baseline'], g['infrastructure_rows']['candidate']) for g in G],
             ['\\quad (4b) infrastructure errors do not rise'] + [yn(g['criteria']['harness']) for g in G],
             ['Worst repository total (task-equivalents)'] + [sgn(g['worst_repository_total'], 2) for g in G],
             ['\\quad (5) no repository loses two or more'] + [yn(g['criteria']['guard_h']) if 'guard_h' in g['criteria'] else 'not registered' for g in G],
             ['\\midrule\nDecision of the frozen rule'] + ['\\textbf{%s}' % g['decision'] for g in G]]
    table(gen, 'gates', 'Every application of the pass rule of Section 5 to a real candidate, from the frozen gate reports. Baseline and candidate are two versions of the withheld agent bundle.',
          'l' + 'r' * len(G), head, lines)
    # data files
    table(gen, 'files', 'Data files released with the paper.', 'lrp{9.2cm}', ['File', 'Records', 'Content'],
          [['\\texttt{%s}' % esc(n), intc(k) if isinstance(k, int) else k, d] for n, k, d in files])
    # columns of the run file
    table(gen, 'columns', 'Columns of \\texttt{e8\\_runs.csv}. The rows of interrupted jobs hold the first ten columns and the two times of day only.',
          'lp{11.4cm}', ['Column', 'Content'], [['\\texttt{%s}' % esc(c), esc(d).replace('-1', '$-1$')] for c, d in COLDOC])
    # statements of the prose that no table cell shows: stop if one of them no longer holds
    loo = {a: (C[a]['repository']['loo_min'], C[a]['repository']['loo_max']) for a in TRAPS}
    check(loo['H'][1] < 0 and all(loo[a][0] < 0 < loo[a][1] for a in 'RCT'), 'leave-one-out signs: H stays negative, R, C and T change sign')
    check(all(C[a]['primary']['interval95']['lo'] < 0 < C[a]['primary']['interval95']['hi'] for a in TRAPS), 'every 95% interval includes zero')
    check(all(C[a]['repository']['interval95']['lo'] < 0 < C[a]['repository']['interval95']['hi'] for a in TRAPS), 'every repository interval includes zero')
    check(not any(S['holm']['reject'].values()) and not any(C[a]['claim'] for a in TRAPS), 'no claim after Holm')
    check(len(MEASURES) == 13, 'thirteen run measures')
    check(all(len(C[a]['sensitivity']) == 3 and min(v['p_exact'] for v in C[a]['sensitivity'].values()) > 0.05 for a in TRAPS),
          'three sensitivity analyses per arm, none with p at or below 0.05')
    # what supp/gates_text.tex says about the applications: stop if one of these statements no longer holds
    check(len(G) == 3 and all(g['decision'] == 'FAIL' for g in G), 'three applications, each with the decision FAIL')
    check(all(not g['criteria']['gain'] and not g['criteria']['p'] for g in G[:2]), 'applications 1 and 2 miss conditions (1) and (2)')
    c3 = G[2]['criteria']
    check(c3['gain'] and not c3['p'] and all(c3[k] for k in ('pos_repos', 'loo', 'time', 'harness')), 'application 3 meets (1), (3) and (4) and misses (2) only')
    check(G[2]['thresholds']['alpha'] < G[2]['bootstrap']['p'] < 0.05 and G[2]['bootstrap']['p_exact_enumeration'] > G[2]['thresholds']['alpha'],
          'application 3: p between the level of the rule and 0.05, by draws and by enumeration')
    check(G[2]['same_tasks_as'] == [G[1]['application']] and not G[1]['same_tasks_as'], 'application 3 uses the tasks of application 2; application 2 shares no panel with application 1')
    check(['guard_h' in g['criteria'] for g in G] == [True, False, False], 'condition (5) was registered in application 1 only')
    check(G[0]['gain'] < 0 and G[1]['gain'] < 0, 'the estimates of applications 1 and 2 are below zero')
    check(G[2]['repository_totals'][-3]['sum_gain'] > G[2]['repository_totals'][-4]['sum_gain'], 'application 3: the three largest repository totals are well defined')
    share = lambda n, d=len(ag) // len(ARMS): num(100.0 * n / d, 1)
    mac = dict(
        SLoadRows=intc(S['load']['rows']), SLoadFlag=num(100 * S['load']['flag_share'], 0), SInfraNoRow=sum(r[4] for r in batch_lines),
        SReRuns=S['slots']['attempts']['2'], STasksNever=never, STasksAlways=always, STasksMixed=len(panel['tasks']) - never - always,
        SHTasksDown=down['H'], SHTasksUp=up['H'], SRunsPerArm=intc(len(ag) // len(ARMS)),
        SCRejectedShare=share(cnt[('C', 'A18')]), SHUnsolvedAtLimitShare=share(cnt[('H', 'A05')]), SFUnsolvedAtLimitShare=share(cnt[('F', 'A05')]),
        SFCompactedShare=num(100.0 * sum(1 for r in ag if r['arm'] == 'F' and r['summaries'] > 0) / (len(ag) // len(ARMS)), 0),
        SFailedCalls=len(failed), SHLooLo=num(abs(100 * loo['H'][0]), 2), SHLooHi=num(abs(100 * loo['H'][1]), 2),
        SFRate=num(100 * sr['F']['rate'], 1), SMinRate=num(100 * min(v['rate'] for v in sr.values()), 1), SMaxRate=num(100 * max(v['rate'] for v in sr.values()), 1),
        SFirstStart=min(r['start_utc'] for r in rows)[:16].replace('T', ' '), SLastEnd=max(r['end_utc'] for r in rows)[:16].replace('T', ' '),
        SExposureMin=num(100 * min(S['exposure'][a]['share_changed'] for a in 'RHT'), 0), SExposureMax=num(100 * max(S['exposure'][a]['share_changed'] for a in 'RHT'), 0),
        SRows=intc(len(rows)), SAgentRows=intc(len(ag)), SInfraRowsWritten=sum(1 for r in rows if r['row_class'] == 'infrastructure'),
        SInfraRowsAll=sum(v['rows'] for v in S['infrastructure_rows'].values()), SJobsStarted=intc(tot_started),
        STasks=S['design']['tasks'], SRepos=S['design']['repositories'], SSlots=intc(S['slots']['planned']),
        SChecksPassed=intc(S['prompt_checks']['passed']), SChecksFailed=S['prompt_checks']['failed'],
        SLoadAbove=S['load']['rows_above'], SLoadShare=num(100 * S['load']['share_above'], 1), SLoadThreshold=S['load']['threshold_s'],
        SNGates=len(G), SReportSha=S['analysis_report_sha256'],
        SFTimeMean=num(S['f_run_measures']['t_agent']['mean'], 0), SFTimeMedian=num(S['f_run_measures']['t_agent']['median'], 0),
        SHTimeMean=num(C['H']['arm_run_measures']['t_agent']['mean'], 0), SHTimeMedian=num(C['H']['arm_run_measures']['t_agent']['median'], 0),
        SHDiffer=C['H']['primary']['n_nonzero'], SNoDifferMin=min(S['design']['tasks'] - C[a]['primary']['n_nonzero'] for a in TRAPS),
        SNoDifferMax=max(S['design']['tasks'] - C[a]['primary']['n_nonzero'] for a in TRAPS),
        SMinRepoP=num(min(C[a]['repository']['p_two_exact'] for a in TRAPS), 2),
        SSolvedAtLimit=len(full), SHUnsolvedAtLimit=cnt[('H', 'A05')], SFUnsolvedAtLimit=cnt[('F', 'A05')],
        SCRejected=cnt[('C', 'A18')], SFCompacted=sum(1 for r in ag if r['arm'] == 'F' and r['summaries'] > 0),
        SHTokOut=num(C['H']['arm_run_measures']['tok_out']['mean'], 0), SFTokOut=num(S['f_run_measures']['tok_out']['mean'], 0),
        SHTimeCapMissing=num(100 * C['H']['sensitivity']['time_cap_as_missing']['d'], 2),
        SHTimeCapMissingP=num(C['H']['sensitivity']['time_cap_as_missing']['p_exact'], 2),
        SHd=num(abs(100 * C['H']['primary']['d']), 2), SHp=num(C['H']['primary']['p_exact'], 2), SHHolm=num(C['H']['holm_adjusted_p'], 2),
        SHSimLo=num(abs(100 * C['H']['primary']['interval_simultaneous']['lo']), 1), STSimHi=num(100 * C['T']['primary']['interval_simultaneous']['hi'], 1),
        SMinAttainP=num(max(C[a]['repository']['min_attainable_p_two'] for a in TRAPS), 5),
        SRChanged=num(100 * S['exposure']['R']['share_changed'], 0), SCChanged=num(100 * S['exposure']['C']['share_changed'], 0),
        SHChanged=num(100 * S['exposure']['H']['share_changed'], 0), STChanged=num(100 * S['exposure']['T']['share_changed'], 0),
        SRequests=intc(sum(v['requests'] for v in S['exposure'].values())))
    (gen / 'supp_macros.tex').write_text(''.join('\\newcommand{\\%s}{%s}\n' % kv for kv in mac.items()))
    # one set of macros per application of the pass rule (the prose of the supplement uses them)
    check(len({(g['thresholds']['gain_min'], g['thresholds']['alpha'], g['thresholds']['n_boot'], g['thresholds']['seed']) for g in G}) == 1,
          'one pass rule: the same thresholds in every application')
    gm = dict(SGateMinGain=num(100 * float(Fraction(G[0]['thresholds']['gain_min'])), 2), SGateAlpha=repr(G[0]['thresholds']['alpha']),
              SGateCount=['', 'once', 'twice', 'three times', 'four times'][len(G)],
              SGateAlphaSplit=num(G[0]['thresholds']['alpha'] / len(G), 4))       # the level divided by the number of applications
    for nm, g in zip(['One', 'Two', 'Three', 'Four'], G):
        gm.update({'SGate%sTasks' % nm: g['tasks'], 'SGate%sRepos' % nm: g['repositories'], 'SGate%sRuns' % nm: intc(g['runs_per_arm']),
                   'SGate%sSessions' % nm: intc(2 * g['runs_per_arm']),      # both arms
                   'SGate%sBase' % nm: g['solved']['baseline'], 'SGate%sCand' % nm: g['solved']['candidate'],
                   'SGate%sGain' % nm: sgn(100 * g['gain'], 2), 'SGate%sP' % nm: num(g['bootstrap']['p'], 2),
                   'SGate%sPExact' % nm: num(g['bootstrap']['p_exact_enumeration'], 2),
                   'SGate%sTimeRatio' % nm: num(g['time']['ratio'], 2), 'SGate%sDecision' % nm: g['decision'],
                   'SGate%sReposUp' % nm: g['repositories_positive'], 'SGate%sReposDown' % nm: g['repositories_negative'],
                   'SGate%sLoo' % nm: sgn(100 * g['loo_min'], 2), 'SGate%sMore' % nm: g['solved']['candidate'] - g['solved']['baseline'],
                   'SGate%sNet' % nm: sgn(sum(r['sum_gain'] for r in g['repository_totals']), 2),
                   'SGate%sTopThree' % nm: sgn(sum(r['sum_gain'] for r in g['repository_totals'][-3:]), 2),
                   'SGate%sTopThreeTasks' % nm: sum(r['tasks'] for r in g['repository_totals'][-3:]),
                   'SGate%sReposSame' % nm: g['repositories'] - g['repositories_positive'] - g['repositories_negative']})
    (gen / 'gate_macros.tex').write_text(''.join('\\newcommand{\\%s}{%s}\n' % kv for kv in gm.items()))
    mac.update(gm)
    return mac


# ----------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    for a in ('--e8-report', '--e8-manip', '--e8-runs', '--panel', '--data', '--gen'): ap.add_argument(a, required=True)
    ap.add_argument('--gate', action='append', default=[])
    o = ap.parse_args()
    data, gen = pathlib.Path(o.data), pathlib.Path(o.gen)
    data.mkdir(parents=True, exist_ok=True); gen.mkdir(parents=True, exist_ok=True)
    rep = json.load(open(o.e8_report)); panel = json.load(open(o.panel))
    check(rep['status'] == 'UNBLINDED' and rep['panel'] == 'E8', 'the report is the unblinded report of the block')
    manip = {}
    for ln in open(o.e8_manip):
        m = json.loads(ln); check(m['job_id'] not in manip, 'two prompt-check records for one job'); manip[m['job_id']] = m
    rows = load_rows(o.e8_runs, panel, manip)
    solved, repos = recount(rows, panel, rep)
    S = summary(rep, sha(o.e8_report), panel, rows)
    gates = [gate_record(*g.split('::', 2)) for g in o.gate]
    for i, g in enumerate(gates):                      # the same panel file = the same tasks
        g['same_tasks_as'] = [h['application'] for h in gates[:i] if h['panel_sha256'] == g['panel_sha256']]

    write_csv(data / 'e8_runs.csv', COLUMNS, [[r[c] for c in COLUMNS] for r in rows])
    write_csv(data / 'e8_tasks.csv', ['task', 'repository', 'stage', 'repeats'] + ['solved_' + a for a in ARMS],
              [[t, panel['task_repo'][t], next(r['stage'] for r in rows if r['task'] == t), REPEATS] + [solved[(a, t)] for a in ARMS] for t in panel['tasks']])
    C = S['comparisons']
    ntask = collections.Counter(panel['task_repo'].values())
    write_csv(data / 'e8_repositories.csv', ['repository', 'tasks'] + ['solved_runs_' + a for a in ARMS] + ['sum_gain_' + a for a in TRAPS],
              [[g, ntask[g]] + [sum(solved[(a, t)] for t in panel['tasks'] if panel['task_repo'][t] == g) for a in ARMS]
               + [float(Fraction(C[a]['repository']['repo_table'][g]['sum_gain_exact'])) for a in TRAPS] for g in repos])
    write_csv(data / 'e8_compare_table.csv', ['arm', 'task', 'repository', 'repeat', 'solved', 'agent_time', 'row_class'],
              [[r['arm'], r['task'], r['repository'], r['repeat'], r['resolved'], r['t_agent_s'], {'agent': 'agent', 'infrastructure': 'infra'}[r['row_class']]] for r in rows])
    (data / 'e8_summary.json').write_text(json.dumps(S, indent=1, sort_keys=True) + '\n')
    (data / 'pass_rule_applications.json').write_text(json.dumps(gates, indent=1, sort_keys=True) + '\n')
    files = [('e8_runs.csv', len(rows), 'one line per result row of the case study: arm, task, repository, repeat, outcome, times, calls, tokens, prompt check'),
             ('e8_compare_table.csv', len(rows), 'the same rows in the input format of \\texttt{drydock-eval compare} (seven columns)'),
             ('e8_tasks.csv', len(panel['tasks']), 'per task: repository and solved repeats (0 to 4) in each arm'),
             ('e8_repositories.csv', len(repos), 'per repository: tasks, solved runs per arm, summed change against F'),
             ('e8_summary.json', '--', 'the analysis results behind Tables~\\ref{tab:s-rates} to~\\ref{tab:s-measures}, copied field by field from the frozen report'),
             ('pass_rule_applications.json', len(gates), 'the applications of the pass rule (Table~\\ref{tab:s-gates})')]
    mac = tables(gen, S, rows, panel, solved, repos, gates, files)
    n_infra = sum(r['row_class'] == 'infrastructure' for r in rows)
    card = ['# Data of the case study and of the pass-rule applications', '',
            'Outcome rows and analysis results of the paper "Contain, Conform, Compare" (Section 6: five arms, %d tasks in %d repositories, '
            '%d repeats; Section 5: the applications of the pass rule). Written by `supp/build_supp.py`, which first recounts the solved '
            'runs, the task-level differences, the per-repository sums and the request counts from the result rows and stops if any of '
            'them differs from the frozen analysis report (sha256 `%s`).' % (len(panel['tasks']), len(repos), REPEATS, S['analysis_report_sha256']), '',
            'The files hold no prompt, request, patch, test output, error text or machine detail. The agent bundle under test is withheld '
            'until the competition ends; the arms of the pass-rule applications are therefore named `baseline` and `candidate`.', '',
            '| File | Records | Content |', '|---|---|---|',
            '| `e8_runs.csv` | %d | one line per written result row: %d analysed runs and %d rows of interrupted jobs (%d further interrupted jobs wrote no row) |'
            % (len(rows), len(rows) - n_infra, n_infra, mac['SInfraNoRow']),
            '| `e8_compare_table.csv` | %d | the same rows in the input format of `drydock-eval compare` |' % len(rows),
            '| `e8_tasks.csv` | %d | per task, in the order of the panel file: repository, stage, and solved repeats (0 to %d) in each arm |' % (len(panel['tasks']), REPEATS),
            '| `e8_repositories.csv` | %d | per repository: tasks, solved runs per arm, summed change against F in task-equivalents (solved runs / %d) |' % (len(repos), REPEATS),
            '| `e8_summary.json` | | the analysis results, copied field by field from the frozen report; estimates are fractions of 1, not percentage points |',
            '| `pass_rule_applications.json` | %d | one record per application of the pass rule, from the frozen gate reports: counts, gain, '
            'test, conditions, decision, and `repository_totals` (per repository: tasks and summed gain in task-equivalents) |' % len(gates), '',
            '## Columns of `e8_runs.csv`', '', '| Column | Content |', '|---|---|'] + ['| `%s` | %s |' % cd for cd in COLDOC] + [
            '', 'Rows of interrupted jobs (`row_class` = `infrastructure`) hold the first ten columns and the two times of day; their other fields are empty.', '',
            '## Columns of `e8_compare_table.csv`', '',
            '`arm`, `task`, `repository`, `repeat`, `solved` (0 or 1), `agent_time` (seconds), `row_class` (`agent` or `infra`).', '',
            '## Reproducing the analysis', '', '```', 'git clone https://github.com/MinhThanghus/drydock-eval', 'pip install "./drydock-eval[compare]"',
            'drydock-eval compare signflip e8_compare_table.csv --reference F --repeats %d' % REPEATS, '```', '',
            'This returns the estimates, exact p-values, Holm-adjusted p-values and repository-level results of the paper. The 95% intervals '
            'of the task-level test are found with frozen Monte Carlo draws and depend on the order of the tasks: the command sorts task '
            'identifiers, the registered analysis used the order of the panel file (the order of `e8_tasks.csv`). `supp/reproduce.py` runs '
            'both and compares them with `e8_summary.json`.', '',
            '## Checksums', '', '`SHA256SUMS` lists the sha256 of every file of this directory except itself.', '']
    (data / 'README.md').write_text('\n'.join(card))
    names = sorted(q.name for q in data.iterdir() if q.is_file() and q.name != 'SHA256SUMS')
    (data / 'SHA256SUMS').write_text(''.join('%s  %s\n' % (sha(data / n), n) for n in names))
    print('rows %d (agent %d); tasks %d; repositories %d; gates %d' % (len(rows), sum(r['row_class'] == 'agent' for r in rows), len(panel['tasks']), len(repos), len(gates)))
    print('all recounts equal the analysis report (sha256 %s)' % S['analysis_report_sha256'])
    for g in gates: print('gate', g['application'], g['decision'], g['tasks'], g['solved'], g['gain_exact'], g['bootstrap']['p'])
    print('macros:', mac)


if __name__ == '__main__':
    main()
