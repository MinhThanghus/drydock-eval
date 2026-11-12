"""Re-run the case-study analysis with the released kit on the released table, and compare it with the registered
analysis (data/e8_summary.json, copied from the frozen report). Writes gen/s_repro.tex and gen/repro_macros.tex.

usage (python -I -B, numpy needed):  reproduce.py <kit source dir> <data dir> <gen dir>

Two runs:
  (a) the kit's command, as a reader would type it:
        drydock-eval compare signflip data/e8_compare_table.csv --reference F --repeats 4 --json out.json
      (the command orders the tasks by identifier);
  (b) the kit's interval routine on the same task differences in the order of data/e8_tasks.csv, which is the order of the
      panel file and of the registered analysis.
The script stops if a quantity that must not depend on the task order differs from the registered value, or if (b) does
not return the registered Monte Carlo quantities.
"""
import csv, json, pathlib, sys, tempfile
from fractions import Fraction

kit, data, gen = (pathlib.Path(a).resolve() for a in sys.argv[1:4])
sys.path.insert(0, str(kit))
from drydock_eval.cli import main as kit_main                         # noqa: E402
from drydock_eval.compare import _vendored_e8 as A8                   # noqa: E402

TRAPS = ['R', 'C', 'H', 'T']
TOL = 1e-12


def die(msg):
    sys.exit('reproduce: ' + msg)


def same(a, b, what):
    if abs(float(a) - float(b)) > TOL: die('%s: kit %r, registered %r' % (what, a, b))


S = json.load(open(data / 'e8_summary.json'))
with tempfile.TemporaryDirectory() as tmp:
    out = pathlib.Path(tmp) / 'signflip.json'
    rc = kit_main(['compare', 'signflip', str(data / 'e8_compare_table.csv'), '--reference', 'F', '--repeats', '4', '--json', str(out)])
    if rc != 0: die('the kit command ended with exit code %r' % rc)
    K = json.load(open(out))

tasks = list(csv.DictReader(open(data / 'e8_tasks.csv')))
rows, move95, movesim, movep = [], 0.0, 0.0, 0.0
for a in TRAPS:
    k, r = K['comparisons'][a]['primary'], S['comparisons'][a]['primary']
    kg, rg = k['repository'], S['comparisons'][a]['repository']
    # ---- must not depend on the task order: equal to the registered values
    if Fraction(k['d_exact']) != Fraction(r['d_exact']): die('d of arm ' + a)
    if Fraction(k['p_exact_fraction']) != Fraction(r['p_exact_fraction']): die('exact p of arm ' + a)
    if (k['n_nonzero'], k['sum_K'], k['n_tasks']) != (r['n_nonzero'], r['sum_K'], r['n_tasks']): die('counts of arm ' + a)
    same(Fraction(K['comparisons'][a]['holm_adjusted_p']), S['comparisons'][a]['holm_adjusted_p'], 'Holm p of arm ' + a)
    for q in ('lo', 'hi'):
        same(k['t_interval95'][q], r['t_interval95'][q], 't interval of arm ' + a)
        same(kg['interval95'][q], rg['interval95'][q], 'repository interval of arm ' + a)
        same(kg['cr1'][q], rg['cr1'][q], 'CR1 interval of arm ' + a)
    for q in ('p_two_exact', 'p_two_webb', 'p_two_mc', 'repo_equal', 'n_clusters_nonzero', 'min_attainable_p_two'):
        same(kg[q], rg[q], '%s of arm %s' % (q, a))
    same(kg['loo_min'], rg['loo_min'], 'leave-one-out minimum'); same(kg['loo_max'], rg['loo_max'], 'leave-one-out maximum')
    for g, v in rg['repo_table'].items():
        if Fraction(kg['repo_table'][g]['sum_gain_exact']) != Fraction(v['sum_gain_exact']): die('repository sum %s %s' % (a, g))
    # ---- (b) Monte Carlo quantities with the tasks in panel order: equal to the registered values
    D = [(int(t['solved_' + a]) - int(t['solved_F'])) / 4 for t in tasks]
    mc = A8.SignFlipMC(D)
    i95 = A8.signflip_interval(D, 0.95, mc=mc); isim = A8.signflip_interval(D, 1 - 0.05 / len(TRAPS), mc=mc)
    same(mc.p(0.0), r['p_mc'], 'Monte Carlo p in panel order, arm ' + a)
    for q in ('lo', 'hi'):
        same(i95[q], r['interval95'][q], '95%% interval in panel order, arm ' + a)
        same(isim[q], r['interval_simultaneous'][q], 'simultaneous interval in panel order, arm ' + a)
    # ---- (a) the command: the same draws meet other tasks
    move95 = max(move95, *(abs(k['interval95'][q] - r['interval95'][q]) for q in ('lo', 'hi')))
    movesim = max(movesim, *(abs(k['interval_simultaneous'][q] - r['interval_simultaneous'][q]) for q in ('lo', 'hi')))
    movep = max(movep, abs(k['p_mc'] - r['p_mc']))
    f = lambda x: '$%s$' % ('%+.2f' % (100 * x)).replace('+', '{+}').replace('-', '{-}')
    rows.append([a, '[%s, %s]' % (f(r['interval95']['lo']), f(r['interval95']['hi'])),
                 '[%s, %s]' % (f(i95['lo']), f(i95['hi'])), '[%s, %s]' % (f(k['interval95']['lo']), f(k['interval95']['hi'])),
                 '%.4f' % r['p_mc'], '%.4f' % mc.p(0.0), '%.4f' % k['p_mc']])

# the step shape of the test's p-value near the upper end of H's interval, in panel order
D = [(int(t['solved_H']) - int(t['solved_F'])) / 4 for t in tasks]
mc = A8.SignFlipMC(D)
hi_reg, hi_cmd = S['comparisons']['H']['primary']['interval95']['hi'], K['comparisons']['H']['primary']['interval95']['hi']
if not hi_cmd < hi_reg: die('the H interval of the command was expected to end below the registered one')
grid = [hi_cmd + (hi_reg - hi_cmd) * i / 40 for i in range(1, 40)]
ps = [mc.p(x) for x in grid]

head = ['Arm', 'Registered', 'Kit, panel order', 'Kit command', 'Registered', 'Kit, panel order', 'Kit command']
tex = ('\\begin{table}[!htb]\n\\centering\n\\caption{Reproduction with the released kit: the two quantities that depend on Monte Carlo draws. '
       '95\\%% intervals of the change against F in percentage points, and the Monte Carlo $p$ that accompanies the exact $p$.}\\label{tab:s-repro}\n'
       '\\footnotesize\n\\begin{tabular}{lrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{95\\%% interval} & \\multicolumn{3}{c}{Monte Carlo $p$} \\\\\n'
       '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n%s \\\\\n\\midrule\n%s \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
       % (' & '.join(head), ' \\\\\n'.join(' & '.join(r) for r in rows)))
(gen / 's_repro.tex').write_text(tex)
mac = dict(SReproMoveNinetyFive='%.2f' % (100 * move95), SReproMoveSim='%.2f' % (100 * movesim), SReproMoveP='%.4f' % movep,
           SReproHRegHi='%.2f' % (100 * hi_reg), SReproHCmdHi='%.2f' % (100 * hi_cmd),
           SReproStepLo='%.4f' % min(ps), SReproStepHi='%.4f' % max(ps),
           SReproDraws='{:,}'.format(K['comparisons']['H']['primary']['mc_draws']).replace(',', '{,}'),
           SReproSeed=str(K['comparisons']['H']['primary']['mc_seed']))
(gen / 'repro_macros.tex').write_text(''.join('\\newcommand{\\%s}{%s}\n' % kv for kv in mac.items()))
print('kit command: d, exact p, Holm p, t interval and every repository-level quantity equal the registered analysis (4 arms)')
print('kit routine, tasks in panel order: Monte Carlo p and both inverted intervals equal the registered analysis (4 arms)')
print('kit command (tasks sorted): largest move of a 95%% end %.4f, of a simultaneous end %.4f (fractions of 1); of the Monte Carlo p %.5f'
      % (move95, movesim, movep))
print('p of the H test across the stretch between the two upper ends, panel order: %.4f to %.4f' % (min(ps), max(ps)))
print('macros:', mac)
