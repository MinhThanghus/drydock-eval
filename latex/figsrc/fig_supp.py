"""Figures of the supplement, drawn from the released data files only (data/e8_tasks.csv, e8_repositories.csv, e8_runs.csv,
pass_rule_applications.json).

usage: fig_supp.py <data dir> <out dir>      writes supp_taskdiff.pdf, supp_repos.pdf, supp_time.pdf, supp_gates.pdf

One blue for the arm under test, one grey for the reference arm F; small multiples, one axis each.
"""
import collections, csv, json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, BLUE, GREY, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#8a8985", "#ffffff"
TRAPS = ["R", "C", "H", "T"]
NAMES = {"R": "R: role", "C": "C: compaction unset", "H": "H: thoughts removed", "T": "T: other template"}
W = 6.5                                                      # text width of the page, inches

data, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
tasks = list(csv.DictReader(open(data / "e8_tasks.csv")))
repos = list(csv.DictReader(open(data / "e8_repositories.csv")))
runs = [r for r in csv.DictReader(open(data / "e8_runs.csv")) if r["row_class"] == "agent"]
assert len(tasks) == 231 and len(repos) == 15 and len(runs) == 4620
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6})


def clean(ax, left=False):
    ax.set_facecolor(SURF)
    for s in ("top", "right") + (() if left else ("left",)): ax.spines[s].set_visible(False)
    for s in ("bottom", "left"): ax.spines[s].set_color(GRID)
    ax.tick_params(length=0, colors=INK2, labelsize=7)


# ---- 1. task-level differences: solved repeats (of 4) in the arm minus in F, per task
fig, axes = plt.subplots(1, 4, figsize=(W, 1.95), facecolor=SURF, sharey=True, gridspec_kw=dict(wspace=0.12))
top = 0
hist = {}
for a in TRAPS:
    hist[a] = collections.Counter(int(t["solved_" + a]) - int(t["solved_F"]) for t in tasks)
    top = max(top, max(v for k, v in hist[a].items() if k))
for ax, a in zip(axes, TRAPS):
    h = hist[a]; ks = [k for k in range(-4, 5) if k]
    clean(ax)
    ax.bar(ks, [h[k] for k in ks], width=0.72, color=BLUE, zorder=2)
    for k in ks:
        if h[k]: ax.text(k, h[k] + top * 0.03, str(h[k]), ha="center", va="bottom", color=INK, fontsize=6.5)
    ax.axvline(0, color=GRID, lw=0.8, zorder=1)
    ax.set_xlim(-4.7, 4.7); ax.set_ylim(0, top * 1.22); ax.set_yticks([])
    ax.set_xticks(ks); ax.set_xticklabels(["−4", "−3", "−2", "−1", "+1", "+2", "+3", "+4"], fontsize=6.2)
    differ = sum(v for k, v in h.items() if k)
    ax.set_title("%s\n%d tasks differ, %d do not" % (NAMES[a], differ, h[0]), loc="left", color=INK, fontsize=7.5, pad=5)
fig.supxlabel("Solved repeats of a task (of 4): arm minus F. Tasks with no difference are not drawn.", color=INK2, fontsize=7.5, y=0.02)
fig.subplots_adjust(left=0.01, right=0.995, top=0.8, bottom=0.24)
fig.savefig(out / "supp_taskdiff.pdf", facecolor=SURF); plt.close(fig)

# ---- 2. per-repository change against F, in task-equivalents
order = sorted(repos, key=lambda r: (int(r["tasks"]), r["repository"]))          # drawn bottom to top
labels = ["%s (%s)" % (r["repository"].split("/")[1], r["tasks"]) for r in order]
lim = max(abs(float(r["sum_gain_" + a])) for r in repos for a in TRAPS) + 0.45
fig, axes = plt.subplots(1, 4, figsize=(W, 3.1), facecolor=SURF, sharey=True, gridspec_kw=dict(wspace=0.1))
for ax, a in zip(axes, TRAPS):
    v = [float(r["sum_gain_" + a]) for r in order]
    clean(ax)
    ax.barh(range(len(order)), v, height=0.62, color=BLUE, zorder=2)
    ax.axvline(0, color=INK2, lw=0.6, zorder=3)
    ax.set_xlim(-lim, lim); ax.set_xticks([-2, 0, 2]); ax.set_xticklabels(["−2", "0", "+2"])
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(labels, color=INK, fontsize=6.8)
    ax.set_title("%s\ntotal %s" % (NAMES[a], ("%+.2f" % sum(v)).replace("-", "−")), loc="left", color=INK, fontsize=7.5, pad=5)
fig.supxlabel("Change against F per repository, in task-equivalents (solved runs / 4). In brackets: tasks.", color=INK2, fontsize=7.5, y=0.015)
fig.subplots_adjust(left=0.2, right=0.995, top=0.87, bottom=0.15)
fig.savefig(out / "supp_repos.pdf", facecolor=SURF); plt.close(fig)

# ---- 3. agent time: share of runs finished by a given time, arm against F
t = {a: sorted(float(r["t_agent_s"]) for r in runs if r["arm"] == a) for a in ["F"] + TRAPS}
fig, axes = plt.subplots(1, 4, figsize=(W, 1.95), facecolor=SURF, sharey=True, gridspec_kw=dict(wspace=0.1))
for ax, a in zip(axes, TRAPS):
    clean(ax, left=True)
    for arm, col, lw in (("F", GREY, 1.6), (a, BLUE, 1.6)):
        x = t[arm]; n = len(x)
        ax.step(x, [(i + 1) / n for i in range(n)], where="post", color=col, lw=lw, label=arm, zorder=3 if arm == a else 2)
    for v in (240, 300): ax.axvline(v, color=INK2, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.set_xlim(0, 310); ax.set_ylim(0, 1.02); ax.set_xticks([0, 120, 240, 300]); ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "50%", "100%"]); ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_title(NAMES[a], loc="left", color=INK, fontsize=7.5, pad=5)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=7, handlelength=1.4, borderaxespad=0.2, labelcolor=INK)
fig.supxlabel("Agent time (s). Dashed: 240 s (load threshold) and 300 s (the bundle's limit). Vertical axis: share of runs finished.", color=INK2, fontsize=7.5, y=0.02)
fig.subplots_adjust(left=0.065, right=0.995, top=0.86, bottom=0.25)
fig.savefig(out / "supp_time.pdf", facecolor=SURF); plt.close(fig)
# ---- 4. applications of the pass rule: summed gain of each repository (candidate minus baseline), sorted
apps = json.load(open(data / "pass_rule_applications.json"))
vals = [[r["sum_gain"] for r in a["repository_totals"]] for a in apps]              # ascending in the file
assert all(v == sorted(v) for v in vals)
lo, hi, wide = min(min(v) for v in vals), max(max(v) for v in vals), max(len(v) for v in vals)
sg = lambda x, nd=2: ("%+.*f" % (nd, x)).replace("-", "−")
fig, axes = plt.subplots(1, len(apps), figsize=(W, 2.05), facecolor=SURF, sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, a, v in zip(axes, apps, vals):
    clean(ax, left=True)
    ax.bar(range(len(v)), v, width=0.68, color=BLUE, zorder=2)
    same = [i for i, y in enumerate(v) if y == 0]
    ax.scatter(same, [0] * len(same), s=7, color=GREY, zorder=4, linewidths=0)
    ax.axhline(0, color=INK2, lw=0.6, zorder=3)
    ax.set_xlim(-0.9, wide - 0.1); ax.set_xticks([])
    ax.set_ylim(lo - 0.25, hi + 0.25); ax.set_yticks([-1, 0, 1, 2]); ax.set_yticklabels(["−1", "0", "+1", "+2"])
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_title("Application %s: net %s\n%d tasks in %d repositories" % (a["application"], sg(sum(v)), a["tasks"], len(v)),
                 loc="left", color=INK, fontsize=7.5, pad=5)
axes[0].set_ylabel("task-equivalents", color=INK2, fontsize=7.5)
fig.supxlabel("One bar per repository, sorted: summed gain of its tasks in task-equivalents (solved runs / 4). Dot: no change.",
              color=INK2, fontsize=7.5, y=0.02)
fig.subplots_adjust(left=0.075, right=0.995, top=0.84, bottom=0.14)
fig.savefig(out / "supp_gates.pdf", facecolor=SURF); plt.close(fig)
print("figures written:", ", ".join(sorted(p.name for p in out.glob("supp_*.pdf"))))
