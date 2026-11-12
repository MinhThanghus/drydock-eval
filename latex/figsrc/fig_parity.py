"""The parity-and-effects figure of the paper (fig/parity_effects.pdf; Figure 1 of paper.md).
Panel A: local minus Kaggle solve rate on the 24-task parity list; the solved counts are typed below (Section 7 of the
paper). Panel B: the four changes against arm F, read from data/e8_summary.json.

usage: fig_parity.py <out file> --e8 data/e8_summary.json      (without --e8 only panel A is drawn)
"""
import json, sys
from math import sqrt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Z = 1.959963984540054
KAGGLE = (8, 48)                      # two Kaggle-kernel runs, 4/24 each, pooled
RUNS = [                              # (group, label, solved of 24), earliest first
    ("no sandbox", "8 workers", 3), ("no sandbox", "1 worker", 5),
    ("container runners", "version 1", 3), ("container runners", "version 2", 4),
    ("container runners", "version 3", 2), ("container runners", "version 4, run a", 4),
    ("container runners", "version 4, run b", 3), ("container runners", "version 7 (final)", 5),
]
INK, INK2, GRID, BLUE, BAND, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#efeeea", "#ffffff"

def wilson(x, n):
    p = x / n; d = 1 + Z * Z / n; c = (p + Z * Z / (2 * n)) / d
    h = Z * sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h

def newcombe(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2; l1, u1 = wilson(x1, n1); l2, u2 = wilson(x2, n2); d = p1 - p2
    return 100 * d, 100 * (d - sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)), 100 * (d + sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))

def main(out, e8=None):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    if e8:
        fig, (ax, axb) = plt.subplots(2, 1, figsize=(7.2, 5.9), dpi=200, facecolor=SURF,
                                      gridspec_kw=dict(height_ratios=[8, 4.2], hspace=0.66))
    else:
        fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200, facecolor=SURF)
    ax.set_facecolor(SURF)
    band = 100 * 2 / 58
    ax.axvspan(-band, band, color=BAND, zorder=0, lw=0)
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    n = len(RUNS)
    for i, (grp, lab, k) in enumerate(RUNS):
        y = n - 1 - i
        d, lo, hi = newcombe(k, 24, *KAGGLE)
        ax.plot([lo, hi], [y, y], color=BLUE, lw=2, solid_capstyle="round", zorder=2)
        ax.plot([d], [y], "o", color=BLUE, ms=7, mec=SURF, mew=1.5, zorder=3)
        ax.text(41, y, f"{k}/24   {d:+.1f} [{lo:+.1f}, {hi:+.1f}]".replace("+0.0 ", "0.0 ").replace("-", "−"), va="center", ha="left", color=INK, fontsize=8)
    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels([f"{g}, {l}" if g == "no sandbox" else f"runner {l}" for g, l, _ in RUNS], color=INK)
    ax.set_xlim(-30, 40); ax.set_ylim(-0.7, n - 0.3)
    ax.set_xticks([-30, -20, -10, 0, 10, 20, 30, 40])
    ax.set_xticklabels([str(v).replace("-", "−") for v in [-30, -20, -10, 0, 10, 20, 30, 40]], color=INK2)
    ax.set_xlabel("Solve rate, local run minus Kaggle kernel (percentage points, 95% Newcombe interval)", color=INK2, fontsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(length=0)
    ax.text(0, n - 0.25, "pass rule's ±3.45-point band", ha="center", va="bottom", color=INK2, fontsize=7.5)
    ax.text(41, n - 0.25, "solved   difference [95% interval]", ha="left", va="bottom", color=INK2, fontsize=7.5, clip_on=False)
    ax.set_title("A. Local minus Kaggle on one 24-task list: every interval is over 33 points wide",
                 loc="left", color=INK, fontsize=9.5, fontweight="bold", pad=16, x=-0.42)
    if e8:
        panel_b(axb, e8)
        fig.subplots_adjust(left=0.215, right=0.70, top=0.905, bottom=0.085)
    else:
        fig.subplots_adjust(left=0.215, right=0.70, top=0.84, bottom=0.15)
    fig.savefig(out, facecolor=SURF)
    for g, l, k in RUNS:
        d, lo, hi = newcombe(k, 24, *KAGGLE); print(f"{g:18s} {l:18s} {k}/24 {d:+.1f} [{lo:+.1f}, {hi:+.1f}] width {hi-lo:.1f}")

ARMS_B = [("R", "R: tool-result role"), ("C", "C: no compaction"), ("H", "H: thoughts removed"), ("T", "T: other template")]

def panel_b(ax, path):
    """E8 effects: d_X = mean over tasks of (solve rate of arm X - solve rate of F), primary 95% interval."""
    rep = json.load(open(path)); comp = rep.get("results", rep)["comparisons"]      # the report, or its released summary
    assert rep["status"] == "UNBLINDED", rep["status"]
    ax.set_facecolor(SURF)
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    n = len(ARMS_B); tasks = set()
    for i, (arm, lab) in enumerate(ARMS_B):
        y = n - 1 - i; pr = comp[arm]["primary"]; tasks.add(pr["n_tasks"])
        d, lo, hi = 100 * pr["d"], 100 * pr["interval95"]["lo"], 100 * pr["interval95"]["hi"]
        hp = comp[arm]["holm_adjusted_p"]
        ax.plot([lo, hi], [y, y], color=BLUE, lw=2, solid_capstyle="round", zorder=2)
        ax.plot([d], [y], "o", color=BLUE, ms=7, mec=SURF, mew=1.5, zorder=3)
        ptxt = "1" if hp >= 0.9995 else f"{hp:.2f}"
        ax.text(7.15, y, f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]   p = {ptxt}".replace("-", "−"), va="center", ha="left", color=INK, fontsize=8, clip_on=False)
        print(f"E8 {arm} d {d:+.2f} [{lo:+.2f}, {hi:+.2f}] Holm p {hp:.4f}")
    assert len(tasks) == 1
    ax.set_yticks([n - 1 - i for i in range(n)]); ax.set_yticklabels([l for _, l in ARMS_B], color=INK)
    ax.set_xlim(-7, 7); ax.set_ylim(-0.7, n - 0.3)
    ticks = [-6, -4, -2, 0, 2, 4, 6]
    ax.set_xticks(ticks); ax.set_xticklabels([str(v).replace("-", "−") for v in ticks], color=INK2)
    ax.set_xlabel("Solve rate, arm minus scorer path F (percentage points, 95% sign-flip interval)", color=INK2, fontsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(length=0)
    ax.text(7.15, n - 0.25, "change [95% interval]   Holm p", ha="left", va="bottom", color=INK2, fontsize=7.5, clip_on=False)
    ax.set_title(f"B. Four prompt divergences on {tasks.pop()} tasks: no difference from the scorer path detected",
                 loc="left", color=INK, fontsize=9.5, fontweight="bold", pad=16, x=-0.42)

if __name__ == "__main__":
    a = sys.argv[1:]
    e8 = a[a.index("--e8") + 1] if "--e8" in a else None
    main([x for i, x in enumerate(a) if x != "--e8" and (i == 0 or a[i - 1] != "--e8")][0], e8)
