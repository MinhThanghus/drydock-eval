"""Figure: canaries passed and high-severity vectors found per runner version. Every count is a cell of Table 1 of the
draft (ledger rows of Section 3): canaries 52, 60, 98, 134, 159, 191, 200 (each version passed all of its own);
demonstrated high-severity vectors 4, 4, 1, 0, 0, 0, 0 (nine in all).   usage: fig_versions.py <out.pdf>"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V = ["1", "2", "3", "4", "5", "6", "7"]
CANARIES = [52, 60, 98, 134, 159, 191, 200]
HIGH = [4, 4, 1, 0, 0, 0, 0]
assert sum(HIGH) == 9
INK, INK2, GRID, BLUE, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#ffffff"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.35), facecolor=SURF, gridspec_kw=dict(wspace=0.28))
for ax, vals, title, top in ((a, CANARIES, "Canaries passed (all, in every version)", 235),
                             (b, HIGH, "High-severity vectors still demonstrated", 4.7)):
    ax.set_facecolor(SURF)
    ax.bar(V, vals, width=0.62, color=BLUE, zorder=2)
    for x, v in zip(V, vals):
        ax.text(x, v + top * 0.02, str(v), ha="center", va="bottom", color=INK, fontsize=8)
    ax.set_ylim(0, top); ax.set_yticks([])
    ax.set_xlabel("Runner version", color=INK2, fontsize=8)
    ax.set_title(title, loc="left", color=INK, fontsize=8.5, fontweight="bold", pad=6)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(length=0, colors=INK2)
fig.subplots_adjust(left=0.02, right=0.99, top=0.86, bottom=0.2)
fig.savefig(sys.argv[1], facecolor=SURF)
