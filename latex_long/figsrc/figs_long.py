"""Figures of the extended paper, drawn from the released data (data/) and from gen/analysis.json, which
scripts/analyze_long.py writes from the same data.

usage: figs_long.py DATA_DIR GEN_DIR OUT_DIR

One blue for the quantity in focus, greys for references; one axis per panel; labels on the marks.
"""
import csv, json, math, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

INK, INK2, GRID, BLUE, GREY, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#8a8985", "#ffffff"
LIGHT = "#eeeeeb"
W = 6.5
data, gen, out = (pathlib.Path(x) for x in sys.argv[1:4])
A = json.load(open(gen / "analysis.json"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6})


def clean(ax, left=False):
    ax.set_facecolor(SURF)
    for s in ("top", "right") + (() if left else ("left",)): ax.spines[s].set_visible(False)
    for s in ("bottom", "left"): ax.spines[s].set_color(GRID)
    ax.tick_params(length=0, colors=INK2, labelsize=7)


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


# ---- 1. serving: decode speed per stream and over all streams, by number of concurrent streams (2k context)
rows = [r for r in A["serving"] if r["experiment"] == "serve-1" and r["context_tokens"] == "2048"]
fig, axes = plt.subplots(1, 2, figsize=(W, 2.35), facecolor=SURF, gridspec_kw=dict(wspace=0.32))
STY = {"E4B": (BLUE, 1.6), "12B": (GREY, 1.4), "31B": (INK, 1.4)}
for ax, key, title in ((axes[0], "per_stream_tok_s", "One stream: tokens per second"),
                       (axes[1], "aggregate_tok_s", "All streams together: tokens per second")):
    clean(ax, left=True)
    for m in ("E4B", "12B", "31B"):
        pts = sorted((int(r["concurrency"]), float(r[key])) for r in rows if r["model"] == m)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=STY[m][0], lw=STY[m][1], marker="o", ms=3.2, zorder=3)
        ax.text(pts[-1][0] * 1.12, pts[-1][1], m, color=INK, fontsize=7.5, va="center")
        if m == "31B":                                   # one disturbed measurement, shown as measured
            d = [q for q in pts if q[0] == 8][0]
            first = key == "per_stream_tok_s"
            ax.annotate("disturbed\nmeasurement", xy=d, xytext=(d[0] * 0.98, d[1] - 9) if first else (11, 8),
                        color=INK2, fontsize=6.3, ha="center" if first else "left", va="top" if first else "bottom",
                        arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5))
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.set_xlim(0.85, 44); ax.set_ylim(0, None)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=7.5, pad=6)
    ax.set_xlabel("Concurrent streams", color=INK2, fontsize=7.5)
fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.18)
fig.savefig(out / "serving.pdf", facecolor=SURF); plt.close(fig)

# ---- 2. six configurations on the 70 development tasks: which tasks each one solved
P = A["pub70"]; names = list(P["series"]); T = len(P["tasks"])
cnt = [sum(P["series"][n][i] for n in names) for i in range(T)]
order = sorted(range(T), key=lambda i: (-cnt[i], P["tasks"][i]))
fig, ax = plt.subplots(figsize=(W, 1.95), facecolor=SURF)
ax.set_facecolor(SURF)
for y, n in enumerate(names):
    for x, i in enumerate(order):
        ax.add_patch(Rectangle((x + 0.08, -y - 0.42), 0.84, 0.84, color=BLUE if P["series"][n][i] else LIGHT, lw=0))
    ax.text(-0.8, -y, n, ha="right", va="center", color=INK, fontsize=7.5)
    ax.text(T + 0.8, -y, "%d" % sum(P["series"][n]), ha="left", va="center", color=INK, fontsize=7.5)
ax.text(T + 0.8, 1.0, "of 70", ha="left", va="center", color=INK2, fontsize=7)
n_all = sum(1 for c in cnt if c == len(names)); n_none = sum(1 for c in cnt if c == 0)
for x0, x1, lab in ((0, n_all, "solved by all six: %d" % n_all), (n_all, T - n_none, "by some: %d" % (T - n_all - n_none)),
                    (T - n_none, T, "by none: %d" % n_none)):
    ax.plot([x0 + 0.2, x1 - 0.2], [0.75, 0.75], color=INK2, lw=0.7)
    ax.text((x0 + x1) / 2, 0.95, lab, ha="center", va="bottom", color=INK2, fontsize=7)
ax.set_xlim(-8, T + 5); ax.set_ylim(-len(names) + 0.4, 1.8); ax.axis("off")
ax.text(T / 2, -len(names) + 0.25, "70 development tasks, ordered by the number of configurations that solved them (one run each)",
        ha="center", va="top", color=INK2, fontsize=7.5)
fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.1)
fig.savefig(out / "pub_matrix.pdf", facecolor=SURF); plt.close(fig)

# ---- 3. the same six configurations: solved share with 95% Wilson intervals
fig, ax = plt.subplots(figsize=(W, 1.9), facecolor=SURF)
clean(ax)
for y, n in enumerate(names):
    k = sum(P["series"][n]); lo, hi = wilson(k, T)
    ax.plot([lo, hi], [-y, -y], color=BLUE if n != "BF5" else GREY, lw=1.4, solid_capstyle="round", zorder=2)
    ax.plot([100.0 * k / T], [-y], "o", color=BLUE if n != "BF5" else GREY, ms=5, mec=SURF, mew=1.0, zorder=3)
    ax.text(hi + 1.0, -y, "%d of 70" % k, va="center", color=INK, fontsize=7.5)
ax.set_yticks([-y for y in range(len(names))]); ax.set_yticklabels(names, color=INK, fontsize=7.5)
ax.set_xlim(30, 85); ax.set_ylim(-len(names) + 0.4, 0.6)
ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_xlabel("Solved development tasks, per cent (dot), with 95% interval (line). Grey: the public reference configuration.",
              color=INK2, fontsize=7.5)
fig.subplots_adjust(left=0.1, right=0.985, top=0.97, bottom=0.24)
fig.savefig(out / "pub_configs.pdf", facecolor=SURF); plt.close(fig)

# ---- 4. hosted scores of our submissions
H = [h for h in A["hosted"] if h["score"] != "pending"]
fig, ax = plt.subplots(figsize=(W, 2.0), facecolor=SURF)
clean(ax)
ax.axvspan(8 / 58.0, 11 / 58.0, color=LIGHT, zorder=0)
ax.text((8 + 11) / 2 / 58.0, 0.72, "ten public draws of the\nBF5 files: 8 to 11 of 58", ha="center", va="bottom", color=INK2, fontsize=6.8)
for y, h in enumerate(H):
    v = float(h["score"])
    ax.plot([v], [-y], "o", color=BLUE if h["model"] == "E4B" else INK, ms=5.5, mec=SURF, mew=1.0, zorder=3)
    ax.text(v + 0.006, -y, "%s  %s" % (h["score"], h["date"][5:]), va="center", color=INK, fontsize=7.5)
ax.set_yticks([-y for y in range(len(H))]); ax.set_yticklabels(["%s (%s)" % (h["arm"], h["model"]) for h in H], color=INK, fontsize=7.5)
ax.set_xlim(0, 0.26); ax.set_ylim(-len(H) + 0.4, 1.75)
ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_xlabel("Public leaderboard score of one hosted run (share of the public test tasks solved)", color=INK2, fontsize=7.5)
fig.subplots_adjust(left=0.13, right=0.985, top=0.97, bottom=0.23)
fig.savefig(out / "hosted.pdf", facecolor=SURF); plt.close(fig)

# ---- 5. lead in the screen against gain in the confirmation
L = A["lead"]
fig, ax = plt.subplots(figsize=(W * 0.7, 2.7), facecolor=SURF)
clean(ax, left=True)
ax.axhline(0, color=INK2, lw=0.7, zorder=1)
ax.axhline(100 * 2 / 58.0, color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=1)
ax.text(-0.4, 100 * 2 / 58.0 + 0.15, "gain the rule asks for: 3.45", ha="left", va="bottom", color=INK2, fontsize=6.8)
OFF = {"v-p2": (0.0, 0.42), "v-t6": (0.0, -0.42), "v-a3": (0.0, 0.0)}
for c, v in L.items():
    ax.plot([0, 1], [v["lead"], v["gain"]], color=BLUE, lw=1.5, marker="o", ms=4.5, mec=SURF, mew=0.9, zorder=3)
    ax.text(-0.04, v["lead"], "%s  %+.1f" % (c, v["lead"]), ha="right", va="center", color=INK, fontsize=7.5)
    ax.text(1.04, v["gain"] + OFF[c][1], "%+.2f  %s" % (v["gain"], c), ha="left", va="center", color=INK, fontsize=7.5)
ax.set_xlim(-0.42, 1.6); ax.set_ylim(-2, 7.6)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Screen\n(48 tasks, 96 runs per arm)", "Confirmation\n(new tasks, 288 or 512 runs per arm)"], color=INK, fontsize=7.5)
ax.set_yticks([-2, 0, 2, 4, 6]); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_title("Candidate minus baseline, percentage points", loc="left", color=INK, fontsize=7.5, pad=6)
fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.17)
fig.savefig(out / "lead_gain.pdf", facecolor=SURF); plt.close(fig)

# ---- 6. the baseline again and again: session totals, and how often each task of panel H was solved in 8 runs
S = A["baseline_sessions"]
rowsA = [("24 tasks, hosted, repeat %d" % (i + 1), k, n) for i, (lab, k, n) in enumerate(A["parity"][:2])]
rowsA += [("24 tasks, workstation, run %d" % (i + 1), k, n) for i, (lab, k, n) in enumerate(A["parity"][2:])]
rowsA += [("S, session %d" % (i + 1), k, 96) for i, k in enumerate(S["S"])]
rowsA += [("C", S["C"][0], 288)]
rowsA += [("H, session %d" % (i + 1), k, 512) for i, k in enumerate(S["H"])]
fig, axes = plt.subplots(1, 2, figsize=(W, 3.05), facecolor=SURF, gridspec_kw=dict(wspace=0.12, width_ratios=[1.45, 1]))
ax = axes[0]; clean(ax)
for y, (lab, k, n) in enumerate(rowsA):
    lo, hi = wilson(k, n)
    col = BLUE if n >= 96 else GREY
    ax.plot([lo, hi], [-y, -y], color=col, lw=1.3, solid_capstyle="round", zorder=2)
    ax.plot([100.0 * k / n], [-y], "o", color=col, ms=4.5, mec=SURF, mew=0.9, zorder=3)
    ax.text(hi + 1.0, -y, "%d/%d" % (k, n), va="center", color=INK, fontsize=6.8)
ax.set_yticks([-y for y in range(len(rowsA))]); ax.set_yticklabels([r[0] for r in rowsA], color=INK, fontsize=6.8)
ax.set_xlim(0, 52); ax.set_ylim(-len(rowsA) + 0.4, 0.6)
ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_title("Baseline: solved runs, per cent (95% interval)", loc="left", color=INK, fontsize=7.5, pad=6)
ax = axes[1]; clean(ax)
hh = A["h_hist"]
ax.bar(range(9), hh, width=0.7, color=BLUE, zorder=2)
for i, v in enumerate(hh):
    ax.text(i, v + 1.5, str(v), ha="center", va="bottom", color=INK, fontsize=6.8)
ax.set_xticks(range(9)); ax.set_yticks([]); ax.set_ylim(0, max(hh) * 1.15)
ax.set_title("Tasks of panel H by solved runs", loc="left", color=INK, fontsize=7.5, pad=6)
ax.set_xlabel("Solved baseline runs of a task, of 8", color=INK2, fontsize=7)
fig.subplots_adjust(left=0.24, right=0.99, top=0.92, bottom=0.13)
fig.savefig(out / "repeat.pdf", facecolor=SURF); plt.close(fig)

# ---- 7. baseline solve rate per repository
RP = sorted(A["repos"], key=lambda r: r["solved"] / r["runs"])
fig, ax = plt.subplots(figsize=(W, 3.5), facecolor=SURF)
clean(ax)
for y, r in enumerate(RP):
    v = 100.0 * r["solved"] / r["runs"]
    ax.barh(y, v, height=0.55, color=BLUE, zorder=2)
    ax.text(v + 0.8, y, "%.0f%%  (%d of %d runs)" % (v, r["solved"], r["runs"]), va="center", color=INK, fontsize=6.8)
ax.set_yticks(range(len(RP))); ax.set_yticklabels(["%s (%d)" % (r["repository"].split("/")[-1], r["tasks"]) for r in RP], color=INK, fontsize=6.8)
ax.set_xlim(0, 100); ax.set_ylim(-0.6, len(RP) - 0.4)
ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_xlabel("Solved runs of the E4B baseline, per cent. In brackets after the name: number of tasks.", color=INK2, fontsize=7.5)
fig.subplots_adjust(left=0.2, right=0.985, top=0.99, bottom=0.11)
fig.savefig(out / "repos.pdf", facecolor=SURF); plt.close(fig)

# ---- 8. time of solved and unsolved runs
TD = A["time_dist"]
POS = {("e4b", "solved"): (52, 0.72, "left"), ("e4b", "unsolved"): (120, 0.27, "left"), ("b31", "solved"): (200, 0.43, "center"),
       ("b31", "unsolved"): (470, 0.75, "right")}        # x in seconds, y as a fraction of the panel height
fig, axes = plt.subplots(1, 2, figsize=(W, 2.2), facecolor=SURF, gridspec_kw=dict(wspace=0.14))
for ax, pre, title, top in ((axes[0], "e4b", "E4B on the workstation: agent time of a run, seconds", 320),
                            (axes[1], "b31", "31B with emulated latency: time of a job, seconds", 620)):
    clean(ax)
    bins = [top / 40.0 * i for i in range(41)]
    for kind, col, lw in (("unsolved", GREY, 1.2), ("solved", BLUE, 1.5)):
        v = [min(x, top - 1e-6) for x in TD[pre + "_" + kind]]
        h, _ = __import__("numpy").histogram(v, bins=bins)
        h = h / float(len(v))
        ax.step(bins[:-1], h, where="post", color=col, lw=lw, zorder=3 if kind == "solved" else 2)
        x, y, ha = POS[(pre, kind)]
        ax.text(x, y, "%s (%s runs)" % (kind, format(len(v), ",")), color=INK, fontsize=7, va="bottom", ha=ha,
                transform=ax.get_xaxis_transform())
    ax.set_xlim(0, top); ax.set_ylim(0, None); ax.set_yticks([])
    ax.set_title(title, loc="left", color=INK, fontsize=7.5, pad=6)
fig.supxlabel("Share of runs in each time bin; the two curves of a panel each sum to one", color=INK2, fontsize=7.5, y=0.02)
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.2)
fig.savefig(out / "time_dist.pdf", facecolor=SURF); plt.close(fig)
print("figures written")
