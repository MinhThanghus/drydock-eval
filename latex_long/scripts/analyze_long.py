#!/usr/bin/env python3
"""Recompute every number, table and figure input of the long paper from the released CSV files.

usage: analyze_long.py DATA_DIR GEN_DIR
Standard library only. Writes GEN_DIR/num.tex (numbers as LaTeX macros), GEN_DIR/tab_*.tex,
GEN_DIR/analysis.json (inputs of the figures), DATA_DIR/experiments.csv, DATA_DIR/arms.csv,
DATA_DIR/submissions.csv.
"""
import collections, csv, json, math, os, sys

DATA, GEN = sys.argv[1:3]
os.makedirs(GEN, exist_ok=True)

def rd(name):
    return list(csv.DictReader(open(os.path.join(DATA, name), encoding="utf-8")))

def fnum(x):
    return float(x) if x not in ("", None) else None

RUNS = rd("runs.csv")
for r in RUNS:
    r["resolved"] = int(r["resolved"])
    r["repeat"] = int(r["repeat"])
    for k in ("wall_s", "t_agent_s", "t_setup_s", "t_verify_s", "emulation_pad_s", "model_calls", "tool_calls",
              "tokens_in", "tokens_out", "max_prompt_tokens", "summaries", "patch_chars"):
        r[k] = fnum(r[k])
CTRL = rd("controls.csv")
SERVE = rd("serving_bench.csv")
SERVERS = rd("serving_servers.csv")
GATES = json.load(open(os.path.join(DATA, "pass_rule_applications.json")))

M = collections.OrderedDict()      # macro name -> text
A = {}                             # analysis.json

def mac(name, val):
    if not name.isalpha():
        raise SystemExit("macro name " + name)
    if name in M:
        raise SystemExit("macro twice: " + name)
    M[name] = str(val)

def intc(n):
    return "{:,}".format(int(n)).replace(",", "{,}")

def pct(k, n, nd=1):
    return ("%." + str(nd) + "f") % (100.0 * k / n)

def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)

def wil(k, n):
    lo, hi = wilson(k, n)
    return "%.0f--%.0f" % (max(0.0, lo) + 0.0, min(100.0, hi))

def sign_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")

def median(xs):
    xs = sorted(x for x in xs if x is not None)
    n = len(xs)
    return (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2) if n else float("nan")

def sel(**kw):
    out = RUNS
    for k, v in kw.items():
        out = [r for r in out if (r[k] in v if isinstance(v, (set, tuple, list)) else r[k] == v)]
    return out

def solved(rows):
    return sum(r["resolved"] for r in rows)

_MINUS = __import__("re").compile(r"(?<![\w.$\-{}])-(\d)")
def table(name, caption, spec, head, lines, size="\\small", note=None, place="t", star=False):
    env = "table*" if star else "table"
    s = ["\\begin{%s}[%s]" % (env, place), "\\centering", size,
         "\\caption{%s}" % caption, "\\label{tab:%s}" % name,
         "\\begin{tabular}{%s}" % spec, "\\toprule", head + " \\\\", "\\midrule"]
    for ln in lines:
        ln = _MINUS.sub(r"\\ensuremath{-}\1", ln)          # a leading hyphen of a number is a minus sign
        s.append(ln if ln.startswith(("\\midrule", "\\addlinespace", "\\cmidrule")) else ln + " \\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    if note:
        s.append("\\par\\vspace{3pt}{\\footnotesize\\raggedright %s\\par}" % note)
    s.append("\\end{%s}" % env)
    open(os.path.join(GEN, "tab_%s.tex" % name), "w").write("\n".join(s) + "\n")

MACH = {"colab_emulated_l4": "one GPU, emulated latency", "kaggle_4xl4": "4$\\times$L4 (hosted notebook)",
        "workstation": "workstation, one GPU", "kaggle_cpu": "hosted notebook, CPU"}
SRC = {"competition_dev": "development tasks", "berth": "Berth"}

# ------------------------------------------------------------------ arms
ARMS = [
    # label, model, category, public, description
    ("BF5", "31B", "reference", "Single agent: the public Budget-Fit V5 configuration, run unchanged"),
    ("BF5-p", "31B", "prompt", "BF5 with one changed section of the task prompt"),
    ("BF5-c", "31B", "prompt", "BF5 with a prompt-only checking step before the patch is submitted"),
    ("BF5-s", "31B", "sampling", "BF5 with changed sampling settings"),
    ("W1", "31B", "architecture", "Wrapped: a root agent calls a coder sub-agent as a tool; the coder has its own context"),
    ("W2", "31B", "architecture", "W1 with a bounded restart of the coder after a malformed tool call"),
    ("W3", "31B", "architecture", "W2 with a root that can only pass the request on and restart the coder"),
    ("W1-b", "31B", "budget", "W1 with a shorter time limit, more tool calls and half the output limit per call"),
    ("W1-p", "31B", "prompt", "W1 with a rewritten coder prompt, a shorter time limit and half the output limit per call"),
    ("SH1", "31B", "architecture", "Wrapped, coder with a shell tool only; first version"),
    ("SH3", "31B", "architecture", "SH1, third version of prompt and limits"),
    ("SH5", "31B", "architecture", "SH1, fifth version of prompt and limits"),
    ("SH6", "31B", "prompt", "SH5 with an added analysis step in the prompt"),
    ("ID24", "31B", "reference", "Public bundle of Team Muhammed Idris, version 24, run unchanged"),
    ("ID24-r", "31B", "tools", "ID24 with one file-writing tool removed, a shorter command timeout, added prompt sections"),
    ("e-single", "E4B", "architecture", "Single agent"),
    ("e-wrap", "E4B", "architecture", "Wrapped, as W1"),
    ("e-wrap-ad", "E4B", "adapter", "e-wrap with a fine-tuned adapter on the coder (withheld)"),
    ("e-BF", "E4B", "prompt", "The public single-agent prompt adapted to E4B (six tools, 28 calls)"),
    ("e-BF-s", "E4B", "sampling", "e-BF with changed sampling settings"),
    ("base", "E4B", "baseline", "The withheld E4B agent bundle; the baseline of every later comparison"),
    ("base-p", "E4B", "prompt", "base with one prompt change"),
    ("v-p1", "E4B", "prompt", "base with one prompt-only change"),
    ("v-p2", "E4B", "prompt", "base with one prompt-only change (candidate 1)"),
    ("v-p3", "E4B", "prompt", "base with one prompt-only change"),
    ("v-p4", "E4B", "prompt", "base with one prompt-only change"),
    ("v-b1", "E4B", "budget", "base with a changed thinking-token limit"),
    ("v-t1", "E4B", "tools", "base with a changed file-editing path, first build"),
    ("v-t2", "E4B", "tools", "as v-t1, second design, first build"),
    ("v-t3", "E4B", "tools", "rebuild of v-t1, option a"),
    ("v-t4", "E4B", "tools", "rebuild of v-t1, option b"),
    ("v-t5", "E4B", "tools", "rebuild of v-t2, option a"),
    ("v-t6", "E4B", "tools", "rebuild of v-t2, option b (candidate 2)"),
    ("v-a1", "E4B", "architecture", "base with an added sub-agent step, first design, first build"),
    ("v-a2", "E4B", "architecture", "as v-a1, second design, first build"),
    ("v-a3", "E4B", "architecture", "rebuild of v-a1 (candidate 3)"),
    ("v-a4", "E4B", "architecture", "rebuild of v-a2"),
]
used = {(r["arm"], r["model"]) for r in RUNS}
for a in ARMS:
    if (a[0], a[1]) not in used:
        raise SystemExit("arm not in the data: " + a[0])
if {a[0] for a in ARMS} != {r["arm"] for r in RUNS}:
    raise SystemExit("arm list differs from the data")
with open(os.path.join(DATA, "arms.csv"), "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["arm", "model", "category", "runs", "description"])
    for a in ARMS:
        w.writerow([a[0], a[1], a[2], len(sel(arm=a[0])), a[3]])
table("arms", "Arms of the experiments. Labels are neutral; the category names what was changed against the "
      "arm it was built from. Prompt text and bundle files are not released.",
      "llrp{0.58\\linewidth}", "Arm & Model & Runs & What it is",
      ["\\texttt{%s} & %s & %s & %s (\\emph{%s})" % (a[0], a[1], intc(len(sel(arm=a[0]))), a[3], a[2]) for a in ARMS],
      size="\\footnotesize", place="p")

# ------------------------------------------------------------------ blocks
BLOCK_NOTE = {
    "pub31-1": "first 30 development tasks", "pub31-2": "same 30 tasks", "pub31-3": "same 30 tasks",
    "pub31-4": "same 30 tasks, both arms in one queue", "pub31-5": "40 further development tasks",
    "pub31-6": "same 40 tasks, both arms in one queue", "pub31-7": "all 70 tasks", "pub31-8": "all 70 tasks; 30 tasks with masked test runners",
    "l4-1": "10 tasks, plus 2 masked", "l4-2": "24 tasks", "pubE4-1": "24 development tasks with small fixes",
    "berth31-time": "10 tasks, serial", "berth31-check": "16 tasks", "berthE4-k1": "24 tasks", "berthE4-k2": "8 tasks",
    "parity-1": "no sandbox, 8 workers", "parity-2": "no sandbox, 1 worker", "parity-3": "container runner, version 1",
    "parity-4": "container runner, version 3", "parity-5": "container runner, version 4, run a",
    "parity-6": "container runner, version 4, run b", "parity-7": "container runner, version 7 (final)",
    "screen-1": "screen set S", "confirm-1": "small panel C", "pilot-2a": "15 tasks of the case-study panel",
    "pilot-2b": "same 15 tasks", "screen-2": "screen set S", "confirm-2": "panel H", "dev-3": "16 tasks of panel C",
    "screen-3": "screen set S", "confirm-3": "panel H, second use",
}
blocks = []
for r in RUNS:
    if r["block"] not in blocks:
        blocks.append(r["block"])
if set(blocks) != set(BLOCK_NOTE):
    raise SystemExit("block list differs")
MACH_SHORT = {"colab_emulated_l4": "emulated", "kaggle_4xl4": "4$\\times$L4", "workstation": "workstation"}
SRC_SHORT = {"competition_dev": "dev.", "berth": "Berth"}
exp_rows, lines = [], []
for b in blocks:
    rows = sel(block=b)
    arms = sorted({r["arm"] for r in rows})
    tasks = {r["task"] for r in rows}
    reps = max(r["repeat"] for r in rows)
    e = {"block": b, "experiment": rows[0]["experiment"], "model": rows[0]["model"], "machine": rows[0]["machine"],
         "task_source": rows[0]["task_source"], "tasks": len(tasks), "repositories": len({r["repository"] for r in rows}),
         "arms": " ".join(arms), "repeats": reps, "runs": len(rows), "solved": solved(rows), "note": BLOCK_NOTE[b]}
    exp_rows.append(e)
    lines.append("\\texttt{%s} & %s & %s & %s & %d & %d & %d & %s & %s" % (
        b, e["model"], MACH_SHORT[e["machine"]], SRC_SHORT[e["task_source"]], len(tasks), len(arms), reps, intc(len(rows)),
        BLOCK_NOTE[b]))
with open(os.path.join(DATA, "experiments.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(exp_rows[0].keys()), lineterminator="\n")
    w.writeheader()
    w.writerows(exp_rows)
table("blocks", "Every block of agent runs in the released file \\texttt{runs.csv}. A block is one session on one machine; "
      "its arms ran in one job queue. Machine: one GPU with emulated L4 latency (emulated), a hosted notebook with four L4 "
      "GPUs (4$\\times$L4), or the single-GPU workstation. Source: the development tasks of the competition (dev.) or Berth. "
      "Rep.: repeats per task and arm.", "llllrrrrp{0.27\\linewidth}",
      "Block & Model & Machine & Source & Tasks & Arms & Rep. & Runs & Note", lines, size="\\footnotesize\\setlength{\\tabcolsep}{4.5pt}", place="p", star=False)
mac("RunsTotal", intc(len(RUNS)))
mac("BlocksTotal", len(blocks))
mac("ArmsTotal", len(ARMS))
mac("RunsThirtyOne", intc(len(sel(model="31B"))))
mac("RunsEFour", intc(len(sel(model="E4B"))))
mac("RunsWorkstation", intc(len(sel(machine="workstation"))))
mac("ControlRows", intc(len(CTRL)))
mac("ServeRows", len(SERVE))
E8ROWS = rd("e8_runs.csv")
mac("CaseRows", intc(len(E8ROWS)))
mac("AllAgentRows", intc(len(RUNS) + len(E8ROWS)))

# ------------------------------------------------------------------ serving
def sv(exp, model, adapter, target, L, c):
    x = [r for r in SERVE if r["experiment"] == exp and r["model"] == model and r["adapter"] == adapter
         and r["request_target"] == target and int(r["context_tokens"]) == L and int(r["concurrency"]) == c]
    if len(x) != 1:
        return None
    return x[0]
lines = []
for model in ("31B", "12B", "E4B"):
    for L in (2048, 8192, 14336):
        for c in (1, 2, 4, 8, 16):
            r = sv("serve-1", model, "none", "base", L, c)
            if r is None or (model != "31B" and c == 2 and L == 2048):
                continue
            lines.append("%s & %s & %d & %s & %.1f & %s" % (model, intc(L), c, intc(round(float(r["prefill_tok_s"]))),
                                                          float(r["per_stream_tok_s"]), intc(round(float(r["aggregate_tok_s"])))))
    lines.append("\\addlinespace")
lines = lines[:-1]
table("serving", "Serving throughput on four L4 GPUs (vLLM, one server per model). Context: prompt tokens already in "
      "the request. Streams: concurrent requests. Each request generates 160 tokens.",
      "lrrrrr", "Model & Context & Streams & Prefill tok/s & Decode tok/s per stream & Decode tok/s, all streams", lines)
if any(int(r["errors"]) for r in SERVE):
    raise SystemExit("serving rows with errors")
A["serving"] = [{k: r[k] for k in r} for r in SERVE]
def svf(*a):
    return float(sv(*a)["per_stream_tok_s"])
mac("ServeBigOne", "%.1f" % svf("serve-1", "31B", "none", "base", 2048, 1))
mac("ServeBigLong", "%.1f" % svf("serve-1", "31B", "none", "base", 14336, 1))
mac("ServeBigFourPer", "%.1f" % svf("serve-1", "31B", "none", "base", 14336, 4))
mac("ServeBigFourAgg", "%.0f" % float(sv("serve-1", "31B", "none", "base", 14336, 4)["aggregate_tok_s"]))
mac("ServeMidOne", "%.1f" % svf("serve-1", "12B", "none", "base", 2048, 1))
mac("ServeSmallOne", "%.1f" % svf("serve-1", "E4B", "none", "base", 2048, 1))
mac("ServeSmallSixteenAgg", "%.0f" % float(sv("serve-1", "E4B", "none", "base", 2048, 16)["aggregate_tok_s"]))
mac("ServeBigSixteenAgg", "%.0f" % float(sv("serve-1", "31B", "none", "base", 2048, 16)["aggregate_tok_s"]))
mac("ServeBigPrefill", intc(round(float(sv("serve-1", "31B", "none", "base", 2048, 1)["prefill_tok_s"]))))
mac("ServeRatioSmallBig", "%.1f" % (svf("serve-1", "E4B", "none", "base", 2048, 1) / svf("serve-1", "31B", "none", "base", 2048, 1)))
srv = {(s["experiment"], s["model"], s["adapter"]): s for s in SERVERS}
mac("ServeBigStartup", "%.0f" % float(srv[("serve-1", "31B", "none")]["startup_s"]))
mac("ServeBigKV", intc(srv[("serve-1", "31B", "none")]["kv_cache_tokens"]))
mac("ServeMidKV", intc(srv[("serve-1", "12B", "none")]["kv_cache_tokens"]))
mac("ServeSmallKV", intc(srv[("serve-1", "E4B", "none")]["kv_cache_tokens"]))
lines, rel = [], []
ADN = {"none": "no adapter (control)", "r16-attention": "rank 16, attention projections",
       "r16-all-linear": "rank 16, all linear layers", "r8-two-modules": "rank 8, two module types"}
for ad in ("none", "r16-attention", "r16-all-linear", "r8-two-modules"):
    tgt = "base" if ad == "none" else "adapter"
    s = srv[("serve-2", "31B", ad)]
    v = [svf("serve-2", "31B", ad, tgt, 2048, 1), svf("serve-2", "31B", ad, tgt, 8192, 1), svf("serve-2", "31B", ad, tgt, 8192, 4)]
    c = [svf("serve-2", "31B", "none", "base", 2048, 1), svf("serve-2", "31B", "none", "base", 8192, 1),
         svf("serve-2", "31B", "none", "base", 8192, 4)]
    ch = [100 * (a / b - 1) for a, b in zip(v, c)]
    rel += ch if ad != "none" else []
    cells = ["%.1f" % v[i] + ("" if ad == "none" else " (%+.0f\\%%)" % ch[i]) for i in range(3)]
    lines.append("%s & %s & %s & %s & %s" % (ADN[ad], intc(s["kv_cache_tokens"]), cells[0], cells[1], cells[2]))
    if ad != "none":
        b = svf("serve-2", "31B", ad, "base", 2048, 1)
        A.setdefault("adapter_base", {})[ad] = b
table("adapter", "Cost of serving a LoRA adapter with the 31B model (same server settings; requests addressed to the "
      "adapter). Decode speed in tokens per second per stream; change against the control in brackets.",
      "lrrrr", "Server & KV cache, tokens & 2k context, 1 stream & 8k, 1 stream & 8k, 4 streams", lines)
mac("AdapterLossMin", "%.0f" % (-max(rel)))
mac("AdapterLossMax", "%.0f" % (-min(rel)))
kvc = int(srv[("serve-2", "31B", "none")]["kv_cache_tokens"])
kva = int(srv[("serve-2", "31B", "r16-attention")]["kv_cache_tokens"])
mac("AdapterKV", intc(kva))
mac("AdapterKVLoss", "%.0f" % (100 * (1 - kva / kvc)))
bb = [A["adapter_base"][k] / svf("serve-2", "31B", "none", "base", 2048, 1) for k in A["adapter_base"]]
mac("AdapterBaseMin", "%.1f" % (100 * (min(bb) - 1)))
mac("AdapterBaseMax", "%.1f" % (100 * (max(bb) - 1)))

# ------------------------------------------------------------------ 31B on development tasks
P30 = {r["task"] for r in sel(block="pub31-1")}
P40 = {r["task"] for r in sel(block="pub31-5")}
if len(P30) != 30 or len(P40) != 40 or P30 & P40:
    raise SystemExit("panels")
def armrow(rows, label):
    n, k = len(rows), solved(rows)
    return "%s & %d & %d & %s & %s & %.0f & %s & %.1f & %d & %d & %d" % (
        label, n, k, pct(k, n, 0), wil(k, n), mean([r["wall_s"] for r in rows]),
        intc(round(mean([r["tokens_out"] for r in rows]))), mean([r["model_calls"] for r in rows]),
        sum(1 for r in rows if r["end_reason"] == "session_timeout"),
        sum(1 for r in rows if not r["patch_chars"]), sum(1 for r in rows if r["summaries"]))
lines = []
order = [("pub31-1", "BF5"), ("pub31-1", "BF5-p"), ("pub31-1", "W1"), ("pub31-2", "W1-b"), ("pub31-3", "W1-p"),
         ("pub31-4", "W3"), ("pub31-4", "SH1"), None, ("pub31-5", "BF5"), ("pub31-5", "W2"), ("pub31-6", "W3"),
         ("pub31-6", "SH3"), None, ("pub31-7", "SH5"), ("pub31-8", "SH6")]
for o in order:
    if o is None:
        lines.append("\\addlinespace")
        continue
    rows = sel(block=o[0], arm=o[1], condition="standard")
    lines.append(armrow(rows, "\\texttt{%s} & \\texttt{%s}" % o))
table("pubarms", "The 31B arms on the development tasks, one run per task, with emulated L4 latency. Top: the first 30 "
      "tasks. Middle: 40 further tasks. Bottom: all 70. Wall: mean seconds per job. Out: mean generated tokens. "
      "T/O: runs stopped by the session time limit. Empty: runs that left no patch. Sum.: runs with at least one history summary.",
      "llrrrrrrrrrr", "Block & Arm & Tasks & Solved & \\% & 95\\% interval & Wall & Out & Calls & T/O & Empty & Sum.",
      lines, size="\\footnotesize", star=False)

def ser(pairs):
    d = {}
    for b, a in pairs:
        for r in sel(block=b, arm=a, condition="standard"):
            if r["task"] in d:
                raise SystemExit("task twice in a series")
            d[r["task"]] = r["resolved"]
    if set(d) != P30 | P40:
        raise SystemExit("series does not cover the 70 tasks")
    return d
SER = collections.OrderedDict([
    ("BF5", ser([("pub31-1", "BF5"), ("pub31-5", "BF5")])),
    ("W1/W2", ser([("pub31-1", "W1"), ("pub31-5", "W2")])),
    ("W3", ser([("pub31-4", "W3"), ("pub31-6", "W3")])),
    ("SH1/SH3", ser([("pub31-4", "SH1"), ("pub31-6", "SH3")])),
    ("SH5", ser([("pub31-7", "SH5")])),
    ("SH6", ser([("pub31-8", "SH6")])),
])
T70 = sorted(P30 | P40)
lines = []
for name, d in SER.items():
    k = sum(d.values())
    a = sum(1 for t in T70 if d[t] and not SER["BF5"][t])
    b = sum(1 for t in T70 if not d[t] and SER["BF5"][t])
    k30, k40 = sum(d[t] for t in P30), sum(d[t] for t in P40)
    lines.append("\\texttt{%s} & %d & %d & %d & %s & %s & %s" % (
        name, k30, k40, k, pct(k, 70, 0), wil(k, 70),
        "--- & --- & ---" if name == "BF5" else "%d & %d & %.2f" % (a, b, sign_p(a, b))))
table("pubseventy", "Six configurations on all 70 development tasks (one run per task). The last three columns compare "
      "each configuration with \\texttt{BF5} on the same tasks: tasks only it solved, tasks only \\texttt{BF5} solved, and the "
      "two-sided exact sign test.", "lrrrrrrrr",
      "Configuration & First 30 & Next 40 & All 70 & \\% & 95\\% interval & Only it & Only \\texttt{BF5} & $p$", lines)
def uan(names):
    cnt = [sum(SER[n][t] for n in names) for t in T70]
    return sum(1 for c in cnt if c), sum(1 for c in cnt if c == len(names)), sum(1 for c in cnt if c == 0)
u4 = uan(["BF5", "W1/W2", "W3", "SH1/SH3"])
u6 = uan(list(SER))
mac("PubUnionFour", u4[0]); mac("PubAlwaysFour", u4[1]); mac("PubNeverFour", u4[2])
mac("PubUnionSix", u6[0]); mac("PubAlwaysSix", u6[1]); mac("PubNeverSix", u6[2])
mac("PubBetweenSix", 70 - u6[1] - u6[2])
mac("PubMinSeventy", min(sum(d.values()) for d in SER.values()))
mac("PubMaxSeventy", max(sum(d.values()) for d in SER.values()))
A["pub70"] = {"tasks": T70, "p30": sorted(P30), "series": {n: [SER[n][t] for t in T70] for n in SER},
              "repo": {t: sel(task=t)[0]["repository"] for t in T70}}
for n, d in SER.items():
    A.setdefault("pub70_tot", {})[n] = sum(d.values())
# same-queue pairs
def pair(block, a, b):
    da = {r["task"]: r["resolved"] for r in sel(block=block, arm=a, condition="standard")}
    db = {r["task"]: r["resolved"] for r in sel(block=block, arm=b, condition="standard")}
    oa = sum(1 for t in da if da[t] and not db[t]); ob = sum(1 for t in da if db[t] and not da[t])
    return sum(da.values()), sum(db.values()), oa, ob, sign_p(oa, ob)
q = pair("pub31-4", "W3", "SH1")
mac("PairAWthree", q[0]); mac("PairASh", q[1]); mac("PairAOnlySh", q[3]); mac("PairAOnlyW", q[2]); mac("PairAP", "%.3f" % q[4])
q = pair("pub31-6", "W3", "SH3")
mac("PairBWthree", q[0]); mac("PairBSh", q[1]); mac("PairBOnlySh", q[3]); mac("PairBOnlyW", q[2]); mac("PairBP", "%.2f" % q[4])
q = pair("pub31-5", "BF5", "W2")
mac("PairCBf", q[0]); mac("PairCW", q[1]); mac("PairCOnlyBf", q[2]); mac("PairCOnlyW", q[3]); mac("PairCP", "%.2f" % q[4])
q = pair("pub31-1", "BF5", "W1")
mac("PairDBf", q[0]); mac("PairDW", q[1]); mac("PairDOnlyBf", q[2]); mac("PairDOnlyW", q[3]); mac("PairDP", "%.2f" % q[4])
# the same arm twice on the same tasks: W1 (pub31-1) and W3 (pub31-4) share the coder; BF5 standard twice? (masked differs)
w1 = {r["task"]: r["resolved"] for r in sel(block="pub31-1", arm="W1")}
w3 = {r["task"]: r["resolved"] for r in sel(block="pub31-4", arm="W3")}
mac("CoderOnlyFirst", sum(1 for t in w1 if w1[t] and not w3[t])); mac("CoderOnlySecond", sum(1 for t in w1 if w3[t] and not w1[t]))
mac("CoderP", "%.2f" % sign_p(sum(1 for t in w1 if w1[t] and not w3[t]), sum(1 for t in w1 if w3[t] and not w1[t])))
# masked runners
lines = []
for arm in ("BF5", "SH5"):
    std = {r["task"]: r["resolved"] for r in RUNS if r["arm"] == arm and r["condition"] == "standard"
           and r["block"] in ("pub31-1", "pub31-7") and r["task"] in P30}
    msk = {r["task"]: r["resolved"] for r in sel(block="pub31-8", arm=arm, condition="masked_test_runners")}
    if set(std) != P30 or set(msk) != P30:
        raise SystemExit("masked sets")
    a = sum(1 for t in P30 if msk[t] and not std[t]); b = sum(1 for t in P30 if std[t] and not msk[t])
    lines.append("\\texttt{%s} & %d & %d & %d & %d" % (arm, sum(std.values()), sum(msk.values()), a, b))
    mac("Mask" + ("Bf" if arm == "BF5" else "Sh") + "Std", sum(std.values()))
    mac("Mask" + ("Bf" if arm == "BF5" else "Sh") + "Msk", sum(msk.values()))
table("masked", "Runs with the test runners masked inside the sandbox, on the first 30 development tasks (solved of 30).",
      "lrrrr", "Arm & Standard & Masked & Only masked & Only standard", lines)

# ------------------------------------------------------------------ real L4 against emulation
lines = []
for blk, arm, src in (("l4-1", "SH5", "pub31-7"), ("l4-2", "SH6", "pub31-8")):
    l4 = {r["task"]: r for r in sel(block=blk, arm=arm, condition="standard")}
    em = {r["task"]: r for r in sel(block=src, arm=arm, condition="standard") if r["task"] in l4}
    if set(em) != set(l4):
        raise SystemExit("l4 tasks")
    n = len(l4)
    both = sum(1 for t in l4 if l4[t]["resolved"] and em[t]["resolved"])
    ol = sum(1 for t in l4 if l4[t]["resolved"] and not em[t]["resolved"])
    oe = sum(1 for t in l4 if em[t]["resolved"] and not l4[t]["resolved"])
    wl, we = mean([l4[t]["wall_s"] for t in l4]), mean([em[t]["wall_s"] for t in l4])
    lines.append("\\texttt{%s} & %d & %d & %d & %d & %d & %d & %.0f & %.0f & %+.0f\\%%" % (
        arm, n, solved(l4.values()), solved(em.values()), both, ol, oe, wl, we, 100 * (wl / we - 1)))
    tag = "Five" if arm == "SH5" else "Six"
    mac("LfourN" + tag, n); mac("LfourReal" + tag, solved(l4.values())); mac("LfourEmu" + tag, solved(em.values()))
    mac("LfourOnlyReal" + tag, ol); mac("LfourOnlyEmu" + tag, oe)
    mac("LfourWallReal" + tag, "%.0f" % wl); mac("LfourWallEmu" + tag, "%.0f" % we)
    mac("LfourWallDiff" + tag, "%.0f" % abs(100 * (wl / we - 1)))
table("lfour", "The same arm and tasks on real 4$\\times$L4 hardware (serial jobs) and under latency emulation (eight workers). "
      "Wall: mean seconds per job.", "lrrrrrrrrr",
      "Arm & Tasks & Solved, L4 & Solved, emul. & Both & Only L4 & Only emul. & Wall, L4 & Wall, emul. & Diff.", lines,
      size="\\footnotesize")

# ------------------------------------------------------------------ hosted submissions
idr = sel(block="berth31-time", arm="ID24")
_bw = [r for r in RUNS if r["arm"] == "base" and r["machine"] == "workstation" and not r["block"].startswith("parity")]
_c3 = sel(block="confirm-3", arm="v-a3")
SUBS = [("2026-11-05", "W3", "31B", "0.12", "%d of 70 development tasks (emulated latency)" % sum(SER["W3"].values())),
        ("2026-11-08", "ID24", "31B", "0.17", "%d of %d Berth tasks (4$\\times$L4)" % (solved(idr), len(idr))),
        ("2026-11-09", "base", "E4B", "0.06", "%s\\%% of %s Berth runs (workstation)" % (pct(solved(_bw), len(_bw)), intc(len(_bw)))),
        ("2026-11-10", "BF5", "31B", "0.18", "%d of 70 development tasks (emulated latency)" % sum(SER["BF5"].values())),
        ("2026-11-11", "v-a3", "E4B", "0.05", "%d of %d Berth runs (panel H)" % (solved(_c3), len(_c3)))]
A["hosted"] = [{"date": x[0], "arm": x[1], "model": x[2], "score": x[3]} for x in SUBS]
_hl = {"W3": sum(SER["W3"].values()) / 70.0, "BF5": sum(SER["BF5"].values()) / 70.0,
       "base": solved(_bw) / float(len(_bw)), "v-a3": solved(_c3) / float(len(_c3))}
_hr = [float(x[3]) / _hl[x[1]] for x in SUBS if x[1] in _hl]
if len(_hr) != 4:
    raise SystemExit("hosted ratios")
mac("HostedRatioMin", "%.2f" % min(_hr)); mac("HostedRatioMax", "%.2f" % max(_hr))
with open(os.path.join(DATA, "submissions.csv"), "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["date_utc", "arm", "model", "public_score"])
    for s in SUBS:
        w.writerow(s[:4])
table("hosted", "Our five hosted submissions (one per day) and the local result of the same configuration. The public "
      "score is the solved share on the public part of the hidden test set, as displayed.",
      "llllp{0.42\\linewidth}", "Date (UTC) & Arm & Model & Public score & Local result of the same arm",
      ["%s & \\texttt{%s} & %s & %s & %s" % s for s in SUBS])

# ------------------------------------------------------------------ Berth, 31B
lines = []
for arm in ("BF5", "BF5-s", "ID24", "ID24-r"):
    rows = sel(block="berth31-time", arm=arm)
    lines.append("\\texttt{%s} & %d/%d & %.0f & %.1f & %.1f & %s & %.1f" % (
        arm, solved(rows), len(rows), mean([r["t_agent_s"] for r in rows]), mean([r["t_setup_s"] for r in rows]),
        mean([r["t_verify_s"] for r in rows]), intc(round(mean([r["tokens_out"] for r in rows]))),
        mean([r["tool_calls"] for r in rows])))
table("timing", "Timing block: four 31B arms on ten Berth tasks, jobs run one after another on 4$\\times$L4 with the "
      "software stack of the scorer. Times are mean seconds per job.", "lrrrrrr",
      "Arm & Solved & Agent & Setup & Verify & Tokens out & Tool calls", lines)
rows = sel(block="berth31-time")
mac("TimingAgentMin", "%.0f" % min(mean([r["t_agent_s"] for r in sel(block="berth31-time", arm=a)]) for a in ("BF5", "BF5-s", "ID24", "ID24-r")))
mac("TimingAgentMax", "%.0f" % max(mean([r["t_agent_s"] for r in sel(block="berth31-time", arm=a)]) for a in ("BF5", "BF5-s", "ID24", "ID24-r")))
def per_task(block, arm):
    d = collections.defaultdict(int)
    for r in sel(block=block, arm=arm):
        d[r["task"]] += r["resolved"]
    return d
a, b = per_task("berth31-check", "BF5"), per_task("berth31-check", "BF5-c")
mac("CheckBase", sum(a.values())); mac("CheckCand", sum(b.values())); mac("CheckRuns", len(sel(block="berth31-check", arm="BF5")))
mac("CheckUp", sum(1 for t in a if b[t] > a[t])); mac("CheckDown", sum(1 for t in a if b[t] < a[t]))
ta, tb = mean([r["t_agent_s"] for r in sel(block="berth31-check", arm="BF5")]), mean([r["t_agent_s"] for r in sel(block="berth31-check", arm="BF5-c")])
mac("CheckTime", "%.0f" % (100 * (tb / ta - 1)))

# ------------------------------------------------------------------ E4B
rows = sel(block="pubE4-1")
d = collections.defaultdict(int)
for r in rows:
    d[r["task"]] += r["resolved"]
mac("SmallPubUnion", sum(1 for t in d if d[t])); mac("SmallPubWall", "%.0f" % mean([r["wall_s"] for r in rows]))
mac("SmallPubTools", "%.0f" % mean([r["tool_calls"] for r in rows]))
_pa = {}
for r in rows:
    _pa.setdefault(r["arm"], {})[r["task"]] = r["resolved"]
mac("SmallPubAllThree", sum(1 for t in d if all(_pa[a][t] for a in _pa)))
for nm, arm in (("BF", "BF5"), ("SH", "SH6")):
    cc_ = collections.defaultdict(int); _n = collections.Counter()
    for r in RUNS:
        if r["arm"] == arm and r["machine"] == "colab_emulated_l4" and r["condition"] == "standard" and r["task"] in d:
            cc_[r["task"]] += r["resolved"]; _n[r["task"]] += 1
    if len(cc_) != 24 or set(_n.values()) != {1}:
        raise SystemExit("31B on the E4B development tasks")
    mac("SmallPubBig" + nm, sum(1 for t in cc_ if cc_[t]))
_a = collections.defaultdict(int); _b = collections.defaultdict(int)
for r in sel(block="berthE4-k1"):
    (_a if r["arm"] == "e-BF" else _b)[r["task"]] += r["resolved"]
mac("KOneBaseBetter", sum(1 for t in _b if _b[t] > _a[t])); mac("KOneOtherBetter", sum(1 for t in _b if _b[t] < _a[t]))
mac("KOneP", "%.2f" % sign_p(sum(1 for t in _b if _b[t] > _a[t]), sum(1 for t in _b if _b[t] < _a[t])))
mac("KTwoRepos", len({r["repository"] for r in sel(block="berthE4-k2")}))
lines = []
for blk, arms in (("pubE4-1", ("e-single", "e-wrap", "e-wrap-ad")), ("berthE4-k1", ("e-BF", "base")),
                  ("berthE4-k2", ("base", "base-p", "e-BF", "e-BF-s"))):
    for arm in arms:
        rows = sel(block=blk, arm=arm)
        t = [r["t_agent_s"] if r["t_agent_s"] is not None else r["wall_s"] for r in rows]
        lines.append("\\texttt{%s} & \\texttt{%s} & %d & %d & %d & %s & %s & %.0f & %.1f" % (
            blk, arm, len({r["task"] for r in rows}), len(rows), solved(rows), pct(solved(rows), len(rows), 0),
            wil(solved(rows), len(rows)), mean(t), mean([r["tool_calls"] for r in rows])))
    lines.append("\\addlinespace")
table("smallfirst", "First E4B experiments on hosted 4$\\times$L4 notebooks. Time: mean seconds per job (whole job in the "
      "first block, agent phase in the others).", "llrrrrrrr",
      "Block & Arm & Tasks & Runs & Solved & \\% & 95\\% interval & Time & Tool calls", lines[:-1], size="\\footnotesize")
# parity
lines, par = [], []
k1 = sel(block="berthE4-k1", arm="base")
for rep in (1, 2):
    rr = [r for r in k1 if r["repeat"] == rep]
    lines.append("hosted 4$\\times$L4, repeat %d & %d/%d & %s & %.0f" % (rep, solved(rr), len(rr), wil(solved(rr), len(rr)), mean([r["t_agent_s"] for r in rr])))
    par.append(("hosted, repeat %d" % rep, solved(rr), len(rr)))
lines.append("\\addlinespace")
for i in range(1, 8):
    rr = sel(block="parity-%d" % i)
    lines.append("workstation, %s & %d/%d & %s & %.0f" % (BLOCK_NOTE["parity-%d" % i], solved(rr), len(rr), wil(solved(rr), len(rr)), mean([r["t_agent_s"] for r in rr])))
    par.append((BLOCK_NOTE["parity-%d" % i], solved(rr), len(rr)))
table("parity", "The baseline on the same 24 Berth tasks, one run per task, on the hosted notebook and on the workstation "
      "under successive runners. Agent: mean seconds of the agent phase.", "lrrr",
      "Run & Solved & 95\\% interval, \\% & Agent", lines,
      note="A further workstation run under runner version 2 solved 4 of 24; its result rows were not kept and it is not in the released file.")
A["parity"] = par
ws = [solved(sel(block="parity-%d" % i)) for i in range(1, 8)]
mac("ParityMin", min(ws)); mac("ParityMax", max(ws)); mac("ParityWsTotal", sum(ws)); mac("ParityWsRuns", 24 * 7)
mac("ParityWsPct", pct(sum(ws), 24 * 7)); mac("ParityHostedTotal", solved(k1)); mac("ParityHostedPct", pct(solved(k1), len(k1)))

# screens
lines, scr = [], {}
SCREENS = [("pilot-2a", None), ("pilot-2b", None), ("dev-3", "base"), ("screen-1", "base"), ("screen-2", "base"), ("screen-3", "base")]
for blk, ref in SCREENS:
    arms = sorted({r["arm"] for r in sel(block=blk)}, key=lambda a: (a != "base", a))
    kb = solved(sel(block=blk, arm="base")) if ref else None
    for arm in arms:
        rows = sel(block=blk, arm=arm)
        k, n = solved(rows), len(rows)
        lead = "---" if (ref is None or arm == "base") else "%+.1f" % (100.0 * (k - kb) / n)
        lines.append("\\texttt{%s} & \\texttt{%s} & %d & %d & %d & %s & %s & %.0f" % (
            blk, arm, len({r["task"] for r in rows}), n, k, pct(k, n), lead, mean([r["t_agent_s"] for r in rows])))
        scr.setdefault(blk, {})[arm] = (k, n)
    lines.append("\\addlinespace")
table("screens", "Pilot, development and screen blocks of the E4B candidates on the workstation. Lead: difference to the "
      "baseline of the same block in percentage points. Agent: mean seconds of the agent phase.", "llrrrrrr",
      "Block & Arm & Tasks & Runs & Solved & \\% & Lead & Agent", lines[:-1], size="\\footnotesize", place="p")
A["screens"] = scr
LEAD = {}
for i, (blk, cand, conf) in enumerate((("screen-1", "v-p2", "confirm-1"), ("screen-2", "v-t6", "confirm-2"), ("screen-3", "v-a3", "confirm-3"))):
    kb, n = scr[blk]["base"]; kc, _ = scr[blk][cand]
    cb, cn = solved(sel(block=conf, arm="base")), len(sel(block=conf, arm="base"))
    cc = solved(sel(block=conf, arm=cand))
    g = GATES[i]
    if abs(g["gain"] - (cc - cb) / cn) > 1e-12:
        raise SystemExit("gain of application %d differs from the rows" % (i + 1))
    LEAD[cand] = {"screen_base": kb, "screen_cand": kc, "screen_n": n, "lead": 100.0 * (kc - kb) / n,
                  "conf_base": cb, "conf_cand": cc, "conf_n": cn, "gain": 100.0 * (cc - cb) / cn, "p": g["bootstrap"]["p"],
                  "rank": sorted([v[0] for v in scr[blk].values()], reverse=True).index(kc) + 1, "arms": len(scr[blk])}
    w = ("One", "Two", "Three")[i]
    mac("Lead" + w, "%+.1f" % LEAD[cand]["lead"]); mac("Gain" + w, "%+.2f" % LEAD[cand]["gain"])
    mac("GateP" + w, "%.4f" % g["bootstrap"]["p"]); mac("ConfBase" + w, cb); mac("ConfCand" + w, cc); mac("ConfRuns" + w, cn)
    mac("ScreenBase" + w, kb); mac("ScreenCand" + w, kc)
A["lead"] = LEAD
lines = []
for i, g in enumerate(GATES):
    conf = ("confirm-1", "confirm-2", "confirm-3")[i]
    cand = ("v-p2", "v-t6", "v-a3")[i]
    L = LEAD[cand]
    ta, tb = mean([r["t_agent_s"] for r in sel(block=conf, arm="base")]), mean([r["t_agent_s"] for r in sel(block=conf, arm=cand)])
    lines.append("%d & \\texttt{%s} & %d & %d & %d & %d & %d & %+.2f & %.4f & %d/%d & %+.2f & %.2f & %s" % (
        i + 1, cand, len({r["task"] for r in sel(block=conf)}), g["repositories"], L["conf_n"], L["conf_base"], L["conf_cand"],
        L["gain"], g["bootstrap"]["p"], g["repositories_positive"], g["repositories_negative"], 100 * g["loo_min"], tb / ta,
        "not confirmed" if g["decision"] == "FAIL" else "confirmed"))
    if g["decision"] != "FAIL":
        raise SystemExit("decision")
table("confirm", "The three applications of the pass rule (four repeats per task and arm). Gain: candidate minus baseline "
      "in percentage points. $p$: one-sided, repository-clustered. $+/-$: repositories with a positive or negative summed gain. "
      "LOO: lowest gain when one repository is left out. Time: ratio of mean agent time, candidate to baseline.",
      "rlrrrrrrrrrrl", "No. & Candidate & Tasks & Repos & Runs/arm & Base & Cand. & Gain & $p$ & $+/-$ & LOO & Time & Verdict",
      lines, size="\\footnotesize\\setlength{\\tabcolsep}{3.6pt}", star=False)
A["gates"] = [{"application": g["application"], "repos": g["repository_totals"]} for g in GATES]

# ------------------------------------------------------------------ repeatability of the baseline
def counts(block, arm="base"):
    d = collections.defaultdict(int)
    for r in sel(block=block, arm=arm):
        d[r["task"]] += r["resolved"]
    return d
def pearson(x, y):
    n = len(x); mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y)); sx = sum((a - mx) ** 2 for a in x); sy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sx * sy)
S = [counts(b) for b in ("screen-1", "screen-2", "screen-3")]
H = [counts(b) for b in ("confirm-2", "confirm-3")]
ST, HT = sorted(S[0]), sorted(H[0])
lines = []
for name, sess, tasks, per in (("S (48 tasks, 2 repeats)", S, ST, 2), ("H (128 tasks, 4 repeats)", H, HT, 4)):
    tot = [sum(s.values()) for s in sess]
    n = len(tasks) * per
    same = sum(1 for t in tasks if len({s[t] for s in sess}) == 1)
    zero = sum(1 for t in tasks if all(s[t] == 0 for s in sess))
    full = sum(1 for t in tasks if all(s[t] == per for s in sess))
    r = pearson([sess[0][t] for t in tasks], [sess[-1][t] for t in tasks])
    lines.append("%s & %d & %s & %s & %.1f & %d & %d & %d & %.2f" % (
        name, len(sess), ", ".join(str(x) for x in tot), ", ".join(pct(x, n) for x in tot),
        100.0 * (max(tot) - min(tot)) / n, same, zero, full, r))
table("repeat", "The unchanged baseline run again on the same tasks in separate sessions on the workstation. Range: "
      "largest minus smallest session total in percentage points. Same: tasks with the same solved count in every session. "
      "Never, always: tasks solved in no run, in every run. $r$: correlation of the per-task counts of the first and last session.",
      "lrllrrrrr", "Task set & Sessions & Solved runs & \\% & Range & Same & Never & Always & $r$", lines, size="\\footnotesize")
mac("RepeatSRange", "%.1f" % (100.0 * (max(sum(s.values()) for s in S) - min(sum(s.values()) for s in S)) / 96))
mac("RepeatHFirst", sum(H[0].values())); mac("RepeatHSecond", sum(H[1].values()))
mac("RepeatHDiff", "%.2f" % (100.0 * abs(sum(H[0].values()) - sum(H[1].values())) / 512))
k8 = {t: H[0][t] + H[1][t] for t in HT}
hist = collections.Counter(k8.values())
A["h_hist"] = [hist.get(i, 0) for i in range(9)]
mac("HNever", hist.get(0, 0)); mac("HAlways", hist.get(8, 0)); mac("HBetween", 128 - hist.get(0, 0) - hist.get(8, 0))
mac("HNeverPct", pct(hist.get(0, 0), 128, 0))
var_task = sum(k * (8 - k) / (8.0 * 7.0) for k in k8.values())      # sum of unbiased p(1-p)
sd_total = math.sqrt(4 * var_task)                                    # runs, 4 repeats per task
mac("NoiseSdRuns", "%.1f" % sd_total); mac("NoiseSdPts", "%.2f" % (100 * sd_total / 512))
mac("NoiseSdDiffPts", "%.2f" % (100 * sd_total * math.sqrt(2) / 512))
mac("NoiseSdDiffRuns", "%.1f" % (sd_total * math.sqrt(2)))
pm = sum(k8.values()) / (8.0 * 128)
mac("HBasePct", "%.1f" % (100 * pm))
sd_binom = math.sqrt(512 * pm * (1 - pm))
mac("NoiseSdBinomPts", "%.2f" % (100 * sd_binom / 512))
mac("NoiseShare", "%.0f" % (100 * (4 * var_task) / (512 * pm * (1 - pm))))
A["h_sessions"] = {"first": [H[0][t] for t in HT], "second": [H[1][t] for t in HT]}
A["baseline_sessions"] = {"S": [sum(s.values()) for s in S], "H": [sum(h.values()) for h in H],
                          "C": [solved(sel(block="confirm-1", arm="base"))]}
# candidate 3 per task
c3 = counts("confirm-3", "v-a3")
b3 = counts("confirm-3", "base")
mac("CandThreeTasksUp", sum(1 for t in HT if c3[t] > b3[t])); mac("CandThreeTasksDown", sum(1 for t in HT if c3[t] < b3[t]))
mac("CandThreeTasksSame", sum(1 for t in HT if c3[t] == b3[t]))
# candidate 3 against run-to-run noise alone (post hoc; not part of the registered rule)
d3 = sum(c3.values()) - sum(b3.values())
sd_diff = sd_total * math.sqrt(2)
mac("CandThreeDiffRuns", d3); mac("CandThreeNoiseRatio", "%.1f" % (d3 / sd_diff))
d3e = sum(c3.values()) - sum(H[0].values())
mac("CandThreeDiffEarlier", d3e); mac("CandThreeNoiseRatioEarlier", "%.1f" % (d3e / sd_diff))
dbs = abs(sum(H[0].values()) - sum(H[1].values()))
mac("BaseSessionsDiffRuns", dbs); mac("BaseSessionsNoiseRatio", "%.1f" % (dbs / sd_diff))

# ------------------------------------------------------------------ what a screen lead is worth under no effect
k6 = {t: S[0][t] + S[1][t] + S[2][t] for t in ST}
var_s = sum(k * (6 - k) / 30.0 for k in k6.values())               # sum of unbiased p(1-p), six runs per task
sd_arm_s = math.sqrt(2 * var_s)                                     # runs, one arm, 2 repeats per task
mac("ScreenSdArmRuns", "%.1f" % sd_arm_s); mac("ScreenSdArmPts", "%.1f" % (100 * sd_arm_s / 96))
mac("ScreenSdDiffPts", "%.1f" % (100 * sd_arm_s * math.sqrt(2) / 96))
def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
def null_lead(m, c=None, steps=4001, span=8.0):
    """Best of m arms minus the baseline, every arm N(0, sd_arm_s): mean, or P(lead >= c) when c is given."""
    h = 2 * span / (steps - 1)
    if c is None:           # E[max of m] by integrating x d(Phi^m); the baseline has mean 0
        tot = 0.0
        for i in range(steps):
            z = -span + i * h
            tot += z * m * _phi(z) ** (m - 1) * math.exp(-z * z / 2) / math.sqrt(2 * math.pi) * h
        return tot * sd_arm_s
    tot = 0.0
    for i in range(steps):   # integrate over the baseline value b (in SD units)
        b = -span + i * h
        tot += math.exp(-b * b / 2) / math.sqrt(2 * math.pi) * (1 - _phi(c / sd_arm_s + b) ** m) * h
    return tot
mac("ScreenNullLeadFive", "%.1f" % (100 * null_lead(5) / 96)); mac("ScreenNullLeadTwo", "%.1f" % (100 * null_lead(2) / 96))
lead_runs = {"One": (25 - 19, 5), "Two": (22 - 21, 2), "Three": (25 - 20, 2)}
chk = {"One": ("screen-1", "v-p2"), "Two": ("screen-2", "v-t6"), "Three": ("screen-3", "v-a3")}
for key, (lead, m) in lead_runs.items():
    blk, arm = chk[key]
    obs = solved(sel(block=blk, arm=arm)) - solved(sel(block=blk, arm="base"))
    if obs != lead:
        raise SystemExit("screen lead %s: %d != %d" % (key, obs, lead))
    narms = len({r["arm"] for r in sel(block=blk)}) - 1
    if narms != m:
        raise SystemExit("screen candidates %s: %d != %d" % (key, narms, m))
    mac("ScreenLeadRuns" + key, lead); mac("ScreenCands" + key, m)
    mac("ScreenNullP" + key, "%.2f" % null_lead(m, lead - 0.5))     # continuity correction: lead of at least this many runs

# ------------------------------------------------------------------ tasks that no arm solved
hall = collections.defaultdict(int); hn = collections.Counter()
for r in sel(block=("confirm-2", "confirm-3")):
    hall[r["task"]] += r["resolved"]; hn[r["task"]] += 1
if set(hn.values()) != {16}:
    raise SystemExit("H runs per task")
mac("HNeverAny", sum(1 for t in HT if hall[t] == 0))
resc = [t for t in HT if k8[t] == 0 and hall[t] > 0]
mac("HNeverRescued", len(resc)); mac("HNeverRescuedRuns", sum(hall[t] for t in HT if k8[t] == 0))
mac("HNeverCandRuns", 8 * sum(1 for t in HT if k8[t] == 0))
sall = collections.defaultdict(int)
for r in sel(block=("screen-1", "screen-2", "screen-3")):
    sall[r["task"]] += r["resolved"]
mac("SNeverAny", sum(1 for t in ST if sall[t] == 0)); mac("SNeverBase", sum(1 for t in ST if k6[t] == 0))
wsr = sel(machine="workstation")
wall_ = collections.defaultdict(int); wn = collections.Counter()
for r in wsr:
    wall_[r["task"]] += r["resolved"]; wn[r["task"]] += 1
mac("WsTasksAny", len(wall_)); mac("WsTasksSolvedAny", sum(1 for t in wall_ if wall_[t]))
mac("WsTasksNeverPct", pct(sum(1 for t in wall_ if not wall_[t]), len(wall_), 0))
mac("WsRunsPerTaskMin", min(wn.values())); mac("WsRunsPerTaskMax", max(wn.values()))
wun = [r for r in wsr if not r["resolved"]]
mac("WsUnsolved", intc(len(wun)))
mac("WsUnsolvedNormalPct", pct(sum(1 for r in wun if r["end_reason"] == "none"), len(wun), 0))
wempty = sum(1 for r in wun if not r["patch_chars"])
mac("WsUnsolvedEmptyPct", pct(wempty, len(wun), 0)); mac("WsUnsolvedWrongPct", pct(len(wun) - wempty, len(wun), 0))
mac("WsUnsolvedTimeoutPct", pct(sum(1 for r in wun if r["end_reason"] == "session_timeout"), len(wun), 1))

# ------------------------------------------------------------------ both models on the same Berth tasks
b31 = [r for r in RUNS if r["model"] == "31B" and r["task_source"] != "competition_dev"]
ct = sorted({r["task"] for r in b31})
if not set(ct) <= set(ST):
    raise SystemExit("31B Berth tasks outside S")
e4c = [r for r in sel(block=("screen-1", "screen-2", "screen-3"), arm="base") if r["task"] in set(ct)]
mac("CommonTasks", len(ct))
for nm, rr in (("Big", b31), ("Small", e4c)):
    tt = sum(r["t_agent_s"] for r in rr); sv_ = solved(rr)
    mac("Common%sRuns" % nm, len(rr)); mac("Common%sSolved" % nm, sv_); mac("Common%sPct" % nm, pct(sv_, len(rr)))
    mac("Common%sMean" % nm, "%.0f" % (tt / len(rr))); mac("Common%sPerSolved" % nm, "%.0f" % (tt / sv_))
c31 = collections.defaultdict(int)
for r in b31:
    c31[r["task"]] += r["resolved"]
mac("CommonBigTasks", sum(1 for t in ct if c31[t])); mac("CommonSmallTasks", sum(1 for t in ct if k6[t]))
mac("CommonOnlySmall", sum(1 for t in ct if k6[t] and not c31[t]))

# ------------------------------------------------------------------ hosted noise (third-party census; our recount of its draws)
HOSTED_SD_TASKS = 1.56          # pooled within-file SD of repeated public draws, 29 draws of 6 file sets
mac("HostedSdTasks", "%.2f" % HOSTED_SD_TASKS)
hsd = 100 * HOSTED_SD_TASKS * math.sqrt(2) / 58
mac("HostedSdDiffPts", "%.1f" % hsd)
mac("HostedDrawsNeeded", int(math.ceil(((1.959964 + 0.841621) * hsd / (100 * 2 / 58.0)) ** 2)))
mac("LocalBlockRatio", "%.0f" % ((hsd / (100 * sd_diff / 512)) ** 2))

# ------------------------------------------------------------------ per repository (baseline, workstation)
base_ws = [r for r in RUNS if r["arm"] == "base" and r["block"] in ("screen-1", "screen-2", "screen-3", "confirm-1", "confirm-2", "confirm-3")]
rep = collections.defaultdict(lambda: [set(), 0, 0])
for r in base_ws:
    x = rep[r["repository"]]
    x[0].add(r["task"]); x[1] += 1; x[2] += r["resolved"]
reps = sorted(rep.items(), key=lambda kv: -kv[1][2] / kv[1][1])
lines = ["\\texttt{%s} & %d & %s & %d & %s & %s" % (k.replace("_", "\\_"), len(v[0]), intc(v[1]), v[2], pct(v[2], v[1]), wil(v[2], v[1])) for k, v in reps]
table("repos", "Solved share of the E4B baseline by repository, over every baseline run of the screen and confirmation "
      "blocks on the workstation. The interval treats runs as independent and is too narrow where few tasks carry many runs.",
      "lrrrrr", "Repository & Tasks & Runs & Solved & \\% & 95\\% interval", lines, size="\\footnotesize", place="p")
A["repos"] = [{"repository": k, "tasks": len(v[0]), "runs": v[1], "solved": v[2]} for k, v in reps]
mac("BaseWsRuns", intc(len(base_ws))); mac("BaseWsSolved", intc(solved(base_ws))); mac("BaseWsPct", pct(solved(base_ws), len(base_ws)))
mac("BaseWsTasks", len({r["task"] for r in base_ws})); mac("BaseWsRepos", len(rep))
mac("RepoTopPct", pct(reps[0][1][2], reps[0][1][1], 0)); mac("RepoZero", sum(1 for k, v in reps if v[2] == 0))
top3 = sorted(rep.values(), key=lambda v: -v[2])[:3]
mac("RepoTopThreeShare", pct(sum(v[2] for v in top3), solved(base_ws), 0))

# ------------------------------------------------------------------ ends, time and cost
GROUPS = [("31B, development tasks, emulated latency", lambda r: r["experiment"] == "pub31", "wall_s"),
          ("31B, development tasks, 4$\\times$L4", lambda r: r["experiment"] == "l4check", "wall_s"),
          ("31B, Berth, 4$\\times$L4", lambda r: r["experiment"] == "berth31", "t_agent_s"),
          ("E4B, development tasks, 4$\\times$L4", lambda r: r["experiment"] == "pubE4", "wall_s"),
          ("E4B, Berth, 4$\\times$L4", lambda r: r["experiment"] == "berthE4", "t_agent_s"),
          ("E4B, Berth, workstation", lambda r: r["machine"] == "workstation", "t_agent_s")]
lines, cost = [], []
for name, f, tk in GROUPS:
    rows = [r for r in RUNS if f(r)]
    n, k = len(rows), solved(rows)
    t = [r[tk] for r in rows]
    ts, tu = mean([r[tk] for r in rows if r["resolved"]]), mean([r[tk] for r in rows if not r["resolved"]])
    lines.append("%s & %s & %s & %.0f & %.0f & %.0f & %.0f & %s & %s & %.1f & %.1f & %.1f & %.1f" % (
        name, intc(n), pct(k, n), mean(t), median(t), ts, tu, intc(round(mean([r["tokens_out"] for r in rows]))),
        intc(round(mean([r["tokens_in"] for r in rows]) / 1000.0)), mean([r["tool_calls"] for r in rows]),
        100.0 * sum(1 for r in rows if r["end_reason"] == "session_timeout") / n,
        100.0 * sum(1 for r in rows if r["end_reason"] == "tool_budget") / n,
        100.0 * sum(1 for r in rows if not r["patch_chars"]) / n))
    cost.append({"group": name, "runs": n, "solved": k, "mean_t": mean(t), "t_solved": ts, "t_unsolved": tu,
                 "per_solved": sum(x for x in t if x is not None) / k})
table("cost", "Time, tokens and endings of the agent runs by model and setting. Time: seconds per run (whole job for the "
      "development tasks, agent phase for Berth). In: prompt tokens summed over the requests of a run, in thousands. "
      "T/O, Budget, Empty: percentage of runs stopped by the time limit, stopped by the tool-call limit, and left without a patch.",
      "p{0.2\\linewidth}rrrrrrrrrrrr",
      "Setting & Runs & Solved \\% & Mean & Median & Solved & Unsolved & Out & In (k) & Tools & T/O & Budget & Empty",
      lines, size="\\scriptsize\\setlength{\\tabcolsep}{3.1pt}", star=False)
A["cost"] = cost
def _endpct(i, reason):
    rows = [r for r in RUNS if GROUPS[i][1](r)]
    return "%.1f" % (100.0 * sum(1 for r in rows if r["end_reason"] == reason) / len(rows))
mac("TimeoutBigDevPct", _endpct(0, "session_timeout")); mac("TimeoutBigBerthPct", _endpct(2, "session_timeout"))
mac("TimeoutSmallWsPct", _endpct(5, "session_timeout")); mac("BudgetSmallWsPct", _endpct(5, "tool_budget"))
mac("BudgetBigDevPct", _endpct(0, "tool_budget"))
mac("CostBigPerSolved", "%.0f" % cost[0]["per_solved"]); mac("CostSmallPerSolved", "%.0f" % cost[5]["per_solved"])
mac("CostBigMean", "%.0f" % cost[0]["mean_t"]); mac("CostSmallMean", "%.0f" % cost[5]["mean_t"])
mac("CostSmallSolvedT", "%.0f" % cost[5]["t_solved"]); mac("CostSmallUnsolvedT", "%.0f" % cost[5]["t_unsolved"])
mac("CostBigSolvedT", "%.0f" % cost[0]["t_solved"]); mac("CostBigUnsolvedT", "%.0f" % cost[0]["t_unsolved"])
wsr = sel(machine="workstation")
A["time_dist"] = {"e4b_solved": [r["t_agent_s"] for r in wsr if r["resolved"]], "e4b_unsolved": [r["t_agent_s"] for r in wsr if not r["resolved"]],
                  "b31_solved": [r["wall_s"] for r in sel(experiment="pub31") if r["resolved"]],
                  "b31_unsolved": [r["wall_s"] for r in sel(experiment="pub31") if not r["resolved"]]}
tot_h = sum(r["t_agent_s"] for r in wsr) / 3600.0
mac("WsAgentHours", "%.0f" % tot_h)
mac("WsTokensOutM", "%.0f" % (sum(r["tokens_out"] for r in wsr) / 1e6)); mac("WsTokensInM", "%.0f" % (sum(r["tokens_in"] for r in wsr) / 1e6))

# ------------------------------------------------------------------ controls
cc = collections.defaultdict(lambda: [0, 0, set()])
for r in CTRL:
    x = cc[(r["block"], r["kind"])]
    x[0] += int(r["resolved"]); x[1] += 1; x[2].add(r["task"])
CN = {"pool": "whole pool, workstation, no container", "runner-final-D": "panels C and H, final runner",
      "runner-final-S": "screen set S, final runner", "runner-final-E": "case-study panel and two tasks of S, final runner",
      "kaggle-cpu-1": "hosted notebook, first batch", "kaggle-cpu-2": "hosted notebook, second batch"}
lines = []
for b in CN:
    g, e = cc[(b, "reference_patch")], cc[(b, "empty_patch")]
    lines.append("%s & %d/%d & %d/%d" % (CN[b], g[0], g[1], e[0], e[1]))
table("controls", "Control runs of the grading path: every task graded once with its reference patch (must pass) and once "
      "with no patch (must fail).", "lrr", "Tasks and machine & Reference patch passes & Empty patch passes", lines)
mac("PoolTasks", cc[("pool", "reference_patch")][1]); mac("PoolEmptyPass", cc[("pool", "empty_patch")][0])
mac("PoolValid", cc[("pool", "reference_patch")][0] - cc[("pool", "empty_patch")][0])
kg = cc[("kaggle-cpu-1", "reference_patch")][2] | cc[("kaggle-cpu-2", "reference_patch")][2]
ke = cc[("kaggle-cpu-1", "empty_patch")][2] | cc[("kaggle-cpu-2", "empty_patch")][2]
mac("HostedCtrlRef", len(kg)); mac("HostedCtrlEmpty", len(ke))
mac("PoolRepos", len({r["repository"] for r in CTRL if r["block"] == "pool"}))
_res = collections.defaultdict(set); _blk = collections.defaultdict(set)
for r in CTRL:
    _res[(r["task"], r["kind"])].add(r["resolved"]); _blk[(r["task"], r["kind"])].add(r["block"])
mac("CtrlPairsMulti", intc(sum(1 for k in _blk if len(_blk[k]) > 1)))
mac("CtrlPairsDisagree", sum(1 for k in _res if len(_res[k]) > 1))
_pref = {r["task"]: int(r["resolved"]) for r in CTRL if r["block"] == "pool" and r["kind"] == "reference_patch"}
_pemp = {r["task"]: int(r["resolved"]) for r in CTRL if r["block"] == "pool" and r["kind"] == "empty_patch"}
_prepo = {r["task"]: r["repository"] for r in CTRL if r["block"] == "pool"}
_valid = {t for t in _pref if _pref[t] == 1 and _pemp.get(t) == 0}
_used_runs = {r["task"] for r in RUNS if r["task_source"] != "competition_dev"}
_used_case = {r["task"] for r in csv.DictReader(open(os.path.join(DATA, "e8_tasks.csv")))}
if (_used_runs | _used_case) - _valid:
    raise SystemExit("a used task is outside the validated pool")
_un = _valid - _used_runs - _used_case
_unc = collections.Counter(_prepo[t] for t in _un)
mac("PoolUsed", len(_used_runs | _used_case)); mac("PoolUntouched", len(_un)); mac("PoolUntouchedRepos", len(_unc))
mac("PoolUntouchedTopRepo", max(_unc.values())); mac("CaseOnlyTasks", len(_used_case - _used_runs))
mac("PoolUntouchedSmallRepos", sum(1 for v in _unc.values() if v == 1))

with open(os.path.join(GEN, "num.tex"), "w") as f:
    f.write("% written by scripts/analyze_long.py from the released data; do not edit\n")
    for k, v in M.items():
        f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
json.dump(A, open(os.path.join(GEN, "analysis.json"), "w"))
print("macros", len(M))
for k in ("PubUnionFour", "PubAlwaysFour", "PubNeverFour", "PubUnionSix", "PubAlwaysSix", "PubNeverSix", "NoiseSdPts", "NoiseSdDiffPts",
          "NoiseShare", "HNever", "HAlways", "LeadOne", "GainOne", "LeadTwo", "GainTwo", "LeadThree", "GainThree", "AdapterLossMin",
          "AdapterLossMax", "AdapterKVLoss", "BaseWsPct", "CostBigPerSolved", "CostSmallPerSolved", "WsAgentHours", "CheckUp", "CheckDown",
          "CheckTime", "LfourWallDiffFive", "LfourWallDiffSix", "RepoTopThreeShare", "CandThreeTasksUp", "CandThreeTasksDown"):
    print(k, M[k])
