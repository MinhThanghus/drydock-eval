#!/usr/bin/env python3
"""Copy the checked text of the short paper and of its supplement into the extended paper.

The extended paper repeats the kit description and the case study of the short paper sentence for sentence. This
script takes those passages from the LaTeX sources of the short paper and writes them to gen/short_*.tex, and it copies
the generated tables and numbers of the supplement. It changes only citation keys, labels and run-in headings, and it
stops if a passage it expects is missing.

usage: reuse_short.py <latex directory of the short paper> <gen directory of the extended paper>
"""
import os
import re
import sys

SRC, GEN = sys.argv[1], sys.argv[2]

CITE = {
    "r1": "googledeepmind2026competition", "r2": "gemmateam2026gemma4", "r3": "afshar2026census",
    "r4": "mobassir2026toosmall", "r5": "adk2026toolresponses", "r6": "google2026gemma4prompt",
    "r7": "dimondev2026hf44", "r8": "delta90002026vllm38488", "r9": "gerganov2026hf135", "r10": "holbrook2026starter",
    "r11": "kwon2023vllm", "r12": "patel2026berth", "r13": "gao2019houdini", "r14": "rabin2025sandboxeval",
    "r15": "marchand2026sandboxescapebench", "r16": "agl2025retokenization", "r17": "damsolanke2026swekit",
    "r18": "tyfeng2026template", "r19": "miller2024errorbars", "r20": "cameron2008bootstrap",
    "r21": "mackinnon2017wild", "r22": "marcofpereira2026lab", "r23": "szeto2026rig",
    "r24": "houchiunyo2026silentzeros", "r25": "hemantkumarbk2026thoughtkeep",
}
LABEL = {
    "sec:1": "sec:intro", "sec:2": "sec:setting", "sec:3": "sec:contain", "sec:4": "sec:conform", "sec:5": "sec:compare",
    "sec:6": "sec:case", "sec:7": "sec:related", "sec:8": "sec:limits", "sec:9": "sec:avail",
    "tab:1": "tab:versions", "tab:2": "tab:checks", "tab:3": "tab:simulated", "tab:4": "tab:casearms",
}
HEAD = "% written by scripts/reuse_short.py from the sources of the short paper; do not edit\n"


def read(rel):
    return open(os.path.join(SRC, rel), encoding="utf-8").read()


def write(name, text):
    open(os.path.join(GEN, name), "w", encoding="utf-8").write(HEAD + text.strip("\n") + "\n")


def convert(text):
    text = re.sub(r"\\cite\{([^}]*)\}", lambda m: "\\citep{%s}" % ",".join(CITE[k.strip()] for k in m.group(1).split(",")), text)
    text = re.sub(r"\\S\\ref\{(sec:\d)\}", lambda m: "\\cref{%s}" % LABEL[m.group(1)], text)
    text = re.sub(r"\\(label|ref)\{((?:sec|tab):\d)\}", lambda m: "\\%s{%s}" % (m.group(1), LABEL[m.group(2)]), text)
    text = re.sub(r"(?m)^\\textbf\{([^{}]*\.)\} ", lambda m: "\\paragraph{%s} " % m.group(1), text)
    return text


def once(text, old, new):
    if text.count(old) != 1:
        raise SystemExit("expected one occurrence of %r, found %d" % (old[:60], text.count(old)))
    return text.replace(old, new)


# ---- the short paper: one piece per section, without its heading
body = read("gen/body.tex")
parts = re.split(r"(?m)^\\section\{[^\n]*\}\\label\{sec:(\d)\}\n", body)
sections = dict(zip(parts[1::2], parts[2::2]))
if sorted(sections) != list("123456789"):
    raise SystemExit("sections of the short paper: %s" % sorted(sections))
for num, name in (("3", "contain"), ("4", "conform"), ("5", "compare"), ("6", "case"), ("7", "related"), ("8", "limits"),
                  ("9", "avail")):
    write("short_%s.tex" % name, convert(sections[num]))

# ---- the supplement: the case study in full, the applications of the pass rule, the recomputation
supp = read("supp/supp_body.tex")
a = supp.index("\\subsection{Execution record}")
b = supp.index("\\section{Applications of the pass rule}")
full = once(supp[a:b], "(\\papersec{6})", "(\\cref{sec:case-design})")
write("short_casefull.tex", full)
c = supp.index("\\section{Recomputing the case study with the released kit}")
d = supp.index("\\section{Released data and code}")
write("short_repro.tex", supp[c:d].split("\n", 1)[1])
write("short_gates.tex", "\n".join(ln for ln in read("supp/gates_text.tex").split("\n") if not ln.startswith("%")))

# ---- generated tables and numbers of the supplement, copied as they are (one caption names a section of the short paper)
for name in sorted(os.listdir(os.path.join(SRC, "gen"))):
    if name.startswith("s_") or name in ("supp_macros.tex", "gate_macros.tex", "repro_macros.tex"):
        text = read("gen/" + name)
        if name == "s_gates.tex":
            text = once(text, "the pass rule of Section 5", "the pass rule of \\cref{sec:compare}")
        open(os.path.join(GEN, name), "w", encoding="utf-8").write(text)
print("pieces written")
