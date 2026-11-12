# Contain, Conform, Compare (Extended Version)

Nguyen Minh Thang. *Contain, Conform, Compare (Extended Version): A Red-Teamed Evaluation Kit for Small Coding Agents
and the Full Record of Its Experiments.* November 2026.

This directory holds the extended version of the paper that describes `drydock-eval`. It repeats the short paper
(directory `latex/` of this branch) and adds the record of the experiments made with and around the kit: the serving
measurements, the experiments with the 31B model, the experiments with E4B, the three attempts to improve the E4B
baseline, and an analysis across them. The kit itself is on the branch `develop` of this repository (tag `v0.4.0`).

## Files

| Path | Content |
|---|---|
| `main.pdf` | the extended paper |
| `main.tex`, `sec/`, `refs.bib` | LaTeX sources; one file per section |
| `gen/` | LaTeX written by the scripts (numbers, tables, passages copied from the short paper); not edited by hand |
| `fig/`, `figsrc/` | figures and the script that draws them |
| `data/` | the released data files, with `README.md` (files and coded values) and `SHA256SUMS` |
| `scripts/` | the scripts behind the numbers (below) |

## Scripts

| Script | What it does | Inputs |
|---|---|---|
| `scripts/analyze_long.py` | computes every number and table of Sections 3, 7, 8, 10 and 11 and stops if a check against the rows fails | `data/` |
| `scripts/recheck_gates.py` | applies the pass rule of the released kit to the rows of the three confirmation blocks and compares the result with the registered decisions | `data/` and the kit at tag `v0.4.0` |
| `figsrc/figs_long.py` | draws the figures of the added sections | `data/`, `gen/analysis.json` |
| `scripts/reuse_short.py` | copies the passages, tables and numbers of the short paper into `gen/` without changing a sentence | the directory `latex/` of this branch |

The files of `data/` were written from the raw result rows by an export script that keeps a fixed list of fields and
replaces every arm name by its label. The raw rows and that script are not released: they hold error texts, test
output, paths, machine details and the names of the withheld bundles. Everything above can be re-run from this
repository:

```
git clone https://github.com/MinhThanghus/drydock-eval kit            # branch develop, the kit
git clone -b paper https://github.com/MinhThanghus/drydock-eval paper
pip install "./kit[compare]" matplotlib

cd paper/latex_long
python scripts/analyze_long.py data gen          # numbers and tables
python scripts/recheck_gates.py ../../kit data gen   # the three decisions, recomputed
python figsrc/figs_long.py data gen fig          # figures
python scripts/reuse_short.py ../latex gen       # passages of the short paper
tectonic -X compile main.tex
```

## What is not here

The agent bundles under test, their prompts, the fine-tuned adapter and its training data, and the request dumps of
the jobs are withheld while the competition runs. Arms are therefore named by neutral labels, and the tasks of the
competition's development set by opaque codes; the table that maps these codes to the tasks is not released.

## Licences

Scripts (`*.py`): Apache-2.0, as the kit. Text, figures and data: CC BY 4.0
(<https://creativecommons.org/licenses/by/4.0/>).
