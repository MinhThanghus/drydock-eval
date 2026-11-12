# Contain, Conform, Compare: paper, supplement and data

This directory holds the paper that describes `drydock-eval`, its supplementary material, and the outcome data of the
paper's case study. The kit itself is on the branch `develop` of this repository (tag `v0.4.0`).

Nguyen Minh Thang. *Contain, Conform, Compare: A Red-Teamed Kit for Evaluating Small Coding Agents on One GPU.*
November 2026.

## Files

| Path | Content |
|---|---|
| `main.pdf` | the paper |
| `supplement.pdf` | supplementary material: the case study in full, every application of the pass rule, the recomputation with the released kit, the data description |
| `full.pdf` | the paper followed by the supplement, in one file |
| `paper.md` | the text of the paper as it was checked; `build_tex.py` converts it to LaTeX without rewriting a sentence |
| `fig1_parity_and_effects_v2.png` | the figure of `paper.md` as an image (Figure 3 of `main.pdf`) |
| `main.tex`, `supplement.tex`, `full.tex`, `preamble.tex` | top-level LaTeX files |
| `gen/` | LaTeX written by the scripts (paper body, tables and numbers of the supplement); not edited by hand |
| `supp/` | text of the supplement and the scripts behind it (below) |
| `fig/`, `figsrc/` | figures and the scripts that draw them |
| `data/` | the released data files, with `README.md` (columns) and `SHA256SUMS` |

## Scripts

| Script | What it does | Inputs |
|---|---|---|
| `build_tex.py` | converts `paper.md` to `gen/meta.tex`, `gen/abstract.tex`, `gen/body.tex`, `gen/refs.tex`; prints the word count | `paper.md` (in this directory) and its sha256; needs `pandoc` |
| `supp/build_supp.py` | recounts the case study from its result rows, stops if a recount differs from the frozen analysis report, then writes `data/` and the tables and numbers of the supplement | the raw result rows and reports, which are **not** released (they hold request dumps, error texts and machine details) |
| `supp/reproduce.py` | runs the released kit on `data/e8_compare_table.csv` and compares every quantity with the registered analysis | `data/` and the kit at tag `v0.4.0` |
| `figsrc/fig_supp.py` | draws the four figures of the supplement | `data/` |
| `figsrc/fig_versions.py` | draws the runner-version figure of the paper | the counts of Table 1, typed in the script |
| `figsrc/fig_parity.py` | draws the parity-and-effects figure of the paper | solved counts typed in the script, and `data/e8_summary.json` |
| `supp/scan_public.py` | scans files with the kit's release check before they are published | the kit at tag `v0.4.0` |

`supp/build_supp.py` is published so that the way each number was made can be read; it cannot be re-run without the
withheld inputs. Everything else can be re-run from this repository:

```
git clone https://github.com/MinhThanghus/drydock-eval kit            # branch develop, the kit
git clone -b paper https://github.com/MinhThanghus/drydock-eval paper
pip install "./kit[compare]" matplotlib

cd paper/latex
drydock-eval compare signflip data/e8_compare_table.csv --reference F --repeats 4    # the analysis of Section 6
python supp/reproduce.py ../../kit data gen        # the same, checked against the registered values
python figsrc/fig_supp.py data fig              # figures of the supplement
python figsrc/fig_parity.py fig/parity_effects.pdf --e8 data/e8_summary.json    # Figure 3 of the paper
python figsrc/fig_versions.py fig/runner_versions.pdf                              # Figure 2 of the paper
python build_tex.py paper.md "$(sha256sum paper.md | cut -d' ' -f1)"    # LaTeX of the paper from its text
tectonic -X compile main.tex && tectonic -X compile supplement.tex && tectonic -X compile full.tex
```

## What is not here

The agent bundle under test, its prompts and the request dumps of the jobs are withheld for now; their release is
planned for 2026-12-03. The two arms of each pass-rule application are therefore named `baseline` and `candidate`.

## Licences

Scripts (`*.py`): Apache-2.0, as the kit. Text, figures and data: CC BY 4.0
(<https://creativecommons.org/licenses/by/4.0/>).
