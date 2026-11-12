"""Build the LaTeX sources of the paper from the checked markdown draft, so that both carry the same words.

usage (python3 -I -B): build_tex.py <DRAFT_vN.md> <sha256 of that file>

Writes gen/meta.tex, gen/abstract.tex, gen/body.tex and gen/refs.tex next to this file (overwritten on every run; they
are build products). main.tex, fig/ and supp/ are written by hand or by figsrc/ and are not touched.

What is converted mechanically (pandoc, then the replacements below; no sentence is rewritten):
  - "## N Title" -> \\section{Title}\\label{sec:N};  "## Abstract" -> abstract;  "## References" -> thebibliography
  - "[3, 4]" -> \\cite{r3,r4};  the section sign + N -> \\ref{sec:N};  "Table N" -> \\ref{tab:N};  "Figure 1" -> \\ref{fig:parity}
  - each pipe table and its "Table N. caption" paragraph -> a table float (column widths: WIDTHS below)
  - the image line -> a figure float with the same caption (file: fig/parity_effects.pdf)
What is added (LaTeX only, listed in ADDED with its words): two figure floats with short captions.
"""
import hashlib, pathlib, re, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
GEN = HERE / 'gen'
WIDTHS = {1: (20, 19, 61), 2: (13, 52, 35), 3: (58, 21, 21), 4: (5, 40, 19, 36)}      # relative column widths per table
ADDED = {                                                                              # LaTeX-only floats: caption text
    'overview': 'The kit: where each module sits.',
    'versions': 'Canaries passed and high-severity vectors found, per runner version.',
}
ANCHOR_OVERVIEW = r'\textbf{Stack.}'            # the overview figure is placed before this paragraph
UNI = {'\u2212': r'\ensuremath{-}', '\u2248': r'\ensuremath{\approx}', '\u2264': r'\ensuremath{\le}',
       '\u00b1': r'\ensuremath{\pm}', '\u00a7': r'\S{}', '\u2014': '---', '\u2013': '--'}


def pandoc(md, to='latex', inline=False):
    # inline=True: one caption or title; "(A) ..." at its start must not become a list
    return subprocess.run(['pandoc', '-f', 'markdown-fancy_lists' if inline else 'markdown', '-t', to, '--wrap=none'],
                          input=md, text=True, capture_output=True, check=True).stdout


def words(md):
    return len(pandoc(md, 'plain').split())


def post(tex):
    """Replacements on pandoc's LaTeX. Every pattern must leave the visible words unchanged."""
    tex = re.sub(r'\{\[\}(\d+(?:, \d+)*)\{\]\}', lambda m: '\\cite{' + ','.join('r' + x.strip() for x in m.group(1).split(',')) + '}', tex)
    tex = re.sub('\u00a7(\\d)', r'\\S\\ref{sec:\1}', tex)
    tex = re.sub(r'\bTable (\d)\b', r'Table~\\ref{tab:\1}', tex)
    tex = re.sub(r'\bFigure 1(?=[AB]?\b)', r'Figure~\\ref{fig:parity}', tex)
    tex = tex.replace(r'd\_X', r'$d_X$')
    tex = re.sub(r'\\texttt\{([0-9a-f]{64})\}', r'\\texttt{\\seqsplit{\1}}', tex)
    for k, v in UNI.items():
        tex = tex.replace(k, v)
    bad = sorted({c for c in tex if ord(c) > 127})
    if bad: sys.exit('characters without a LaTeX mapping: %r' % bad)
    return tex


LT = re.compile(r'\\begin\{longtable\}\[\]\{(.*?)\}\n\\toprule\\noalign\{\}\n(.*?)\\midrule\\noalign\{\}\n\\endhead\n'
                r'\\bottomrule\\noalign\{\}\n\\endlastfoot\n(.*?)\\end\{longtable\}', re.S)


def table_float(tex, n, caption):
    m = LT.search(tex)
    if not m: sys.exit('table %d: pandoc layout not recognised' % n)
    spec, head, rows = m.groups()
    out = ('\\begin{table}[!tb]\n\\centering\n\\caption{%s}\\label{tab:%d}\n\\footnotesize\n\\begin{tabular}{%s}\n\\toprule\n%s\\midrule\n%s'
           '\\bottomrule\n\\end{tabular}\n\\end{table}' % (caption, n, spec, head, rows))
    return tex[:m.start()] + out + tex[m.end():]


def main():
    draft, want = pathlib.Path(sys.argv[1]), sys.argv[2]
    raw = draft.read_bytes()
    if hashlib.sha256(raw).hexdigest() != want: sys.exit('the draft is not the file with that sha256')
    md = raw.decode('utf-8')
    if 'PENDING' in md: print('WARNING: the draft still holds a PENDING placeholder')
    lines = md.split('\n')
    assert lines[0].startswith('# ') and lines[2].startswith('*') and lines[2].endswith('*'), 'title / subtitle layout'
    title, subtitle = lines[0][2:].strip(), lines[2].strip('*').strip()
    parts = re.split(r'^## (.+)$', '\n'.join(lines[3:]), flags=re.M)
    assert not parts[0].strip(), 'text before the first section'
    secs = [(parts[i].strip(), parts[i + 1].strip('\n')) for i in range(1, len(parts), 2)]
    assert secs[0][0] == 'Abstract' and secs[-1][0] == 'References'

    GEN.mkdir(exist_ok=True)
    tt = post(pandoc(title, inline=True)).strip(); st = post(pandoc(subtitle, inline=True)).strip()
    (GEN / 'meta.tex').write_text('\\newcommand{\\papertitle}{%s}\n\\newcommand{\\papersubtitle}{%s}\n' % (tt, st))
    (GEN / 'abstract.tex').write_text('\\begin{abstract}\n%s\\end{abstract}\n' % post(pandoc(secs[0][1])))

    body, ntab, nfig = [], 0, 0
    for name, text in secs[1:-1]:
        m = re.match(r'^(\d+) (.+)$', name)
        assert m, 'section heading without a number: %r' % name
        # table captions: the paragraph "Table N. caption" that follows a pipe table
        caps = {}
        def take(mm):
            caps[int(mm.group(1))] = mm.group(2).strip(); return '\n'
        text = re.sub(r'^Table (\d)\. (.+)$', take, text, flags=re.M)
        # relative column widths: pandoc reads them from the dashes of the separator line
        seen = []
        def seps(mm):
            n = ntab + len(seen) + 1; seen.append(n)
            return '|' + '|'.join('-' * w for w in WIDTHS[n]) + '|'
        text = re.sub(r'^\|(?:-+\|)+$', seps, text, flags=re.M)
        assert sorted(caps) == seen, 'tables and captions do not pair up in section %s: %s %s' % (name, sorted(caps), seen)
        # the image line -> a marker; its caption is converted on its own
        figs = []
        def img(mm):
            figs.append(mm.group(1)); return 'FIGUREMARKER%d' % len(figs)
        text = re.sub(r'^!\[Figure 1\. (.+)\]\((\S+)\)$', img, text, flags=re.M)
        tex = pandoc(text)
        for n in seen:
            tex = table_float(tex, n, pandoc(caps[n], inline=True).strip())
            if n == 1:
                tex = tex.replace('\\end{table}', '\\end{table}\n\n\\input{fig/versions}', 1)
        for i, cap in enumerate(figs, 1):
            nfig += 1
            tex = tex.replace('FIGUREMARKER%d' % i,
                              '\\begin{figure}[!tb]\n\\centering\n\\includegraphics[width=\\linewidth]{fig/parity_effects.pdf}\n'
                              '\\caption{%s}\\label{fig:parity}\n\\end{figure}' % pandoc(cap, inline=True).strip())
        ntab += len(seen)
        if ANCHOR_OVERVIEW in tex:
            tex = tex.replace(ANCHOR_OVERVIEW, '\\input{fig/overview}\n\n' + ANCHOR_OVERVIEW, 1)
        body.append('\\section{%s}\\label{sec:%s}\n\n%s' % (post(pandoc(m.group(2))).strip(), m.group(1), post(tex)))
    out = '\n'.join(body)
    assert ntab == 4 and nfig == 1, (ntab, nfig)
    assert out.count('\\input{fig/overview}') == 1 and out.count('\\input{fig/versions}') == 1
    (GEN / 'body.tex').write_text(out)

    refs = []
    for ln in secs[-1][1].split('\n'):
        if not ln.strip(): continue
        m = re.match(r'^(\d+)\. (.+)$', ln)
        assert m and int(m.group(1)) == len(refs) + 1, 'reference list numbering: %r' % ln[:40]
        refs.append('\\bibitem{r%s} %s' % (m.group(1), post(pandoc(m.group(2))).strip()))
    cited = {int(x) for x in re.findall(r'r(\d+)', ','.join(re.findall(r'\\cite\{([^}]*)\}', out + (GEN / 'abstract.tex').read_text())))}
    assert cited == set(range(1, len(refs) + 1)), 'cited %s, listed 1..%d' % (sorted(set(range(1, len(refs) + 1)) ^ cited), len(refs))
    (GEN / 'refs.tex').write_text('\\begin{thebibliography}{%d}\n\\small\n%s\n\\end{thebibliography}\n' % (len(refs), '\n'.join(refs)))

    w_all = words(md); w_refs = words('## References\n' + secs[-1][1]); w_add = sum(len(v.split()) for v in ADDED.values())
    print('draft %s sha256 %s' % (draft.name, want))
    print('sections %d, tables %d, figure from the draft %d, references %d (all cited)' % (len(secs) - 2, ntab, nfig, len(refs)))
    print('words (pandoc plain text): draft %d (reference list %d of them); captions added in LaTeX %d; total %d'
          % (w_all, w_refs, w_add, w_all + w_add))


if __name__ == '__main__':
    main()
