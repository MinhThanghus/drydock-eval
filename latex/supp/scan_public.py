"""Scan files before publication: the release rules of the kit plus deny files that stay private.

usage (python -I -B):  scan_public.py <kit source dir> [--deny-file FILE ...] <path> [<path> ...]

Every text file under the paths is scanned line by line with the kit's rules (home paths, tokens, addresses, remote
access, internal identifiers) and with the rules of the deny files. A PDF is scanned through `pdftotext` and
`pdfinfo`. Other files are listed as not scanned. Exit code 1 on any finding.
"""
import argparse, importlib.util, pathlib, subprocess, sys

TEXT = {'.tex', '.py', '.md', '.csv', '.json', '.txt', '.bib', '.sh', '.cfg', '.toml'}
NAMES = {'SHA256SUMS', 'LICENSE', 'NOTICE', 'README'}
SKIP_DIRS = {'.git', '__pycache__'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kit'); ap.add_argument('--deny-file', action='append', default=[]); ap.add_argument('paths', nargs='+')
    o = ap.parse_args()
    spec = importlib.util.spec_from_file_location('release_check', str(pathlib.Path(o.kit) / 'scripts' / 'release_check.py'))
    rc = importlib.util.module_from_spec(spec); sys.modules['release_check'] = rc; spec.loader.exec_module(rc)
    rules = rc.Rules()
    for f in o.deny_file: rc.load_deny_file(f, rules)
    files = []
    for p in map(pathlib.Path, o.paths):
        files += [p] if p.is_file() else sorted(q for q in p.rglob('*') if q.is_file() and not (set(q.parts) & SKIP_DIRS))
    found, scanned, skipped = [], 0, []
    for q in files:
        if q.suffix == '.pdf':
            text = ''
            for cmd in (['pdftotext', '-layout', str(q), '-'], ['pdfinfo', str(q)]):
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0: sys.exit('scan_public: %s failed on %s' % (cmd[0], q))
                text += r.stdout + '\n'
        elif q.suffix in TEXT or q.name in NAMES:
            text = q.read_text(encoding='utf-8')
        else:
            skipped.append(str(q)); continue
        scanned += 1
        found += rc.scan_text(str(q), text, rules, need_header=False)
    for f in found: print(f.render())
    print('%d finding(s) in %d file(s) scanned; %d deny-file rule(s); not scanned (binary): %d' % (len(found), scanned, rules.deny_rules, len(skipped)))
    for s in skipped: print('  not scanned:', s)
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
