# Copyright 2026 The drydock-eval Authors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``drydock-eval conform record|ref|check|demo|mutate|adk-roles|graph|graph-ref`` (``tools-ref`` is kept as an alias
of ``ref``)."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .checks import ALL_CHECKS


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("conform", help="check that a local stack sends what the Kaggle scorer sends",
                       description="Conformance checks: record what your local stack sends to the model and compare "
                                   "it with the Kaggle scorer path.")
    cs = p.add_subparsers(dest="conform_cmd", metavar="<command>")
    cs.required = True

    r = cs.add_parser("record", help="run the logging proxy in front of an OpenAI-compatible server")
    r.add_argument("--upstream", required=True, help="model server base URL, e.g. http://127.0.0.1:8000")
    r.add_argument("--listen", default="127.0.0.1:8099", help="host:port to listen on (default 127.0.0.1:8099)")
    r.add_argument("--out", required=True, help="capture file to append to (JSONL)")
    r.add_argument("--request-token-ids", action="store_true",
                   help="RECOMMENDED: ask the server for prompt token ids (adds return_token_ids=true to the forwarded "
                        "request; vLLM returns prompt_token_ids; the recorded request is left unchanged), so the "
                        "template check compares ids instead of lengths")
    r.set_defaults(func=cmd_record)

    f = cs.add_parser("ref", help="make a reference (tools, request fields, message shape) from a capture of your own "
                                  "harness")
    f.add_argument("captures", nargs="+")
    f.add_argument("--out", required=True)
    f.add_argument("--hash-only", action="store_true", help="store tool hashes only, not the schemas")
    f.set_defaults(func=cmd_ref)

    c = cs.add_parser("check", help="run the conformance checks over captures")
    c.add_argument("captures", nargs="+", help="capture files or directories (.jsonl/.jsonl.gz); client-side LiteLLM "
                                               "call dumps are imported too")
    c.add_argument("--profile", default=None, help="built-in profile (default kaggle-e4b-2026: scorer-path properties, "
                                                   "no sampling) or a TOML file that overrides it, for example with "
                                                   "your submission's sampling (format: README)")
    c.add_argument("--ref", help="reference made with 'drydock-eval conform ref' from a capture of your own harness "
                                 "(tools, sampling expectation, message shape)")
    c.add_argument("--server-thought-keys", choices=["reasoning", "reasoning,reasoning_content"], default=None,
                   help="history-thought keys your server reads: 'reasoning' for vLLM <= 0.21 (the scorer's 0.19.1, "
                        "default), 'reasoning,reasoning_content' for vLLM >= 0.22")
    c.add_argument("--system-prompt", default="auto", choices=["auto", "same", "any"],
                   help="how the shape check judges the system prompt's text against --ref: 'auto' (default) FAILs a "
                        "different text when the reference has one system text, WARNs when it varies; 'same' FAILs any "
                        "text the reference never sent; 'any' skips it (your agent configuration differs on purpose)")
    c.add_argument("--only", default=",".join(ALL_CHECKS), help=f"comma list of checks (default: {','.join(ALL_CHECKS)})")
    c.add_argument("--template", help="reference chat template (.jinja) for the template check; not shipped with drydock-eval")
    c.add_argument("--template-sha256", help="expected sha256 of --template (required unless --allow-unverified-template)")
    c.add_argument("--allow-unverified-template", action="store_true")
    c.add_argument("--tokenizer", help="tokenizer.json for the template check ('toy' for the kit's toy tokenizer)")
    c.add_argument("--special-tokens", help="tokenizer_config.json to read bos/eos/... token strings from")
    c.add_argument("--content-format", default=None, choices=["openai", "string"],
                   help="how the server hands message content to the template (default: from the profile, 'openai')")
    c.add_argument("--tools-ref", help="same as --ref (older tool-only references are accepted too)")
    c.add_argument("--vllm-log", help="vLLM log with request logging; attaches server prompt token ids by order "
                                      "(one-worker captures only)")
    c.add_argument("--post-invocation", default="always", choices=["always", "if-not-compacted", "never"],
                   help="end-of-invocation compaction behaviour to replay (google-adk 1.36.1: always)")
    c.add_argument("--json", help="write the full report as JSON")
    c.add_argument("-v", "--verbose", action="store_true", help="print every finding")
    c.set_defaults(func=cmd_check)

    t = cs.add_parser("tools-ref", help="alias of 'ref'")
    t.add_argument("captures", nargs="+")
    t.add_argument("--out", required=True)
    t.add_argument("--hash-only", action="store_true", help="store tool hashes only, not the schemas")
    t.set_defaults(func=cmd_ref)

    d = cs.add_parser("demo", help="live CPU demo with a real google-adk agent against the mock server")
    d.add_argument("--part", default="all", choices=["role", "compaction", "all"])
    d.add_argument("--out-dir", help="write the demo captures here")
    d.add_argument("--template", help="optional reference chat template to also count real tokens")
    d.add_argument("--template-sha256")
    d.add_argument("--allow-unverified-template", action="store_true")
    d.add_argument("--tokenizer")
    d.add_argument("--json")
    d.set_defaults(func=cmd_demo)

    m = cs.add_parser("mutate", help="mutation suite: inject each drift into live runs, measure detection")
    m.add_argument("--scenarios", help="comma list (default: all of short,long,nudge,parallel)")
    m.add_argument("--mutations", help="comma list of mutation names (default: all)")
    m.add_argument("--jobs", type=int, default=1, help="processes, one scenario each (default 1)")
    m.add_argument("--list", action="store_true", help="list the mutations and exit")
    m.add_argument("--json")
    m.add_argument("-v", "--verbose", action="store_true")
    m.set_defaults(func=cmd_mutate)

    a = cs.add_parser("adk-roles", help="which tool-result role each google-adk version sends for a model string")
    a.add_argument("--model", action="append", help="LiteLLM model string (repeatable)")
    a.set_defaults(func=cmd_adk_roles)

    from .graphcheck import DEFAULT_PAIRS, DEFAULT_QUERIES, DEFAULT_SEED
    g = cs.add_parser("graph", help="graph/embedding fidelity: prompt-section conditions, node-key/vector alignment, "
                                    "vector statistics against the official files' numbers",
                      description="Check a task's code-graph JSON and embedding NPZ the way the competition harness "
                                  "uses them. Give the files directly (--graph/--embeddings), or the directories "
                                  "with --repo/--base-commit (one task) or --tasks (every task in a JSONL).")
    g.add_argument("--graph", help="graph JSON (direct mode: file names are not checked)")
    g.add_argument("--embeddings", help="embedding NPZ (direct mode)")
    g.add_argument("--graph-dir", help="graph directory, as in your evaluator's config")
    g.add_argument("--embeddings-dir", help="embedding directory, as in your evaluator's config")
    g.add_argument("--repo", help="the task's 'repo' field, e.g. owner/name")
    g.add_argument("--base-commit", help="the task's 'base_commit' field")
    g.add_argument("--tasks", help="tasks JSONL with repo and base_commit: check every task")
    g.add_argument("--tools", help="declared tool names (comma list), 'none', or a reference made with "
                                   "'drydock-eval conform ref' (its tool list is used); default: unknown")
    g.add_argument("--reference", default="kaggle-dev-2026",
                   help="reference numbers: built-in 'kaggle-dev-2026' (default; measured on the official files, "
                        "numbers only), a JSON made with 'graph-ref', or 'none'")
    g.add_argument("--against", help="a second NPZ for the same graph (for example the official vectors when you "
                                     "make stand-ins for an official task): overlap@10 of the neighbour lists")
    g.add_argument("--pairs", type=int, default=DEFAULT_PAIRS, help=f"random node pairs (default {DEFAULT_PAIRS})")
    g.add_argument("--queries", type=int, default=DEFAULT_QUERIES,
                   help=f"sampled search_similar_code queries (default {DEFAULT_QUERIES})")
    g.add_argument("--seed", type=int, default=DEFAULT_SEED)
    g.add_argument("--jobs", type=int, default=1, help="processes for --tasks (default 1)")
    g.add_argument("--json", help="write the full report as JSON")
    g.add_argument("-v", "--verbose", action="store_true", help="print every finding")
    g.set_defaults(func=cmd_graph)

    gr = cs.add_parser("graph-ref", help="reference numbers (statistics only, no data) from a directory of graph/"
                                         "embedding pairs, e.g. your copy of the official files")
    gr.add_argument("--graph-dir", required=True)
    gr.add_argument("--embeddings-dir", required=True)
    gr.add_argument("--out", required=True)
    gr.add_argument("--name", default="local")
    gr.add_argument("--source", default="", help="free-text provenance stored in the reference")
    gr.add_argument("--per-file", action="store_true",
                    help="also store per-file rows with file stems (for local use; do not publish them)")
    gr.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    gr.add_argument("--queries", type=int, default=DEFAULT_QUERIES)
    gr.add_argument("--seed", type=int, default=DEFAULT_SEED)
    gr.add_argument("--jobs", type=int, default=1)
    gr.set_defaults(func=cmd_graph_ref)


# --- commands ------------------------------------------------------------------------------------------------------

def _template_setup(args: argparse.Namespace, profile=None):
    from .checks import TemplateSetup
    from .render import ChatTemplate, DEFAULT_SPECIAL_TOKENS
    from .tokenizer import load_tokenizer
    if not getattr(args, "template", None) and not getattr(args, "tokenizer", None):
        return None
    if not args.template or not args.tokenizer:
        raise SystemExit("the template check needs both --template and --tokenizer")
    if args.template == "toy":
        template = ChatTemplate.toy()
    else:
        template = ChatTemplate.from_path(args.template, args.template_sha256, args.allow_unverified_template)
    special = dict(DEFAULT_SPECIAL_TOKENS)
    if getattr(args, "special_tokens", None):
        with open(args.special_tokens, encoding="utf-8") as fh:
            cfg = json.load(fh)
        for key in list(special):
            val = cfg.get(key)
            if isinstance(val, dict):
                val = val.get("content")
            if isinstance(val, str):
                special[key] = val
    fmt = getattr(args, "content_format", None) or (profile.content_format if profile else "openai")
    return TemplateSetup(template, load_tokenizer(args.tokenizer), fmt, special)


def cmd_record(args: argparse.Namespace) -> int:
    from .recorder import Recorder
    host, _, port = args.listen.rpartition(":")
    rec = Recorder(args.upstream, out_path=args.out, host=host or "127.0.0.1", port=int(port),
                   request_token_ids=args.request_token_ids)
    print(f"drydock-eval recorder: {rec.url} -> {args.upstream}; appending to {args.out}")
    print(f"point your client at {rec.url}/v1 (or {rec.url}/s/<label>/v1 to label a session); Ctrl-C to stop")
    print("keep the recorder as the last hop before the server: settings a later proxy or the server applies are "
          "invisible here")
    if not args.request_token_ids:
        print("note: without --request-token-ids the template check can only compare prompt lengths")
    try:
        rec.serve_forever()
    except KeyboardInterrupt:
        pass
    print(f"recorded {len(rec.exchanges)} exchange(s)")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from .capture import attach_vllm_log_ids, load_capture
    from .checks import run_checks
    from .profile import load_profile
    from .report import FAIL, format_text, overall, to_json
    try:
        profile = load_profile(args.profile)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if args.server_thought_keys:
        profile.server_reasoning_keys = tuple(args.server_thought_keys.split(","))
        profile.server = "vLLM >= 0.22" if "reasoning_content" in args.server_thought_keys else "vLLM <= 0.21"
    exchanges = load_capture(args.captures)
    if not exchanges:
        print("no exchanges found in the given captures", file=sys.stderr)
        return 2
    meta: dict = {"profile": profile.name, "captures": [os.path.basename(c) for c in args.captures]}
    notes = []
    if args.vllm_log:
        att = attach_vllm_log_ids(exchanges, args.vllm_log)
        meta["vllm_log"] = att
        notes.append(f"vLLM log: {att['logged_prompts']} logged prompts, attached ids to {att['attached']} of "
                     f"{att['exchanges']} exchanges" + (f"; stopped: {att['stopped']}" if att["stopped"] else ""))
    only = tuple(x.strip() for x in args.only.split(",") if x.strip())
    setup = _template_setup(args, profile)
    reference = None
    ref_path = args.ref or args.tools_ref
    if ref_path:
        with open(ref_path, encoding="utf-8") as fh:
            reference = json.load(fh)
    results, inv = run_checks(exchanges, profile, only=only, template=setup, reference=reference,
                              post_invocation=args.post_invocation, system_prompt=args.system_prompt)
    meta["inventory"] = inv
    header = (f"drydock-eval conform check: {inv['exchanges']} exchanges ({inv['agent_calls']} agent calls in "
              f"{inv['trajectories']} trajectories, {inv['summarizer_calls']} summarizer calls"
              + (f", {inv['failed_exchanges']} failed exchanges left out" if inv["failed_exchanges"] else "")
              + f"); profile {profile.name}; server reads history thoughts from {list(profile.server_reasoning_keys)}")
    if notes:
        header += "\n" + "\n".join(notes)
    print(format_text(results, header, verbose=args.verbose))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(to_json(results, meta))
    return 1 if overall(results) == FAIL else 0


def cmd_ref(args: argparse.Namespace) -> int:
    from .capture import load_capture
    from .checks import make_reference
    ref = make_reference(load_capture(args.captures), include_schemas=not args.hash_only,
                         source=",".join(os.path.basename(c) for c in args.captures))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(ref, fh, indent=1, ensure_ascii=False)
    tools = ref.get("tools") or {}
    agent = ref["sampling"]["agent"]
    print(f"wrote {args.out} (from a {ref['level']} capture): {agent['calls']} agent calls; request fields "
          f"{', '.join(agent['observed_keys'])}; "
          + (f"{len(tools.get('tools') or [])} tools, list sha256 {tools.get('list_sha256')}" if tools else "no tools")
          + (f" (capture has {tools['distinct_lists_in_capture']} distinct tool lists; used the most common)"
             if tools and tools.get("distinct_lists_in_capture", 1) > 1 else ""))
    if ref["level"] != "wire":
        print("note: a client-call capture cannot show request fields added later in the HTTP stack; prefer a capture "
              "made with 'drydock-eval conform record'")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .adk_harness import adk_available
    if not adk_available():
        print("the demo needs google-adk and litellm: pip install 'drydock-eval[adk]'", file=sys.stderr)
        return 2
    from .demo import demo_compaction, demo_role
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    setup = _template_setup(args)
    out = {}
    if args.part in ("role", "all"):
        out["role"] = demo_role(args.out_dir, setup)
        print(out["role"]["text"])
        print()
    if args.part in ("compaction", "all"):
        out["compaction"] = demo_compaction(args.out_dir)
        print(out["compaction"]["text"])
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=str)
    return 0


def cmd_mutate(args: argparse.Namespace) -> int:
    from .mutate import format_matrix, mutations, run_suite
    if args.list:
        for m in mutations():
            print(f"{m.name:40s} {m.layer:7s} {m.target or '(control)':11s} {m.description}")
        return 0
    from .adk_harness import adk_available
    if not adk_available():
        print("the mutation suite runs a real google-adk agent: pip install 'drydock-eval[adk]'", file=sys.stderr)
        return 2
    scen = [s.strip() for s in args.scenarios.split(",")] if args.scenarios else None
    muts = [s.strip() for s in args.mutations.split(",")] if args.mutations else None
    summary = run_suite(scen, muts, jobs=args.jobs, verbose=args.verbose)
    print(format_matrix(summary))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1, default=str)
    return 1 if summary["baseline_false_alarms"] or summary["control_false_alarms"] else 0


def cmd_adk_roles(args: argparse.Namespace) -> int:
    from . import adk_roles
    from .adk_harness import GEMMA4_NAME_MODEL, KAGGLE_STYLE_MODEL
    models = args.model or [KAGGLE_STYLE_MODEL, "hosted_vllm/google/gemma-4-e4b-it", GEMMA4_NAME_MODEL]
    for row in adk_roles.version_table(models):
        print(f"{row['versions']:16s} {row['rule']}")
        for m, role in row["roles"].items():
            print(f"    {role:15s} {m}")
    installed = adk_roles.installed_adk_version()
    if installed:
        probe = adk_roles.probe_installed_adk(models)
        print(f"installed google-adk {installed}: " + (", ".join(f"{m} -> {r}" for m, r in probe.items()) if probe
                                                       else "converter not available for a live probe"))
    print("upstream: " + "; ".join(adk_roles.UPSTREAM_REFERENCES))
    return 0


def _stats_line(rows: list[dict]) -> str:
    from .graphcheck import summarise
    s = summarise(rows)
    parts = []
    for key, label in (("rand_cos_mean", "random-pair cosine mean"), ("edge_auc", "edge AUC"),
                       ("top10_sim_median", "median top-10 similarity"), ("norm_median", "median norm")):
        if key in s:
            lo, med, hi = s[key]
            parts.append(f"{label} {lo:.3f} / {med:.3f} / {hi:.3f}")
    return f"vector statistics over {s['files']} file pair(s), min / median / max: " + "; ".join(parts)


def cmd_graph(args: argparse.Namespace) -> int:
    from . import graphcheck as gc
    from .report import FAIL, format_text, overall
    reference = gc.load_reference(args.reference)
    tools = gc.tool_names_from(args.tools)
    kw = dict(pairs=args.pairs, queries=args.queries, seed=args.seed)
    ref_note = (f"reference {reference['name']} ({reference['pooled']['files']} files; WARN band "
                f"{gc.warn_band(reference)})" if reference else "no reference")
    tools_note = ("declared tools unknown" if tools is None
                  else f"declared graph tools: {', '.join(sorted(set(gc.GRAPH_TOOLS) & tools)) or 'none'}")
    meta: dict = {"reference": reference.get("name") if reference else None,
                  "declared_tools": sorted(tools) if tools is not None else None}
    if args.graph or args.embeddings:
        if not (args.graph and args.embeddings):
            raise SystemExit("direct mode needs both --graph and --embeddings")
        if args.tasks or args.graph_dir or args.embeddings_dir:
            raise SystemExit("give either --graph/--embeddings or --graph-dir/--embeddings-dir, not both")
        entry = gc.analyse_pair(args.graph, args.embeddings, against=args.against, **kw)
        align, vec = gc.results_for(entry, reference)
        results = [gc.check_section(None, tools, direct=(args.graph, args.embeddings)), align, vec]
        meta["files"] = entry
        header = f"drydock-eval conform graph: {os.path.basename(args.graph)} + {os.path.basename(args.embeddings)}"
    elif args.graph_dir and args.embeddings_dir:
        if args.tasks:
            tasks = gc.read_tasks(args.tasks)
            if args.against:
                raise SystemExit("--against works with one task only")
        elif args.repo:
            tasks = [{"instance_id": None, "repo": args.repo, "base_commit": args.base_commit}]
        else:
            raise SystemExit("with --graph-dir/--embeddings-dir give --repo (and --base-commit) or --tasks")
        resolved = [(t, gc.resolve(args.graph_dir, args.embeddings_dir, t["repo"], t["base_commit"])) for t in tasks]
        file_pairs = sorted({(r["graph"]["tool_file"], r["embeddings"]["tool_file"]) for _, r in resolved
                             if r["graph"]["tool_file"] and r["embeddings"]["tool_file"]})
        entries = gc.analyse_many(file_pairs, jobs=args.jobs, against=args.against, **kw)
        by_pair = {(e["graph"], e["embeddings"]): e for e in entries}
        sections = []
        for t, r in resolved:
            label = t["instance_id"] or f"{t['repo']}@{(t['base_commit'] or '')[:10]}"
            sections.append((label, gc.check_section(r, tools)))
        pair_results = []
        for (gp, ep), e in sorted(by_pair.items()):
            label = os.path.basename(gp)[:-5]
            align, vec = gc.results_for(e, reference)
            pair_results.append((label, align, vec))
        if len(tasks) == 1:
            results = [sections[0][1]]
            if pair_results:
                results += [pair_results[0][1], pair_results[0][2]]
            else:
                from .report import SKIP, CheckResult
                results += [CheckResult("graph-align", SKIP, "no file pair for the tools to load"),
                            CheckResult("graph-vectors", SKIP, "no file pair for the tools to load")]
        else:
            results = [gc.aggregate("graph-section", sections),
                       gc.aggregate("graph-align", [(lbl, a) for lbl, a, _ in pair_results]),
                       gc.aggregate("graph-vectors", [(lbl, v) for lbl, _, v in pair_results])]
        meta["tasks"] = [{"task": lbl, **r.to_json()} for lbl, r in sections]
        meta["file_pairs"] = entries
        header = (f"drydock-eval conform graph: {len(tasks)} task(s), {len(file_pairs)} distinct file pair(s) the "
                  f"tools would load")
        rows = [e["stats"] for e in entries if e.get("stats")]
        if len(rows) > 1:
            header += "\n" + _stats_line(rows)
    else:
        raise SystemExit("give --graph and --embeddings, or --graph-dir and --embeddings-dir")
    header += f"\n{tools_note}; {ref_note}"
    print(format_text(results, header, verbose=args.verbose))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"overall": overall(results), "meta": meta, "checks": [r.to_json() for r in results]}, fh,
                      indent=1, ensure_ascii=False, default=str)
    return 1 if overall(results) == FAIL else 0


def cmd_graph_ref(args: argparse.Namespace) -> int:
    import datetime
    from collections import Counter
    from . import graphcheck as gc
    from .. import __version__
    gs = {f[:-5] for f in os.listdir(args.graph_dir) if f.endswith(".json")}
    es = {f[:-4] for f in os.listdir(args.embeddings_dir) if f.endswith(".npz")}
    stems = sorted(gs & es)
    if not stems:
        print("no <stem>.json / <stem>.npz pairs found", file=sys.stderr)
        return 2
    pairs = [(os.path.join(args.graph_dir, s + ".json"), os.path.join(args.embeddings_dir, s + ".npz")) for s in stems]
    entries = gc.analyse_many(pairs, jobs=args.jobs, pairs=args.pairs, queries=args.queries, seed=args.seed)
    rows, groups = [], {}
    for s, e in zip(stems, entries):
        if e.get("stats"):
            rows.append(e["stats"])
            groups.setdefault(gc.repo_group(s), []).append(e["stats"])
    ref = {
        "format": gc.REFERENCE_FORMAT,
        "name": args.name,
        "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        "kit_version": __version__,
        "source": args.source,
        "method": {"pairs": args.pairs, "edge_pairs": gc.DEFAULT_EDGE_PAIRS, "queries": args.queries, "k": gc.TOP_K,
                   "seed": args.seed, "values": "[min, median, max] over files"},
        "pairs_found": len(stems),
        "unpaired": {"graph_only": len(gs - es), "embeddings_only": len(es - gs)},
        "align_status": dict(Counter(e["align"]["status"] for e in entries)),
        "pooled": gc.summarise(rows),
        "groups": {g: gc.summarise(r) for g, r in sorted(groups.items())},
    }
    if args.per_file:
        ref["per_file"] = [{"stem": s, "align": e["align"]["status"], **(e.get("stats") or {})}
                           for s, e in zip(stems, entries)]
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(ref, fh, indent=1)
    print(f"wrote {args.out}: {len(rows)} of {len(stems)} pairs with statistics; groups "
          + ", ".join(f"{g} {len(r)}" for g, r in sorted(groups.items())))
    print(_stats_line(rows))
    return 0
