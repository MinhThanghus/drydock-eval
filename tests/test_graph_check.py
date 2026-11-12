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
"""Tests for the graph/embedding fidelity check (``drydock-eval conform graph``) on synthetic arrays only."""

from __future__ import annotations

import json
import math
import os
import random
import re
import struct
import zipfile

import pytest

from drydock_eval.cli import main
from drydock_eval.conform import graphcheck as gc
from drydock_eval.conform.report import FAIL, PASS, SKIP, WARN

COMMIT = "c0ffee" * 6 + "abcd"
ALL_TOOLS = set(gc.GRAPH_TOOLS)


# --- synthetic fixtures -------------------------------------------------------------------------------------------

def write_graph(path, ids, edges, *, names=None, inline=None, edge_key="edges", extra_nodes=()):
    nodes = []
    for i in ids:
        n = {"id": i, "name": (names or {}).get(i, i), "text": f"def {i.rsplit('.', 1)[-1]}():\n    return 1"}
        if inline and i in inline:
            n["embedding"] = inline[i]
        nodes.append(n)
    nodes.extend(extra_nodes)
    data = {"directed": True, "multigraph": True, "graph": {"repo_name": "synthetic"}, "nodes": nodes,
            edge_key: [{"source": a, "target": b, "type": "calls", "key": 0} for a, b in edges]}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def chain(n, seed=0):
    """A call graph of n nodes: a chain plus a few random links, stored in both directions like the official files."""
    ids = [f"pkg.mod{i // 20}.f{i}" for i in range(n)]
    rng = random.Random(seed)
    pairs = {(i, i + 1) for i in range(n - 1)} | {tuple(sorted(rng.sample(range(n), 2))) for _ in range(n // 10)}
    edges = [(ids[a], ids[b]) for a, b in sorted(pairs)] + [(ids[b], ids[a]) for a, b in sorted(pairs)]
    return ids, edges


def gaussian(ids, dim=256, seed=1, scale=1.0):
    rng = random.Random(seed)
    return {i: [rng.gauss(0, scale) for _ in range(dim)] for i in ids}


def clustered(ids, dim=256, seed=3, size=20, offset=35.0, spread=(0.3, 2.0), jitter=0.1):
    """Official-like vectors: runs of ``size`` consecutive chain nodes share a cluster centre, so they are
    near-duplicates (as many official nodes are: about 9% of random official pairs have cosine >= 0.99 and the median
    top-10 similarity is about 0.999); the centres share an offset whose weight varies per cluster, so cosines spread
    out as the official ones do."""
    rng = random.Random(seed)
    out, centre = {}, []
    for k, i in enumerate(ids):
        if k % size == 0:
            w = rng.uniform(*spread)
            centre = [w * offset] + [rng.gauss(0, 1) for _ in range(dim - 1)]
        out[i] = [c + rng.gauss(0, jitter) for c in centre]
    return out


def collapsed(ids, dim=256, seed=2, window=3, offset=85.0):
    """Graph-consistent, collapsed vectors: a shared offset plus noise shared along the chain. The mean random-pair
    cosine and the edge AUC look official, but almost no pair reaches 0.99 and the top-10 similarities are far lower
    (the calibrated stand-ins of the drydock-eval 0.2.0 review)."""
    rng = random.Random(seed)
    noise = [[rng.gauss(0, 1) for _ in range(dim)] for _ in ids]
    out = {}
    for k, i in enumerate(ids):
        v = [offset] + [0.0] * (dim - 1)
        for j in range(max(0, k - window), min(len(ids), k + window + 1)):
            v = [a + b for a, b in zip(v, noise[j])]
        out[i] = v
    return out


def make_pair(tmp_path, ids, edges, vectors, *, stem=None, **graph_kw):
    stem = stem or f"repo_{COMMIT}"
    gdir, edir = tmp_path / "graphs", tmp_path / "embeddings"
    gdir.mkdir(exist_ok=True)
    edir.mkdir(exist_ok=True)
    gp, ep = gdir / f"{stem}.json", edir / f"{stem}.npz"
    write_graph(gp, ids, edges, **graph_kw)
    gc.write_npz(str(ep), vectors)
    return str(gp), str(ep)


def f32(values):
    return [struct.unpack("<f", struct.pack("<f", x))[0] for x in values]


def object_npy() -> bytes:
    header = "{'descr': '|O', 'fortran_order': False, 'shape': (2,), }"
    header += " " * (64 - (10 + len(header) + 1) % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode() + b"\x80\x04N."


# --- NPZ reader ---------------------------------------------------------------------------------------------------

def test_npz_roundtrip_float32_float64_and_uncompressed(tmp_path):
    vals = {"a.b": [k / 8 for k in range(-8, 8)], "c": [0.0, 1.5, -2.25]}
    for dtype, compress in (("<f4", True), ("<f8", False), (">f4", True)):
        p = tmp_path / f"x{dtype[1:]}{compress}.npz"
        gc.write_npz(str(p), vals, dtype=dtype, compress=compress)
        got = gc.read_npz(str(p))
        assert got.harness_loads and got.keys == ["a.b", "c"]
        assert got.vectors == vals
        assert set(got.shapes) == {(16,), (3,)}


def test_npz_object_entry_loses_the_whole_file(tmp_path):
    ids = ["pkg.f", "pkg.g", "pkg.h"]
    gp = tmp_path / "g.json"
    write_graph(gp, ids, [("pkg.f", "pkg.g")])
    ep = tmp_path / "e.npz"
    gc.write_npz(str(ep), {"pkg.f": [1.0, 2.0], "pkg.g": [3.0, 4.0]}, raw_members={"pkg.h.npy": object_npy()})
    npz = gc.read_npz(str(ep))
    assert not npz.harness_loads and "not numeric" in npz.entry_errors[0]
    al = gc.harness_vectors(gc.read_graph(str(gp)), npz)
    assert al.npz_lost and set(al.source) == {"zero"} and not any(any(v) for v in al.vectors)
    entry = gc.analyse_pair(str(gp), str(ep))
    assert entry["align"]["status"] == FAIL and "loses the whole file" in entry["align"]["findings"][0]
    assert entry["stats"] is None and "loses the whole NPZ" in entry["skip"]


def test_not_a_zip_and_bad_graph_fail(tmp_path):
    gp, ep = make_pair(tmp_path, ["a.b", "a.c", "a.d"], [("a.b", "a.c")], gaussian(["a.b", "a.c", "a.d"], dim=4))
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a zip" * 20)
    r = gc.analyse_pair(gp, str(bad))
    assert r["align"]["status"] == FAIL and "not a readable" in r["align"]["summary"]
    for content, needle in (({"nodes": [{"name": "x"}], "edges": []}, "no 'id'"),
                            ({"nodes": [{"id": "x"}]}, "no 'edges' or 'links'"),
                            ([1, 2], "no 'nodes'")):
        g = tmp_path / "g_bad.json"
        g.write_text(json.dumps(content))
        r = gc.analyse_pair(str(g), ep)
        assert r["align"]["status"] == FAIL and needle in r["align"]["summary"]


# --- alignment ----------------------------------------------------------------------------------------------------

def test_align_pass_when_keys_equal_ids(tmp_path):
    ids, edges = chain(60)
    gp, ep = make_pair(tmp_path, ids, edges, gaussian(ids))
    r = gc.analyse_pair(gp, ep, pairs=500, queries=10)
    assert r["align"]["status"] == PASS
    assert r["align"]["details"]["vector_source"] == {"own": 60}
    assert r["align"]["details"]["edges"] == len({frozenset(e) for e in edges}) > 59


def test_align_missing_keys_borrow_by_suffix_and_case_or_get_zero(tmp_path):
    ids = ["pkg.mod.f", "pkg.mod.g", "pkg.mod.C", "util.h"]
    vec = gaussian(["pkg.mod.f", "x.util.h", "y.z.util.h", "a.PKG.MOD.c"], dim=8)
    gp, ep = make_pair(tmp_path, ids, [("pkg.mod.f", "pkg.mod.g")], vec)
    graph, npz = gc.read_graph(gp), gc.read_npz(ep)
    al = gc.harness_vectors(graph, npz)
    assert al.source == ["own", "zero", "borrowed-ci", "borrowed-suffix"]
    assert al.vectors[3] == f32(vec["x.util.h"])       # the shortest matching key wins
    assert al.vectors[2] == f32(vec["a.PKG.MOD.c"])
    assert al.vectors[1] == [0.0] * 8
    res = gc.check_align(graph, npz, al)
    assert res.status == FAIL
    assert "3 of 4 nodes have no NPZ key of their own: 1 borrow another node's vector by suffix, 1 by " \
           "case-insensitive match, 0 through a node text, 1 get a zero vector" in res.findings[0]


def test_align_mixed_lengths_fail_and_follow_the_first_node(tmp_path):
    ids = ["a.f", "a.g", "a.h"]
    vec = {"a.f": [1.0, 2.0, 3.0, 4.0], "a.g": [1.0] * 8, "a.h": [2.0] * 8}
    gp, ep = make_pair(tmp_path, ids, [("a.f", "a.g")], vec)
    al = gc.harness_vectors(gc.read_graph(gp), gc.read_npz(ep))
    assert al.adjusted == 2 and {len(v) for v in al.vectors} == {4}   # truncated to the first node's length
    r = gc.analyse_pair(gp, ep)
    assert r["align"]["status"] == FAIL and any("different lengths" in f for f in r["align"]["findings"])


def test_align_warnings_for_format_differences(tmp_path):
    ids = ["a.f", "a.g", "a.h", "a.k"]
    vec = {f"repo;{i}": [1.0 + k, 2.0, 0.5, 1.0] for k, i in enumerate(ids)}
    vec["a.h"] = [0.0, 0.0, 0.0, 0.0]      # a zero vector under its own key wins over the prefixed one
    vec["unused.key"] = [1.0, 1.0, 1.0, 1.0]
    gp, ep = make_pair(tmp_path, ids, [("a.f", "a.g"), ("a.g", "a.ghost")], vec, names={"a.k": "renamed"},
                       inline={"a.g": [9.0, 9.0, 9.0, 9.0]}, edge_key="links",
                       extra_nodes=[{"id": "a.f", "name": "a.f", "text": "dup"}])
    graph, npz = gc.read_graph(gp), gc.read_npz(ep)
    assert graph.duplicate_ids == 1 and graph.implicit_nodes == 1 and graph.edge_key == "links"
    al = gc.harness_vectors(graph, npz)
    assert al.source[:4] == ["own-prefixed", "inline", "own", "own-prefixed"]
    res = gc.check_align(graph, npz, al)
    text = " | ".join(res.findings)
    # the implicit node 'a.ghost' has no key: FAIL, and every format difference is listed
    assert res.status == FAIL and "1 of 5 nodes have no NPZ key" in text
    for needle in ("match no graph node", "'<repo>;' prefix", "vector length [4]", "are all zero",
                   "inline 'embedding'", "'name' different", "duplicate node ids", "not declared as nodes"):
        assert needle in text, needle


def test_align_warn_only_when_every_node_has_its_own_vector(tmp_path):
    ids = ["a.f", "a.g", "a.h"]
    gp, ep = make_pair(tmp_path, ids, [("a.f", "a.g")], {**gaussian(ids, dim=16), "extra.x": [1.0] * 16})
    r = gc.analyse_pair(gp, ep, pairs=100, queries=3)
    assert r["align"]["status"] == WARN
    assert any("vector length [16]" in f for f in r["align"]["findings"])


# --- prompt section -----------------------------------------------------------------------------------------------

def _layout(tmp_path, files):
    root = tmp_path / "layout"
    for rel, size in files.items():
        p = root / rel.replace("<C>", COMMIT)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * size)
    (root / "graphs").mkdir(parents=True, exist_ok=True)
    (root / "embeddings").mkdir(parents=True, exist_ok=True)
    return gc.resolve(str(root / "graphs"), str(root / "embeddings"), "owner/repo", COMMIT)


@pytest.mark.parametrize("gsize,expect", [(101, True), (100, False)])
def test_section_needs_files_larger_than_100_bytes(tmp_path, gsize, expect):
    res = _layout(tmp_path, {"graphs/repo_<C>.json": gsize, "embeddings/repo_<C>.npz": 500})
    assert (res["graph"]["prompt_ok"] and res["embeddings"]["prompt_ok"]) is expect
    r = gc.check_section(res, {"search_similar_code"})
    assert r.status == (PASS if expect else FAIL)


def test_section_depends_on_declared_graph_tools(tmp_path):
    res = _layout(tmp_path, {})
    assert gc.check_section(res, {"read_file", "run_command"}).status == SKIP
    assert gc.check_section(res, {"get_code_subgraph"}).status == FAIL
    unknown = gc.check_section(res, None)
    assert unknown.status == WARN and "NOT added" in unknown.summary


def test_section_earned_by_one_file_but_tools_load_an_empty_one(tmp_path):
    res = _layout(tmp_path, {"graphs/repo_<C>.json": 0, "graphs/owner_repo.json": 500, "embeddings/repo_<C>.npz": 500})
    r = gc.check_section(res, ALL_TOOLS)
    assert r.status == FAIL and "empty file" in r.summary
    assert res["graph"]["prompt_file"].endswith("owner_repo.json")
    assert res["graph"]["tool_file"].endswith(f"repo_{COMMIT}.json")


def test_section_and_tools_read_different_valid_files(tmp_path):
    res = _layout(tmp_path, {"graphs/owner_repo.json": 500, f"graphs/owner/repo_<C>.json": 500,
                             "embeddings/repo_<C>.npz": 500})
    r = gc.check_section(res, ALL_TOOLS)
    assert r.status == WARN and "different file" in r.summary


def test_section_missing_when_only_nested_names_exist(tmp_path):
    res = _layout(tmp_path, {"graphs/owner/repo_<C>.json": 500, "embeddings/owner/repo_<C>.npz": 500})
    r = gc.check_section(res, ALL_TOOLS)
    assert r.status == FAIL and any("would still load" in f for f in r.findings)


def test_section_direct_mode_checks_sizes_only(tmp_path):
    g, e = tmp_path / "g.json", tmp_path / "e.npz"
    g.write_bytes(b"x" * 300)
    e.write_bytes(b"x" * 50)
    r = gc.check_section(None, None, direct=(str(g), str(e)))
    assert r.status == WARN and "embeddings file is 50 bytes" in r.findings[0]


# --- vector statistics --------------------------------------------------------------------------------------------

def test_auc_and_dot_helpers():
    assert gc._auc([2.0, 3.0], [0.0, 1.0]) == 1.0
    assert gc._auc([1.0], [1.0]) == 0.5
    assert gc._auc([0.0], [1.0]) == 0.0
    a, b = [0.5, -1.25, 3.0], [2.0, 4.0, -0.5]
    assert math.isclose(gc._dot(a, b), gc._dot_py(a, b))


def test_gaussian_and_calibrated_stand_ins_warn_and_clustered_official_like_vectors_pass(tmp_path):
    ref = gc.load_reference("kaggle-dev-2026")
    ids, edges = chain(200)
    g_dir, c_dir, k_dir = tmp_path / "g", tmp_path / "c", tmp_path / "k"
    for d in (g_dir, c_dir, k_dir):
        d.mkdir()
    gp, ep = make_pair(g_dir, ids, edges, gaussian(ids))
    s = gc.analyse_pair(gp, ep, pairs=4000, queries=40)["stats"]
    assert abs(s["rand_cos_mean"]) < 0.02 and 0.4 < s["edge_auc"] < 0.6
    res = gc.check_vectors(s, ref)
    assert res.status == WARN
    assert {"rand_cos_mean", "rand_cos_ge_099", "top10_sim_median", "edge_auc"} <= set(res.details["outside"])
    assert "edge AUC" in " ".join(res.findings) and res.findings[-1] == gc.PASS_MEANING
    # a calibrated stand-in: official mean cosine and edge AUC, but no near-duplicates (passed drydock-eval 0.2.0)
    gp2, ep2 = make_pair(c_dir, ids, edges, collapsed(ids))
    s2 = gc.analyse_pair(gp2, ep2, pairs=4000, queries=40)["stats"]
    assert 0.7 < s2["rand_cos_mean"] < 0.9 and s2["edge_auc"] > 0.85
    res2 = gc.check_vectors(s2, ref)
    assert res2.status == WARN and {"rand_cos_ge_099", "top10_sim_median"} <= set(res2.details["outside"])
    gp3, ep3 = make_pair(k_dir, ids, edges, clustered(ids))
    s3 = gc.analyse_pair(gp3, ep3, pairs=4000, queries=40)["stats"]
    res3 = gc.check_vectors(s3, ref)
    assert res3.status == PASS, res3.findings
    assert "does not show" in res3.summary and res3.findings == [gc.PASS_MEANING]
    assert gc.check_vectors(s3, None).status == SKIP


def test_warn_band_is_two_sided():
    ref = gc.load_reference("kaggle-dev-2026")
    band = gc.warn_band(ref)
    p = ref["pooled"]
    assert band["rand_cos_sd"] == [round(p["rand_cos_sd"][0] / 2, 4), round(p["rand_cos_sd"][2] * 2, 4)]
    assert band["rand_cos_ge_099"][0] == round(p["rand_cos_ge_099"][0] / 2, 4)
    assert band["top10_sim_median"][0] < p["top10_sim_median"][0] <= p["top10_sim_median"][2] < band["top10_sim_median"][1]
    assert band["edge_auc"][0] == band["edge_auc_min"] and p["edge_auc"][2] < band["edge_auc"][1] < 1.0
    base = {"rand_cos_mean": 0.8, "rand_cos_sd": 0.15, "rand_cos_ge_099": 0.1, "edge_auc": 0.9,
            "top10_sim_median": 0.999}
    assert gc.check_vectors(base, ref).status == PASS
    for key, value in (("edge_auc", 0.99), ("rand_cos_sd", 0.01), ("rand_cos_ge_099", 0.0),
                       ("top10_sim_median", 0.95), ("top10_sim_median", 1.0)):
        res = gc.check_vectors({**base, key: value}, ref)
        assert res.status == WARN and res.details["outside"] == [key], (key, value)


def test_vector_stats_are_deterministic_and_against_measures_overlap(tmp_path):
    ids, edges = chain(80)
    vec = collapsed(ids)
    gp, ep = make_pair(tmp_path, ids, edges, vec)
    other = tmp_path / "other.npz"
    gc.write_npz(str(other), gaussian(ids))
    a = gc.analyse_pair(gp, ep, pairs=1000, queries=20, against=ep)
    b = gc.analyse_pair(gp, ep, pairs=1000, queries=20, against=ep)
    assert a["stats"] == b["stats"] and a["stats"]["overlap_at_10_vs_against"] == 1.0
    c = gc.analyse_pair(gp, ep, pairs=1000, queries=20, against=str(other))
    assert c["stats"]["overlap_at_10_vs_against"] < 0.3
    d = gc.analyse_pair(gp, ep, pairs=1000, queries=20, seed=5)
    assert d["stats"]["rand_cos_mean"] != a["stats"]["rand_cos_mean"]


def test_top_k_ties_break_by_name():
    unit = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    names = ["q", "b", "a", "z"]
    assert [j for j, _ in gc._top_k(unit, names, 0, 2)] == [2, 1]


# --- reference ----------------------------------------------------------------------------------------------------

def test_builtin_reference_holds_numbers_only():
    ref = gc.load_reference("kaggle-dev-2026")
    assert ref["pooled"]["files"] == 127 and set(ref["groups"]) == {"fastapi", "httpx", "requests", "rich"}
    lo, _, hi = ref["pooled"]["rand_cos_mean"]
    assert 0.7 < lo < hi < 0.95
    assert gc.warn_band(ref)["rand_cos_mean"] == [round(lo - gc.COS_MARGIN, 4), round(min(1.0, hi + gc.COS_MARGIN), 4)]
    path = os.path.join(os.path.dirname(gc.__file__), "data", gc.BUILTIN_REFERENCES["kaggle-dev-2026"])
    text = open(path, encoding="utf-8").read()
    assert not re.search(r"[0-9a-f]{40}", text) and "per_file" not in text and "Apache-2.0" in text
    assert gc.load_reference("none") is None


def test_tool_names_from_list_none_and_reference(tmp_path):
    assert gc.tool_names_from(None) is None
    assert gc.tool_names_from("none") == set()
    assert gc.tool_names_from("read_file, search_similar_code") == {"read_file", "search_similar_code"}
    p = tmp_path / "ref.json"
    p.write_text(json.dumps({"tools": {"tools": [{"name": "get_code_neighbors", "sha256": "x"}]}}))
    assert gc.tool_names_from(str(p)) == {"get_code_neighbors"}


# --- CLI ----------------------------------------------------------------------------------------------------------

def test_cli_graph_direct_mode_exit_codes(tmp_path, capsys):
    ids, edges = chain(50)
    gp, ep = make_pair(tmp_path, ids, edges, gaussian(ids))
    out = tmp_path / "r.json"
    assert main(["conform", "graph", "--graph", gp, "--embeddings", ep, "--pairs", "500", "--queries", "5",
                 "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "[WARN] graph-vectors" in text and "stand-in vectors" in text
    report = json.loads(out.read_text())
    assert report["overall"] == WARN and [c["name"] for c in report["checks"]] == \
        ["graph-section", "graph-align", "graph-vectors"]
    bad = tmp_path / "bad.npz"
    gc.write_npz(str(bad), {"other.key": [1.0] * 256})
    assert main(["conform", "graph", "--graph", gp, "--embeddings", str(bad), "--pairs", "100", "--queries", "2"]) == 1


def test_cli_graph_tasks_mode_and_graph_ref(tmp_path, capsys):
    ids, edges = chain(200)
    root = tmp_path / "data"
    root.mkdir()
    make_pair(root, ids, edges, clustered(ids), stem=f"repo_{COMMIT}")
    make_pair(root, ids, edges, gaussian(ids), stem=f"other_{'1' * 40}")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("\n".join(json.dumps(t) for t in (
        {"instance_id": "t1", "repo": "owner/repo", "base_commit": COMMIT},
        {"instance_id": "t2", "repo": "owner/other", "base_commit": "1" * 40},
        {"instance_id": "t3", "repo": "owner/missing", "base_commit": "2" * 40})) + "\n")
    out = tmp_path / "tasks.json"
    rc = main(["conform", "graph", "--graph-dir", str(root / "graphs"), "--embeddings-dir", str(root / "embeddings"),
               "--tasks", str(tasks), "--tools", "search_similar_code", "--pairs", "500", "--queries", "5",
               "--json", str(out)])
    assert rc == 1                                   # t3 has no files: the section is missing
    report = json.loads(out.read_text())
    section, align, vectors = report["checks"]
    assert section["details"]["counts"] == {"PASS": 2, "FAIL": 1} and "t3" in section["findings"][0]
    assert align["details"]["counts"] == {"PASS": 2}
    assert vectors["details"]["counts"] == {"PASS": 1, "WARN": 1}
    assert len(report["meta"]["file_pairs"]) == 2
    ref_out = tmp_path / "myref.json"
    assert main(["conform", "graph-ref", "--graph-dir", str(root / "graphs"), "--embeddings-dir",
                 str(root / "embeddings"), "--out", str(ref_out), "--pairs", "500", "--queries", "5"]) == 0
    ref = json.loads(ref_out.read_text())
    assert ref["format"] == gc.REFERENCE_FORMAT and ref["pooled"]["files"] == 2 and "per_file" not in ref
    assert set(ref["groups"]) == {"repo", "other"}
    assert gc.load_reference(str(ref_out))["name"] == "local"
