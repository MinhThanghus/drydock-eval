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
"""Graph and embedding fidelity: do a task's code-graph JSON and embedding NPZ make the competition harness's graph
tools behave as they do on the Kaggle scorer?

The behaviour modelled here was read from the competition harness wheel 0.2.11 and is re-implemented, not copied
(file:line in the README). In short:

- **Prompt section.** The first user message gets a "Code Intelligence Tools" section only when at least one graph
  tool is declared (or the agent declares no tool list) and a graph file and an embedding file each exist under one
  of four names and are larger than 100 bytes.
- **Files the tools load.** The tools pick the first of six names that exists, with no size test.
- **Per-node vector.** An inline ``embedding`` in the graph JSON wins. Otherwise the node id is looked up in the NPZ:
  exact key, then (for names without whitespace) the shortest key ending in ``.<id>``, ``;<id>`` or ``/<id>``, then a
  case-insensitive match. A node with no match gets a zero vector. If any NPZ entry cannot be loaded with
  ``allow_pickle=False``, the whole file is lost and every node gets a zero vector.
- **search_similar_code.** A query without whitespace is resolved to a node and every other node is ranked by cosine
  (ties by name). The cosine is printed to the agent as ``similarity``.

Three checks come out of this: ``graph-section`` (prompt-section conditions and file resolution), ``graph-align``
(node-key/vector alignment) and ``graph-vectors`` (vector statistics against reference numbers measured on the
official files, which are never shipped). Standard library only: NPZ files are read with ``zipfile`` and ``struct``.
"""

from __future__ import annotations

import ast
import heapq
import io
import json
import math
import operator
import os
import random
import re
import struct
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from .report import FAIL, PASS, SKIP, WARN, CheckResult

GRAPH_TOOLS = ("search_similar_code", "get_code_neighbors", "get_code_subgraph")
SIZE_LIMIT = 100          # a file counts for the prompt section only when it is LARGER than this many bytes
OFFICIAL_DIM = 256        # vector length of the official files
TOP_K = 10                # search_similar_code's default k
DEFAULT_PAIRS = 20000
DEFAULT_EDGE_PAIRS = 5000
DEFAULT_QUERIES = 200
DEFAULT_SEED = 0
# WARN bands for graph-vectors: each statistic must lie inside the official per-file range [min, max] widened as
# follows (two-sided; see warn_band). Margins were set so that every official file passes when the band is built from
# the other three dev repositories only (leave-one-repository-out); they are not tuned on stand-ins.
COS_MARGIN = 0.10         # random-pair cosine mean: [min - 0.10, max + 0.10]
AUC_MARGIN = 0.05         # edge AUC, lower side: min - 0.05
SCALE_FACTOR = 2.0        # random-pair cosine sd and share of pairs >= 0.99: [min / 2, max * 2]
TOP10_FACTOR = 3.0        # median top-10 similarity, as distance from 1: [(1 - max) / 3, (1 - min) * 3]
AUC_HI_FACTOR = 2.0       # edge AUC, upper side, as distance from 1: 1 - (1 - max) / 2
MIN_EDGES_FOR_AUC = 20
REFERENCE_FORMAT = "drydock-eval graph reference 1"
BUILTIN_REFERENCES = {"kaggle-dev-2026": "graph_reference_kaggle_dev_2026.json"}

_ws = re.compile(r"\s")


def _dot_py(a, b) -> float:
    return sum(map(operator.mul, a, b))


_dot = getattr(math, "sumprod", None) or _dot_py


# --- NPZ reading without numpy ------------------------------------------------------------------------------------

_NPY_CODES = {"f2": "e", "f4": "f", "f8": "d", "i1": "b", "i2": "h", "i4": "i", "i8": "q",
              "u1": "B", "u2": "H", "u4": "I", "u8": "Q", "b1": "?"}


class NpyError(ValueError):
    pass


def parse_npy(raw: bytes) -> tuple[tuple[int, ...], str, list[float]]:
    """Parse one .npy array: (shape, dtype string, flattened values as floats). Object, string and structured
    dtypes are refused, as numpy refuses them with allow_pickle=False or a float32 cast does."""
    if raw[:6] != b"\x93NUMPY":
        raise NpyError("not an .npy array")
    major = raw[6]
    if major == 1:
        (hlen,), start = struct.unpack("<H", raw[8:10]), 10
    elif major in (2, 3):
        (hlen,), start = struct.unpack("<I", raw[8:12]), 12
    else:
        raise NpyError(f"unknown .npy format version {major}")
    try:
        header = ast.literal_eval(raw[start:start + hlen].decode("utf-8" if major == 3 else "latin-1"))
        descr, shape = header["descr"], tuple(int(x) for x in header["shape"])
    except Exception as exc:  # noqa: BLE001
        raise NpyError(f"bad .npy header: {exc}") from exc
    if not isinstance(descr, str):
        raise NpyError("structured dtype")
    order, code = descr[0], descr[1:]
    fmt = _NPY_CODES.get(code)
    if fmt is None:
        raise NpyError(f"dtype {descr} is not numeric (needs pickle or cannot be cast to float32)")
    count = 1
    for d in shape:
        count *= d
    size = struct.calcsize(fmt)
    body = raw[start + hlen:]
    if len(body) < count * size:
        raise NpyError("truncated array data")
    endian = ">" if order == ">" else "<"
    values = struct.unpack(f"{endian}{count}{fmt}", body[:count * size]) if count else ()
    return shape, descr, [float(v) for v in values]


def _npy_bytes(values: list[float], dtype: str = "<f4", shape: tuple[int, ...] | None = None) -> bytes:
    shape = (len(values),) if shape is None else shape
    header = "{'descr': '%s', 'fortran_order': False, 'shape': %s, }" % (dtype, repr(tuple(shape)))
    pad = 64 - (10 + len(header) + 1) % 64
    header = header + " " * pad + "\n"
    fmt = _NPY_CODES[dtype[1:]]
    return (b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin-1")
            + struct.pack(f"{dtype[0] if dtype[0] in '<>' else '<'}{len(values)}{fmt}", *values))


def write_npz(path: str, arrays: dict[str, list[float]], dtype: str = "<f4", compress: bool = True,
              raw_members: dict[str, bytes] | None = None) -> None:
    """Write an NPZ archive (one 1-D array per key) without numpy; ``raw_members`` adds arbitrary zip members.
    Used by the tests and for synthetic fixtures."""
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", mode) as zf:
        for key, vals in arrays.items():
            zf.writestr(key + ".npy", _npy_bytes(list(vals), dtype))
        for name, data in (raw_members or {}).items():
            zf.writestr(name, data)


@dataclass
class NpzData:
    path: str
    size: int
    keys: list[str] = field(default_factory=list)
    vectors: dict[str, list[float]] = field(default_factory=dict)
    shapes: Counter = field(default_factory=Counter)
    dtypes: Counter = field(default_factory=Counter)
    entry_errors: list[str] = field(default_factory=list)   # entries numpy (allow_pickle=False) + float32 cannot load
    fatal: str | None = None                               # not a readable zip at all

    @property
    def harness_loads(self) -> bool:
        """The harness loads the archive in one go; a single bad entry loses every vector."""
        return self.fatal is None and not self.entry_errors


def read_npz(path: str) -> NpzData:
    out = NpzData(path=path, size=os.path.getsize(path) if os.path.exists(path) else -1)
    try:
        with open(path, "rb") as fh:
            blob = fh.read()
        zf = zipfile.ZipFile(io.BytesIO(blob))
    except Exception as exc:  # noqa: BLE001
        out.fatal = f"not a readable NPZ/zip archive: {type(exc).__name__}: {exc}"
        return out
    with zf:
        for name in zf.namelist():
            key = name[:-4] if name.endswith(".npy") else name
            out.keys.append(key)
            try:
                raw = zf.read(name)
                shape, descr, vals = parse_npy(raw)
            except Exception as exc:  # noqa: BLE001
                out.entry_errors.append(f"{key!r}: {exc}")
                continue
            out.shapes[shape] += 1
            out.dtypes[descr] += 1
            out.vectors[key] = vals  # flattened; a non-1-D shape is reported by the align check
    return out


# --- graph JSON (networkx node-link) ------------------------------------------------------------------------------

@dataclass
class GraphData:
    path: str
    size: int
    ids: list[str] = field(default_factory=list)       # str(node id), in networkx node order
    names: list[str] = field(default_factory=list)     # what search_similar_code prints and ranks ties by
    texts: list[str] = field(default_factory=list)
    inline: dict[int, list[float]] = field(default_factory=dict)
    raw_nodes: list[dict] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)   # distinct undirected non-loop pairs
    edge_records: int = 0
    self_loops: int = 0
    edge_types: Counter = field(default_factory=Counter)
    duplicate_ids: int = 0
    implicit_nodes: int = 0                            # edge endpoints never declared as nodes (networkx adds them)
    edge_key: str = ""
    error: str | None = None


def _node_key(v: Any) -> Any:
    return tuple(_node_key(x) for x in v) if isinstance(v, list) else v


def read_graph(path: str) -> GraphData:
    g = GraphData(path=path, size=os.path.getsize(path) if os.path.exists(path) else -1)
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        g.error = f"not readable JSON: {type(exc).__name__}: {exc}"
        return g
    if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
        g.error = "no 'nodes' list (not networkx node-link JSON)"
        return g
    g.edge_key = "edges" if "edges" in data else ("links" if "links" in data else "")
    if not g.edge_key or not isinstance(data.get(g.edge_key), list):
        g.error = "no 'edges' or 'links' list: networkx node_link_graph fails, so the graph tools fail"
        return g
    index: dict[Any, int] = {}
    attrs: list[dict] = []
    for obj in data["nodes"]:
        if not isinstance(obj, dict) or "id" not in obj:
            g.error = "a node has no 'id': networkx node_link_graph fails, so the graph tools fail"
            return g
        g.raw_nodes.append(obj)
        nid = _node_key(obj["id"])
        rest = {k: v for k, v in obj.items() if k != "id"}
        if nid in index:
            g.duplicate_ids += 1
            attrs[index[nid]].update(rest)
        else:
            index[nid] = len(attrs)
            attrs.append(dict(rest))
            g.ids.append(str(nid))
    pairs: set[tuple[int, int]] = set()
    for e in data[g.edge_key]:
        if not isinstance(e, dict) or "source" not in e or "target" not in e:
            g.error = "an edge has no 'source' or 'target': networkx node_link_graph fails"
            return g
        g.edge_records += 1
        g.edge_types[str(e.get("type"))] += 1
        ends = []
        for end in (_node_key(e["source"]), _node_key(e["target"])):
            if end not in index:
                index[end] = len(attrs)
                attrs.append({})
                g.ids.append(str(end))
                g.implicit_nodes += 1
            ends.append(index[end])
        s, t = ends
        if s == t:
            g.self_loops += 1
        else:
            pairs.add((min(s, t), max(s, t)))
    g.edges = sorted(pairs)
    for i, a in enumerate(attrs):
        g.names.append(str(a["name"]) if "name" in a else g.ids[i])
        text = a.get("text")
        g.texts.append(text if isinstance(text, str) else "")
        emb = a.get("embedding")
        if isinstance(emb, list) and emb:
            flat = _flatten(emb)
            if flat:
                g.inline[i] = flat
    return g


def _flatten(v: Any) -> list[float]:
    out: list[float] = []
    stack = [v]
    while stack:
        x = stack.pop()
        if isinstance(x, list):
            stack.extend(reversed(x))
        else:
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                return []
    return out


# --- the harness's per-node vector ----------------------------------------------------------------------------------

@dataclass
class Alignment:
    vectors: list[list[float]]          # effective vector per graph node, after the harness's length adjustment
    source: list[str]                   # inline | own | own-prefixed | borrowed-suffix | borrowed-ci | borrowed-text | zero
    borrowed_from: dict[int, str]
    target_dim: int
    adjusted: int                       # vectors padded or truncated to the first node's length
    used_keys: set[str]
    npz_lost: bool


def harness_vectors(graph: GraphData, npz: NpzData | None) -> Alignment:
    """The vector each graph node gets in the harness's graph tools (see the module docstring)."""
    cache: dict[str, list[float]] = {}
    origin: dict[str, tuple[str, str]] = {}
    lost = npz is None or not npz.harness_loads
    if not lost:
        for k in npz.keys:
            v = npz.vectors[k]
            cache[k] = v
            origin[k] = ("key", k)
            if ";" in k:
                bare = k.split(";", 1)[1]
                if bare not in cache:
                    cache[bare] = v
                    origin[bare] = ("bare", k)
        for obj in graph.raw_nodes:    # node texts are indexed too (keyed by name, else id)
            nid = str(obj.get("name") or obj.get("id") or "")
            text = obj.get("text") or obj.get("code")
            if nid and text and isinstance(text, str) and nid in cache:
                cache[text] = cache[nid]
                origin[text] = ("text", origin[nid][1])
    symbol_keys = [k for k in cache if not _ws.search(k)]

    def lookup(q: str) -> tuple[list[float] | None, str, str]:
        if q in cache:
            kind, key = origin[q]
            if kind == "text":
                return cache[q], "borrowed-text", key
            return cache[q], ("own" if kind == "key" else "own-prefixed"), key
        if cache and not _ws.search(q):
            bare = q.split(";", 1)[1] if ";" in q else q
            hits = [k for k in symbol_keys if k.endswith("." + bare) or k.endswith(";" + bare) or k.endswith("/" + bare)]
            if hits:
                best = min(hits, key=lambda x: (len(x.split(".")), len(x), x))
                return cache[best], "borrowed-suffix", origin[best][1]
            low = bare.lower()
            hits = [k for k in symbol_keys if k.lower() == low or k.lower().endswith("." + low)
                    or k.lower().endswith(";" + low) or k.lower().endswith("/" + low)]
            if hits:
                best = min(hits, key=lambda x: (len(x.split(".")), len(x), x))
                return cache[best], "borrowed-ci", origin[best][1]
        return None, "zero", ""

    found: list[tuple[list[float] | None, str, str]] = []
    for i, nid in enumerate(graph.ids):
        if i in graph.inline:
            found.append((graph.inline[i], "inline", ""))
        else:
            found.append(lookup(nid))
    target_dim = next((len(graph.inline[i]) for i in sorted(graph.inline)), None)
    if target_dim is None:
        target_dim = next((len(v) for v, _, _ in found if v), OFFICIAL_DIM)
    raw = [v if v else [0.0] * target_dim for v, _, _ in found]
    detected = len(raw[0]) if raw else target_dim
    adjusted = 0
    vectors = []
    for v in raw:
        if len(v) != detected:
            adjusted += 1
            v = (v + [0.0] * detected)[:detected]
        vectors.append(v)
    source = [s for _, s, _ in found]
    borrowed = {i: k for i, (_, s, k) in enumerate(found) if s.startswith("borrowed")}
    used = {k for _, s, k in found if k}
    return Alignment(vectors, source, borrowed, target_dim, adjusted, used, lost)


# --- vector statistics --------------------------------------------------------------------------------------------

def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _auc(pos: list[float], neg: list[float]) -> float:
    """P(pos > neg) + 0.5 P(pos == neg) (Mann-Whitney), ties after rounding to 1e-9."""
    allv = sorted([(round(v, 9), 1) for v in pos] + [(round(v, 9), 0) for v in neg])
    rank_sum = 0.0
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        rank_sum += avg * sum(1 for x in range(i, j + 1) if allv[x][1] == 1)
        i = j + 1
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def _unit(vectors: list[list[float]]) -> tuple[list[list[float]], list[float]]:
    norms = [math.sqrt(_dot(v, v)) for v in vectors]
    unit = [[x / n for x in v] if n > 0 else [0.0] * len(v) for v, n in zip(vectors, norms)]
    return unit, norms


def _top_k(unit: list[list[float]], names: list[str], q: int, k: int) -> list[tuple[int, float]]:
    uq = unit[q]
    sims = [_dot(uq, u) for u in unit]
    n = len(unit)
    top = heapq.nsmallest(k, (j for j in range(n) if j != q), key=lambda j: (-sims[j], names[j]))
    return [(j, sims[j]) for j in top]


def vector_stats(vectors: list[list[float]], names: list[str], edges: list[tuple[int, int]], *,
                 pairs: int = DEFAULT_PAIRS, edge_pairs: int = DEFAULT_EDGE_PAIRS, queries: int = DEFAULT_QUERIES,
                 seed: int = DEFAULT_SEED, against: list[list[float]] | None = None) -> dict:
    """Statistics of the vectors search_similar_code ranks. Deterministic for a given seed."""
    n = len(vectors)
    unit, norms = _unit(vectors)
    s_norms = sorted(norms)
    st: dict[str, Any] = {
        "nodes": n, "dim": len(vectors[0]) if vectors else 0,
        "norm_min": s_norms[0], "norm_median": _quantile(s_norms, 0.5), "norm_max": s_norms[-1],
        "zero_norm": sum(1 for x in norms if x == 0),
        "method": {"pairs": pairs, "edge_pairs": edge_pairs, "queries": queries, "k": TOP_K, "seed": seed},
    }
    counts = Counter(tuple(v) for v in vectors)
    st["shared_vector_share"] = sum(c for c in counts.values() if c > 1) / n
    rng = random.Random(seed)
    rand = []
    for _ in range(pairs):
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        rand.append(_dot(unit[i], unit[j]))
    sr = sorted(rand)
    mean = sum(rand) / len(rand)
    st.update({
        "rand_cos_mean": mean,
        "rand_cos_sd": math.sqrt(sum((x - mean) ** 2 for x in rand) / len(rand)),
        "rand_cos_p05": _quantile(sr, 0.05), "rand_cos_p50": _quantile(sr, 0.5), "rand_cos_p95": _quantile(sr, 0.95),
        "rand_cos_ge_099": sum(1 for x in rand if x >= 0.99) / len(rand),
    })
    st["edges"] = len(edges)
    if len(edges) >= MIN_EDGES_FOR_AUC:
        sample = edges if len(edges) <= edge_pairs else random.Random(seed + 1).sample(edges, edge_pairs)
        ecos = [_dot(unit[a], unit[b]) for a, b in sample]
        st["edge_cos_mean"] = sum(ecos) / len(ecos)
        st["edge_auc"] = _auc(ecos, rand)
    else:
        st["edge_cos_mean"] = None
        st["edge_auc"] = None
    adj: list[set[int]] = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    k = min(TOP_K, n - 1)
    qs = random.Random(seed + 2).sample(range(n), min(queries, n))
    top_means, top1, nb_share, chance, overlap = [], [], [], [], []
    a_unit = _unit(against)[0] if against is not None else None
    for q in qs:
        top = _top_k(unit, names, q, k)
        sims = [s for _, s in top]
        top_means.append(sum(sims) / len(sims))
        top1.append(sims[0])
        if adj[q]:
            nb_share.append(sum(1 for j, _ in top if j in adj[q]) / len(top))
            chance.append(len(adj[q]) / (n - 1))
        if a_unit is not None:
            other = {j for j, _ in _top_k(a_unit, names, q, k)}
            overlap.append(len(other & {j for j, _ in top}) / k)
    st["top10_sim_median"] = _quantile(sorted(top_means), 0.5)
    st["top1_sim_median"] = _quantile(sorted(top1), 0.5)
    st["top10_neighbour_share"] = sum(nb_share) / len(nb_share) if nb_share else None
    st["top10_neighbour_chance"] = sum(chance) / len(chance) if chance else None
    st["queries_with_neighbours"] = len(nb_share)
    if a_unit is not None:
        st["overlap_at_10_vs_against"] = sum(overlap) / len(overlap) if overlap else None
    return st


# --- reference numbers --------------------------------------------------------------------------------------------

REF_STATS = ("nodes", "norm_min", "norm_median", "norm_max", "rand_cos_mean", "rand_cos_sd", "rand_cos_p05",
             "rand_cos_p50", "rand_cos_p95", "rand_cos_ge_099", "edge_auc", "top10_sim_median", "top1_sim_median",
             "top10_neighbour_share", "top10_neighbour_chance", "shared_vector_share")


def repo_group(stem: str) -> str:
    """'<repo_short>_<40-hex commit>' -> '<repo_short>'."""
    m = re.match(r"^(.*)_[0-9a-f]{40}$", stem)
    return m.group(1) if m else stem


def summarise(rows: list[dict]) -> dict:
    """[min, median, max] of each statistic over files."""
    out: dict[str, Any] = {"files": len(rows)}
    for key in REF_STATS:
        vals = sorted(r[key] for r in rows if r.get(key) is not None)
        if vals:
            out[key] = [round(vals[0], 4), round(_quantile(vals, 0.5), 4), round(vals[-1], 4)]
    return out


def load_reference(spec: str | None) -> dict | None:
    if spec in (None, "", "none"):
        return None
    if spec in BUILTIN_REFERENCES:
        path = os.path.join(os.path.dirname(__file__), "data", BUILTIN_REFERENCES[spec])
    else:
        path = spec
    with open(path, encoding="utf-8") as fh:
        ref = json.load(fh)
    if ref.get("format") != REFERENCE_FORMAT:
        raise ValueError(f"{path}: not a {REFERENCE_FORMAT!r} file")
    return ref


def warn_band(reference: dict) -> dict:
    """Two-sided bands [low, high] per statistic, from the reference's per-file [min, median, max]."""
    pooled = reference["pooled"]
    lo, _, hi = pooled["rand_cos_mean"]
    band = {"rand_cos_mean": [round(lo - COS_MARGIN, 4), round(min(1.0, hi + COS_MARGIN), 4)]}
    if "rand_cos_sd" in pooled:
        lo, _, hi = pooled["rand_cos_sd"]
        band["rand_cos_sd"] = [round(lo / SCALE_FACTOR, 4), round(hi * SCALE_FACTOR, 4)]
    if "rand_cos_ge_099" in pooled:
        lo, _, hi = pooled["rand_cos_ge_099"]
        band["rand_cos_ge_099"] = [round(lo / SCALE_FACTOR, 4), round(min(1.0, hi * SCALE_FACTOR), 4)]
    if "top10_sim_median" in pooled:
        lo, _, hi = pooled["top10_sim_median"]
        band["top10_sim_median"] = [round(1 - (1 - lo) * TOP10_FACTOR, 4), round(1 - (1 - hi) / TOP10_FACTOR, 4)]
    if "edge_auc" in pooled:
        lo, _, hi = pooled["edge_auc"]
        band["edge_auc"] = [round(lo - AUC_MARGIN, 4), round(1 - (1 - hi) / AUC_HI_FACTOR, 4)]
        band["edge_auc_min"] = band["edge_auc"][0]  # drydock-eval 0.2.0 name
    return band


# What each band protects, for the WARN text: (label, what a value outside the band means for the agent).
_BAND_TEXT = {
    "rand_cos_mean": ("random-pair cosine mean", "unrelated nodes look {dir} similar than on the scorer, so the "
                                                 "similarities search_similar_code prints are on another scale"),
    "rand_cos_sd": ("random-pair cosine sd", "similarities are {dir} spread out than on the scorer, so the gaps "
                                             "between ranked neighbours differ"),
    "rand_cos_ge_099": ("share of random pairs with cosine >= 0.99", "{dir} unrelated nodes look near-identical than "
                                                                     "on the scorer, where many do"),
    "top10_sim_median": ("median top-10 similarity", "search_similar_code prints {dir} similarities than the "
                                                     "scorer does"),
    "edge_auc": ("edge AUC", "call-graph neighbours are {dir} similar to each other than on the scorer, so the "
                             "neighbour lists follow the call graph {dir2} closely"),
}


# --- file resolution (prompt section and tool loader) -----------------------------------------------------------

def prompt_candidates(base_dir: str, repo: str, base_commit: str | None, ext: str) -> list[str]:
    short = repo.split("/")[-1] if repo else ""
    slug = repo.replace("/", "_") if repo else ""
    out = []
    if base_commit:
        out += [os.path.join(base_dir, f"{short}_{base_commit}{ext}"), os.path.join(base_dir, f"{slug}_{base_commit}{ext}")]
    out += [os.path.join(base_dir, f"{slug}{ext}"), os.path.join(base_dir, f"{short}{ext}")]
    return out


def loader_candidates(base_dir: str, repo: str, base_commit: str | None, ext: str) -> list[str]:
    given_a_file = base_dir.endswith(ext)   # a file path given in place of a directory is used as is
    if given_a_file:
        return [base_dir]
    commit = None if (base_commit and repo.endswith(f"_{base_commit}")) else base_commit
    short = repo.rsplit("/", 1)[-1]
    slug = repo.replace("/", "_")
    out = []
    if commit:
        out += [os.path.join(base_dir, f"{short}_{commit}{ext}"), os.path.join(base_dir, f"{slug}_{commit}{ext}"),
                os.path.join(base_dir, f"{repo}_{commit}{ext}")]
    out += [os.path.join(base_dir, f"{repo}{ext}"), os.path.join(base_dir, f"{slug}{ext}"),
            os.path.join(base_dir, f"{short}{ext}")]
    return out


def _size(p: str) -> int:
    return os.path.getsize(p) if os.path.isfile(p) else -1


def resolve(graph_dir: str, emb_dir: str, repo: str, base_commit: str | None) -> dict:
    """Which files the prompt-section test accepts and which files the tools load."""
    out: dict[str, Any] = {}
    for kind, base, ext in (("graph", graph_dir, ".json"), ("embeddings", emb_dir, ".npz")):
        pc = prompt_candidates(base, repo, base_commit, ext)
        ok = [p for p in pc if _size(p) > SIZE_LIMIT]
        lc = loader_candidates(base, repo, base_commit, ext)
        loaded = next((p for p in lc if os.path.exists(p)), lc[0])
        out[kind] = {"prompt_ok": bool(ok), "prompt_file": ok[0] if ok else None,
                     "prompt_candidates": [(os.path.basename(p), _size(p)) for p in pc],
                     "tool_file": loaded if os.path.exists(loaded) else None, "tool_file_size": _size(loaded)}
    return out


# --- the three checks ---------------------------------------------------------------------------------------------

REFS_SECTION = ["harness 0.2.11 harness/agent_runner.py:336-337,416-466 (section); graph/embedding_utils.py:53-90 "
                "(tool file lookup)"]
REFS_ALIGN = ["harness 0.2.11 graph/graph_utils.py:154-175,279-334 (graph load, per-node vector); "
              "graph/embedding_utils.py:186-210,316-436 (NPZ load and key lookup); graph/retrieval_utils.py:89-124 "
              "(length adjustment)"]
REFS_VECTORS = ["harness 0.2.11 graph/retrieval_utils.py:771-912 (cosine ranking, ties by name); tools/graph.py:198-288 "
                "(similarity printed to 4 decimals)"]


def check_section(res: dict | None, tools: set[str] | None, direct: tuple[str, str] | None = None) -> CheckResult:
    """Prompt-section conditions. ``tools``: declared tool names, or None when unknown."""
    declared = sorted(set(GRAPH_TOOLS) & tools) if tools is not None else None
    if tools is not None and not declared:
        return CheckResult("graph-section", SKIP, "no graph tool declared: the scorer adds no section and offers no "
                           "graph tools, so the files do not matter", references=REFS_SECTION)
    problems: list[str] = []   # the prompt or the tools differ from the scorer
    cautions: list[str] = []   # the tools read a different (non-empty) file than the one that earned the section
    notes: list[str] = []
    details: dict[str, Any] = {"declared_graph_tools": declared}
    if direct is not None:
        gsz, esz = _size(direct[0]), _size(direct[1])
        ok = gsz > SIZE_LIMIT and esz > SIZE_LIMIT
        details.update({"graph_size": gsz, "embeddings_size": esz, "mode": "files"})
        notes.append("file names were not checked against the harness's lookup (pass --graph-dir/--embeddings-dir "
                     "with --repo and --base-commit for that)")
        problems += [f"{k} file {'missing' if s < 0 else f'is {s} bytes'} (needs more than {SIZE_LIMIT})"
                     for k, s in (("graph", gsz), ("embeddings", esz)) if s <= SIZE_LIMIT]
    else:
        details.update(res or {})
        details["mode"] = "dirs"
        g, e = res["graph"], res["embeddings"]
        ok = g["prompt_ok"] and e["prompt_ok"]
        for kind, r in (("graph", g), ("embeddings", e)):
            if not r["prompt_ok"]:
                problems.append(f"no {kind} file larger than {SIZE_LIMIT} bytes under the names the harness tries: "
                                + ", ".join(f"{n} ({'absent' if s < 0 else f'{s} B'})" for n, s in r["prompt_candidates"]))
            if r["tool_file"] is None:
                problems.append(f"the graph tools find no {kind} file")
            elif r["prompt_ok"] and os.path.abspath(r["tool_file"]) != os.path.abspath(r["prompt_file"]):
                tsz = r["tool_file_size"]
                msg = (f"the section is earned by {os.path.basename(r['prompt_file'])} but the tools load "
                       f"{os.path.relpath(r['tool_file'], os.path.dirname(r['prompt_file']))} ({tsz} B)")
                if tsz <= SIZE_LIMIT:
                    problems.append(msg)
                else:
                    cautions.append(msg)
            elif not r["prompt_ok"]:
                notes.append(f"the tools would still load {kind} file {r['tool_file']} ({r['tool_file_size']} B), "
                             "but the prompt does not advertise them")
    where = (f"if your config declares one of {', '.join(GRAPH_TOOLS)} (or no tool list)" if declared is None
             else f"declared: {', '.join(declared)}")
    if ok and not problems and not cautions:
        return CheckResult("graph-section", PASS, f"the 'Code Intelligence Tools' section is added, as on the scorer "
                           f"({where})", details, notes, REFS_SECTION)
    if ok and not problems:
        return CheckResult("graph-section", WARN, "the section is added, but the tools read a different file from the "
                           "one that earned it", details, cautions + notes, REFS_SECTION)
    status = FAIL if declared else WARN
    if ok:
        summary = "the section is added, but the tools would load a different, empty file"
    else:
        summary = (f"the section is NOT added ({where}); on the scorer every public task has both files, so the "
                   "prompt differs")
    return CheckResult("graph-section", status, summary, details, problems + cautions + notes, REFS_SECTION)


def check_align(graph: GraphData, npz: NpzData, al: Alignment | None) -> CheckResult:
    det: dict[str, Any] = {"graph_bytes": graph.size, "npz_bytes": npz.size}
    if graph.error:
        return CheckResult("graph-align", FAIL, f"graph JSON: {graph.error}", det, references=REFS_ALIGN)
    if npz.fatal:
        return CheckResult("graph-align", FAIL, f"embeddings: {npz.fatal}; every node gets a zero vector", det,
                           references=REFS_ALIGN)
    fails: list[str] = []
    warns: list[str] = []
    n = len(graph.ids)
    src = Counter(al.source)
    det.update({"nodes": n, "npz_keys": len(npz.keys), "edges": len(graph.edges), "edge_records": graph.edge_records,
                "self_loops": graph.self_loops, "edge_types": dict(graph.edge_types), "vector_source": dict(src),
                "dims": {str(k): v for k, v in npz.shapes.items()}, "dtypes": dict(npz.dtypes),
                "target_dim": al.target_dim, "adjusted": al.adjusted})
    if npz.entry_errors:
        fails.append(f"{len(npz.entry_errors)} NPZ entr{'y' if len(npz.entry_errors) == 1 else 'ies'} cannot be loaded "
                     f"with allow_pickle=False as float32 (first: {npz.entry_errors[0]}); the harness then loses the "
                     f"whole file and all {n} nodes get zero vectors")
    if n == 0:
        fails.append("the graph has no nodes")
    own = src.get("own", 0) + src.get("own-prefixed", 0)
    if not npz.entry_errors and n:
        missing = n - own - src.get("inline", 0)
        if missing:
            ex = [f"{graph.ids[i]} -> {k}" for i, k in list(al.borrowed_from.items())[:3]]
            fails.append(f"{missing} of {n} nodes have no NPZ key of their own: {src.get('borrowed-suffix', 0)} borrow "
                         f"another node's vector by suffix, {src.get('borrowed-ci', 0)} by case-insensitive match, "
                         f"{src.get('borrowed-text', 0)} through a node text, {src.get('zero', 0)} get a zero vector "
                         f"(official files: keys equal node ids)" + (f"; e.g. {'; '.join(ex)}" if ex else ""))
    if len({len(v) for v in npz.vectors.values()} | {len(v) for v in graph.inline.values()}) > 1 or al.adjusted:
        fails.append(f"vectors of different lengths ({dict(Counter(len(v) for v in npz.vectors.values()))}); the "
                     f"harness pads or truncates {al.adjusted} of them to {len(al.vectors[0]) if al.vectors else 0}")
    bad_shape = {s: c for s, c in npz.shapes.items() if len(s) != 1}
    if bad_shape:
        fails.append(f"arrays that are not 1-D: {bad_shape} (official: shape ({OFFICIAL_DIM},))")
    nonfinite = sum(1 for v in al.vectors if not all(math.isfinite(x) for x in v))
    if nonfinite:
        fails.append(f"{nonfinite} node vectors contain NaN or inf (cosine ranking becomes undefined)")
    det["nonfinite"] = nonfinite
    if npz.harness_loads:
        extra = [k for k in npz.keys if k not in al.used_keys and k not in set(graph.ids)]
        if extra:
            warns.append(f"{len(extra)} NPZ keys match no graph node (unused unless a lookup borrows them), e.g. "
                         f"{extra[0]!r}")
        prefixed = sum(1 for k in npz.keys if ";" in k)
        if prefixed:
            warns.append(f"{prefixed} NPZ keys carry a '<repo>;' prefix (official keys are bare node ids)")
    dims = {len(v) for v in al.vectors}
    if dims and dims != {OFFICIAL_DIM}:
        warns.append(f"vector length {sorted(dims)} (official: {OFFICIAL_DIM}); cosine still works")
    zero = sum(1 for i, v in enumerate(al.vectors) if al.source[i] != "zero" and not any(v))
    if zero:
        warns.append(f"{zero} supplied vectors are all zero (official minimum norm 3.65); such nodes score 0 against "
                     "every query")
    if graph.inline:
        warns.append(f"{len(graph.inline)} graph nodes carry an inline 'embedding', which the tools use instead of the "
                     "NPZ (official graphs have none)")
    name_ne = sum(1 for i in range(n) if graph.names[i] != graph.ids[i])
    if name_ne:
        warns.append(f"{name_ne} nodes have a 'name' different from their id; vectors are looked up by id but results "
                     "are printed and tie-broken by name (official: name == id)")
    if graph.duplicate_ids:
        warns.append(f"{graph.duplicate_ids} duplicate node ids (networkx merges them)")
    if graph.implicit_nodes:
        warns.append(f"{graph.implicit_nodes} edge endpoints are not declared as nodes (networkx adds them with no "
                     "text)")
    det["name_differs_from_id"] = name_ne
    if fails:
        return CheckResult("graph-align", FAIL, f"{n} nodes, {len(npz.keys)} NPZ keys: the tools do not get one vector "
                           "per node as on the scorer", det, fails + warns, REFS_ALIGN)
    if warns:
        return CheckResult("graph-align", WARN, f"{n} nodes, each with its own vector; {len(warns)} difference(s) from "
                           "the official format", det, warns, REFS_ALIGN)
    return CheckResult("graph-align", PASS, f"{n} nodes, {len(npz.keys)} NPZ keys: one {OFFICIAL_DIM}-d vector per "
                       "node id, as in the official files", det, [], REFS_ALIGN)


def _fmt(x: Any, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


PASS_MEANING = ("A PASS shows only that five summary statistics of the vectors (random-pair cosine mean, sd and share "
                "at or above 0.99, edge AUC, median top-10 similarity) lie inside bands around the official files' "
                "per-file range; it is no evidence that a stand-in is good. The bands can be targeted directly: "
                "stand-ins built from node names and the call graph alone, tuned to the shipped band numbers, pass "
                "with overlap@10 of 0.07-0.31 against the official neighbour lists. A PASS does not show that "
                "search_similar_code returns the scorer's neighbour lists (use --against with the official vectors: "
                "overlap@10), that the vectors come from the same embedding model, that the graph has the official "
                "node set, or that agents solve the same tasks; and the bands were validated on the 4 public dev "
                "repositories only, not on the hidden test repositories.")


def check_vectors(stats: dict | None, reference: dict | None, skip_reason: str = "") -> CheckResult:
    if stats is None:
        return CheckResult("graph-vectors", SKIP, skip_reason or "no vector statistics", references=REFS_VECTORS)
    det = {"stats": stats}
    seen = (f"random-pair cosine mean {_fmt(stats['rand_cos_mean'])} (sd {_fmt(stats['rand_cos_sd'])}, "
            f"{100 * stats['rand_cos_ge_099']:.1f}% >= 0.99); edge AUC {_fmt(stats['edge_auc'])}; median top-10 "
            f"similarity {_fmt(stats['top10_sim_median'], 4)}")
    if "overlap_at_10_vs_against" in stats:
        seen += f"; overlap@10 with --against {_fmt(stats['overlap_at_10_vs_against'])}"
    if reference is None:
        return CheckResult("graph-vectors", SKIP, f"no reference given; {seen}", det, references=REFS_VECTORS)
    pooled = reference["pooled"]
    band = warn_band(reference)
    det["reference"] = reference.get("name")
    det["warn_band"] = band
    warns = []
    outside = []
    for key, (label, meaning) in _BAND_TEXT.items():
        if key not in band or stats.get(key) is None:
            continue
        lo, hi = band[key]
        x = stats[key]
        if lo <= x <= hi:
            continue
        outside.append(key)
        low = x < lo
        pct = key == "rand_cos_ge_099"
        shown = (lambda v: f"{100 * v:.2f}%") if pct else (lambda v: _fmt(v, 4))
        off = pooled[key]
        text = meaning.format(dir="less" if low else "more", dir2="less" if low else "more")
        if key == "rand_cos_ge_099":
            text = meaning.format(dir="fewer" if low else "more")
        elif key == "top10_sim_median":
            text = meaning.format(dir="lower" if low else "higher")
        warns.append(f"{label} {shown(x)} is {'below' if low else 'above'} the band {shown(lo)}-{shown(hi)} (official "
                     f"per-file range {shown(off[0])}-{shown(off[2])}): {text}")
    det["outside"] = outside
    if warns:
        return CheckResult("graph-vectors", WARN, f"stand-in vectors: search_similar_code would behave unlike the scorer "
                           f"({len(warns)} of {sum(1 for k in _BAND_TEXT if k in band)} statistics outside the official "
                           f"bands; {seen})", det, warns + [PASS_MEANING], REFS_VECTORS)
    return CheckResult("graph-vectors", PASS, f"every statistic inside the bands around the official files ({seen}); "
                       "this does not show that the neighbour lists equal the scorer's", det, [PASS_MEANING],
                       REFS_VECTORS)


# --- one task / one file pair ----------------------------------------------------------------------------------------

def analyse_pair(graph_path: str, npz_path: str, *, against: str | None = None, pairs: int = DEFAULT_PAIRS,
                 edge_pairs: int = DEFAULT_EDGE_PAIRS, queries: int = DEFAULT_QUERIES,
                 seed: int = DEFAULT_SEED) -> dict:
    """Read one graph/NPZ pair as the harness would and compute alignment and statistics (JSON-safe dict)."""
    graph = read_graph(graph_path)
    npz = read_npz(npz_path)
    al = None if graph.error or npz.fatal else harness_vectors(graph, npz)
    align = check_align(graph, npz, al)
    stats = None
    skip = ""
    if al is None:
        skip = "files unreadable"
    elif al.npz_lost:
        skip = "the harness loses the whole NPZ (every vector is zero)"
    elif len(graph.ids) < 3:
        skip = "fewer than 3 nodes"
    elif align.details.get("nonfinite"):
        skip = "non-finite vectors"
    else:
        other = None
        against_note = None
        if against:
            other_npz = read_npz(against)
            if other_npz.harness_loads:
                other_al = harness_vectors(graph, other_npz)
                if len({len(v) for v in other_al.vectors}) == 1:
                    other = other_al.vectors
                else:
                    against_note = "--against vectors have mixed lengths"
            else:
                against_note = f"--against unreadable: {other_npz.fatal or other_npz.entry_errors[0]}"
        stats = vector_stats(al.vectors, graph.names, graph.edges, pairs=pairs, edge_pairs=edge_pairs,
                             queries=queries, seed=seed, against=other)
        if against_note:
            stats["against_error"] = against_note
    return {"graph": graph_path, "embeddings": npz_path, "align": align.to_json(), "stats": stats, "skip": skip}


def _analyse_star(args: tuple) -> dict:
    gp, ep, kw = args
    return analyse_pair(gp, ep, **kw)


def analyse_many(pairs_: list[tuple[str, str]], jobs: int = 1, **kw) -> list[dict]:
    work = [(g, e, kw) for g, e in pairs_]
    if jobs <= 1 or len(work) <= 1:
        return [_analyse_star(w) for w in work]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(_analyse_star, work, chunksize=1))


def results_for(entry: dict, reference: dict | None) -> tuple[CheckResult, CheckResult]:
    align = CheckResult(**entry["align"])
    vec = check_vectors(entry["stats"], reference, entry.get("skip", ""))
    return align, vec


def tool_names_from(spec: str | None) -> set[str] | None:
    """--tools: a comma list, 'none', or a conform reference JSON whose tool list is used."""
    if spec is None:
        return None
    if spec.endswith(".json") and os.path.isfile(spec):
        with open(spec, encoding="utf-8") as fh:
            ref = json.load(fh)
        tools = ref.get("tools") if isinstance(ref.get("tools"), dict) else ref
        return {t["name"] for t in (tools or {}).get("tools") or [] if isinstance(t, dict) and "name" in t}
    if spec.strip().lower() == "none":
        return set()
    return {x.strip() for x in spec.split(",") if x.strip()}


def read_tasks(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                t = json.loads(line)
                out.append({"instance_id": t.get("instance_id"), "repo": t.get("repo") or "",
                            "base_commit": t.get("base_commit")})
    return out


def aggregate(name: str, per_task: list[tuple[str, CheckResult]]) -> CheckResult:
    """One line per check over many tasks: the worst status, counts, and the first findings per task."""
    order = [FAIL, WARN, PASS, SKIP]
    counts = Counter(r.status for _, r in per_task)
    worst = next((s for s in order if counts.get(s)), SKIP)
    findings = []
    for label, r in per_task:
        if r.status in (FAIL, WARN):
            findings.append(f"{label}: [{r.status}] {r.summary}" + (f" | {r.findings[0]}" if r.findings else ""))
    summary = f"{len(per_task)} item(s): " + ", ".join(f"{counts[s]} {s}" for s in order if counts.get(s))
    return CheckResult(name, worst, summary, {"counts": dict(counts)}, findings,
                       per_task[0][1].references if per_task else [])
