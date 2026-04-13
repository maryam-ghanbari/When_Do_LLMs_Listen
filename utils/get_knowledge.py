import pandas as pd
from conceptnet import merged_relations, merged_relations_clean, forward_rels, inverse_rels
from tqdm import tqdm
from collections import defaultdict, deque
import argparse
from collections import OrderedDict
import scipy
import networkx as nx
import json
import numpy as np
import pickle
from typing import Iterable, Optional, Union, List, Tuple, Dict, Set
from scipy.sparse import coo_matrix
from sentence_transformers import SentenceTransformer

# ── Globals ────────────────────────────────────────────────────────────────────
concept2id = None
id2concept = None
relation2id = None
id2relation = None

cpnet = None
cpnet_all = None
cpnet_simple = None

pruned_graph = '/content/KGSweetSpot/data/cpnet/conceptnet.en.pruned.graph'
vocab_graph = '/content/KGSweetSpot/data/cpnet/concept.txt'
df_statement = '/content/KGSweetSpot/data/csqa/statement/dev.statement.jsonl'
df_grounded = '/content/KGSweetSpot/data/csqa/grounded/dev.grounded.jsonl'
graph_path = '/content/KGSweetSpot/data/csqa/graph/dev.graph.adj.pk'
pickleFile = open("/content/KGSweetSpot/data/csqa/graph/dev.graph.adj.pk", "rb")
test_pruned_graph_df = pd.read_pickle(pickleFile)


def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = forward_rels
    relation2id = {r: i for i, r in enumerate(id2relation)}

def load_cpnet(cpnet_graph_path):
    global cpnet, cpnet_simple
    cpnet = nx.read_gpickle(cpnet_graph_path)
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


# 3) helper to invert a triple
def invert_triple(subject: str, rel: str, obj: str):
    inv = inverse_rels.get(rel, None)
    if inv is None:
        # fallback: prepend “is”/append “by” or just mirror
        inv = rel + " (inverse)"
    # swap subject/object
    return obj, inv, subject


def concepts2adj(node_ids: List[int]) -> Tuple[coo_matrix, np.ndarray]:
    """
    Build multi-relational adjacency for the induced subgraph on node_ids.
    Returns (adj, concepts_array).
    """
    cids = list(node_ids)
    idx = {c:i for i,c in enumerate(cids)}
    R = len(id2relation)
    N = len(cids)
    tensor = np.zeros((R, N, N), dtype=np.uint8)

    for u in cids:
        if u not in cpnet: continue
        ui = idx[u]
        for v, edges in cpnet[u].items():
            if v not in idx: continue
            vi = idx[v]
            for e in edges.values():
                r = e.get('rel', -1)
                if 0 <= r < R:
                    tensor[r, ui, vi] = 1

    adj = coo_matrix(tensor.reshape(R*N, N))
    return adj, np.array(cids, dtype=np.int32)


# --- helpers to group and to select relation list -----------------------------
def _group_every_k(lst, k):
    if len(lst) % k != 0:
        raise ValueError(f"List length {len(lst)} is not a multiple of {k}")
    return [lst[i:i+k] for i in range(0, len(lst), k)]

def _select_id2relation(subgraphs, forward_rels, merged_rels):
    # infer R from the first subgraph
    first = subgraphs[0]
    N = int(first["concepts"].shape[0])
    R = int(first["adj"].shape[0] // N)
    if len(forward_rels) == R:
        return forward_rels
    if len(merged_rels) == R:
        return merged_rels
    raise ValueError(f"Cannot match relation list: R={R}, "
                     f"len(forward_rels)={len(forward_rels)}, len(merged_rels)={len(merged_rels)}")


def all_simple_relation_paths(
    subgraph: Dict,
    starts: Optional[Union[str, Iterable[int], Iterable[str]]] = 'q',
    ends:   Optional[Union[str, Iterable[int], Iterable[str]]] = 'a',
    max_hops: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[str]:
    """
    Enumerate ALL vertex-simple paths from any start to any end (up to max_hops).
    Traversal is bidirectional: forward uses id2relation[r], reverse uses inverse_rels[id2relation[r]].
    Paths are linearized as: A (rel1) B (rel2) C
    """

    def _pretty(s: str) -> str:
        # Clean display: "department_store" -> "department store"
        return s.replace("_", " ").strip()

    adj = subgraph['adj'].tocoo(copy=False)
    concepts = subgraph['concepts']
    N = concepts.shape[0]
    if N == 0:
        return []

    # --- helpers ---
    def _normalize(which, default_mask_key=None) -> Set[int]:
        if which is None:
            if default_mask_key:
                mask = subgraph[default_mask_key]
                return {i for i, f in enumerate(mask) if f}
            return set(range(N))
        if isinstance(which, str):
            key = {'q': 'qmask', 'a': 'amask'}.get(which.lower())
            if key is None:
                raise ValueError("starts/ends must be 'q','a', indices, or concept strings")
            mask = subgraph[key]
            return {i for i, f in enumerate(mask) if f}
        which = list(which)
        if not which:
            return set()
        if isinstance(which[0], int):
            return {i for i in which if 0 <= i < N}
        # concept names -> local indices
        gid2local = {gid: i for i, gid in enumerate(concepts)}
        out = set()
        for name in which:
            gid = concept2id.get(name)
            if gid is not None and gid in gid2local:
                out.add(gid2local[gid])
        return out

    def _lin(vseq: List[int], etexts: List[str]) -> str:
        # Pretty-print concept names and relation texts (remove underscores)
        names = [_pretty(id2concept[concepts[i]]) for i in vseq]
        parts = [names[0]]
        for rtxt, nxt in zip(etexts, names[1:]):
            parts.append(f"({_pretty(rtxt)})")
            parts.append(nxt)
        return " ".join(parts)

    # Rebuild bidirectional adjacency with relation texts
    R = len(id2relation)
    out: Dict[int, List[Tuple[int, str]]] = {u: [] for u in range(N)}
    row, col, data = adj.row, adj.col, adj.data
    # row encodes r*N + s
    for rN_plus_s, t, val in zip(row, col, data):
        if not val:
            continue
        r = rN_plus_s // N
        s = rN_plus_s %  N
        rel_f = id2relation[r]
        rel_inv = inverse_rels.get(rel_f, f"inverse of {rel_f}")
        # forward s -> t
        out[s].append((t, rel_f))
        # reverse t -> s
        out[t].append((s, rel_inv))

    S = _normalize(starts, 'qmask')
    T = _normalize(ends,   'amask')
    if not S or not T:
        return []

    results: List[str] = []
    visited = [False] * N

    def dfs(u: int, hops: int, vseq: List[int], etexts: List[str]):
        if limit is not None and len(results) >= limit:
            return
        if u in T and hops >= 1 and (max_hops is None or hops <= max_hops):
            results.append(_lin(vseq, etexts))
        if max_hops is not None and hops == max_hops:
            return
        for v, rtxt in out[u]:
            if visited[v]:
                continue
            visited[v] = True
            vseq.append(v)
            etexts.append(rtxt)
            dfs(v, hops + 1, vseq, etexts)
            etexts.pop()
            vseq.pop()
            visited[v] = False

    for s in S:
        visited[s] = True
        dfs(s, 0, [s], [])
        visited[s] = False

    return results


def all_shortest_relation_paths(
    subgraph: Dict,
    starts: Optional[Union[str, Iterable[int], Iterable[str]]] = 'q',
    ends:   Optional[Union[str, Iterable[int], Iterable[str]]] = 'a',
    max_hops: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[str]:
    """
    Enumerate ALL shortest (fewest-hops) paths from any start to any end.
    Traversal is bidirectional: forward uses id2relation[r], reverse uses inverse_rels[id2relation[r]].
    If max_hops is provided, only return shortest paths whose length <= max_hops.

    Returns a list of linearized strings: "A (rel1) B (rel2) C"
    """
    def _pretty(s: str) -> str:
        # "department_store" -> "department store"
        return s.replace("_", " ").strip()

    adj = subgraph['adj'].tocoo(copy=False)
    concepts = subgraph['concepts']
    N = concepts.shape[0]
    if N == 0:
        return []

    # --- helpers ---
    def _normalize(which, default_mask_key=None) -> Set[int]:
        if which is None:
            if default_mask_key:
                mask = subgraph[default_mask_key]
                return {i for i, f in enumerate(mask) if f}
            return set(range(N))
        if isinstance(which, str):
            key = {'q': 'qmask', 'a': 'amask'}.get(which.lower())
            if key is None:
                raise ValueError("starts/ends must be 'q','a', indices, or concept strings")
            mask = subgraph[key]
            return {i for i, f in enumerate(mask) if f}
        which = list(which)
        if not which:
            return set()
        if isinstance(which[0], int):
            return {i for i in which if 0 <= i < N}
        gid2local = {gid: i for i, gid in enumerate(concepts)}
        out = set()
        for name in which:
            gid = concept2id.get(name)
            if gid is not None and gid in gid2local:
                out.add(gid2local[gid])
        return out

    def _lin(vseq: List[int], etexts: List[str]) -> str:
        # Pretty-print concept names and relation texts
        names = [_pretty(id2concept[concepts[i]]) for i in vseq]
        parts = [names[0]]
        for rtxt, nxt in zip(etexts, names[1:]):
            parts.append(f"({_pretty(rtxt)})")
            parts.append(nxt)
        return " ".join(parts)

    # Rebuild bidirectional adjacency with relation texts
    R = len(id2relation)
    out: Dict[int, List[Tuple[int, str]]] = {u: [] for u in range(N)}
    row, col, data = adj.row, adj.col, adj.data
    for rN_plus_s, t, val in zip(row, col, data):
        if not val:
            continue
        r = rN_plus_s // N
        s = rN_plus_s %  N
        rel_f = id2relation[r]
        rel_inv = inverse_rels.get(rel_f, f"inverse of {rel_f}")
        out[s].append((t, rel_f))     # forward
        out[t].append((s, rel_inv))   # reverse

    S = _normalize(starts, 'qmask')
    T = _normalize(ends,   'amask')
    if not S or not T:
        return []

    # Multi-source BFS to compute hop distances
    INF = 10**12
    dist = [INF] * N
    q = deque()
    for s in S:
        dist[s] = 0
        q.append(s)
    while q:
        u = q.popleft()
        du = dist[u]
        for v, _ in out[u]:
            if dist[v] == INF:
                dist[v] = du + 1
                q.append(v)

    # Shortest distance to any end
    min_d = min((dist[t] for t in T if dist[t] < INF), default=INF)
    if min_d == INF:
        return []
    if max_hops is not None and min_d > max_hops:
        return []

    # Build shortest-path DAG: edges u->v with dist[v] == dist[u] + 1
    sp_edges: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    for u in range(N):
        if dist[u] == INF:
            continue
        for v, rtxt in out[u]:
            if dist[v] == dist[u] + 1:
                sp_edges[u].append((v, rtxt))

    # Enumerate all shortest paths from any s to any t with dist[t] == min_d
    T_ok = {t for t in T if dist[t] == min_d}
    results: List[str] = []

    def dfs(u: int, vseq: List[int], etexts: List[str]):
        if limit is not None and len(results) >= limit:
            return
        if u in T_ok and len(etexts) == min_d:
            results.append(_lin(vseq, etexts))
            return
        for v, rtxt in sp_edges.get(u, []):
            if dist[v] != dist[u] + 1:
                continue
            vseq.append(v)
            etexts.append(rtxt)
            dfs(v, vseq, etexts)
            etexts.pop()
            vseq.pop()

    for s in S:
        if dist[s] == 0 and 0 <= min_d:
            dfs(s, [s], [])

    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--grounded-out',  required=True)
    p.add_argument('--mode',
                   choices=['gold_only', 'gold_plus_strongest', 'gold_plus_weakest', 'merge_all', 'wrong_only'],
                   required=True,
                   help=("gold_only: only the correct option's paths; "
                         "gold_plus_strongest: gold + distractor with most paths; "
                         "gold_plus_weakest: gold + distractor with fewest paths; "
                         "merge_all: union of all options' paths; "
                         "wrong_only: union of all distractors' paths (exclude gold)."))
    p.add_argument('--method',        choices=['simple_paths', 'shortest_paths'], required=True)
    p.add_argument('--max-path-len',  type=int, default=2, help='Max hops for simple paths (and an upper bound filter for shortest).')
    p.add_argument('--starts',        type=str, default='q',
                   help="Start nodes selector: 'q' | 'a' | 'qmask' | 'amask' | 'all'")
    p.add_argument('--ends',          type=str, default='a',
                   help="End nodes selector:   'q' | 'a' | 'qmask' | 'amask' | 'all'")
    p.add_argument('--limit',         type=int, default=None,
                   help='Optional max number of paths to return per option (applied during enumeration).')
    p.add_argument('--choices-per-q', type=int, default=5)
    args = p.parse_args()

    # 1) Load resources
    load_resources(vocab_graph)
    load_cpnet(pruned_graph)

    # normalize selectors: allow qmask/amask/all as convenience
    def norm_sel(s):
        s = (s or '').lower()
        if s in ('q', 'qmask'):
            return 'q'
        if s in ('a', 'amask'):
            return 'a'
        if s in ('all',):
            return None  # means: all nodes
        return s

    starts_sel = norm_sel(args.starts)
    ends_sel   = norm_sel(args.ends)

    # 2) Load grounded lines (per-option) and subgraphs
    records = []
    with open(df_grounded, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    with open(graph_path, 'rb') as f:
        subgraphs = pickle.load(f)

    # 3) Select relation naming to match the subgraph tensor planes
    global id2relation
    id2relation = _select_id2relation(subgraphs, forward_rels, merged_relations)

    # 4) Bind path enumeration method
    def run_simple(sg):
        return all_simple_relation_paths(
            sg, starts=starts_sel, ends=ends_sel,
            max_hops=args.max_path_len, limit=args.limit
        )
    def run_shortest(sg):
        return all_shortest_relation_paths(
            sg, starts=starts_sel, ends=ends_sel,
            max_hops=args.max_path_len, limit=args.limit
        )
    methods = {'simple_paths': run_simple, 'shortest_paths': run_shortest}

    # 5) Group per question (assumes QA-GNN ordering: 5 options per question)
    K = args.choices_per_q
    df = pd.read_json(df_statement, lines=True)
    if len(subgraphs) != len(records):
        raise ValueError(f"Mismatch: len(subgraphs)={len(subgraphs)} vs len(grounded)={len(records)}")
    if len(subgraphs) != len(df) * K:
        raise ValueError(f"Expected {len(df)*K} subgraphs for {len(df)} questions × {K} choices, got {len(subgraphs)}")

    sg_groups  = _group_every_k(subgraphs, K)
    rec_groups = _group_every_k(records, K)

    # helper: concatenate and de-duplicate while preserving order
    def dedup_concat(list_of_lists):
        seen, out = set(), []
        for L in list_of_lists:
            for p in L:
                if p not in seen:
                    seen.add(p)
                    out.append(p)
        return out

    # 6) Build per-question outputs
    output_list = []
    for qi, ((_, row), sg5, gr5) in enumerate(tqdm(zip(df.iterrows(), sg_groups, rec_groups),
                                                   total=len(df), desc="Linearizing per-question knowledges")):
        # Enumerate paths per option
        per_option_paths = [methods[args.method](sg) for sg in sg5]

        # gold option index
        if "answerKey" not in row or not isinstance(row["answerKey"], str):
            raise ValueError(f"Row {qi} missing 'answerKey' needed for --mode {args.mode}")
        gold_idx = ord(row["answerKey"]) - ord("A")

        # simple connectivity = number of paths per option (diagnostic)
        scores = [len(paths) for paths in per_option_paths]

        if args.mode == 'gold_only':
            knowledges = per_option_paths[gold_idx]

        elif args.mode == 'gold_plus_strongest':
            strongest_idx, strongest_val = None, None
            for j, s in enumerate(scores):
                if j == gold_idx: 
                    continue
                if strongest_idx is None or s > strongest_val:
                    strongest_idx, strongest_val = j, s
            knowledges = dedup_concat([per_option_paths[gold_idx],
                                       per_option_paths[strongest_idx] if strongest_idx is not None else []])

        elif args.mode == 'gold_plus_weakest':
            weakest_idx, weakest_val = None, None
            for j, s in enumerate(scores):
                if j == gold_idx:
                    continue
                if weakest_idx is None or s < weakest_val:
                    weakest_idx, weakest_val = j, s
            knowledges = dedup_concat([per_option_paths[gold_idx],
                                       per_option_paths[weakest_idx] if weakest_idx is not None else []])

        elif args.mode == 'merge_all':
            knowledges = dedup_concat(per_option_paths)

        else:  # args.mode == 'wrong_only'
            knowledges = dedup_concat([per_option_paths[j] for j in range(K) if j != gold_idx])

        # Common fields
        stem   = row["question"]["stem"]
        cands  = [ch["text"] for ch in row["question"]["choices"]]
        answer = cands[gold_idx]

        new_id   = str(qi + 1)
        orig_qid = (row.get("id") if isinstance(row, dict) else None)

        output_list.append({
            "id":         new_id,
            "query":      stem,
            "cands":      cands,
            "answer":     answer,
            "qc":         gr5[0].get("qc", []),          # same across options
            "ac":         gr5[gold_idx].get("ac", []),   # keep gold's answer concepts for reference
            "knowledges": knowledges,
            "diag_path_counts": scores,
            "diag_mode": args.mode
        })

    # 7) Write the QA+Knowledge JSON
    with open(args.grounded_out, 'w', encoding='utf-8') as f:
        json.dump(output_list, f, indent=2, ensure_ascii=False)
    print(f"✔ Wrote {len(output_list)} QA+Knowledge entries to {args.grounded_out}")
    print(f"Mode={args.mode} | Method={args.method} | max_path_len={args.max_path_len} | per-option limit={args.limit}")

if __name__ == "__main__":
    main()