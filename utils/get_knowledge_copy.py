# extract_knowledge_scenarios.py
import argparse, json, pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, deque
from scipy.sparse import coo_matrix
from tqdm import tqdm

# ---- Your paths (unchanged defaults; override with args/env if needed) ----
PRUNED_GRAPH = '/content/KGSweetSpot/data/cpnet/conceptnet.en.pruned.graph'
VOCAB_GRAPH  = '/content/KGSweetSpot/data/cpnet/concept.txt'
DF_STATEMENT = '/content/KGSweetSpot/data/csqa/statement/dev.statement.jsonl'
DF_GROUNDED  = '/content/KGSweetSpot/data/csqa/grounded/dev.grounded.jsonl'
GRAPH_PATH   = '/content/KGSweetSpot/data/csqa/graph/dev.graph.adj.pk'

# ConceptNet relation lists you already have
from conceptnet import merged_relations, forward_rels, inverse_rels

# ---- Globals ----
concept2id = None
id2concept = None
id2relation = None
cpnet = None

def load_resources(cpnet_vocab_path: str):
    global concept2id, id2concept, id2relation
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}
    id2relation = forward_rels  # match QA-GNN forward planes by default

def load_cpnet(cpnet_graph_path: str):
    global cpnet
    cpnet = nx.read_gpickle(cpnet_graph_path)

def _group_every_k(lst, k):
    if len(lst) % k != 0:
        raise ValueError(f"List length {len(lst)} is not a multiple of {k}")
    return [lst[i:i+k] for i in range(0, len(lst), k)]

def _select_id2relation(subgraphs, forward_rels, merged_rels):
    first = subgraphs[0]
    N = int(first["concepts"].shape[0])
    R = int(first["adj"].shape[0] // N)
    if len(forward_rels) == R: return forward_rels
    if len(merged_rels)  == R: return merged_rels
    raise ValueError(f"Cannot match relation list: R={R}")

def _normalize_nodes(which, subgraph, default_mask_key=None) -> Set[int]:
    N = subgraph['concepts'].shape[0]
    if which is None:
        if default_mask_key:
            mask = subgraph[default_mask_key]
            return {i for i, f in enumerate(mask) if f}
        return set(range(N))
    if isinstance(which, str):
        key = {'q':'qmask','a':'amask'}.get(which.lower())
        mask = subgraph[key]
        return {i for i, f in enumerate(mask) if f}
    which = list(which)
    if not which: return set()
    if isinstance(which[0], int):
        return {i for i in which if 0 <= i < N}
    gid2local = {gid: i for i, gid in enumerate(subgraph['concepts'])}
    out = set()
    for name in which:
        gid = concept2id.get(name)
        if gid is not None and gid in gid2local: out.add(gid2local[gid])
    return out

def _pretty(s: str) -> str:
    return s.replace("_", " ").strip()

def _lin_path(subgraph, vseq: List[int], etexts: List[str]) -> str:
    names = [_pretty(id2concept[subgraph['concepts'][i]]) for i in vseq]
    parts = [names[0]]
    for rtxt, nxt in zip(etexts, names[1:]):
        parts.append(f"({_pretty(rtxt)})"); parts.append(nxt)
    return " ".join(parts)

def all_simple_relation_paths(subgraph: Dict, starts='q', ends='a',
                              max_hops: int = 2, limit: Optional[int] = None) -> List[str]:
    """Vertex-simple paths up to max_hops (bidirectional with inverse labels)."""
    adj = subgraph['adj'].tocoo(copy=False)
    N = subgraph['concepts'].shape[0]
    if N == 0: return []
    out: Dict[int, List[Tuple[int, str]]] = {u: [] for u in range(N)}
    row, col, data = adj.row, adj.col, adj.data
    for rNs, t, val in zip(row, col, data):
        if not val: continue
        r = rNs // N; s = rNs % N
        rel_f = id2relation[r]
        rel_inv = inverse_rels.get(rel_f, f"inverse of {rel_f}")
        out[s].append((t, rel_f))
        out[t].append((s, rel_inv))
    S = _normalize_nodes(starts, subgraph, 'qmask')
    T = _normalize_nodes(ends,   subgraph, 'amask')
    if not S or not T: return []
    results: List[str] = []
    visited = [False] * N
    def dfs(u, hops, vseq, etexts):
        if limit is not None and len(results) >= limit: return
        if u in T and hops >= 1 and hops <= max_hops:
            results.append(_lin_path(subgraph, vseq, etexts))
        if hops == max_hops: return
        for v, rtxt in out[u]:
            if visited[v]: continue
            visited[v] = True
            vseq.append(v); etexts.append(rtxt)
            dfs(v, hops+1, vseq, etexts)
            etexts.pop(); vseq.pop()
            visited[v] = False
    for s in S:
        visited[s] = True
        dfs(s, 0, [s], [])
        visited[s] = False
    return results

def load_base_bin(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    return {str(it.get("id", "")).strip(): it for it in arr if "id" in it}

def norm_txt(s: str) -> str:
    return str(s).strip().lower()

def _ranked_indices_from_base(record: dict, cands: List[str]) -> List[int]:
    """Return indices of choices sorted by base-model probability (desc)."""
    # 1) Prefer 'ranked_idx' if present
    if 'ranked_idx' in record and isinstance(record['ranked_idx'], list) and record['ranked_idx']:
        idxs = [int(i) for i in record['ranked_idx']]
        # sanity: keep only valid
        idxs = [i for i in idxs if 0 <= i < len(cands)]
        # make unique in order
        seen = set(); uniq = []
        for i in idxs:
            if i not in seen:
                seen.add(i); uniq.append(i)
        # if missing any, append remaining by natural order
        remaining = [i for i in range(len(cands)) if i not in seen]
        return uniq + remaining

    # 2) Else sort by 'probs' if available
    probs = record.get('probs', None)
    if isinstance(probs, list) and len(probs) == len(cands):
        arr = np.asarray(probs, dtype=float)
        return list(np.argsort(-arr))

    # 3) Else try reconstruct from pred2..pred5 strings
    lst = []
    if 'pred' in record:
        try:
            lst.append(next(i for i,t in enumerate(cands) if norm_txt(t)==norm_txt(record['pred'])))
        except StopIteration:
            pass
    for k in ['pred2','pred3','pred4','pred5']:
        if k in record:
            try:
                lst.append(next(i for i,t in enumerate(cands) if norm_txt(t)==norm_txt(record[k])))
            except StopIteration:
                pass
    # de-dup and fill the rest
    seen = set(lst); lst = [i for i in lst if 0 <= i < len(cands)]
    lst += [i for i in range(len(cands)) if i not in seen]
    return lst

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grounded-out', required=True,
                    help='Output JSON of QA+knowledge for selected scenario.')
    ap.add_argument('--scenario', choices=['support_pred','rival_sim','rival_rank'], required=True,
                    help='support_pred: support base prediction; '
                         'rival_sim: choose rival with closest connectivity; '
                         'rival_rank: choose rival by base rank (use --rival-rank).')
    ap.add_argument('--rival-rank', type=int, default=2,
                    help='When --scenario=rival_rank, choose this rank (2..5) as the rival.')
    ap.add_argument('--base-low-json', required=True,
                    help='Path to base no-KG JSON whose ids define the subset; use its ranked predictions.')
    ap.add_argument('--method', choices=['simple_paths'], default='simple_paths')
    ap.add_argument('--max-path-len', type=int, default=2)
    ap.add_argument('--starts', type=str, default='q')
    ap.add_argument('--ends',   type=str, default='a')
    ap.add_argument('--limit',  type=int, default=None)
    ap.add_argument('--choices-per-q', type=int, default=5)
    args = ap.parse_args()

    # guard rival-rank
    if args.scenario == 'rival_rank' and args.rival_rank not in (2,3,4,5):
        print(f"[warn] --rival-rank={args.rival_rank} invalid; forcing to 2")
        args.rival_rank = 2

    load_resources(VOCAB_GRAPH)
    load_cpnet(PRUNED_GRAPH)

    # Load dataset
    df = pd.read_json(DF_STATEMENT, lines=True)
    with open(DF_GROUNDED, 'r', encoding='utf-8') as f:
        grounded = [json.loads(line) for line in f if line.strip()]
    with open(GRAPH_PATH, 'rb') as f:
        subgraphs = pickle.load(f)

    # Align relation planes
    global id2relation
    id2relation = _select_id2relation(subgraphs, forward_rels, merged_relations)

    # Per-option groups
    K = args.choices_per_q
    if len(subgraphs) != len(grounded): raise ValueError("Mismatch: subgraphs vs grounded")
    if len(subgraphs) != len(df) * K:  raise ValueError("Need one subgraph per option")
    sg_groups  = _group_every_k(subgraphs, K)
    gr_groups  = _group_every_k(grounded, K)

    # Only ids present in base file; get base predictions
    base_map = load_base_bin(args.base_low_json)
    target_ids = set(base_map.keys())

    # enumerator (simple paths up to max_hops)
    def enumerate_paths(sg):
        return all_simple_relation_paths(sg, starts=args.starts, ends=args.ends,
                                         max_hops=args.max_path_len, limit=args.limit)

    out, no_match = [], 0
    for qi, ((_, row), sg5, gr5) in enumerate(tqdm(zip(df.iterrows(), sg_groups, gr_groups),
                                                   total=len(df),
                                                   desc=f"Extracting {args.scenario}")):
        qid = str(qi + 1)
        if qid not in target_ids:
            continue

        cands = [ch["text"] for ch in row["question"]["choices"]]
        gold_idx = ord(row["answerKey"]) - ord("A")
        answer   = cands[gold_idx]

        rec = base_map[qid]
        ranked = _ranked_indices_from_base(rec, cands)
        if not ranked:
            no_match += 1
            continue

        pred_idx = ranked[0]

        # enumerate paths per option
        per_option_paths = [enumerate_paths(sg) for sg in sg5]
        counts = [len(p) for p in per_option_paths]

        if args.scenario == 'support_pred':
            target_idx = pred_idx

        elif args.scenario == 'rival_sim':
            # closest-by-count rival (exclude pred)
            pred_cnt = counts[pred_idx]
            others = [j for j in range(K) if j != pred_idx]
            if others:
                diffs = sorted((abs(counts[j]-pred_cnt), j) for j in others)
                target_idx = diffs[0][1]
            else:
                target_idx = (pred_idx + 1) % K

        else:  # rival_rank
            r = args.rival_rank
            if r == 1:
                r = 2  # ensure rival != top-1
            if r-1 < len(ranked):
                target_idx = ranked[r-1]
                if target_idx == pred_idx:
                    # just in case of ties or malformed ranking
                    target_idx = ranked[min(r, len(ranked)-1)]
            else:
                # fallback: nearest-by-count rival
                pred_cnt = counts[pred_idx]
                others = [j for j in range(K) if j != pred_idx]
                target_idx = sorted((abs(counts[j]-pred_cnt), j) for j in others)[0][1]

        knowledges = per_option_paths[target_idx]

        out.append({
            "id":         qid,
            "scenario":   args.scenario,
            "rival_rank": args.rival_rank if args.scenario == 'rival_rank' else None,
            "query":      row["question"]["stem"],
            "cands":      cands,
            "answer":     answer,
            "qc":         gr5[0].get("qc", []),
            "ac":         gr5[target_idx].get("ac", []),   # ACs for the chosen option
            "knowledges": knowledges,                      # <- feed to inference (h=1)
            "diag_path_counts": counts,
        })

    Path(args.grounded_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.grounded_out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"✔ Wrote {len(out)} items to {args.grounded_out}")
    print(f"Unmatched / unrankeable base predictions: {no_match}")

if __name__ == "__main__":
    main()
