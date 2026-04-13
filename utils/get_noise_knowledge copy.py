# /content/KGSweetSpot/utils/get_noise_knowledge.py
# Generates noisy knowledge statements with the same schema you already use.
# Requires the same cpnet resources as your current extract script.

import argparse, json, pickle, random
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

# --- default resource paths (override via CLI) ---
PRUNED_GRAPH = '/content/KGSweetSpot/data/cpnet/conceptnet.en.pruned.graph'
VOCAB_GRAPH  = '/content/KGSweetSpot/data/cpnet/concept.txt'
DF_STATEMENT = '/content/KGSweetSpot/data/csqa/statement/dev.statement.jsonl'
DF_GROUNDED  = '/content/KGSweetSpot/data/csqa/grounded/dev.grounded.jsonl'
GRAPH_PATH   = '/content/KGSweetSpot/data/csqa/graph/dev.graph.adj.pk'

# Your relation spaces
from conceptnet import merged_relations, forward_rels, inverse_rels

# --- Globals ---
cpnet: Optional[nx.MultiDiGraph] = None
id2concept: List[str] = []
concept2id: Dict[str, int] = {}
id2relation: List[str] = []   # will match edge attribute 'rel' indices

# ---------------- Utilities ----------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def load_vocab(path: str):
    global id2concept, concept2id
    with open(path, "r", encoding="utf8") as f:
        id2concept = [w.strip() for w in f]
    concept2id = {w: i for i, w in enumerate(id2concept)}

def load_cpnet(path: str):
    global cpnet, GRAPH_NODES, GRAPH_NODE_SET
    cpnet = nx.read_gpickle(path)
    if not isinstance(cpnet, (nx.MultiDiGraph, nx.MultiGraph)):
        cpnet = nx.MultiDiGraph(cpnet)
    GRAPH_NODES = list(cpnet.nodes())
    GRAPH_NODE_SET = set(GRAPH_NODES)

def choose_relation_space(space: str):
    """space in {'merged','forward','auto'}"""
    global id2relation
    if space == 'merged':
        id2relation = merged_relations
    elif space == 'forward':
        id2relation = forward_rels
    else:
        # 'auto': infer by the max rel id present on edges we sample
        # fallback to merged if unsure
        id2relation = merged_relations

def _pretty(s: str) -> str:
    return str(s).replace("_", " ").strip()

def _edge_label(u: int, v: int) -> Optional[str]:
    """Pick one edge u->v and return its forward label (no inversion here)."""
    if not cpnet.has_edge(u, v): 
        return None
    data_dict = cpnet.get_edge_data(u, v)
    if not data_dict: 
        return None
    # pick a random multi-edge
    e = random.choice(list(data_dict.values()))
    rid = e.get('rel', None)
    if rid is None: 
        return None
    if rid < 0 or rid >= len(id2relation):
        return None
    return id2relation[rid]

def _inv_label(lbl: str) -> str:
    return inverse_rels.get(lbl, f"inverse of {lbl}")

def _lin_trip(u: int, m: int, t: int, use_inverse_if_needed: bool = True) -> Optional[str]:
    """Linearize u -> m -> t; if a direction doesn't exist, try inverse label if allowed."""
    # hop 1
    r1 = _edge_label(u, m)
    if r1 is None and use_inverse_if_needed:
        r1_back = _edge_label(m, u)
        if r1_back is None: 
            return None
        r1 = _inv_label(r1_back)
    elif r1 is None:
        return None

    # hop 2
    r2 = _edge_label(m, t)
    if r2 is None and use_inverse_if_needed:
        r2_back = _edge_label(t, m)
        if r2_back is None:
            return None
        r2 = _inv_label(r2_back)
    elif r2 is None:
        return None

    return f"{_pretty(id2concept[u])} ({_pretty(r1)}) {_pretty(id2concept[m])} ({_pretty(r2)}) {_pretty(id2concept[t])}"

def _neighbors_out(u: int) -> List[int]:
    if not cpnet.has_node(u):
        return []
    # Prefer directed successors; fall back to undirected neighbors
    try:
        return list(cpnet.successors(u))
    except Exception:
        return list(cpnet.neighbors(u))

def _neighbors_in(u: int) -> List[int]:
    if not cpnet.has_node(u):
        return []
    try:
        return list(cpnet.predecessors(u))
    except Exception:
        return list(cpnet.neighbors(u))

# ---------------- Ranking helpers (reused logic) ----------------
def norm_txt(s: str) -> str:
    return str(s).strip().lower()

def ranked_indices_from_base(record: dict, cands: List[str]) -> List[int]:
    # Prefer explicit ranked_idx
    if isinstance(record.get('ranked_idx'), list):
        idxs = [int(i) for i in record['ranked_idx'] if 0 <= int(i) < len(cands)]
        seen = set(); uniq = []
        for i in idxs:
            if i not in seen:
                uniq.append(i); seen.add(i)
        return uniq + [i for i in range(len(cands)) if i not in seen]
    # Else derive from probs
    probs = record.get('probs')
    if isinstance(probs, list) and len(probs) == len(cands):
        arr = np.asarray(probs, dtype=float)
        return list(np.argsort(-arr))
    # Else reconstruct from pred/pred2..
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
    seen = set(lst)
    lst = [i for i in lst if 0 <= i < len(cands)]
    lst += [i for i in range(len(cands)) if i not in seen]
    return lst

# ---------------- Noise samplers ----------------
def sample_global_noise_paths(k: int, max_tries_per_path: int = 200) -> List[str]:
    """Random 2-hop anywhere in KG."""
    out = []
    N = len(id2concept)
    for _ in range(k):
        ok = None
        for _try in range(max_tries_per_path):
            u = random.randrange(N)
            nbrs1 = _neighbors_out(u)
            if not nbrs1: 
                continue
            m = random.choice(nbrs1)
            nbrs2 = _neighbors_out(m)
            if not nbrs2:
                continue
            t = random.choice(nbrs2)
            if t == u:  # allow cycles if you want; I exclude to diversify
                continue
            s = _lin_trip(u, m, t)
            if s:
                ok = s; break
        if ok:
            out.append(ok)
    return out

def sample_qc_nonac_paths(qc_set: Set[int], ac_set: Set[int], k: int,
                          avoid_qc_ends: bool = True,
                          max_tries_per_path: int = 300) -> List[str]:
    """Start in QC, 2-hop path that does NOT end in any AC (optionally also not in QC)."""
    out = []
    qc = list(qc_set)
    if not qc:
        return out
    for _ in range(k):
        ok = None
        for _try in range(max_tries_per_path):
            u = random.choice(qc)  # start from QC
            m_candidates = _neighbors_out(u)
            if not m_candidates:
                continue
            m = random.choice(m_candidates)
            t_candidates = _neighbors_out(m)
            if not t_candidates:
                continue
            t = random.choice(t_candidates)
            if t in ac_set:
                continue
            if avoid_qc_ends and (t in qc_set):
                continue
            s = _lin_trip(u, m, t)
            if s:
                ok = s; break
        if ok:
            out.append(ok)
    return out

def sample_nonqc_to_ac_paths(ac_target: int, qc_set: Set[int], k: int,
                             avoid_qc_mid: bool = True,
                             max_tries_per_path: int = 500) -> List[str]:
    """
    Start from a NON-QC node, middle also NOT QC (if avoid_qc_mid), end at the target AC.
    We build s -> m -> ac_target. We try both directions on edges when linearizing.
    """
    out = []
    # neighbors that can reach ac_target in one hop
    back_nbrs = list(set(_neighbors_in(ac_target)) | set(_neighbors_out(ac_target)))
    if not back_nbrs:
        return out

    for _ in range(k):
        ok = None
        for _try in range(max_tries_per_path):
            m = random.choice(back_nbrs)
            if avoid_qc_mid and m in qc_set:
                continue
            # choose s that connects to m
            s_candidates = list(set(_neighbors_in(m)) | set(_neighbors_out(m)))
            if not s_candidates:
                continue
            s_node = random.choice(s_candidates)
            if s_node in qc_set:
                continue
            if s_node == ac_target or s_node == m:
                continue
            # linearize s -> m -> t
            s_txt = _lin_trip(s_node, m, ac_target)
            if s_txt:
                ok = s_txt; break
        if ok:
            out.append(ok)
    return out

# ---------------- Main generator ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--noise-type', choices=['noise_global','noise_qc_nonac','noise_nonqc_to_ac'], required=True)
    ap.add_argument('--base-json', required=True,
                    help='Path to base no-KG JSON (e.g., your bin_high_gt0.85.json). Used to select ids & ranking.')
    ap.add_argument('--grounded-out', required=True,
                    help='Where to write the noisy knowledge JSON.')
    ap.add_argument('--choices-per-q', type=int, default=5)
    ap.add_argument('--noise-k', type=int, default=12, help='Number of noisy statements per item.')
    ap.add_argument('--seed', type=int, default=42)

    # resources (override if needed)
    ap.add_argument('--vocab', default=VOCAB_GRAPH)
    ap.add_argument('--cpnet', default=PRUNED_GRAPH)
    ap.add_argument('--relation-space', choices=['merged','forward','auto'], default='merged')

    # data (for QC/AC / subgraphs)
    ap.add_argument('--df-statement', default=DF_STATEMENT)
    ap.add_argument('--df-grounded',  default=DF_GROUNDED)
    ap.add_argument('--subgraphs-pk', default=GRAPH_PATH)

    # only for noise_nonqc_to_ac: which choice to end at
    ap.add_argument('--target', default='pred',
                    help="Which choice to target in noise_nonqc_to_ac. "
                         "Options: pred, rank2, rank3, rank4, rank5, gold")
    args = ap.parse_args()

    set_seed(args.seed)
    load_vocab(args.vocab)
    load_cpnet(args.cpnet)
    choose_relation_space(args.relation_space)

    # load dataset rows, grounded (qc/ac per option), and subgraphs (for masks)
    df = pd.read_json(args.df_statement, lines=True)
    with open(args.df_grounded, 'r', encoding='utf-8') as f:
        grounded = [json.loads(line) for line in f if line.strip()]
    with open(args.subgraphs_pk, 'rb') as f:
        subgraphs = pickle.load(f)

    K = args.choices_per_q
    assert len(subgraphs) == len(grounded), "Mismatch: subgraphs vs grounded"
    assert len(subgraphs) == len(df) * K,   "Expect one subgraph per choice"

    # group per question
    def group_every_k(lst, k): return [lst[i:i+k] for i in range(0, len(lst), k)]
    sg_groups = group_every_k(subgraphs, K)
    gr_groups = group_every_k(grounded,  K)

    # map base ids -> base records (for subset & ranking)
    with open(args.base_json, 'r', encoding='utf-8') as f:
        base_bin = json.load(f)
    base_map = {str(it.get('id','')).strip(): it for it in base_bin if 'id' in it}
    keep_ids = set(base_map.keys())

    out, skipped = [], 0
    for qi, ((_, row), sg5, gr5) in enumerate(tqdm(zip(df.iterrows(), sg_groups, gr_groups),
                                                   total=len(df),
                                                   desc=f"Building {args.noise_type}")):
        qid = str(qi + 1)
        if qid not in keep_ids:
            continue

        cands = [ch["text"] for ch in row["question"]["choices"]]
        gold_idx = ord(row["answerKey"]) - ord("A")
        answer   = cands[gold_idx]

        base_rec = base_map[qid]
        ranked = ranked_indices_from_base(base_rec, cands)
        if not ranked:
            skipped += 1
            continue

        # QC & AC sets (global ids) from grounded/subgraph
        # Use the first option’s qc; AC depends on option index
        qc_names = gr5[0].get("qc", [])
        qc_set   = {concept2id[n] for n in qc_names if n in concept2id}

        # We’ll provide one set of knowledges per item, like your other scenarios.
        knowledges: List[str] = []

        if args.noise_type == 'noise_global':
            knowledges = sample_global_noise_paths(args.noise_k)

        elif args.noise_type == 'noise_qc_nonac':
            # any AC from any option is excluded as an end node
            ac_all_names = set()
            for g in gr5:
                for nm in g.get("ac", []):
                    ac_all_names.add(nm)
            ac_set = {concept2id[n] for n in ac_all_names if n in concept2id}
            knowledges = sample_qc_nonac_paths(qc_set, ac_set, args.noise_k)

        else:  # noise_nonqc_to_ac
            # pick which choice to “attach” to
            if args.target == 'pred':
                target_idx = ranked[0]
            elif args.target.startswith('rank'):
                r = int(args.target.replace('rank',''))
                r = max(1, min(5, r))
                target_idx = ranked[r-1] if r-1 < len(ranked) else ranked[0]
            elif args.target == 'gold':
                target_idx = gold_idx
            else:
                target_idx = ranked[0]

            ac_names = gr5[target_idx].get("ac", [])
            # If multiple AC nodes map to this choice, pick any target per path try
            ac_ids = [concept2id[n] for n in ac_names if n in concept2id]
            if not ac_ids:
                knowledges = []
            else:
                # sample paths; each time choose a random AC id from this set
                for _ in range(args.noise_k):
                    t = random.choice(ac_ids)
                    paths = sample_nonqc_to_ac_paths(t, qc_set, 1)
                    if paths:
                        knowledges.extend(paths)

        out.append({
            "id":         qid,
            "scenario":   args.noise_type,
            "target":     args.target if args.noise_type == 'noise_nonqc_to_ac' else None,
            "query":      row["question"]["stem"],
            "cands":      cands,
            "answer":     answer,
            "qc":         list(qc_names),
            # for consistency, store the AC set we actually constrained against / pointed to
            "ac":         (gr5[ranked[0]].get("ac", []) if args.noise_type=='noise_nonqc_to_ac' else []),
            "knowledges": knowledges,
            "diag_path_counts": None  # no natural "support counts" for noise
        })

    Path(args.grounded_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.grounded_out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"✔ Wrote {len(out)} items to {args.grounded_out} (skipped {skipped})")

if __name__ == "__main__":
    main()
