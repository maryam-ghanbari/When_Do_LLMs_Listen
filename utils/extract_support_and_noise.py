#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For each incorrectly answered CSQA item:
  1) Extract all 1- and 2-hop paths from QC nodes to the correct answer's AC nodes
     in the ConceptNet pruned graph (directional with inverse labeling if needed).
  2) Count #support statements.
  3) Sample global random 2-hop noise paths sized as: same | double | triple.

Output: a JSON list where each entry contains:
  {
    "id": "...",
    "query": "...",
    "correct_choice": "...",
    "qc": [...],
    "ac": [...],                       # AC nodes for the correct choice
    "support_knowledges": [...],       # linearized 1/2-hop QC→AC paths
    "support_count": N,
    "noise_size_policy": "same|double|triple",
    "noise_knowledges": [...],         # random 2-hop anywhere
    "noise_count": M
  }
"""

import argparse, json, pickle, random, re
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import math

# ---- Default resource paths (override by CLI) ----
PRUNED_GRAPH = "/content/KGSweetSpot/data/cpnet/conceptnet.en.pruned.graph"
VOCAB_FILE   = "/content/KGSweetSpot/data/cpnet/concept.txt"
DF_STATEMENT = "/content/KGSweetSpot/data/csqa/statement/dev.statement.jsonl"
DF_GROUNDED  = "/content/KGSweetSpot/data/csqa/grounded/dev.grounded.jsonl"

# Your relation sets
from conceptnet import merged_relations, forward_rels, inverse_rels

# ---- Globals ----
cpnet: Optional[nx.MultiDiGraph] = None
id2concept: List[str] = []
concept2id: Dict[str, int] = {}
id2relation: List[str] = []

# ---- Utilities ----
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def load_vocab(path: str):
    global id2concept, concept2id
    with open(path, "r", encoding="utf-8") as f:
        id2concept = [w.strip() for w in f]
    concept2id = {w: i for i, w in enumerate(id2concept)}

def load_cpnet(path: str):
    global cpnet
    cpnet = nx.read_gpickle(path)
    if not isinstance(cpnet, (nx.MultiDiGraph, nx.MultiGraph)):
        cpnet = nx.MultiDiGraph(cpnet)

def choose_relation_space(space: str):
    global id2relation
    if space == "merged":
        id2relation = merged_relations
    elif space == "forward":
        id2relation = forward_rels
    else:
        id2relation = merged_relations  # default

def _pretty(s: str) -> str:
    return str(s).replace("_", " ").strip()

def _edge_label(u: int, v: int) -> Optional[str]:
    if (cpnet is None) or (not cpnet.has_edge(u, v)):
        return None
    data = cpnet.get_edge_data(u, v)
    if not data: return None
    e = random.choice(list(data.values()))
    rid = e.get("rel", None)
    if rid is None: return None
    if rid < 0 or rid >= len(id2relation): return None
    return id2relation[rid]

def _inv_label(lbl: str) -> str:
    return inverse_rels.get(lbl, f"inverse of {lbl}")

def _neighbors_out(u: int) -> List[int]:
    if (cpnet is None) or (not cpnet.has_node(u)): return []
    try:
        return list(cpnet.successors(u))
    except Exception:
        return list(cpnet.neighbors(u))

def _neighbors_in(u: int) -> List[int]:
    if (cpnet is None) or (not cpnet.has_node(u)): return []
    try:
        return list(cpnet.predecessors(u))
    except Exception:
        return list(cpnet.neighbors(u))

def _lin_trip(a: int, b: int, c: int, allow_inverse=True) -> Optional[str]:
    # hop a->b
    r1 = _edge_label(a, b)
    if r1 is None and allow_inverse:
        r1_back = _edge_label(b, a)
        if r1_back is None: return None
        r1 = _inv_label(r1_back)
    elif r1 is None:
        return None
    # hop b->c
    r2 = _edge_label(b, c)
    if r2 is None and allow_inverse:
        r2_back = _edge_label(c, b)
        if r2_back is None: return None
        r2 = _inv_label(r2_back)
    elif r2 is None:
        return None
    return f"{_pretty(id2concept[a])} ({_pretty(r1)}) {_pretty(id2concept[b])} ({_pretty(r2)}) {_pretty(id2concept[c])}"

def _lin_pair(a: int, b: int, allow_inverse=True) -> Optional[str]:
    r = _edge_label(a, b)
    if r is None and allow_inverse:
        r_back = _edge_label(b, a)
        if r_back is None: return None
        r = _inv_label(r_back)
    elif r is None:
        return None
    return f"{_pretty(id2concept[a])} ({_pretty(r)}) {_pretty(id2concept[b])}"

# ---- Path extraction ----
def extract_qc_to_ac_paths(qc_ids: Set[int], ac_ids: Set[int],
                           max_mid: int = 5000) -> List[str]:
    """
    Return ALL 1-hop and 2-hop (QC -> mid? -> AC) linearized strings.
    Allows inverse labeling when the direct direction is missing.
    """
    out: List[str] = []
    seen: Set[Tuple[int,int,int]] = set()

    # 1-hop QC <-> AC
    for q in qc_ids:
        for a in ac_ids:
            s = _lin_pair(q, a, allow_inverse=True)
            if s:
                key = (q, -1, a)
                if key not in seen:
                    seen.add(key); out.append(s)

    # 2-hop QC -> m -> AC  (m can be any neighbor of q or a)
    # We’ll explore q’s out/in neighbors as mids and try to connect to a.
    for q in qc_ids:
        nbr_q = set(_neighbors_out(q)) | set(_neighbors_in(q))
        if len(nbr_q) > max_mid:  # sanity cap
            nbr_q = set(random.sample(list(nbr_q), max_mid))
        for m in nbr_q:
            for a in ac_ids:
                s = _lin_trip(q, m, a, allow_inverse=True)
                if s:
                    key = (q, m, a)
                    if key not in seen:
                        seen.add(key); out.append(s)

    return out

def sample_global_noise_paths(k: int, tries_per=300) -> List[str]:
    out = []
    if cpnet is None: return out
    N = len(id2concept)
    for _ in range(max(0, k)):
        ok = None
        for _try in range(tries_per):
            u = random.randrange(N)
            n1 = _neighbors_out(u)
            if not n1: continue
            m = random.choice(n1)
            n2 = _neighbors_out(m)
            if not n2: continue
            t = random.choice(n2)
            if t == u: continue
            s = _lin_trip(u, m, t, allow_inverse=True)
            if s:
                ok = s; break
        if ok: out.append(ok)
    return out

# ---- Core runner ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incorrect-json", required=True,
                    help="Items the model got wrong (from your earlier split).")
    ap.add_argument("--df-statement", default=DF_STATEMENT)
    ap.add_argument("--df-grounded",  default=DF_GROUNDED)
    ap.add_argument("--vocab",        default=VOCAB_FILE)
    ap.add_argument("--cpnet",        default=PRUNED_GRAPH)
    ap.add_argument("--relation-space", choices=["merged","forward","auto"], default="merged")
    ap.add_argument("--out", required=True, help="Where to write the results JSON.")
    ap.add_argument("--size", choices=["none", "1/4","half","same","double","triple", "5times"], default="same",
                help="Noise size relative to #support paths.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    load_vocab(args.vocab)
    load_cpnet(args.cpnet)
    choose_relation_space(args.relation_space)

    # Load incorrect items
    with open(args.incorrect_json, "r", encoding="utf-8") as f:
        wrong_items = json.load(f)
    # Load statement/grounded so we can fetch QC/AC per question index
    df = pd.read_json(args.df_statement, lines=True)
    with open(args.df_grounded, "r", encoding="utf-8") as f:
        grounded = [json.loads(line) for line in f if line.strip()]

    # Group grounded per question: 5 options per question (CSQA)
    K = 5
    def group_every_k(lst, k): return [lst[i:i+k] for i in range(0, len(lst), k)]
    gr_groups = group_every_k(grounded, K)

    # Helper: map answer text to option index
    def find_choice_idx(cands: List[str], ans_text: str) -> Optional[int]:
        if not isinstance(cands, list) or not isinstance(ans_text, str): return None
        an = norm(ans_text)
        try:
            return next(i for i, t in enumerate(cands) if norm(t) == an)
        except StopIteration:
            return None

    mult = {"none": 0.0, "1/4": 0.25, "half": 0.5, "same": 1.0, "double": 2.0, "triple": 3.0, "5times": 5.0}[args.size]

    results = []
    for it in tqdm(wrong_items, desc="Extracting support & noise for incorrect cases"):
        # Align to dataset index: we assume ids are 1-based positions in dev set
        try:
            qid = int(str(it.get("id", "0")))
        except Exception:
            continue
        if not (1 <= qid <= len(df)):
            continue

        row = df.iloc[qid-1]
        cands = row["question"]["choices"]
        cands = [ch["text"] for ch in cands] if isinstance(cands, list) else it.get("cands", [])
        gold  = row["answerKey"]
        # Convert gold letter to index if present, else via text match with it["answer"]
        if isinstance(gold, str) and len(gold) == 1 and gold.isalpha():
            gold_idx = ord(gold.upper()) - ord("A")
        else:
            gold_idx = find_choice_idx(cands, it.get("answer", ""))

        if gold_idx is None or not (0 <= gold_idx < len(cands)):  # fallback match
            gold_idx = find_choice_idx(cands, it.get("answer", ""))

        if gold_idx is None:
            # cannot align gold choice -> skip
            continue

        # QC nodes (shared across options): use option 0's qc as in your previous scripts
        gr5 = gr_groups[qid-1]
        qc_names = gr5[0].get("qc", [])
        qc_ids   = {concept2id[n] for n in qc_names if n in concept2id}

        # AC nodes for the *correct* choice
        ac_names = gr5[gold_idx].get("ac", [])
        ac_ids   = {concept2id[n] for n in ac_names if n in concept2id}

        # Extract support (QC→AC) paths: 1-hop and 2-hop
        support_knowledges = extract_qc_to_ac_paths(qc_ids, ac_ids)
        support_count = len(support_knowledges)

        # Decide noise size
        target_noise = 0 if support_count <= 0 else int(math.ceil(support_count * mult))
        noise_knowledges = sample_global_noise_paths(target_noise)

        results.append({
            "id": str(qid),
            "query": row["question"]["stem"],
            "cands": cands,  # <-- add this line
            "answer": cands[gold_idx] if 0 <= gold_idx < len(cands) else None,
            "qc": list(qc_names),
            "ac": list(ac_names),
            "support_knowledges": support_knowledges,
            "support_count": support_count,
            "noise_size_policy": args.size,
            "noise_knowledges": noise_knowledges,
            "noise_count": len(noise_knowledges),
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✔ Wrote {len(results)} items to {args.out}")

if __name__ == "__main__":
    main()
