import torch
import networkx as nx
import itertools
import json
from tqdm import tqdm
from .conceptnet import merged_relations
import numpy as np
from scipy import sparse
import pickle
from scipy.sparse import csr_matrix, coo_matrix
from multiprocessing import Pool
from collections import OrderedDict


from .maths import *

__all__ = ['generate_graph']

concept2id = None
id2concept = None
relation2id = None
id2relation = None

cpnet = None
cpnet_all = None
cpnet_simple = None


def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
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

def concepts2adj(node_ids):
    global id2relation
    cids = np.array(node_ids, dtype=np.int32)
    n_rel = len(id2relation)
    n_node = cids.shape[0]
    adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
    for s in range(n_node):
        for t in range(n_node):
            s_c, t_c = cids[s], cids[t]
            if cpnet.has_edge(s_c, t_c):
                for e_attr in cpnet[s_c][t_c].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                        adj[e_attr['rel']][s][t] = 1
    # cids += 1  # note!!! index 0 is reserved for padding
    adj = coo_matrix(adj.reshape(-1, n_node))
    return adj, cids

#####################################################################################################

def concepts_to_adj_matrices_1hop_neighbours(data):
    """
    Given a set of question concept IDs and answer concept IDs, return a 1-hop subgraph including all qa_nodes:
        [(adj, concept_ids, qmask), ...]
    Each subgraph is centered on one question concept.
    """
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for u in set(qc_ids) | set(ac_ids):
        if u in cpnet.nodes:
            extra_nodes |= set(cpnet[u])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return {
        'adj': adj,
        'concepts': concepts,
        'qmask': qmask,
        'amask': amask
    }

def concepts_to_adj_matrices_1hop_neighbours_qc_only(data):
    qc_ids, _ = data
    extra_nodes = set()

    # Retrieve only the 1-hop neighbors of question concepts (qc_ids)
    for u in set(qc_ids):  # Only iterate over question concepts
        if u in cpnet.nodes:
            extra_nodes |= set(cpnet[u])  # Add all one-hop neighbors of question concepts

    extra_nodes = extra_nodes - qc_ids  # Ensure we do not include original QCs
    schema_graph = sorted(qc_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    adj, concepts = concepts2adj(schema_graph)
    return {
        'adj': adj,
        'concepts': concepts,
        'qmask': qmask
    }


def concepts_to_adj_matrices_1hop_neighbours_without_relatedto(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for u in set(qc_ids) | set(ac_ids):
        if u in cpnet.nodes:
            for v in cpnet[u]:
                for data in cpnet[u][v].values():
                    if data['rel'] not in (15, 32):
                        extra_nodes.add(v)
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return {
        'adj': adj,
        'concepts': concepts,
        'qmask': qmask,
        'amask': amask
    }


def concepts_to_adj_matrices_2hop_neighbours(data):
    """
    Given a set of question concept IDs and answer concept IDs, return a 1-hop subgraph including all qa_nodes:
        [(adj, concept_ids, qmask), ...]
    Each subgraph is centered on one question concept.
    """
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for u in set(qc_ids) | set(ac_ids):
        if u in cpnet.nodes:
            extra_nodes |= set(cpnet[u])
    for v in extra_nodes:
        if v in cpnet.nodes:
            extra_nodes |= set(cpnet[v])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return {
        'adj': adj,
        'concepts': concepts,
        'qmask': qmask,
        'amask': amask
    }

#####################################################################################################

def concepts_to_adj_matrices_2hop_qa_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qc_ids:
        for aid in ac_ids:
            if qid != aid and qid in cpnet.nodes and aid in cpnet.nodes:
                extra_nodes |= set(cpnet[qid]) & set(cpnet[aid])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return {
        'adj': adj,
        'concepts': concepts,
        'qmask': qmask,
        'amask': amask
    }


def concepts_to_adj_matrices_2hop_all_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qa_nodes:
        for aid in qa_nodes:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return {
        'adj': adj,
        'concepts': concepts,
        'qmask': qmask,
        'amask': amask
    }

def concepts_to_adj_matrices_3hop_qa_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qc_ids:
        for aid in ac_ids:
            if qid != aid and qid in cpnet.nodes and aid in cpnet.nodes:
                for u in cpnet[qid]:
                    for v in cpnet[aid]:
                        if cpnet.has_edge(u, v):  # ac is a 3-hop neighbour of qc
                            extra_nodes.add(u)
                            extra_nodes.add(v)
                        if u == v:  # ac is a 2-hop neighbour of qc
                            extra_nodes.add(u)
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return {
        'adj': adj,
        'concepts': concepts,
        'qmask': qmask,
        'amask': amask
    }

#####################################################################################################
#                     functions below this line will be called by preprocess.py                     #
#####################################################################################################

def generate_adj_data_from_grounded_concepts(grounded_path, cpnet_graph_path, cpnet_vocab_path, output_path, num_processes):
    """
    This function will save
        (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
        (2) concepts ids
        (3) qmask that specifices whether a node is a question concept
        (4) amask that specifices whether a node is a answer concept
    to the output path in python pickle format

    grounded_path: str
    cpnet_graph_path: str
    cpnet_vocab_path: str
    output_path: str
    num_processes: int
    """
    print(f'generating adj data for {grounded_path}...')

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    qa_data = []
    with open(grounded_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            dic = json.loads(line)
            q_ids = set(concept2id[c] for c in dic['qc'])
            a_ids = set(concept2id[c] for c in dic['ac'])
            q_ids = q_ids - a_ids
            qa_data.append((q_ids, a_ids))

    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(concepts_to_adj_matrices_2hop_qa_pair, qa_data), total=len(qa_data)))

    # res is a list of tuples, each tuple consists of four elements (adj, concepts, qmask, amask)
    with open(output_path, 'wb') as fout:
        pickle.dump(res, fout)

    print(f'adj data saved to {output_path}')
    print()