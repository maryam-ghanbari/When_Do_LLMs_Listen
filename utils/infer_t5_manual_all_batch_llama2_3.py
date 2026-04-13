#!/usr/bin/env python3
import argparse
import json
from tqdm import tqdm
import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import time
import math
import random
from torch.nn.functional import cross_entropy
import torch.nn.functional as F

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

# ---------------- Repro ----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

global empty_counter
empty_counter = 0


# ---------------- Prompt builder (minimal changes: add default branch) ----------------
def build_source(args, query, cands, knowledge=None):
    # exactly the same logic, with a generic fallback for non-T5 models
    source, targets = None, None
    if args.task == 'csqa':
        # UnifiedQA branch
        if 'unifiedqa-t5' in args.model_type or (args.model_ckpt is not None and 't5' in (args.model_type or '').lower()):
            if knowledge:
                source = (
                    f"{knowledge} \n"
                    + f"{query} \n "
                    + " ".join(f"({chr(ord('A') + i)}) {cand}"
                              for i, cand in enumerate(cands))
                )
            else:
                source = (
                    f"{query} \n "
                    + " ".join(f"({chr(ord('A') + i)}) {cand}"
                              for i, cand in enumerate(cands))
                )
            targets = cands

        # Flan-T5 branch
        elif 'flan-t5' in args.model_type:
            if knowledge:
                source = (
                    "Given the context and the question, identify the most appropriate answer from the choices only based on the provided context.\n" + \
                    f"Context: {knowledge} \n Question: {query} \n Answer Choices: " + \
                    ' '.join([f'({chr(ord("A") + i)}) {cand}' for i, cand in enumerate(cands)]) + \
                    "\nRespond with the exact answer text."
                )
            else:
                source = (
                    # "Answer the provided question.\n" + \
                    f"Question: {query} \n Answer Choices: " + \
                    ' '.join([f'({chr(ord("A") + i)}) {cand}' for i, cand in enumerate(cands)]) + \
                    "\nRespond with the exact answer text."
                )
            targets = cands

        # ---------- NEW: default prompt for non-T5 models (e.g., Llama/Mistral) ----------
        else:
            if knowledge:
                source = (
                    "Given the context and the question, identify the most appropriate answer from the choices only based on the provided context.\n" + \
                    f"Context: {knowledge} \n Question: {query} \n Answer Choices: " + \
                         ' '.join([f'({chr(ord("A") + i)}) {cand}' for i, cand in enumerate(cands)]) + \
                    "\nRespond with the exact answer text."
                )
            else:
                source = (
                    f"Question: {query} \n Answer Choices: " + \
                    ' '.join([f'({chr(ord("A") + i)}) {cand}' for i, cand in enumerate(cands)]) + \
                    "\nRespond with the exact answer text."
                )
            targets = cands

    if source is None:
        raise Exception(
            f"Prompt‐building not implemented for {args.task} {args.model_type}"
        )
    return source, targets


# ---------------- Existing enc-dec scorer (unchanged) ----------------
def score_for_input_multi_block(
    args, tokenizer, model,
    query: str,
    cands: list[str],
    knowledges_batch: list[str],
):
    B = len(knowledges_batch)
    N = len(cands)
    pad_id = tokenizer.pad_token_id
    V = model.config.vocab_size

    # 1) Build B source-strings
    sources = []
    for k in knowledges_batch:
        src, targets = build_source(args, query, cands, knowledge=(k or None))
        sources.append(src)

    # 2) Tokenize all B sources at once → encoder inputs
    enc = tokenizer(
        sources,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    input_ids  = enc.input_ids.cuda()         # (B, L)
    attn_mask  = enc.attention_mask.cuda()    # (B, L)

    with torch.no_grad():
        enc_hs = model.encoder(
            input_ids=input_ids,
            attention_mask=attn_mask
        ).last_hidden_state                   # (B, L, D)

    # 3) Tokenize N candidates → labels (full-text targets; average-loss handles length)
    lbl    = tokenizer(
        cands,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    labels = lbl.input_ids.cuda()             # (N, T)

    # 4) Build decoder inputs by shifting right
    dec_inputs = model._shift_right(labels)   # (N, T)

    # 5) Expand encoder outputs to (B, N, L, D) then flatten → (B*N, L, D)
    enc_hs_flat = (
        enc_hs
        .unsqueeze(1)                         # (B, 1, L, D)
        .expand(B, N, -1, -1)                 # (B, N, L, D)
        .reshape(B*N, enc_hs.size(1), enc_hs.size(2))
    )
    # same for attention mask → (B*N, L)
    attn_flat = (
        attn_mask
        .unsqueeze(1)                         # (B, 1, L)
        .expand(B, N, -1)                     # (B, N, L)
        .reshape(B*N, attn_mask.size(1))
    )

    # 6) Expand decoder inputs to (B, N, T) and flatten → (B*N, T)
    dec_in_flat = (
        dec_inputs
        .unsqueeze(0)                        # (1, N, T)
        .expand(B, -1, -1)                   # (B, N, T)
        .reshape(B*N, dec_inputs.size(1))    # (B*N, T)
    )

    # 7) One big decoder + LM-head pass
    with torch.no_grad():
        dec_hs_flat = model.decoder(
            input_ids=dec_in_flat,
            encoder_hidden_states=enc_hs_flat,
            encoder_attention_mask=attn_flat
        ).last_hidden_state                   # (B*N, T, D)

        lm_logits = model.lm_head(dec_hs_flat) # (B*N, T, V)

    # 8) Per-token losses
    logits_flat = lm_logits.view(-1, V)       # (B*N*T, V)
    labels_flat = (
        labels
        .unsqueeze(0)                         # (1, N, T)
        .expand(B, -1, -1)                    # (B, N, T)
        .reshape(-1)                          # (B*N*T,)
    )

    loss_flat = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=pad_id,
        reduction='none'
    ).view(B*N, -1)                           # (B*N, T)

    # 9) Mask padding & sum/average
    mask        = (labels_flat != pad_id).view(B*N, -1).float()
    loss_per_ex = (loss_flat * mask).sum(dim=1)  # (B*N,)
    if args.average_loss:
        token_counts = mask.sum(dim=1).clamp_min(1)
        loss_per_ex  = loss_per_ex / token_counts

    # 10) Negate → scores, softmax *within* each block
    scores = -loss_per_ex.view(B, N)           # (B, N)
    probs  = torch.softmax(scores, dim=1)      # (B, N)

    return scores, probs


# ---------------- NEW: causal-LM scorer (Llama/Mistral/etc.) ----------------
def score_for_input_multi_block_causal(
    args, tokenizer, model,
    query: str,
    cands: list[str],
    knowledges_batch: list[str],
):
    """
    Compute per-candidate NLL for decoder-only LMs by appending the answer string
    to the prompt and scoring only the answer tokens. --average-loss is respected.
    """
    assert not model.config.is_encoder_decoder
    B, N = len(knowledges_batch), len(cands)

    # 1) Build B prompts
    prompts = []
    for k in knowledges_batch:
        src, _ = build_source(args, query, cands, knowledge=(k or None))
        prompts.append(src)

    # 2) Use full-text answer targets (same logic as enc-dec path)
    targets = list(cands)

    # 3) Build sequences: [prompt] + " " + [target]
    seq_input_ids, seq_labels = [], []
    for b in range(B):
        prompt_ids = tokenizer(prompts[b], add_special_tokens=False).input_ids
        for t in range(N):
            tgt_ids = tokenizer(" " + targets[t], add_special_tokens=False).input_ids
            ids = prompt_ids + tgt_ids
            labels = [-100]*len(prompt_ids) + tgt_ids
            seq_input_ids.append(ids)
            seq_labels.append(labels)

    # 4) Pad (right) and move to CUDA
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    batch = tokenizer.pad({"input_ids": seq_input_ids}, padding=True, return_tensors="pt")
    input_ids = batch["input_ids"].cuda()
    attention_mask = (input_ids != tokenizer.pad_token_id).long().cuda()

    max_len = input_ids.size(1)
    labels_tensor = torch.full((B*N, max_len), -100, dtype=torch.long)
    for i, lbl in enumerate(seq_labels):
        labels_tensor[i, :len(lbl)] = torch.tensor(lbl, dtype=torch.long)
    labels = labels_tensor.cuda()

    # 5) Forward & causal loss over answer span
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # (B*N, L, V)

    logits_shifted = logits[:, :-1, :].contiguous()
    labels_shifted = labels[:, 1:].contiguous()

    V = logits_shifted.size(-1)
    loss_flat = F.cross_entropy(
        logits_shifted.view(-1, V),
        labels_shifted.view(-1),
        ignore_index=-100,
        reduction='none'
    ).view(B*N, -1)

    mask = (labels_shifted != -100).float()
    loss_per_ex = (loss_flat * mask).sum(dim=1)
    if args.average_loss:
        denom = mask.sum(dim=1).clamp_min(1)
        loss_per_ex = loss_per_ex / denom

    scores = (-loss_per_ex).view(B, N)
    probs  = torch.softmax(scores, dim=1)
    return scores, probs


# ---------------- Aggregation (unchanged) ----------------
def score_for_query(
    args, tokenizer, model,
    query: str,
    knowledges: list[str],
    cands:   list[str],
):
    global empty_counter
    # 1) figure out your h/v blocks exactly as before
    n = len(knowledges)
    random.shuffle(knowledges)
    h, v = args.h, args.v

    if h == -1 and v == -1:
        raise ValueError('h and v cannot both be -1!')

    if h != -1 and h <= 0:
        raise ValueError('h must be >= 1 or -1')
    if v != -1 and v <= 0:
        raise ValueError('v must be >= 1 or -1')

    if v == -1:
        # avoid v = 0 when n < h
        v = max(1, math.ceil(n / max(1, h)))
    if h == -1:
        h = max(1, math.ceil(n / max(1, v)))


    if n > 0:
        if getattr(args, 'merge_knowledge', False):
            merged = args.merge_sep.join(knowledges)
            blocks = [merged]
        else:
            blocks = [' '.join(knowledges[i : i + h]) for i in range(0, h * v, h)]
    else:
        print("No knowledge provided; processing without context.")
        empty_counter += 1
        blocks = [""]

    B_max = getattr(args, 'block_batch_size', 64)
    all_scores, all_probs = [], []

    for start in range(0, len(blocks), B_max):
        batch_blocks = blocks[start : start + B_max]

        # ---------- minimal change: choose scorer by model type ----------
        if model.config.is_encoder_decoder:
            scores_b, probs_b = score_for_input_multi_block(
                args, tokenizer, model,
                query=query,
                cands=cands,
                knowledges_batch=batch_blocks
            )
        else:
            scores_b, probs_b = score_for_input_multi_block_causal(
                args, tokenizer, model,
                query=query,
                cands=cands,
                knowledges_batch=batch_blocks
            )

        all_scores.append(scores_b)
        all_probs .append(probs_b)

    scores_tensor = torch.cat(all_scores, dim=0)
    probs_tensor  = torch.cat(all_probs,  dim=0)
    return scores_tensor, probs_tensor


def checker(args, answer, pred):
    return 1 if answer == pred else 0


def process_item(args, tokenizer, model, item):
    query = item['query'] if 'query' in item else item['question']
    if 'cands' in item:
        cands = item['cands']
    elif args.task == 'csqa2':
        cands = ['yes', 'no']
    else:
        raise Exception('process_item() not implemented for {args.task}!')

    # --- BASE (no knowledge): always compute & store full ranking ---
    global empty_counter
    _ec = empty_counter
    base_scores_, base_probs_ = score_for_query(args, tokenizer, model, query, [], cands)
    empty_counter = _ec  # don't count base pass as "empty knowledge"

    if args.aggfunc == 'best_score':
        base_scores = torch.mean(base_scores_, dim=0)
        base_probs  = torch.softmax(base_scores, dim=0)
    elif args.aggfunc == 'best_prob':
        base_probs  = torch.mean(base_probs_, dim=0); base_probs = base_probs / base_probs.sum()
        base_scores = torch.mean(base_scores_, dim=0)
    elif args.aggfunc == 'poe':
        base_logp   = torch.sum(torch.log(base_probs_ + 1e-12), dim=0)
        base_probs  = torch.softmax(base_logp, dim=0)
        base_scores = torch.mean(base_scores_, dim=0)
    elif args.aggfunc == 'moe':
        base_probs  = torch.sum(base_probs_, dim=0); base_probs = base_probs / base_probs.sum()
        base_scores = torch.mean(base_scores_, dim=0)
    else:
        raise ValueError(f"Unknown aggfunc {args.aggfunc}")

    base_order = torch.argsort(base_probs, descending=True).tolist()
    item["ranked"] = [{"choice": cands[i], "prob": float(base_probs[i].item())} for i in base_order]
    item["base_pred"] = cands[base_order[0]]
    item["base_pred_conf"] = float(base_probs[base_order[0]].item())
    item["base_probs"] = base_probs.tolist()

    # --- WITH KNOWLEDGE (from the input item, if any) ---
    knowledges = item.get('knowledges', [])
    scores_, probs_ = score_for_query(args, tokenizer, model, query, knowledges, cands)

    if args.aggfunc == 'best_score':
        scores = torch.mean(scores_, dim=0)
        probs  = torch.softmax(scores, dim=0)
    elif args.aggfunc == 'best_prob':
        probs  = torch.mean(probs_, dim=0); probs = probs / probs.sum()
        scores = torch.mean(scores_, dim=0)
    elif args.aggfunc == 'poe':
        logp   = torch.sum(torch.log(probs_ + 1e-12), dim=0)
        probs  = torch.softmax(logp, dim=0)
        scores = torch.mean(scores_, dim=0)
    elif args.aggfunc == 'moe':
        probs  = torch.sum(probs_, dim=0); probs = probs / probs.sum()
        scores = torch.mean(scores_, dim=0)
    else:
        raise ValueError(f"Unknown aggfunc {args.aggfunc}")

    # with-knowledge ranking/pred
    order = torch.argsort(probs, descending=True).tolist()
    item["ranked_with_ks"] = [{"choice": cands[i], "prob": float(probs[i].item())} for i in order]
    p_idx = order[0]
    item['pred'] = cands[p_idx]
    item['pred_conf'] = float(probs[p_idx].item())

    # per-KS impact on chosen-with-KS prediction
    per_ks = probs_[:, p_idx] if probs_.ndim == 2 else probs_
    item["prob_mean"] = float(per_ks.mean().item())
    item["prob_max"]  = float(per_ks.max().item())
    item["prob_min"]  = float(per_ks.min().item())
    item["per_ks_probs"] = [float(x) for x in per_ks.tolist()]

    # gold prob, margin, entropy
    if 'answer' in item:
        gold_txt = item['answer'].strip().lower()
        gold_idx = next((i for i, t in enumerate(cands) if t.strip().lower() == gold_txt), -1)
        if gold_idx >= 0:
            item['true_conf'] = float(probs[gold_idx].item())
        item['ok'] = 1 if item['pred'] == item['answer'] else 0

    top2_vals, _ = torch.topk(probs, k=min(2, probs.numel()))
    item['conf_margin']  = float(top2_vals[0] - (top2_vals[1] if top2_vals.numel() > 1 else 0.0))
    item['conf_entropy'] = float(-(probs * torch.log(probs + 1e-12)).sum().item())

    # store tensors
    item['probs_'] = probs_.tolist()
    item['probs']  = probs.tolist()
    item['scores'] = scores.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['csqa', 'csqa2', 'qasc'])
    # ---------- minimal change: allow any HF id for model-type ----------
    parser.add_argument('--model-type', type=str, required=True)
    parser.add_argument('--model-ckpt', type=str, default=None)
    # in main() parser:
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--device-map', type=str, default=None)         # e.g. "auto"
    parser.add_argument('--offload-folder', type=str, default=None)     # e.g. "/tmp/offload"
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--average-loss', action='store_true')
    parser.add_argument('--h', type=int, default=1)
    parser.add_argument('--v', type=int, default=-1)
    parser.add_argument('--aggfunc', type=str, default='best_prob', choices=['best_score', 'best_prob', 'poe', 'moe'])
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--merge-knowledge', action='store_true',
                        help='If set, merge all knowledge strings into a single block separated by \\n')
    parser.add_argument('--merge-sep', type=str, default='\n',
                        help='Separator to use when merging knowledge strings (default: newline)')
    parser.add_argument('--n', type=int, default=None)
    args = parser.parse_args()
    args.output_path = f'/content/KGSweetSpot/Inference_results/data/inference_{"" if args.model_ckpt is None else "ft"}{args.model_type.split("/")[-1]}.{args.input_path.split("/")[-1]}'

    # ---------- minimal change: generic loading for enc-dec or causal ----------
    model_id = args.model_ckpt if args.model_ckpt is not None else args.model_type
    cfg = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # choose a compute dtype
    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    # optional quantization config
    quantization_config = None
    if (args.load_in_4bit or args.load_in_8bit) and BitsAndBytesConfig is not None:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_quant_type="nf4" if args.load_in_4bit else None,
            bnb_4bit_use_double_quant=True if args.load_in_4bit else None,
            bnb_4bit_compute_dtype=compute_dtype if args.load_in_4bit else None,
        )

    common_kwargs = dict(
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )

    # If quantized or you want spreading across devices, let HF place layers:
    if args.device_map or quantization_config is not None:
        common_kwargs["device_map"] = args.device_map or "auto"
        if args.offload_folder:
            common_kwargs["offload_folder"] = args.offload_folder
    if quantization_config is not None:
        common_kwargs["quantization_config"] = quantization_config

    if cfg.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **common_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **common_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    # Only push to CUDA manually if you did NOT set device_map
    if not args.device_map and quantization_config is None and torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    if args.interactive:
        while True:
            example = input(f'Enter a {args.task} example: ')
            if args.task == 'csqa':
                splits = example.split(' -- ')
                query, cands = splits[0], splits[1:]
                item = {'query': query, 'cands': cands}
                process_item(args, tokenizer, model, item)
                print(item['pred'], item['probs'])
            elif args.task == 'csqa2':
                item = {'query': example}
                process_item(args, tokenizer, model, item)
                print(item['pred'], item['probs'])
            else:
                raise Exception(f'Interactive mode not implemented for {args.task}')

    with open(args.input_path) as f:
        ds = json.load(f)
        if args.n is not None:
            ds = ds[:args.n]

    pbar = tqdm(ds)
    num, den = 0, 0
    for idx, item in enumerate(pbar, start=1):
        item['id'] = str(idx)
        process_item(args, tokenizer, model, item)
        if 'ok' in item:
            num += item['ok']
            den += 1
            pbar.set_postfix({'acc': num / den})
    print('Total number of questions with empty knowledge: ', empty_counter)

    # --- dataset-level summaries (unchanged) ---
    import numpy as np, math
    ys, pred_confs, true_probs, margins, ents = [], [], [], [], []
    other_hits = []
    mode_tag = (ds[0].get('diag_mode') if ds and isinstance(ds[0], dict) else None)

    gold_strong_all = gold_strong_unique = 0
    gold_weak_all   = gold_weak_unique   = 0
    has_counts = 0

    strict_ok = []
    strict_seen = 0
    strict_excluded = 0

    def _mean(x): return float(np.mean(x)) if x else float('nan')

    for it in ds:
        if 'ok' in it:           ys.append(it['ok'])
        if 'pred_conf' in it:    pred_confs.append(it['pred_conf'])
        if 'true_conf' in it:    true_probs.append(it['true_conf'])
        if 'conf_margin' in it:  margins.append(it['conf_margin'])
        if 'conf_entropy' in it: ents.append(it['conf_entropy'])

        counts = it.get('diag_path_counts', None)
        cands  = it.get('cands', None)
        ans    = it.get('answer', None)

        gold_idx = None
        if isinstance(counts, list) and counts and isinstance(cands, list) and ans is not None and len(counts) == len(cands):
            try:
                gold_idx = next(i for i, t in enumerate(cands) if str(t).strip().lower() == str(ans).strip().lower())
            except StopIteration:
                gold_idx = None

        if gold_idx is not None:
            has_counts += 1
            max_val = max(counts); min_val = min(counts)
            is_max        = (counts[gold_idx] == max_val)
            is_min        = (counts[gold_idx] == min_val)
            is_strict_max = is_max and (counts.count(max_val) == 1)
            is_strict_min = is_min and (counts.count(min_val) == 1)

            if is_max:
                gold_strong_all += 1
                if is_strict_max: gold_strong_unique += 1
            if is_min:
                gold_weak_all += 1
                if is_strict_min: gold_weak_unique += 1

            if mode_tag in ('gold_plus_strongest', 'gold_plus_weakest'):
                strict_seen += 1
                if (mode_tag == 'gold_plus_strongest' and is_max) or \
                   (mode_tag == 'gold_plus_weakest'   and is_min):
                    strict_excluded += 1
                else:
                    if 'ok' in it:
                        strict_ok.append(it['ok'])

        if mode_tag in ('gold_plus_strongest', 'gold_plus_weakest'):
            try:
                if cands is None or ans is None:
                    cands = it['cands']; ans = it['answer']
                pred = it.get('pred', None)
                if gold_idx is None:
                    gold_idx = next(i for i, t in enumerate(cands) if str(t).strip().lower() == str(ans).strip().lower())

                counts_here = it.get('diag_path_counts', None)
                if isinstance(counts_here, list) and len(counts_here) == len(cands):
                    distr = [j for j in range(len(cands)) if j != gold_idx]
                    if mode_tag == 'gold_plus_strongest':
                        target_val = max(counts_here[j] for j in distr)
                        cand_others = [j for j in distr if counts_here[j] == target_val]
                    else:
                        target_val = min(counts_here[j] for j in distr)
                        cand_others = [j for j in distr if counts_here[j] == target_val]
                    other_idx = min(cand_others) if cand_others else distr[0]
                else:
                    other_idx = 0 if gold_idx != 0 else 1

                other_hits.append(1 if pred is not None and pred == cands[other_idx] else 0)
            except Exception:
                pass

    print("=== Summary ===")
    if mode_tag in ('gold_plus_strongest', 'gold_plus_weakest'):
        print(f"Accuracy (gold):     {_mean(ys):.4f} over {len(ys)} items")
        print(f"Accuracy (other):    {_mean(other_hits):.4f} over {len(other_hits)} items")
    else:
        print(f"Accuracy:            {_mean(ys):.4f} over {len(ys)} items")

    if mode_tag == 'gold_plus_strongest' and strict_seen > 0:
        print(f"Strict Accuracy (exclude strongest): "
              f"{_mean(strict_ok):.4f} over {len(strict_ok)} items "
              f"(excluded {strict_excluded} of {strict_seen})")
    elif mode_tag == 'gold_plus_weakest' and strict_seen > 0:
        print(f"Strict Accuracy (exclude weakest):   "
              f"{_mean(strict_ok):.4f} over {len(strict_ok)} items "
              f"(excluded {strict_excluded} of {strict_seen})")

    print(f"Avg pred confidence: {_mean(pred_confs):.4f}")
    print(f"Avg gold confidence: {_mean(true_probs):.4f}")
    print(f"Avg margin:          {_mean(margins):.4f}")
    print(f"Avg entropy:         {_mean(ents):.4f} (max ~{math.log(len(ds[0]['cands'])) if ds and 'cands' in ds[0] else 'log(#choices)'} )")

    if has_counts > 0:
        if mode_tag == 'gold_plus_weakest':
            print("--- In gold_plus_weakest mode ---")
            print(f"Gold weakest (incl. ties):   {gold_weak_all} / {has_counts}")
            print(f"Gold strictly weakest:       {gold_weak_unique} / {has_counts}")
        elif mode_tag == 'gold_plus_strongest':
            print("--- In gold_plus_strongest mode ---")
            print(f"Gold strongest (incl. ties): {gold_strong_all} / {has_counts}")
            print(f"Gold strictly strongest:     {gold_strong_unique} / {has_counts}")

    with open(args.output_path, 'w') as f:
        json.dump(ds, f, indent=4)


if __name__ == '__main__':
    start_time = time.time()
    with torch.no_grad():
        main()
    end_time = time.time()
    print("Overall time: ", end_time - start_time)
