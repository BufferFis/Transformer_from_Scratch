import math
import re
from pathlib import Path
from typing import List, Tuple
import argparse
import sys
import traceback
from tqdm import tqdm

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from config import get_config
from model import build_transformer

# --------- Utilities ---------
def _find_latest_checkpoint(model_folder: str, model_basename: str) -> Path | None:
    folder = Path(model_folder)
    if not folder.exists():
        return None
    pattern = re.compile(rf"^{re.escape(model_basename)}(\d+)\.pt$")
    candidates: List[Tuple[int, Path]] = []
    for f in folder.iterdir():
        if f.is_file():
            m = pattern.match(f.name)
            if m:
                candidates.append((int(m.group(1)), f))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def _load_tokenizers(config):
    tok_src_path = Path(config["tokenizer_file"].format(config["lang_src"]))
    tok_tgt_path = Path(config["tokenizer_file"].format(config["lang_tgt"]))
    if not tok_src_path.exists() or not tok_tgt_path.exists():
        raise FileNotFoundError("Tokenizer files not found. Train first to build tokenizers.")
    tokenizer_src = Tokenizer.from_file(str(tok_src_path))
    tokenizer_tgt = Tokenizer.from_file(str(tok_tgt_path))
    return tokenizer_src, tokenizer_tgt

def _subsequent_mask(size: int, device=None) -> torch.Tensor:
    # lower-triangular causal mask (size, size) -> True for allowed positions
    m = torch.tril(torch.ones(size, size, device=device)).bool()
    return m

def _build_encoder_input(src_ids: List[int], seq_len: int, sos_id: int, eos_id: int, pad_id: int) -> torch.Tensor:
    # [SOS] src ... [EOS] PAD* -> length seq_len
    enc_len = len(src_ids) + 2
    if enc_len > seq_len:
        raise ValueError(f"Source too long ({enc_len}) for seq_len={seq_len}")
    pad_count = seq_len - enc_len
    enc = [sos_id] + src_ids + [eos_id] + [pad_id] * pad_count
    return torch.tensor(enc, dtype=torch.long)

def _make_src_mask(encoder_input: torch.Tensor, pad_id: int) -> torch.Tensor:
    # Expect encoder_input shape (1, S); produce (1, 1, 1, S)
    return (encoder_input != pad_id).unsqueeze(1).unsqueeze(2)

def _make_tgt_mask(tgt_input: torch.Tensor, pad_id: int) -> torch.Tensor:
    # Padding mask (1, 1, L) AND causal mask (L, L) -> broadcast to (1, L, L)
    pad_mask = (tgt_input != pad_id).unsqueeze(0).unsqueeze(0)  # (1,1,L)
    causal = _subsequent_mask(tgt_input.size(0), device=tgt_input.device)  # (L,L)
    return pad_mask & causal  # (1, L, L) via broadcasting

@torch.no_grad()
def greedy_decode(model, src: torch.Tensor, src_mask: torch.Tensor, max_len: int, sos_id: int, eos_id: int, pad_id: int, device):
    # src: (1, S), src_mask: (1,1,1,S)
    model.eval()
    memory = model.encode(src.to(device), src_mask.to(device))
    ys = torch.tensor([[sos_id]], dtype=torch.long, device=device)  # (1,1)
    for _ in range(max_len - 1):
        tgt_mask = _make_tgt_mask(ys[0], pad_id).to(device)  # (1, L, L)
        out = model.decode(memory, src_mask.to(device), ys, tgt_mask)
        logits = model.project(out)  # (1, L, V)
        next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)
        if next_token == eos_id:
            break
    return ys[0].tolist()

def _strip_special(ids: List[int], sos_id: int, eos_id: int, pad_id: int) -> List[int]:
    out = []
    for t in ids:
        if t in (sos_id, pad_id):
            continue
        if t == eos_id:
            break
        out.append(t)
    return out

# --------- Metrics ---------
def _tokenize(s: str) -> List[str]:
    return s.strip().split()

def _lcs(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]

def rouge_l_score(ref: str, hyp: str) -> float:
    ref_toks = _tokenize(ref)
    hyp_toks = _tokenize(hyp)
    if not ref_toks or not hyp_toks:
        return 0.0
    lcs = _lcs(ref_toks, hyp_toks)
    prec = lcs / max(len(hyp_toks), 1)
    rec = lcs / max(len(ref_toks), 1)
    if prec == 0 and rec == 0:
        return 0.0
    beta2 = 1.2 * 1.2  # typical for ROUGE-L
    return (1 + beta2) * prec * rec / (rec + beta2 * prec + 1e-12)

def compute_metrics(references: List[str], hypotheses: List[str]) -> dict:
    # BLEU
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        refs_tok = [[_tokenize(r)] for r in references]  # list of list of refs (tok)
        hyps_tok = [_tokenize(h) for h in hypotheses]
        bleu = corpus_bleu(refs_tok, hyps_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method3)
    except Exception:
        bleu = float("nan")

    # METEOR
    try:
        from nltk.translate.meteor_score import meteor_score
        meteor_vals = [meteor_score([r], h) for r, h in zip(references, hypotheses)]
        meteor = sum(meteor_vals) / max(len(meteor_vals), 1)
    except Exception:
        meteor = float("nan")

    # ROUGE-L
    rouge_vals = [rouge_l_score(r, h) for r, h in zip(references, hypotheses)]
    rouge_l = sum(rouge_vals) / max(len(rouge_vals), 1)

    return {"bleu": bleu, "meteor": meteor, "rouge_l": rouge_l}

# --------- End-to-end evaluation ---------
def evaluate(num_samples: int | None = 200):
    print("Starting evaluation...", flush=True)
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Tokenizers
    tokenizer_src, tokenizer_tgt = _load_tokenizers(config)
    print("Tokenizers loaded.", flush=True)
    src_pad = tokenizer_src.token_to_id("[PAD]")
    src_sos = tokenizer_src.token_to_id("[SOS]")
    src_eos = tokenizer_src.token_to_id("[EOS]")
    tgt_pad = tokenizer_tgt.token_to_id("[PAD]")
    tgt_sos = tokenizer_tgt.token_to_id("[SOS]")
    tgt_eos = tokenizer_tgt.token_to_id("[EOS]")

    # Model
    model = build_transformer(
        src_vocab_size=tokenizer_src.get_vocab_size(),
        tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
        src_seq_len=config["seq_len"],
        tgt_seq_len=config["seq_len"],
        d_model=config["d_model"],
    ).to(device)

    # Load latest checkpoint if available
    latest = _find_latest_checkpoint(config["model_folder"], config["model_basename"])
    if latest and latest.exists():
        state = torch.load(latest, map_location=device)
        sd = state.get("model_state_dict", state.get("state_dict", None))
        if sd is not None:
            model.load_state_dict(sd)
        print(f"Loaded checkpoint: {latest.name}", flush=True)
    else:
        print("No checkpoint found. Evaluating with randomly initialized model.", flush=True)

    # Data (validation split)
    ds = load_dataset(config["datasource"], f'{config["lang_src"]}-{config["lang_tgt"]}', split="train")
    total = len(ds)
    val_size = max(1, total - int(0.9 * total))
    val_ds = ds.select(range(total - val_size, total))
    print(f"Validation samples available: {val_size}", flush=True)

    refs, hyps = [], []
    limit = val_size if num_samples is None else min(num_samples, val_size)

    for i in tqdm(range(limit), desc="Evaluating", unit="sample"):
        item = val_ds[i]["translation"]
        src_text = item[config["lang_src"]]
        ref_text = item[config["lang_tgt"]]

        # Encode source
        src_ids = tokenizer_src.encode(src_text).ids
        enc = _build_encoder_input(src_ids, config["seq_len"], src_sos, src_eos, src_pad).unsqueeze(0)  # (1, S)
        src_mask = _make_src_mask(enc, src_pad)

        out_ids = greedy_decode(
            model=model,
            src=enc,
            src_mask=src_mask,
            max_len=config["seq_len"],
            sos_id=tgt_sos,
            eos_id=tgt_eos,
            pad_id=tgt_pad,
            device=device,
        )
        clean_ids = _strip_special(out_ids, tgt_sos, tgt_eos, tgt_pad)
        hyp_text = tokenizer_tgt.decode(clean_ids)

        refs.append(ref_text)
        hyps.append(hyp_text)

    metrics = compute_metrics(refs, hyps)
    print(f"Evaluated {len(hyps)} samples", flush=True)
    print(f"BLEU:     {metrics['bleu']:.4f}" if not math.isnan(metrics["bleu"]) else "BLEU:     unavailable (install nltk)", flush=True)
    print(f"METEOR:   {metrics['meteor']:.4f}" if not math.isnan(metrics["meteor"]) else "METEOR:   unavailable (install nltk)", flush=True)
    print(f"ROUGE-L:  {metrics['rouge_l']:.4f}", flush=True)
    return metrics

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--num-samples", type=int, default=200, help="Number of validation samples to evaluate")
        args = parser.parse_args()
        evaluate(num_samples=args.num_samples)
    except Exception as e:
        print(f"Evaluation failed: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()