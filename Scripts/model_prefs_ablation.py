#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model_prefs_ablation.py
-----------------------
Score binomial ordering preferences across checkpoints for:
  - Ablated models (znhoughton/opt-babylm-{size}-ablated-20eps-seed964)
  - Finetuned models (znhoughton/opt-babylm-{size}-ablated-finetuned-20eps-seed964)

Output per checkpoint: one CSV in Data/model_pref_results/ with columns:
  model, model_type, checkpoint, step, tokens,
  binom_alpha,           ← binomial with words in alphabetical order
  total_training_occurrences,  ← 0 for ablated; 2×bin for finetuned
  relfreq,               ← NA for ablated; 0.5 for finetuned
  prompt, binom,
  alpha_logprob, nonalpha_logprob, preference

Resume-safe: skips checkpoints whose CSV already contains all (prompt, binom) pairs.
Multi-GPU: shards checkpoints across GPUs (one process per GPU).
"""

import os
import csv
import traceback
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import shutil

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUT_DIR      = str(PROJECT_ROOT / "Data" / "model_pref_results")
BINOMS_CSV   = str(PROJECT_ROOT / "Data" / "novel_binomials_curated.csv")
FREQ_LOG_CSV = str(PROJECT_ROOT / "Data" / "frequency_log.csv")

# Cache globals (populated once per worker process)
_BINOMS_DF  = None
_FREQ_INDEX = None   # (word1, word2) -> {"bin": int, "overall_freq": int}

# Enable compile on Linux only (Inductor backend not stable on Windows).
ENABLE_COMPILE    = True
USE_TORCH_COMPILE = ENABLE_COMPILE and (os.name != "nt")
COMPILE_MODE      = "reduce-overhead"

# ── Prompts ───────────────────────────────────────────────────────────────────
LIST_OF_PROMPTS = [
    " ",
    "Well, ",
    "So, ",
    "Then ",
    "Possibly ",
    "Or even ",
    "Maybe a ",
    "Perhaps a ",
    "At times ",
    "Suddenly, the ",
    "Honestly just ",
    "Especially the ",
    "For instance ",
    "In some cases ",
    "Every now and then ",
    "Occasionally you'll find ",
    "There can be examples like ",
    "You might notice things like ",
    "People sometimes mention ",
    "Sometimes just ",
    "Nothing specific comes to mind except the ",
    "It reminded me loosely of the ",
    "There was a vague reference to the ",
    "Unexpectedly the ",
    "It's easy to overlook the ",
    "There used to be talk of ",
    "Out in the distance was the ",
    "What puzzled everyone was the ",
    "At some point I overheard ",
    "Without warning came ",
    "A friend once described the ",
    "The scene shifted toward ",
    "Nobody expected to hear about ",
    "Things eventually turned toward ",
    "The conversation eventually returned to ",
    "I only remember a hint of the ",
    "I couldn't quite place the ",
    "It somehow led back to the ",
    "What stood out most was the ",
    "The oddest part involved the ",
    "Later on, people were discussing ",
    "There was this fleeting idea about ",
    "I once heard someone bring up the ",
    "There was a moment involving the ",
    "It all started when we noticed the ",
    "Another example floated around concerning the ",
    "I came across something about the ",
    "A situation arose involving the ",
    "The conversation drifted toward the ",
    "At one point we ended up discussing ",
    "Out of nowhere came a mention of the ",
]

# ── Model configs ─────────────────────────────────────────────────────────────
#
# checkpoint_source values:
#   "step_tags"  — HF tags named "step-{N}" (ablated models, saved every 20M tokens)
#   "final"      — no intermediate checkpoints; load the head of main branch once
#
# tokens_per_step  = block_size × per_device_batch × grad_accum × n_gpus
#   125m  ablated: 1024 × 400 × 1 × 2 = 819,200
#   350m  ablated: 1024 × 200 × 1 × 2 = 409,600
#   1.3b  ablated: 1024 × 250 × 1 × 4 = 1,024,000
#   (finetuned use the same batch sizes but only one checkpoint is saved)

MODEL_CONFIGS = {
    # ── Ablated models ────────────────────────────────────────────────────────
    "znhoughton/opt-babylm-125m-ablated-20eps-seed964": {
        "tokens_per_step":    819_200,
        "tokenizer":          "znhoughton/opt-babylm-125m-ablated-20eps-seed964",
        "checkpoint_source":  "step_tags",
        "model_type":         "ablated",
        "log_sample":         False,
    },
    "znhoughton/opt-babylm-350m-ablated-20eps-seed964": {
        "tokens_per_step":    409_600,
        "tokenizer":          "znhoughton/opt-babylm-350m-ablated-20eps-seed964",
        "checkpoint_source":  "step_tags",
        "model_type":         "ablated",
        "log_sample":         False,
    },
    "znhoughton/opt-babylm-1.3b-ablated-20eps-seed964": {
        "tokens_per_step":    1_024_000,
        "tokenizer":          "znhoughton/opt-babylm-1.3b-ablated-20eps-seed964",
        "checkpoint_source":  "step_tags",
        "model_type":         "ablated",
        "log_sample":         False,
    },
    # ── Finetuned models (hub_strategy=end → single final checkpoint) ─────────
    "znhoughton/opt-babylm-125m-ablated-finetuned-20eps-seed964": {
        "tokens_per_step":    819_200,  # same batch as ablated 125m
        "tokenizer":          "znhoughton/opt-babylm-125m-ablated-20eps-seed964",
        "checkpoint_source":  "final",
        "model_type":         "finetuned",
        "log_sample":         False,
    },
    "znhoughton/opt-babylm-350m-ablated-finetuned-20eps-seed964": {
        "tokens_per_step":    409_600,
        "tokenizer":          "znhoughton/opt-babylm-350m-ablated-20eps-seed964",
        "checkpoint_source":  "final",
        "model_type":         "finetuned",
        "log_sample":         False,
    },
    "znhoughton/opt-babylm-1.3b-ablated-finetuned-20eps-seed964": {
        "tokens_per_step":    204_800,  # 1024 × 100 × 1 × 2 (finetune_ablated.sh: batch=100, 2 GPUs)
        "tokenizer":          "znhoughton/opt-babylm-1.3b-ablated-20eps-seed964",
        "checkpoint_source":  "final",
        "model_type":         "finetuned",
        "log_sample":         False,
    },
}


# =========================
# HELPERS
# =========================

def detect_num_gpus() -> int:
    if not torch.cuda.is_available():
        return 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible is None or visible.strip() == "":
        return torch.cuda.device_count()
    return len([v for v in visible.split(",") if v.strip() != ""])


def load_freq_index() -> Dict:
    """Load frequency_log.csv into a (word1, word2) → row dict."""
    idx = {}
    try:
        df = pd.read_csv(FREQ_LOG_CSV)
        for row in df.itertuples(index=False):
            key = (row.word1.strip().lower(), row.word2.strip().lower())
            idx[key] = {"bin": row.bin, "overall_freq": row.overall_freq}
    except FileNotFoundError:
        print(f"⚠️  {FREQ_LOG_CSV} not found — total_training_occurrences will be NA for finetuned models")
    return idx


def get_model_checkpoints(repo_id: str, tokens_per_step: int) -> List[Dict[str, Any]]:
    """Fetch step-* tags from HuggingFace and build checkpoint list."""
    api = HfApi()
    refs = api.list_repo_refs(repo_id)

    checkpoints = []
    for tag in refs.tags:
        if not tag.name.startswith("step-"):
            continue
        try:
            step = int(tag.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append({
            "checkpoint": tag.name,
            "tag":        tag.name,
            "step":       step,
            "tokens":     step * tokens_per_step,
        })

    checkpoints.sort(key=lambda x: x["step"])

    if checkpoints:
        print(
            f"📦 {repo_id}: found {len(checkpoints)} step-tag checkpoints "
            f"(steps {checkpoints[0]['step']} → {checkpoints[-1]['step']})"
        )
    else:
        print(f"⚠️  No step-* tags found for {repo_id}")

    return checkpoints


def get_final_checkpoint() -> List[Dict[str, Any]]:
    """For finetuned models: a single pseudo-checkpoint at the model's head."""
    return [{
        "checkpoint": "final",
        "tag":        None,   # None → HF will use the default (main) branch
        "step":       0,
        "tokens":     0,
    }]


def log_sample_checkpoints(checkpoints: List[Dict[str, Any]], n: int = 20) -> List[Dict[str, Any]]:
    """Log-uniformly sample n checkpoints by index."""
    total = len(checkpoints)
    if total <= n:
        return checkpoints
    indices = sorted(set(
        min(int(round(np.exp(x))) - 1, total - 1)
        for x in np.linspace(np.log(1), np.log(total), n)
    ))
    sampled = [checkpoints[i] for i in indices]
    print(f"  📊 Log-sampled {len(sampled)}/{total} checkpoints")
    return sampled


def check_prompts_in_file(filepath: str, expected_prompts: List[str]) -> Dict[str, Any]:
    """Return which (prompt, binom) pairs are missing from an existing result CSV."""
    binoms_df = pd.read_csv(BINOMS_CSV)
    expected_binoms = set(f"{r.word1} and {r.word2}" for r in binoms_df.itertuples())

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"    ⚠️ Error reading {filepath}: {e}")
        return {"missing_pairs": None}  # signal: rerun everything

    missing_pairs = []
    for prompt in expected_prompts:
        prompt_df  = df[df["prompt"] == prompt]
        seen_binoms = set(prompt_df["binom"].unique())
        for binom in expected_binoms - seen_binoms:
            missing_pairs.append((prompt, binom))

    if missing_pairs:
        print(f"    ⚠️ Missing {len(missing_pairs)} (prompt, binom) pairs")
    else:
        print("    ✅ All prompts complete")

    return {"missing_pairs": missing_pairs}


@torch.inference_mode()
def to_tokens_and_logprobs(
    model,
    tokenizer,
    input_texts: List[str],
    device: str,
    batch_size: int = 256,
    desc: Optional[str] = None,
    leave: bool = False,
) -> List[float]:
    """
    Returns list of total sequence log-probabilities for each text.
    OOM-safe: halves batch size and retries.
    """
    all_logprobs: List[float] = []
    n = len(input_texts)
    total_batches = (n + batch_size - 1) // batch_size

    i = 0
    pbar = tqdm(total=total_batches, desc=desc, leave=leave)
    while i < n:
        batch_texts = input_texts[i:i + batch_size]
        try:
            enc = tokenizer(
                batch_texts,
                padding="longest",
                return_tensors="pt",
                pad_to_multiple_of=8,
            )
            first_device = next(model.parameters()).device
            input_ids      = enc["input_ids"].to(first_device)
            attention_mask = enc["attention_mask"].to(first_device)

            outputs  = model(input_ids, attention_mask=attention_mask)
            logits   = outputs.logits
            logprobs = torch.log_softmax(logits, dim=-1)

            logprobs    = logprobs[:, :-1, :]
            target_ids  = input_ids[:, 1:]
            target_mask = attention_mask[:, 1:]

            token_logprobs = logprobs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            token_logprobs = token_logprobs * target_mask

            all_logprobs.extend(token_logprobs.sum(dim=-1).tolist())
            i += batch_size
            pbar.update(1)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if batch_size == 1:
                    raise RuntimeError("OOM even at batch_size=1") from e
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                pbar.set_description(f"{desc} (OOM -> bs={batch_size})")
            else:
                raise

    pbar.close()
    return all_logprobs


def pick_start_batch_size(model_name: str) -> int:
    name = model_name.lower()
    if "1.3b" in name or "1b" in name:
        return 1024
    if "350m" in name:
        return 2048
    return 4096   # 125m and smaller


def get_model_prefs(
    prompt: str,
    model_name: str,
    checkpoint_info: Dict[str, Any],
    model_type: str,
    tokenizer,
    model,
    device: str,
) -> pd.DataFrame:
    global _BINOMS_DF, _FREQ_INDEX
    if _BINOMS_DF is None:
        _BINOMS_DF = pd.read_csv(BINOMS_CSV)
    if _FREQ_INDEX is None:
        _FREQ_INDEX = load_freq_index()

    df = _BINOMS_DF.copy()

    df["AandB"] = prompt + df["word1"] + " and " + df["word2"]
    df["BandA"] = prompt + df["word2"] + " and " + df["word1"]

    combined_texts = df["AandB"].tolist() + df["BandA"].tolist()
    start_bs       = pick_start_batch_size(model_name)

    scores = to_tokens_and_logprobs(
        model=model,
        tokenizer=tokenizer,
        input_texts=combined_texts,
        device=device,
        batch_size=start_bs,
        desc="      scoring (AandB+BandA)",
        leave=False,
    )

    n = len(df)
    alpha_scores    = scores[:n]
    nonalpha_scores = scores[n:]

    rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        w1, w2 = row.word1.strip().lower(), row.word2.strip().lower()

        # Alphabetical form for easy identification
        binom_alpha = " and ".join(sorted([w1, w2]))

        # Frequency info from log
        freq_key = (w1, w2)
        freq_entry = _FREQ_INDEX.get(freq_key) or _FREQ_INDEX.get((w2, w1))
        if model_type == "finetuned" and freq_entry:
            total_training_occ = freq_entry["overall_freq"]
            relfreq            = 0.5
        else:
            total_training_occ = 0
            relfreq            = None   # NA for ablated

        rows.append({
            "model":                      model_name,
            "model_type":                 model_type,
            "checkpoint":                 checkpoint_info["checkpoint"],
            "step":                       checkpoint_info["step"],
            "tokens":                     checkpoint_info["tokens"],
            "binom_alpha":                binom_alpha,
            "total_training_occurrences": total_training_occ,
            "relfreq":                    relfreq,
            "prompt":                     prompt,
            "binom":                      f"{w1} and {w2}",
            "alpha_logprob":              alpha_scores[i],
            "nonalpha_logprob":           nonalpha_scores[i],
            "preference":                 alpha_scores[i] - nonalpha_scores[i],
        })

    return pd.DataFrame(rows)


def atomic_write_csv(df: pd.DataFrame, out_path: str) -> None:
    tmp_path = out_path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)


# =========================
# WORKER
# =========================

@dataclass
class WorkItem:
    model_name:     str
    tokenizer_id:   str
    tokens_per_step: int
    model_type:     str
    checkpoint:     Dict[str, Any]


def run_work_item(item: WorkItem, device: str, out_dir: str) -> None:
    model_name  = item.model_name
    model_type  = item.model_type
    ckpt        = item.checkpoint
    eval_id     = f"{model_name.split('/')[-1]}_{ckpt['checkpoint']}"
    out_path    = os.path.join(out_dir, f"{eval_id}.csv")

    # ── Early-exit: skip if already complete ──────────────────────────────────
    if os.path.exists(out_path):
        print(f"\n📄 Found existing file: {eval_id}")
        check_result = check_prompts_in_file(out_path, LIST_OF_PROMPTS)
        if check_result["missing_pairs"] is not None and not check_result["missing_pairs"]:
            print("  ✅ All prompts complete, skipping")
            return

    # ── Load tokenizer ────────────────────────────────────────────────────────
    try:
        tokenizer = AutoTokenizer.from_pretrained(item.tokenizer_id, use_fast=False)
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(item.tokenizer_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Determine which prompts still need running ─────────────────────────────
    if os.path.exists(out_path):
        check_result = check_prompts_in_file(out_path, LIST_OF_PROMPTS)
        if check_result["missing_pairs"] is None:
            prompts_to_run = LIST_OF_PROMPTS
            existing_df    = None
        else:
            missing_pairs  = check_result["missing_pairs"]
            prompts_to_run = sorted(set(p for p, _ in missing_pairs))
            existing_df    = pd.read_csv(out_path)
    else:
        prompts_to_run = LIST_OF_PROMPTS
        existing_df    = None

    tokens_str = f"{ckpt['tokens']:,}" if ckpt['tokens'] else "final"
    print(f"\n🚀 Evaluating {ckpt['checkpoint']} ({tokens_str} tokens) [{model_type}] — {len(prompts_to_run)} prompts")
    print("  📥 Loading model...")

    tmp_cache = tempfile.mkdtemp(prefix="hf_ckpt_")
    model     = None
    try:
        load_kwargs = dict(
            low_cpu_mem_usage=True,
            cache_dir=tmp_cache,
        )
        if ckpt["tag"] is not None:
            load_kwargs["revision"] = ckpt["tag"]

        if device == "auto":
            load_kwargs["device_map"] = "auto"
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.float16 if device != "cpu" else torch.float32
            load_kwargs["device_map"]  = device

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs).eval()

        if USE_TORCH_COMPILE:
            try:
                model = torch.compile(model, mode=COMPILE_MODE)
                print("  ⚡ Model compiled with torch.compile")
            except Exception as e:
                print(f"  ⚠️ torch.compile failed (continuing uncompiled): {e}")

        dfs = []
        print("  🔄 Processing prompts...")
        for prompt_idx, prompt in enumerate(
            tqdm(prompts_to_run, desc="  prompts", leave=False), 1
        ):
            df_result = get_model_prefs(
                prompt, model_name, ckpt, model_type, tokenizer, model, device
            )
            dfs.append(df_result)
            print(f"    ✅ Completed prompt {prompt_idx}/{len(prompts_to_run)}")

        new_df = pd.concat(dfs, ignore_index=True)

        if existing_df is not None:
            existing_keys = set(zip(existing_df["prompt"], existing_df["binom"]))
            new_df = new_df[
                ~new_df.apply(
                    lambda r: (r["prompt"], r["binom"]) in existing_keys, axis=1
                )
            ]

        final_df = (
            pd.concat([existing_df, new_df], ignore_index=True)
            if existing_df is not None else new_df
        )

        print("  💾 Saving results...")
        atomic_write_csv(final_df, out_path)
        print(f"  ✅ Saved {len(final_df)} total rows ({len(new_df)} new) to {out_path}")

    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()
        shutil.rmtree(tmp_cache, ignore_errors=True)


def worker_main(rank: int, items: List[WorkItem]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(0)
    device = "cuda:0"

    os.makedirs(OUT_DIR, exist_ok=True)
    failures = []

    print(f"\n{'='*30}")
    print(f"🚀 Worker rank={rank} using GPU={rank}")
    print(f"{'='*30}\n")

    for item in items:
        try:
            print("=" * 60)
            print(f"🔍 [{rank}] {item.model_name} | {item.checkpoint['checkpoint']}")
            print("=" * 60)
            run_work_item(item, device=device, out_dir=OUT_DIR)
        except Exception as e:
            failures.append((item.model_name, item.checkpoint.get("checkpoint"), str(e)))
            print(f"❌ [{rank}] Failure: {e}")
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    print(f"\n🏁 Worker {rank} complete.")
    if failures:
        print(f"⚠️ Worker {rank} had {len(failures)} failures:")
        for m, c, err in failures:
            print(f" • {m} {c}: {err}")


# =========================
# DRIVER
# =========================

def build_work_items() -> List[WorkItem]:
    items: List[WorkItem] = []

    for model_name, config in MODEL_CONFIGS.items():
        print("=" * 60)
        print(f"🔍 Model: {model_name}  [{config['model_type']}]")
        print("=" * 60)

        source     = config.get("checkpoint_source", "step_tags")
        tps        = config["tokens_per_step"]
        model_type = config["model_type"]

        try:
            if source == "step_tags":
                checkpoints = get_model_checkpoints(model_name, tps)
            elif source == "final":
                checkpoints = get_final_checkpoint()
            else:
                print(f"⚠️  Unknown checkpoint_source '{source}' for {model_name}, skipping.")
                continue
        except Exception as e:
            print(f"🚨 Error fetching checkpoints for {model_name}: {e}")
            print("⛔ Skipping this model.\n")
            continue

        if not checkpoints:
            print(f"⚠️  No checkpoints found for {model_name}, skipping.\n")
            continue

        if config.get("log_sample", False):
            checkpoints = log_sample_checkpoints(checkpoints, n=20)

        for ckpt in checkpoints:
            items.append(WorkItem(
                model_name=model_name,
                tokenizer_id=config["tokenizer"],
                tokens_per_step=tps,
                model_type=model_type,
                checkpoint=ckpt,
            ))

    print(f"\n🧾 Total work items: {len(items)} checkpoints\n")
    return items


def shard_items(items: List[WorkItem], num_shards: int) -> List[List[WorkItem]]:
    return [items[i::num_shards] for i in range(num_shards)]


def main():
    print("🔧 Building work items (all checkpoints across models)...")
    items = build_work_items()
    if not items:
        print("No checkpoints to run. Exiting.")
        return

    num_gpus = detect_num_gpus()
    print(f"\n🖥️  Detected {num_gpus} GPU(s)")
    os.makedirs(OUT_DIR, exist_ok=True)

    if num_gpus >= 2:
        print(f"\n🚀 Sharding {len(items)} checkpoints across {num_gpus} GPUs")
        shards = shard_items(items, num_gpus)
        ctx    = mp.get_context("spawn")
        procs  = []
        for rank in range(num_gpus):
            p = ctx.Process(target=worker_main, args=(rank, shards[rank]))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    elif num_gpus == 1:
        print("\n🚀 Running on single GPU")
        for item in items:
            run_work_item(item, device="cuda:0", out_dir=OUT_DIR)
    else:
        print("\n⚠️  No GPU detected — running on CPU (slow)")
        for item in items:
            run_work_item(item, device="cpu", out_dir=OUT_DIR)

    print("\n🏁 RUN COMPLETE")
    print(f"  Results → {OUT_DIR}/")


if __name__ == "__main__":
    main()
