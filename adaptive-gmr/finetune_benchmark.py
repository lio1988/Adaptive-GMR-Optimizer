#!/usr/bin/env python3
"""
finetune_benchmark.py

Publication-quality fine-tuning benchmark: DistilBERT on AG News.

Compares AdamW, AdamW+Clip, AdaptiveGMRAdamW, AdaptiveGMRAdamW+Clip
under identical hyperparameters and seeds.

Usage:
    python finetune_benchmark.py

Dependencies:
    pip install torch transformers datasets scipy pandas matplotlib tqdm scikit-learn
"""

from __future__ import annotations

import gc
import math
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adaptive_gmr import AdaptiveGMRAdamW  # noqa: E402

# ---------------------------------------------------------------------------
# Shared hyperparameters — identical for every optimizer
# ---------------------------------------------------------------------------
SEEDS: List[int] = [0, 1, 2, 3, 4]
OPTIMIZERS = ("AdamW", "AdamW+Clip", "AdaptiveGMRAdamW", "AdaptiveGMRAdamW+Clip")

EPOCHS = 3
BATCH_SIZE = 16
MAX_SEQ_LEN = 128
LR = 2e-5
WARMUP_STEPS = 50
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 0.01
GMR_ALPHA = 1.0
CLIP_VALUE = 1.0
NUM_LABELS = 4

OUTPUT_DIR = Path(__file__).resolve().parent / "finetune_output"
FIGURES_DIR = OUTPUT_DIR / "figures"

OPTIMIZER_COLORS = {
    "AdamW": "#E74C3C",
    "AdamW+Clip": "#C0392B",
    "AdaptiveGMRAdamW": "#27AE60",
    "AdaptiveGMRAdamW+Clip": "#1E8449",
}


@dataclass(frozen=True)
class RunConfig:
    optimizer_name: str
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.optimizer_name}_seed{self.seed}"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def uses_clip(name: str) -> bool:
    return name.endswith("+Clip")


def base_optimizer_name(name: str) -> str:
    return name.replace("+Clip", "")


def make_optimizer(name: str, model: nn.Module):
    base = base_optimizer_name(name)
    if base == "AdamW":
        return AdamW(model.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY)
    if base == "AdaptiveGMRAdamW":
        return AdaptiveGMRAdamW(
            model.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY, alpha=GMR_ALPHA
        )
    raise ValueError(name)


def make_scheduler(optimizer, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < WARMUP_STEPS:
            return float(step + 1) / float(max(1, WARMUP_STEPS))
        progress = (step - WARMUP_STEPS) / float(max(1, total_steps - WARMUP_STEPS))
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def build_ag_news_loaders(tokenizer: DistilBertTokenizer, device: torch.device):
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LEN,
        )

    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ds["test"], batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


@torch.no_grad()
def global_grad_norm(model: nn.Module) -> float:
    return math.sqrt(
        sum(p.grad.detach().pow(2).sum().item() for p in model.parameters() if p.grad is not None)
    )


@torch.no_grad()
def global_parameter_norm(model: nn.Module) -> float:
    return math.sqrt(sum(p.data.pow(2).sum().item() for p in model.parameters()))


@torch.no_grad()
def snapshot_parameters(model: nn.Module) -> List[torch.Tensor]:
    return [p.data.clone() for p in model.parameters()]


@torch.no_grad()
def compute_update_norm(before: List[torch.Tensor], model: nn.Module) -> float:
    sq = 0.0
    for prev, p in zip(before, model.parameters()):
        diff = p.data - prev
        sq += diff.pow(2).sum().item()
    return math.sqrt(sq)


def fresh_model(device: torch.device, seed: int) -> DistilBertForSequenceClassification:
    set_global_seed(seed)
    return DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=NUM_LABELS
    ).to(device)


@torch.no_grad()
def evaluate(model, val_loader, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_labels = [], []

    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        if not torch.isfinite(loss):
            return {
                "val_loss": float("nan"),
                "accuracy": float("nan"),
                "f1": float("nan"),
            }
        total_loss += loss.item()
        n_batches += 1
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["labels"].cpu().numpy())

    val_loss = total_loss / max(n_batches, 1)
    return {
        "val_loss": val_loss,
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1": float(f1_score(all_labels, all_preds, average="macro")),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run_experiment(
    config: RunConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[List[dict], dict]:
    set_global_seed(config.seed)
    model = fresh_model(device, config.seed)
    optimizer = make_optimizer(config.optimizer_name, model)
    total_steps = EPOCHS * len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    records: List[dict] = []
    global_step = 0
    run_start = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                break
            loss.backward()

            grad_norm = global_grad_norm(model)
            param_norm = global_parameter_norm(model)
            params_before = snapshot_parameters(model)

            if uses_clip(config.optimizer_name):
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_VALUE)

            t0 = time.perf_counter()
            optimizer.step()
            step_ms = (time.perf_counter() - t0) * 1000.0
            scheduler.step()
            update_norm = compute_update_norm(params_before, model)
            global_step += 1

            metrics = evaluate(model, val_loader, device) if global_step % max(1, len(train_loader)) == 0 else {}
            records.append(
                {
                    "run_id": config.run_id,
                    "optimizer": config.optimizer_name,
                    "seed": config.seed,
                    "epoch": epoch + 1,
                    "step": global_step,
                    "train_loss": loss.item(),
                    "val_loss": metrics.get("val_loss", float("nan")),
                    "accuracy": metrics.get("accuracy", float("nan")),
                    "f1": metrics.get("f1", float("nan")),
                    "grad_norm": grad_norm,
                    "parameter_norm": param_norm,
                    "update_norm": update_norm,
                    "step_time_ms": step_ms,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )

    total_time = time.perf_counter() - run_start
    final_eval = evaluate(model, val_loader, device)
    peak_gpu = float("nan")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_gpu = torch.cuda.max_memory_allocated(device) / (1024**3)

    def _mean(key: str) -> float:
        vals = [r[key] for r in records if math.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "run_id": config.run_id,
        "optimizer": config.optimizer_name,
        "seed": config.seed,
        "final_train_loss": records[-1]["train_loss"] if records else float("nan"),
        "final_val_loss": final_eval["val_loss"],
        "final_accuracy": final_eval["accuracy"],
        "final_f1": final_eval["f1"],
        "mean_grad_norm": _mean("grad_norm"),
        "mean_parameter_norm": _mean("parameter_norm"),
        "mean_update_norm": _mean("update_norm"),
        "total_runtime_s": total_time,
        "peak_gpu_mem_gb": peak_gpu,
        "steps_completed": len(records),
    }

    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return records, summary


# ---------------------------------------------------------------------------
# Statistics & plotting
# ---------------------------------------------------------------------------
def welch_ci(values: np.ndarray) -> Tuple[float, float, float]:
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean, mean
    se = stats.sem(values, ddof=1)
    t = stats.t.ppf(0.975, len(values) - 1)
    return mean, mean - t * se, mean + t * se


def statistical_tests(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["final_val_loss", "final_accuracy", "final_f1", "mean_grad_norm", "mean_update_norm", "total_runtime_s"]
    pairs = [("AdamW", "AdaptiveGMRAdamW"), ("AdamW+Clip", "AdaptiveGMRAdamW+Clip")]

    for opt_a, opt_b in pairs:
        a_df = summary_df[summary_df["optimizer"] == opt_a]
        b_df = summary_df[summary_df["optimizer"] == opt_b]
        for metric in metrics:
            a = a_df[metric].dropna().to_numpy(float)
            b = b_df[metric].dropna().to_numpy(float)
            if len(a) < 2 or len(b) < 2:
                continue
            t, p = stats.ttest_ind(a, b, equal_var=False)
            a_m, a_lo, a_hi = welch_ci(a)
            b_m, b_lo, b_hi = welch_ci(b)
            lower_better = "loss" in metric or "time" in metric or "norm" in metric
            diff = b_m - a_m
            rows.append(
                {
                    "comparison": f"{opt_b} vs {opt_a}",
                    "metric": metric,
                    "baseline_mean": a_m,
                    "baseline_ci_low": a_lo,
                    "baseline_ci_high": a_hi,
                    "candidate_mean": b_m,
                    "candidate_ci_low": b_lo,
                    "candidate_ci_high": b_hi,
                    "mean_diff": diff,
                    "welch_t": t,
                    "p_value": p,
                    "significant_95": p < 0.05,
                    "candidate_better": (diff < 0) if lower_better else (diff > 0),
                }
            )
    return pd.DataFrame(rows)


def plot_metric(results_df: pd.DataFrame, metric: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for opt in OPTIMIZERS:
        sub = results_df[results_df["optimizer"] == opt]
        curve = sub.groupby("step")[metric].agg(["mean", "std"]).reset_index()
        if curve.empty:
            continue
        color = OPTIMIZER_COLORS[opt]
        ax.plot(curve["step"], curve["mean"], label=opt, color=color, linewidth=2)
        ax.fill_between(
            curve["step"],
            curve["mean"] - curve["std"].fillna(0),
            curve["mean"] + curve["std"].fillna(0),
            color=color,
            alpha=0.15,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} — DistilBERT on AG News (mean ± std over seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bars(summary_df: pd.DataFrame) -> None:
    metrics = [("final_accuracy", "Accuracy"), ("final_f1", "Macro F1"), ("final_val_loss", "Val Loss")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (col, title) in zip(axes, metrics):
        means = [summary_df[summary_df["optimizer"] == o][col].mean() for o in OPTIMIZERS]
        stds = [summary_df[summary_df["optimizer"] == o][col].std(ddof=1) for o in OPTIMIZERS]
        ax.bar(OPTIMIZERS, means, yerr=stds, capsize=4, color=[OPTIMIZER_COLORS[o] for o in OPTIMIZERS])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.3)
    plt.suptitle("Final Metrics — DistilBERT / AG News", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "summary_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_conclusions(summary_df: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("FINETUNE BENCHMARK CONCLUSIONS (unbiased)")
    print("=" * 70)
    for opt in OPTIMIZERS:
        sub = summary_df[summary_df["optimizer"] == opt]
        print(
            f"- {opt}: acc={sub['final_accuracy'].mean():.4f}  "
            f"f1={sub['final_f1'].mean():.4f}  "
            f"val_loss={sub['final_val_loss'].mean():.4f}  "
            f"runtime={sub['total_runtime_s'].mean():.1f}s"
        )
    if not stats_df.empty:
        sig = stats_df[(stats_df["significant_95"]) & (stats_df["metric"].isin(["final_accuracy", "final_f1"]))]
        if sig.empty:
            print("- No statistically significant accuracy/F1 differences (Welch, p<0.05).")
        for _, row in sig.iterrows():
            better = "better" if row["candidate_better"] else "worse"
            print(f"- {row['comparison']} on {row['metric']}: candidate {better} (p={row['p_value']:.4f}).")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = [RunConfig(opt, seed) for seed in SEEDS for opt in OPTIMIZERS]
    print(f"Device: {device} | runs={len(configs)} | optimizers={OPTIMIZERS}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    print("Loading AG News ...")
    train_loader, val_loader = build_ag_news_loaders(tokenizer, device)

    all_records, all_summaries = [], []
    for cfg in tqdm(configs, desc="Fine-tune runs"):
        recs, summ = run_experiment(cfg, train_loader, val_loader, device)
        all_records.extend(recs)
        all_summaries.append(summ)

    results_df = pd.DataFrame(all_records)
    summary_df = pd.DataFrame(all_summaries)
    stats_df = statistical_tests(summary_df)

    results_df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    stats_df.to_csv(OUTPUT_DIR / "statistical_tests.csv", index=False)

    for metric, label, fname in [
        ("train_loss", "Training Loss", "training_loss.png"),
        ("val_loss", "Validation Loss", "validation_loss.png"),
        ("accuracy", "Accuracy", "accuracy.png"),
        ("f1", "Macro F1", "f1.png"),
        ("grad_norm", "Gradient Norm", "gradient_norm.png"),
        ("parameter_norm", "Parameter Norm", "parameter_norm.png"),
        ("update_norm", "Update Norm", "update_norm.png"),
    ]:
        plot_metric(results_df, metric, label, fname)

    plot_summary_bars(summary_df)
    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print_conclusions(summary_df, stats_df)


if __name__ == "__main__":
    main()
