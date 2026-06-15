#!/usr/bin/env python3
"""
llm_robustness_benchmark.py

Publication-quality GPT-2 (124M) robustness study on WikiText-2.

Compares AdamW, AdaptiveGMRAdamW, clipped variants, and Adafactor under
controlled perturbations across multiple training lengths.

Usage:
    python llm_robustness_benchmark.py

Dependencies:
    pip install torch transformers datasets scipy pandas matplotlib tqdm
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
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import Adafactor, GPT2LMHeadModel, GPT2Tokenizer

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adaptive_gmr import AdaptiveGMRAdamW  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment configuration — shared identically across all optimizers
# ---------------------------------------------------------------------------
SEEDS: List[int] = [0, 1, 2, 3, 4]
SPIKE_SCALES: List[int] = [100, 500, 1000, 5000]
PERTURBATION_MODES = ("baseline", "random_grad", "attention_spike", "sparse_spike")

# Original experiments (unchanged)
ORIGINAL_OPTIMIZERS = ("AdamW", "AdaptiveGMRAdamW")
# New optimizers added on top
ADDITIONAL_OPTIMIZERS = ("AdamW+Clip", "AdaptiveGMRAdamW+Clip", "Adafactor")
ALL_OPTIMIZERS = ORIGINAL_OPTIMIZERS + ADDITIONAL_OPTIMIZERS

TRAIN_STEP_OPTIONS: List[int] = [50, 300, 500]
EVAL_EVERY = 5
BATCH_SIZE = 4
SEQ_LEN = 128
LR = 5e-5
WARMUP_STEPS = 5
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 0.0
GMR_ALPHA = 1.0
SPARSE_FRACTION = 0.05
CLIP_VALUE = 1.0
GRAD_EXPLOSION_THRESHOLD = 1e6
PARAM_EXPLOSION_THRESHOLD = 1e4

OUTPUT_DIR = Path(__file__).resolve().parent / "robustness_output"
FIGURES_DIR = OUTPUT_DIR / "figures"


@dataclass(frozen=True)
class RunConfig:
    """Single reproducible experiment specification."""

    optimizer_name: str
    perturbation_mode: str
    spike_scale: int
    seed: int
    train_steps: int

    @property
    def spike_step(self) -> int:
        """Perturbation at midpoint — preserves legacy step 25 for 50-step runs."""
        return max(1, self.train_steps // 2)

    @property
    def run_id(self) -> str:
        scale_tag = "none" if self.perturbation_mode == "baseline" else str(self.spike_scale)
        return (
            f"{self.optimizer_name}_{self.perturbation_mode}"
            f"_s{scale_tag}_steps{self.train_steps}_seed{self.seed}"
        )


# ---------------------------------------------------------------------------
# Reproducibility & data
# ---------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_wikitext_loaders(tokenizer: GPT2Tokenizer) -> Tuple[DataLoader, DataLoader]:
    """Tokenize WikiText-2 and chunk into fixed-length blocks."""
    from datasets import load_dataset

    def tokenize_and_chunk(texts: List[str]) -> torch.Tensor:
        ids: List[int] = []
        for text in texts:
            if not text.strip():
                continue
            ids.extend(tokenizer.encode(text, add_special_tokens=False))
        n_chunks = len(ids) // SEQ_LEN
        if n_chunks == 0:
            raise RuntimeError("WikiText-2 produced no full-length chunks.")
        trimmed = ids[: n_chunks * SEQ_LEN]
        return torch.tensor(trimmed, dtype=torch.long).view(-1, SEQ_LEN)

    train_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    val_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")

    train_loader = DataLoader(
        TensorDataset(tokenize_and_chunk(train_ds["text"])),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(0),
    )
    val_loader = DataLoader(
        TensorDataset(tokenize_and_chunk(val_ds["text"])),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Optimizer & scheduler factories
# ---------------------------------------------------------------------------
def uses_clip(name: str) -> bool:
    return name.endswith("+Clip")


def base_optimizer_name(name: str) -> str:
    return name.replace("+Clip", "")


def make_scheduler(optimizer, total_steps: int) -> LambdaLR:
    """Linear warmup + linear decay — identical schedule for every optimizer."""

    def lr_lambda(step: int) -> float:
        if step < WARMUP_STEPS:
            return float(step + 1) / float(max(1, WARMUP_STEPS))
        progress = (step - WARMUP_STEPS) / float(max(1, total_steps - WARMUP_STEPS))
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def make_optimizer(name: str, model: nn.Module):
    """Create optimizer; clip is applied in the training loop, not here."""
    base = base_optimizer_name(name)

    if base == "AdamW":
        return AdamW(
            model.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY
        )
    if base == "AdaptiveGMRAdamW":
        return AdaptiveGMRAdamW(
            model.parameters(),
            lr=LR,
            betas=BETAS,
            eps=EPS,
            weight_decay=WEIGHT_DECAY,
            alpha=GMR_ALPHA,
        )
    if base == "Adafactor":
        return Adafactor(
            model.parameters(),
            lr=LR,
            eps=(EPS, 1e-3),
            clip_threshold=1.0,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=WEIGHT_DECAY,
        )
    raise ValueError(f"Unknown optimizer: {name}")


def fresh_model(device: torch.device, seed: int) -> GPT2LMHeadModel:
    set_global_seed(seed)
    return GPT2LMHeadModel.from_pretrained("gpt2").to(device)


# ---------------------------------------------------------------------------
# Metric helpers (torch.no_grad where possible, minimal copies)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for (batch,) in val_loader:
        batch = batch.to(device)
        loss = model(input_ids=batch, labels=batch).loss
        if not torch.isfinite(loss):
            return float("nan"), float("nan")
        total_loss += loss.item()
        n_batches += 1
    val_loss = total_loss / max(n_batches, 1)
    return val_loss, math.exp(min(val_loss, 20.0))


@torch.no_grad()
def global_grad_norm(model: nn.Module) -> float:
    sq_sum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq_sum += p.grad.detach().pow(2).sum().item()
    return math.sqrt(sq_sum)


@torch.no_grad()
def global_parameter_norm(model: nn.Module) -> float:
    sq_sum = sum(p.data.pow(2).sum().item() for p in model.parameters())
    return math.sqrt(sq_sum)


@torch.no_grad()
def flatten_grads(model: nn.Module) -> Optional[torch.Tensor]:
    parts = [p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
    return torch.cat(parts) if parts else None


@torch.no_grad()
def gradient_cosine_similarity(before: torch.Tensor, after: torch.Tensor) -> float:
    dot = torch.dot(before, after).item()
    nb, na = before.norm().item(), after.norm().item()
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return dot / (na * nb)


@torch.no_grad()
def snapshot_parameters(model: nn.Module) -> List[torch.Tensor]:
    """Lightweight snapshots for update-norm computation."""
    return [p.data.clone() for p in model.parameters()]


@torch.no_grad()
def compute_update_stats(
    before: List[torch.Tensor], model: nn.Module
) -> Tuple[float, float]:
    """Return (L2 update norm, max absolute update)."""
    update_sq = 0.0
    max_update = 0.0
    for prev, p in zip(before, model.parameters()):
        diff = p.data - prev
        update_sq += diff.pow(2).sum().item()
        max_update = max(max_update, diff.abs().max().item())
    return math.sqrt(update_sq), max_update


def apply_perturbation(
    model: nn.Module,
    mode: str,
    spike_scale: int,
    seed: int,
    spike_step: int,
) -> None:
    """Inject gradient perturbation at the designated spike step."""
    if mode == "baseline":
        return

    gen = torch.Generator()
    gen.manual_seed(seed + spike_step)

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if mode == "random_grad":
            std = p.data.std().clamp_min(1e-8)
            p.grad.data = torch.randn_like(p.grad, generator=gen) * std * spike_scale
        elif mode == "attention_spike":
            if "attn.c_proj.weight" in name:
                p.grad.data.mul_(spike_scale)
        elif mode == "sparse_spike":
            mask = torch.rand(p.grad.shape, generator=gen, device=p.grad.device) < SPARSE_FRACTION
            p.grad.data[mask] *= spike_scale


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def eval_interval(train_steps: int) -> int:
    return max(EVAL_EVERY, train_steps // 10)


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------
def run_single_experiment(
    config: RunConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[List[dict], dict]:
    set_global_seed(config.seed)
    model = fresh_model(device, config.seed)
    optimizer = make_optimizer(config.optimizer_name, model)
    scheduler = make_scheduler(optimizer, config.train_steps)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    step_records: List[dict] = []
    train_iter = infinite_loader(train_loader)
    run_start = time.perf_counter()
    last_train_loss = float("nan")

    # Divergence flags accumulated across the run
    saw_nan = saw_inf = saw_grad_explosion = saw_param_explosion = False

    for step in range(config.train_steps):
        model.train()
        (batch,) = next(train_iter)
        batch = batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=batch, labels=batch).loss

        if not torch.isfinite(loss):
            saw_nan = saw_nan or math.isnan(loss.item())
            saw_inf = saw_inf or math.isinf(loss.item())
            last_train_loss = float("nan")
            break

        loss.backward()

        # --- Metrics: before perturbation / preprocessing ---
        g_before = flatten_grads(model)
        grad_norm = global_grad_norm(model)
        param_norm = global_parameter_norm(model)
        params_before = snapshot_parameters(model)

        if grad_norm > GRAD_EXPLOSION_THRESHOLD:
            saw_grad_explosion = True
        if param_norm > PARAM_EXPLOSION_THRESHOLD:
            saw_param_explosion = True

        # --- Perturbation (spike at midpoint) ---
        if step == config.spike_step and config.perturbation_mode != "baseline":
            apply_perturbation(
                model, config.perturbation_mode, config.spike_scale, config.seed, config.spike_step
            )
            grad_norm = global_grad_norm(model)

        # --- Optimizer preprocessing: gradient clipping ---
        if uses_clip(config.optimizer_name):
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_VALUE)

        # Cosine similarity: g before perturbation vs g after perturbation+clip
        g_after = flatten_grads(model)
        grad_cosine = (
            gradient_cosine_similarity(g_before, g_after)
            if g_before is not None and g_after is not None
            else float("nan")
        )

        t0 = time.perf_counter()
        optimizer.step()
        step_time_ms = (time.perf_counter() - t0) * 1000.0
        scheduler.step()

        update_norm, max_update = compute_update_stats(params_before, model)
        param_norm_after = global_parameter_norm(model)
        last_train_loss = loss.item()

        val_loss, val_ppl = (float("nan"), float("nan"))
        if (step + 1) % eval_interval(config.train_steps) == 0 or step == config.train_steps - 1:
            val_loss, val_ppl = evaluate(model, val_loader, device)

        step_records.append(
            {
                "run_id": config.run_id,
                "optimizer": config.optimizer_name,
                "perturbation_mode": config.perturbation_mode,
                "spike_scale": config.spike_scale if config.perturbation_mode != "baseline" else 0,
                "seed": config.seed,
                "train_steps": config.train_steps,
                "step": step + 1,
                "train_loss": last_train_loss,
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
                "grad_norm": grad_norm,
                "parameter_norm": param_norm_after,
                "update_norm": update_norm,
                "max_update": max_update,
                "gradient_cosine": grad_cosine,
                "step_time_ms": step_time_ms,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )

    total_time_s = time.perf_counter() - run_start
    peak_gpu_gb = float("nan")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_gpu_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    def _series(key: str) -> List[float]:
        return [r[key] for r in step_records if math.isfinite(r[key])]

    finite_train = _series("train_loss")
    finite_val = _series("val_loss")
    finite_ppl = _series("val_perplexity")
    finite_gnorm = _series("grad_norm")
    finite_pnorm = _series("parameter_norm")
    finite_unorm = _series("update_norm")
    finite_max_upd = _series("max_update")
    finite_cosine = _series("gradient_cosine")
    finite_step_t = _series("step_time_ms")

    summary = {
        "run_id": config.run_id,
        "optimizer": config.optimizer_name,
        "perturbation_mode": config.perturbation_mode,
        "spike_scale": config.spike_scale if config.perturbation_mode != "baseline" else 0,
        "seed": config.seed,
        "train_steps": config.train_steps,
        "final_train_loss": finite_train[-1] if finite_train else float("nan"),
        "final_val_loss": finite_val[-1] if finite_val else float("nan"),
        "final_val_perplexity": finite_ppl[-1] if finite_ppl else float("nan"),
        "mean_grad_norm": float(np.mean(finite_gnorm)) if finite_gnorm else float("nan"),
        "std_grad_norm": float(np.std(finite_gnorm, ddof=1)) if len(finite_gnorm) > 1 else 0.0,
        "mean_parameter_norm": float(np.mean(finite_pnorm)) if finite_pnorm else float("nan"),
        "mean_update_norm": float(np.mean(finite_unorm)) if finite_unorm else float("nan"),
        "max_update_norm": float(np.max(finite_max_upd)) if finite_max_upd else float("nan"),
        "mean_gradient_cosine": float(np.mean(finite_cosine)) if finite_cosine else float("nan"),
        "mean_step_time_ms": float(np.mean(finite_step_t)) if finite_step_t else float("nan"),
        "total_training_time_s": total_time_s,
        "peak_gpu_mem_gb": peak_gpu_gb,
        "diverged": not finite_train or not math.isfinite(finite_train[-1]),
        "divergence_nan": saw_nan,
        "divergence_inf": saw_inf,
        "divergence_grad_explosion": saw_grad_explosion,
        "divergence_param_explosion": saw_param_explosion,
        "steps_completed": len(step_records),
    }

    del model, optimizer, scheduler, g_before, g_after, params_before
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return step_records, summary


# ---------------------------------------------------------------------------
# Experiment grid — original experiments preserved, new ones added
# ---------------------------------------------------------------------------
def generate_run_configs() -> List[RunConfig]:
    """
    Build the full experiment grid.

    Original experiments: AdamW & AdaptiveGMRAdamW at all perturbations × seeds.
    Extensions: additional optimizers and TRAIN_STEP_OPTIONS [50, 300, 500].
    """
    configs: List[RunConfig] = []
    for train_steps in TRAIN_STEP_OPTIONS:
        for seed in SEEDS:
            for opt in ALL_OPTIMIZERS:
                configs.append(RunConfig(opt, "baseline", 0, seed, train_steps))
                for mode in PERTURBATION_MODES:
                    if mode == "baseline":
                        continue
                    for scale in SPIKE_SCALES:
                        configs.append(RunConfig(opt, mode, scale, seed, train_steps))
    return configs


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def welch_ci(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, mean, mean
    se = stats.sem(values, ddof=1)
    t_crit = stats.t.ppf((1 + confidence) / 2.0, n - 1)
    margin = t_crit * se
    return mean, mean - margin, mean + margin


def _gmr_better(metric: str, diff: float) -> bool:
    """True when a lower value favours AdaptiveGMRAdamW."""
    lower_is_better = (
        "loss" in metric
        or "perplexity" in metric
        or "time" in metric
        or "norm" in metric
        or metric == "max_update_norm"
    )
    return diff < 0 if lower_is_better else diff > 0


def statistical_analysis(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Welch t-tests: AdamW vs AdaptiveGMRAdamW (and clipped variants)."""
    rows = []
    group_cols = ["perturbation_mode", "spike_scale", "train_steps"]
    metrics = [
        "final_val_perplexity",
        "final_val_loss",
        "mean_grad_norm",
        "mean_update_norm",
        "max_update_norm",
        "mean_parameter_norm",
        "mean_gradient_cosine",
        "total_training_time_s",
        "mean_step_time_ms",
    ]
    pairs = [
        ("AdamW", "AdaptiveGMRAdamW"),
        ("AdamW+Clip", "AdaptiveGMRAdamW+Clip"),
    ]

    for (mode, scale, steps), grp in summary_df.groupby(group_cols):
        for opt_a, opt_b in pairs:
            a_df = grp[grp["optimizer"] == opt_a]
            b_df = grp[grp["optimizer"] == opt_b]
            if len(a_df) < 2 or len(b_df) < 2:
                continue
            for metric in metrics:
                a = a_df[metric].dropna().to_numpy(dtype=float)
                b = b_df[metric].dropna().to_numpy(dtype=float)
                if len(a) < 2 or len(b) < 2:
                    continue
                t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
                a_mean, a_lo, a_hi = welch_ci(a)
                b_mean, b_lo, b_hi = welch_ci(b)
                rows.append(
                    {
                        "perturbation_mode": mode,
                        "spike_scale": scale,
                        "train_steps": steps,
                        "comparison": f"{opt_b} vs {opt_a}",
                        "metric": metric,
                        "baseline_mean": a_mean,
                        "baseline_ci_low": a_lo,
                        "baseline_ci_high": a_hi,
                        "candidate_mean": b_mean,
                        "candidate_ci_low": b_lo,
                        "candidate_ci_high": b_hi,
                        "mean_diff_candidate_minus_baseline": b_mean - a_mean,
                        "welch_t": t_stat,
                        "p_value": p_value,
                        "significant_95": p_value < 0.05,
                        "candidate_better": _gmr_better(metric, b_mean - a_mean),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
OPTIMIZER_COLORS = {
    "AdamW": "#E74C3C",
    "AdaptiveGMRAdamW": "#27AE60",
    "AdamW+Clip": "#C0392B",
    "AdaptiveGMRAdamW+Clip": "#1E8449",
    "Adafactor": "#8E44AD",
}


def plot_metric(
    results_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    filename: str,
    log_y: bool = False,
) -> None:
    """One figure per train_steps, 2×2 subplots for perturbation modes."""
    for train_steps in TRAIN_STEP_OPTIONS:
        sub_all = results_df[results_df["train_steps"] == train_steps]
        if sub_all.empty:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        axes = axes.flatten()

        for ax, mode in zip(axes, PERTURBATION_MODES):
            sub = sub_all[sub_all["perturbation_mode"] == mode]
            if sub.empty:
                ax.set_title(f"{mode} (no data)")
                continue

            if mode == "baseline":
                for opt in ALL_OPTIMIZERS:
                    curve = (
                        sub[sub["optimizer"] == opt]
                        .groupby("step")[metric]
                        .agg(["mean", "std"])
                        .reset_index()
                    )
                    if curve.empty:
                        continue
                    color = OPTIMIZER_COLORS.get(opt, "#333333")
                    x, y = curve["step"], curve["mean"]
                    ax.plot(x, y, label=opt, color=color, linewidth=1.8)
                    ax.fill_between(x, y - curve["std"].fillna(0), y + curve["std"].fillna(0), color=color, alpha=0.1)
            else:
                for scale in SPIKE_SCALES:
                    for opt in ("AdamW", "AdaptiveGMRAdamW"):
                        curve = (
                            sub[(sub["optimizer"] == opt) & (sub["spike_scale"] == scale)]
                            .groupby("step")[metric]
                            .mean()
                            .reset_index()
                        )
                        if curve.empty:
                            continue
                        ax.plot(curve["step"], curve[metric], linewidth=1.2, label=f"{opt}×{scale}")

            ax.set_title(mode.replace("_", " ").title())
            ax.set_xlabel("Step")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if log_y:
                ax.set_yscale("symlog", linthresh=1e-3)
            ax.legend(fontsize=6, loc="best")

        stem = Path(filename).stem
        suffix = Path(filename).suffix
        out_name = f"{stem}_steps{train_steps}{suffix}"
        fig.suptitle(f"{ylabel} — steps={train_steps} (mean over seeds)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / out_name, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_final_perplexity_bars(summary_df: pd.DataFrame) -> None:
    for train_steps in TRAIN_STEP_OPTIONS:
        sub_all = summary_df[summary_df["train_steps"] == train_steps]
        if sub_all.empty:
            continue
        fig, axes = plt.subplots(1, len(PERTURBATION_MODES), figsize=(16, 4))
        for ax, mode in zip(axes, PERTURBATION_MODES):
            sub = sub_all[sub_all["perturbation_mode"] == mode]
            if mode == "baseline":
                means = [sub[sub["optimizer"] == o]["final_val_perplexity"].mean() for o in ALL_OPTIMIZERS]
                ax.bar(ALL_OPTIMIZERS, means, color=[OPTIMIZER_COLORS[o] for o in ALL_OPTIMIZERS])
                ax.tick_params(axis="x", rotation=30)
            else:
                x = np.arange(len(SPIKE_SCALES))
                w = 0.35
                for i, opt in enumerate(("AdamW", "AdaptiveGMRAdamW")):
                    vals = [
                        sub[(sub["optimizer"] == opt) & (sub["spike_scale"] == s)]["final_val_perplexity"].mean()
                        for s in SPIKE_SCALES
                    ]
                    ax.bar(x + (i - 0.5) * w, vals, width=w, label=opt, color=OPTIMIZER_COLORS[opt])
                ax.set_xticks(x)
                ax.set_xticklabels([str(s) for s in SPIKE_SCALES])
                ax.set_xlabel("Spike scale")
                ax.legend(fontsize=7)
            ax.set_title(mode.replace("_", " "))
            ax.set_ylabel("Final val perplexity")
            ax.grid(True, axis="y", alpha=0.3)
        plt.suptitle(f"Final Validation Perplexity — steps={train_steps}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"final_val_perplexity_bars_steps{train_steps}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def print_conclusions(summary_df: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("AUTOMATED CONCLUSIONS (unbiased)")
    print("=" * 70)

    baseline = summary_df[
        (summary_df["perturbation_mode"] == "baseline") & (summary_df["train_steps"] == 50)
    ]
    if not baseline.empty:
        for opt in ("AdamW", "AdaptiveGMRAdamW"):
            ppl = baseline[baseline["optimizer"] == opt]["final_val_perplexity"].mean()
            if math.isfinite(ppl):
                print(f"- Baseline (50 steps) {opt} mean val perplexity: {ppl:.3f}")

    if stats_df.empty:
        print("- Insufficient data for statistical tests.")
    else:
        sig = stats_df[(stats_df["significant_95"]) & (stats_df["metric"] == "final_val_perplexity")]
        if sig.empty:
            print("- No statistically significant perplexity differences (Welch, p<0.05).")
        for _, row in sig.iterrows():
            direction = "better" if row["candidate_better"] else "worse"
            print(
                f"- [{row['comparison']}, {row['perturbation_mode']}, scale={int(row['spike_scale'])}, "
                f"steps={int(row['train_steps'])}] candidate {direction} on perplexity "
                f"(p={row['p_value']:.4f})."
            )

    div = summary_df[summary_df["diverged"] | summary_df["divergence_nan"] | summary_df["divergence_grad_explosion"]]
    if not div.empty:
        for opt, n in div.groupby("optimizer").size().items():
            print(f"- WARNING: {opt} had divergence flags in {n} run(s).")
    else:
        print("- No divergence flags recorded.")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = generate_run_configs()
    print(f"Device: {device}")
    print(f"Runs: {len(configs)} | train_steps={TRAIN_STEP_OPTIONS} | optimizers={ALL_OPTIMIZERS}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading WikiText-2 ...")
    train_loader, val_loader = build_wikitext_loaders(tokenizer)

    all_steps: List[dict] = []
    all_summaries: List[dict] = []

    for config in tqdm(configs, desc="Experiments"):
        steps, summary = run_single_experiment(config, train_loader, val_loader, device)
        all_steps.extend(steps)
        all_summaries.append(summary)

    results_df = pd.DataFrame(all_steps)
    summary_df = pd.DataFrame(all_summaries)
    stats_df = statistical_analysis(summary_df)

    results_df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    stats_df.to_csv(OUTPUT_DIR / "statistical_tests.csv", index=False)

    print(f"\nWrote {OUTPUT_DIR / 'results.csv'} ({len(results_df)} rows)")
    print(f"Wrote {OUTPUT_DIR / 'summary.csv'} ({len(summary_df)} rows)")
    print(f"Wrote {OUTPUT_DIR / 'statistical_tests.csv'} ({len(stats_df)} rows)")

    for metric, label, fname, log_y in [
        ("train_loss", "Training Loss", "training_loss.png", True),
        ("val_loss", "Validation Loss", "validation_loss.png", True),
        ("val_perplexity", "Validation Perplexity", "validation_perplexity.png", True),
        ("grad_norm", "Gradient Norm", "gradient_norm.png", True),
        ("update_norm", "Update Norm ||Δw||", "update_norm.png", True),
        ("parameter_norm", "Parameter Norm ||w||", "parameter_norm.png", True),
        ("gradient_cosine", "Gradient Cosine Similarity", "gradient_cosine.png", False),
        ("max_update", "Max Absolute Update", "max_update.png", True),
    ]:
        plot_metric(results_df, metric, label, fname, log_y=log_y)

    plot_final_perplexity_bars(summary_df)
    print(f"Figures saved to {FIGURES_DIR}/")
    print_conclusions(summary_df, stats_df)


if __name__ == "__main__":
    main()
