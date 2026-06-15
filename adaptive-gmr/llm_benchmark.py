"""LLM spike benchmark: AdamW vs AdaptiveGMR on GPT-2 (124M)."""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adaptive_gmr import AdaptiveGMR

SEED = 42
LR = 1e-4
BATCH_SIZE = 4
SEQ_LEN = 128
NUM_STEPS = 50
SPIKE_STEP = 25
VOCAB_SIZE = 50257  # GPT-2 vocab size


def make_dataloader(device: torch.device) -> DataLoader:
    torch.manual_seed(SEED)
    tokens = torch.randint(0, VOCAB_SIZE, (NUM_STEPS * BATCH_SIZE, SEQ_LEN))
    dataset = TensorDataset(tokens)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


def make_optimizer(optimizer_cls, model):
    if optimizer_cls is AdamW:
        return optimizer_cls(
            model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
    return optimizer_cls(
        model.parameters(), lr=LR, beta1=0.9, beta2=0.999, eps=1e-8, sigma=1.0
    )


def fresh_model(device: torch.device) -> GPT2LMHeadModel:
    torch.manual_seed(SEED)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return model.to(device)


def run_training(optimizer_cls, train_loader, device: torch.device):
    model = fresh_model(device)
    optimizer = make_optimizer(optimizer_cls, model)
    name = "AdamW" if optimizer_cls is AdamW else "AdaptiveGMR"

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    losses = []
    data_iter = iter(train_loader)

    for step in range(NUM_STEPS):
        try:
            (batch,) = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            (batch,) = next(data_iter)

        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=batch, labels=batch)
        loss = outputs.loss
        loss.backward()

        if step == SPIKE_STEP:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data = torch.randn_like(p.grad) * 1e6

        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val if math.isfinite(loss_val) else float("nan"))
        print(f"  [{name}] step {step + 1:02d}/{NUM_STEPS}  loss={loss_val:.4f}")

    peak_gb = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return losses, peak_gb


def plot_results(adam_losses, gmr_losses, output_path: str):
    steps = list(range(1, NUM_STEPS + 1))
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(steps, adam_losses, label="AdamW", color="#E74C3C", linewidth=2)
    ax.plot(steps, gmr_losses, label="AdaptiveGMR", color="#27AE60", linewidth=2)

    ax.axvline(
        SPIKE_STEP + 1,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label=f"Spike (step {SPIKE_STEP + 1})",
    )

    finite = [v for v in adam_losses + gmr_losses if math.isfinite(v) and v > 0]
    if finite:
        ymin = min(finite)
        ymax = max(finite)
        ax.set_ylim(max(ymin * 0.5, 1e-3), ymax * 3)

    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (symlog)")
    ax.set_title(
        "GPT-2 (124M) Training — AdamW vs AdaptiveGMR\n"
        f"lr={LR}, batch={BATCH_SIZE}x{SEQ_LEN}, white-noise grad spike (1e6) at step {SPIKE_STEP + 1}"
    )
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved {output_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading GPT-2 and running {NUM_STEPS} steps per optimizer (lr={LR})...\n")

    train_loader = make_dataloader(device)

    print("=== AdamW ===")
    adam_losses, adam_peak = run_training(AdamW, train_loader, device)

    print("\n=== AdaptiveGMR ===")
    gmr_losses, gmr_peak = run_training(AdaptiveGMR, train_loader, device)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    if device.type == "cuda":
        print(f"AdamW Peak Memory:       {adam_peak:.2f} GB")
        print(f"AdaptiveGMR Peak Memory: {gmr_peak:.2f} GB")
        delta_mb = (gmr_peak - adam_peak) * 1024
        print(f"Memory delta:            {delta_mb:+.1f} MB")
    else:
        print("CUDA not available — GPU memory profiling skipped.")

    adam_final = adam_losses[-1]
    gmr_final = gmr_losses[-1]
    print(f"\nAdamW final loss:       {adam_final}")
    print(f"AdaptiveGMR final loss: {gmr_final}")

    post_spike_adam = [v for v in adam_losses[SPIKE_STEP + 1 :] if math.isfinite(v)]
    post_spike_gmr = [v for v in gmr_losses[SPIKE_STEP + 1 :] if math.isfinite(v)]
    if not post_spike_adam:
        print("AdamW diverged (NaN/Inf) after spike.")
    if post_spike_gmr:
        print(f"AdaptiveGMR post-spike loss at step {NUM_STEPS}: {gmr_losses[-1]:.4f}")

    plot_results(adam_losses, gmr_losses, "llm_spike_test.png")


if __name__ == "__main__":
    main()
