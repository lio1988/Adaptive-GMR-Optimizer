"""Rigorous fair comparison: AdamW vs AdaptiveGMR (clean + spiked validation curves)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adaptive_gmr import AdaptiveGMR

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
LR = 1e-3
NUM_SAMPLES = 1000
INPUT_DIM = 10
HIDDEN_DIM = 64
OUTPUT_DIM = 1
BATCH_SIZE = 64
NUM_EPOCHS = 50
SPIKE_EPOCH = 4  # 0-based index → spike on epoch 5
SPIKE_MAGNITUDE = 1000.0
TRAIN_FRACTION = 0.8

torch.manual_seed(SEED)


class MLP(nn.Module):
    def __init__(self, in_dim=INPUT_DIM, hidden=HIDDEN_DIM, out_dim=OUTPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def make_dataloaders():
    x = torch.randn(NUM_SAMPLES, INPUT_DIM)
    true_w = torch.randn(INPUT_DIM, OUTPUT_DIM)
    y = x @ true_w + 0.1 * torch.randn(NUM_SAMPLES, OUTPUT_DIM)

    dataset = TensorDataset(x, y)
    n_train = int(NUM_SAMPLES * TRAIN_FRACTION)
    n_val = NUM_SAMPLES - n_train
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def fresh_model():
    torch.manual_seed(SEED)
    return MLP()


def make_optimizer(optimizer_cls, model):
    """Both optimizers share identical lr and matched Adam hyperparameters."""
    if optimizer_cls is AdamW:
        return optimizer_cls(
            model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
    return optimizer_cls(
        model.parameters(),
        lr=LR,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        sigma=1.0,
    )


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        total_loss += criterion(model(xb), yb).item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def train(
    optimizer_cls,
    train_loader,
    val_loader,
    *,
    inject_spike: bool,
    device: torch.device,
):
    model = fresh_model().to(device)
    optimizer = make_optimizer(optimizer_cls, model)
    criterion = nn.MSELoss()

    val_losses = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()

            if inject_spike and epoch == SPIKE_EPOCH and batch_idx == 0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data += SPIKE_MAGNITUDE

            optimizer.step()

        val_losses.append(evaluate(model, val_loader, criterion, device))

    return val_losses


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_dataloaders()

    runs = [
        ("AdamW Clean", AdamW, False),
        ("AdaptiveGMR Clean", AdaptiveGMR, False),
        ("AdamW Spiked", AdamW, True),
        ("AdaptiveGMR Spiked", AdaptiveGMR, True),
    ]

    results = {}
    for label, opt_cls, spiked in runs:
        print(f"Running: {label} ...")
        results[label] = train(
            opt_cls,
            train_loader,
            val_loader,
            inject_spike=spiked,
            device=device,
        )
        print(f"  final val loss = {results[label][-1]:.6f}")

    epochs = range(1, NUM_EPOCHS + 1)
    colors = {
        "AdamW Clean": "#2E86C1",
        "AdaptiveGMR Clean": "#27AE60",
        "AdamW Spiked": "#E74C3C",
        "AdaptiveGMR Spiked": "#F39C12",
    }
    styles = {
        "AdamW Clean": "-",
        "AdaptiveGMR Clean": "--",
        "AdamW Spiked": "-",
        "AdaptiveGMR Spiked": "--",
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    for label, losses in results.items():
        ax.plot(
            epochs,
            losses,
            label=label,
            color=colors[label],
            linestyle=styles[label],
            linewidth=2,
        )

    ax.axvline(
        SPIKE_EPOCH + 1,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label=f"Spike (epoch {SPIKE_EPOCH + 1})",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss (log scale)")
    ax.set_title(
        "Fair Comparison: AdamW vs AdaptiveGMR\n"
        f"lr={LR}, same init, 80/20 split, spike={SPIKE_MAGNITUDE} at epoch {SPIKE_EPOCH + 1}"
    )
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("benchmark_fair.png", dpi=150)
    print("\nSaved benchmark_fair.png")


if __name__ == "__main__":
    main()
