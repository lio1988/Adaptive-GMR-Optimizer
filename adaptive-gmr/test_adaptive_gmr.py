import torch
import torch.nn as nn
from adaptive_gmr import AdaptiveGMR


def test_basic():
    """Runs without crashing."""
    model = nn.Linear(10, 1)
    opt = AdaptiveGMR(model.parameters(), lr=3e-4)
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    opt.step()
    print(f"Basic test passed | loss={loss.item():.4f}")


def test_spike_protection():
    """Spike injection — must not diverge."""
    model = nn.Linear(10, 1)
    opt = AdaptiveGMR(model.parameters(), lr=3e-4)
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)

    for step in range(20):
        opt.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()

        if step == 10:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data += 500.0
            print(f"  Spike injected at step {step}")

        opt.step()

    assert torch.isfinite(loss), "Loss diverged!"
    print(f"Spike test passed | final loss={loss.item():.4f}")


def test_sigma_adaptation():
    """Sigma adapts away from initial value."""
    model = nn.Linear(10, 1)
    opt = AdaptiveGMR(model.parameters(), lr=3e-4, alpha=0.9)
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)

    sigmas = []
    for _ in range(50):
        opt.zero_grad()
        nn.MSELoss()(model(x), y).backward()
        opt.step()
        stats = opt.get_sigma_stats()
        if stats:
            sigmas.append(stats["mean"])

    assert sigmas[-1] != 1.0, "Sigma did not adapt!"
    print(f"Sigma adaptation: {sigmas[0]:.4f} -> {sigmas[-1]:.4f}")


def test_state_dict():
    """Save/load state."""
    model = nn.Linear(10, 1)
    opt = AdaptiveGMR(model.parameters(), lr=3e-4)
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    nn.MSELoss()(model(x), y).backward()
    opt.step()

    state = opt.state_dict()
    opt2 = AdaptiveGMR(model.parameters(), lr=3e-4)
    opt2.load_state_dict(state)
    print("state_dict test passed")


def test_repr():
    model = nn.Linear(10, 1)
    opt = AdaptiveGMR(model.parameters(), lr=3e-4)
    print(f"repr: {opt}")


def test_reset_sigma():
    model = nn.Linear(10, 1)
    opt = AdaptiveGMR(model.parameters(), lr=3e-4, alpha=0.9)
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)

    for _ in range(20):
        opt.zero_grad()
        nn.MSELoss()(model(x), y).backward()
        opt.step()

    stats_before = opt.get_sigma_stats()
    opt.reset_sigma()
    stats_after = opt.get_sigma_stats()
    assert abs(stats_after["mean"] - 1.0) < 1e-5
    assert stats_before["mean"] != stats_after["mean"] or stats_before["mean"] == 1.0
    print("reset_sigma test passed")


def test_gradient_accumulation():
    model = nn.Linear(10, 1)
    opt = AdaptiveGMR(model.parameters(), lr=3e-4, accumulation_steps=4)
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)

    for _ in range(8):
        nn.MSELoss()(model(x), y).backward()
        opt.step()

    assert opt._global_step == 2
    print("gradient accumulation test passed")


def test_per_layer_lr():
    model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))
    opt = AdaptiveGMR(
        [
            {"params": model[0].parameters(), "lr": 1e-3},
            {"params": model[1].parameters(), "lr": 1e-4},
        ]
    )
    assert opt.param_groups[0]["lr"] == 1e-3
    assert opt.param_groups[1]["lr"] == 1e-4
    print("per-layer lr test passed")


if __name__ == "__main__":
    print("=" * 50)
    print("AdaptiveGMR Test Suite")
    print("=" * 50)
    test_basic()
    test_spike_protection()
    test_sigma_adaptation()
    test_state_dict()
    test_repr()
    test_reset_sigma()
    test_gradient_accumulation()
    test_per_layer_lr()
    print("\nAll tests passed!")
