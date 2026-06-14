# Adaptive-GMR-Optimizer

`Adaptive-GMR-Optimizer` is a drop-in replacement for PyTorch's AdamW. It incorporates the **Geman-McClure** robust estimator to dynamically scale gradients, providing an "auto-immune" response to training instabilities (gradient spikes) without manual tuning.

## Why use this?
- **Stability:** Prevents `NaN` losses and gradient explosion.
- **Adaptive:** No need to guess `clip_norm` values; the optimizer "senses" the training state.
- **Drop-in:** Compatible with any standard PyTorch `nn.Module`.

## Quick Start
```python
from adaptive_gmr import AdaptiveGMRAdamW

# Replace torch.optim.AdamW with AdaptiveGMRAdamW
optimizer = AdaptiveGMRAdamW(model.parameters(), lr=1e-3, alpha=0.5)
