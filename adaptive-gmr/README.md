# AdaptiveGMR

SGD optimizer with per-parameter **Geman-McClure** gradient scaling. Automatically dampens gradient spikes while preserving direction.

## Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pytest numpy matplotlib
```

## Quick Start

```python
import torch.nn as nn
from adaptive_gmr import AdaptiveGMR

model = nn.Linear(10, 1)
optimizer = AdaptiveGMR(model.parameters(), lr=3e-4, alpha=0.9)
```

## Run Tests

```bash
python test_adaptive_gmr.py
# or
pytest test_adaptive_gmr.py -v
```

## Benchmark

```bash
python benchmark.py
```

## Features

- Per-parameter adaptive sigma (GMR scaling)
- Spike protection without manual gradient clipping
- `sigma_warmup_steps` — hold sigma at init for first N steps
- Gradient accumulation (`accumulation_steps`)
- Per-layer learning rates via param groups
- `reset_sigma()` — reset all sigmas to initial value
- DeepSpeed ZeRO-3 compatible (skips params with `grad_fn` but no `grad`)
- Per-parameter gradient norm history (last 10 steps)
