# AdaptiveGMRAdamW: Robust Optimizer for LLMs

The `AdaptiveGMRAdamW` is a drop-in replacement for AdamW designed to automatically suppress gradient spikes using the Geman-McClure robust estimator. 

### Why GMR?
Standard gradient clipping is often too rigid. Our implementation uses an adaptive threshold (GMR) that dynamically scales gradients, effectively "healing" the model during unstable training phases without manual hyperparameter tuning.

### Installation
```bash
pip install .
