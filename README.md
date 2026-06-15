# Adaptive-GMR-Optimizer

Ένας παραγωγικός (production-ready) optimizer για Deep Learning μοντέλα, που ενσωματώνει **Geman-McClure Robust Statistics** για προστασία από **loss spikes**.

## Κύρια Χαρακτηριστικά
* **Spike Protection**: Προστατεύει τα "moments" του AdamW από μόλυνση κατά τη διάρκεια ξαφνικών spikes.
* **Zero Memory Overhead**: Το scaling των gradients γίνεται `in-place`, χωρίς να αντιγράφονται παράμετροι στη VRAM.
* **Distributed Compatible**: Σχεδιασμένος για DDP, FSDP και ZeRO, με scaling πριν το reduction.
* **Drop-in Replacement**: Κληρονομεί τον `torch.optim.AdamW`, οπότε λειτουργεί άμεσα με το υπάρχον training loop σου.

## Εγκατάσταση
```bash
pip install .
