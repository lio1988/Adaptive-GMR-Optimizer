# Adaptive-GMR-Optimizer

Ένας παραγωγικός (production-ready) optimizer για Deep Learning μοντέλα, για προστασία από **loss spikes**.

## Κύρια Χαρακτηριστικά
* **Spike Protection**: Προστατεύει τα "moments" του AdamW από μόλυνση κατά τη διάρκεια ξαφνικών spikes.
* **Zero Memory Overhead**: Το scaling των gradients γίνεται `in-place`, χωρίς να αντιγράφονται παράμετροι στη VRAM.
* **Distributed Compatible**: Σχεδιασμένος για DDP, FSDP και ZeRO, με scaling πριν το reduction.
* **Drop-in Replacement**: Κληρονομεί τον `torch.optim.AdamW`, οπότε λειτουργεί άμεσα με το υπάρχον training loop σου.

## Εγκατάσταση

Συνιστάται η χρήση virtualenv/conda και Python 3.8 ή νεότερης.

1) Εγκατάσταση απευθείας από το GitHub (για εξωτερικούς χρήστες):

```bash
pip install git+https://github.com/lio1988/Adaptive-GMR-Optimizer.git
```

2) Τοπική εγκατάσταση / ανάπτυξη (developer):

- Εγκατάσταση (σταθερή):
```bash
pip install .
```
- Editable εγκατάσταση (για ενεργή ανάπτυξη):
```bash
pip install -e .
```

Σημείωση: Βεβαιωθείτε ότι έχετε εγκαταστήσει μια συμβατή έκδοση του PyTorch πριν την εγκατάσταση, π.χ. `pip install torch torchvision`.