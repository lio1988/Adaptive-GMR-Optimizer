import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# Αγνόηση των warnings για NaN losses, καθώς τα περιμένουμε εσκεμμένα στο τεστ
warnings.filterwarnings("ignore")

# Εισαγωγή του δικού σου optimizer (βεβαιώσου ότι ο φάκελος adaptive_gmr είναι προσβάσιμος)
from adaptive_gmr import AdaptiveGMRAdamW

# ==========================================
# 1. Ορισμός Αρχιτεκτονικών (Mini Models)
# ==========================================
class MiniMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

class MiniCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 10)
        )
    def forward(self, x):
        return self.net(x.view(-1, 1, 8, 8)) # Dummy image 8x8

class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, 32)
        layer = nn.TransformerEncoderLayer(d_model=32, nhead=2, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(32, 10)
    def forward(self, x):
        x = self.emb(x.long()) # Shape: [batch, seq_len]
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))

def get_model(model_name):
    if model_name == 'MLP': return MiniMLP()
    elif model_name == 'CNN': return MiniCNN()
    elif model_name == 'Transformer': return MiniTransformer()
    raise ValueError("Άγνωστο μοντέλο")

# ==========================================
# 2. Κύρια Συνάρτηση Εκπαίδευσης & Spike Injection
# ==========================================
def train_with_spike(model, opt_name, spike_size, spike_epoch=15, total_epochs=30):
    # Dummy Dataset Initialization
    batch_size = 32
    if isinstance(model, MiniTransformer):
        data = torch.randint(0, 100, (batch_size, 16)) # Sequence data
    else:
        data = torch.randn(batch_size, 64) # Feature / Image data
    target = torch.randint(0, 10, (batch_size,))
    
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer Initialization
    lr = 1e-3
    if opt_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif opt_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif opt_name == 'Adaptive GMR':
        optimizer = AdaptiveGMRAdamW(model.parameters(), lr=lr, alpha=0.5)
    
    model.train()
    final_loss = None
    
    for epoch in range(total_epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # --- SPIKE INJECTION ---
        if epoch == spike_epoch:
            for p in model.parameters():
                if p.grad is not None:
                    # Εισαγωγή θορύβου στα gradients
                    p.grad.data += torch.randn_like(p.grad.data) * spike_size
        
        optimizer.step()
        final_loss = loss.item()
        
        # Αν το loss γίνει NaN, η εκπαίδευση "έσκασε"
        if torch.isnan(torch.tensor(final_loss)) or torch.isinf(torch.tensor(final_loss)):
            return float('inf'), False
            
    return final_loss, True

# ==========================================
# 3. Εκτέλεση του Benchmark (Ablation Study)
# ==========================================
def run_benchmark():
    models = ['MLP', 'CNN', 'Transformer']
    spike_sizes = [100.0, 1000.0, 5000.0]
    optimizers = ['SGD', 'Adam', 'AdamW', 'Adaptive GMR']
    num_seeds = 15
    
    print("Ξεκινάει το Benchmarking Framework...")
    print(f"Αριθμός επαναλήψεων (Seeds) ανά πείραμα: {num_seeds}\n")
    
    results = []
    
    # Χρησιμοποιούμε dict για να μαζέψουμε τα data στο format που θέλουμε
    summary_table = {opt: {spike: 0 for spike in spike_sizes} for opt in optimizers}

    for model_name in models:
        for spike in spike_sizes:
            for opt_name in optimizers:
                survived_count = 0
                
                # Progress bar για κάθε συνδυασμό
                desc = f"{model_name:11} | Spike: {spike:6} | {opt_name:12}"
                for seed in tqdm(range(num_seeds), desc=desc, leave=False):
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    
                    model = get_model(model_name)
                    loss, survived = train_with_spike(model, opt_name, spike)
                    
                    if survived:
                        survived_count += 1
                
                survival_rate = (survived_count / num_seeds) * 100
                summary_table[opt_name][spike] += survival_rate / len(models) # Μέσος όρος επιβίωσης στα 3 μοντέλα
                
                results.append({
                    'Model': model_name,
                    'Optimizer': opt_name,
                    'Spike_Size': spike,
                    'Survival_Rate(%)': survival_rate
                })

    # Εξαγωγή αναλυτικών αποτελεσμάτων
    df_detailed = pd.DataFrame(results)
    df_detailed.to_csv("benchmark_detailed_results.csv", index=False)
    
    # Εκτύπωση του τελικού "Executive" Πίνακα
    print("\n" + "="*50)
    print("ΤΕΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ (Μέσο Survival Rate σε όλα τα Μοντέλα)")
    print("="*50)
    
    # Δημιουργία τελικού πίνακα για εύκολη αντιγραφή στο Markdown
    final_df = pd.DataFrame(summary_table).T
    final_df.columns = [f"Spike {int(s)}" for s in spike_sizes]
    final_df = final_df.map(lambda x: f"{x:.1f}%")
    
    print(final_df.to_markdown())
    print("\nΤα αναλυτικά δεδομένα αποθηκεύτηκαν στο 'benchmark_detailed_results.csv'")

if __name__ == "__main__":
    run_benchmark()
