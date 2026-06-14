import torch
import torch.nn as nn
import torch.optim as optim
# Αντικατέστησε με το δικό σου import αν χρειάζεται
# from adaptive_gmr import AdaptiveGMRAdamW 

class MiniLLM(nn.Module):
    """Ένα μικρό Transformer μοντέλο (εκπροσωπεί αρχιτεκτονική LLM)"""
    def __init__(self, vocab_size=1000, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc_out(x)

def run_transformer_stress_test(optimizer_type='GMR'):
    torch.manual_seed(42)
    model = MiniLLM()
    
    if optimizer_type == 'GMR':
        # [Εδώ προϋποθέτει ότι έχεις ορίσει την κλάση AdaptiveGMRAdamW στο script]
        optimizer = AdaptiveGMRAdamW(model.parameters(), lr=5e-4, alpha=0.5)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=5e-4)
        
    criterion = nn.CrossEntropyLoss()
    
    # Dummy δεδομένα: batch_size=16, seq_length=32
    data = torch.randint(0, 1000, (16, 32))
    target = torch.randint(0, 1000, (16, 32))

    print(f"\n--- Training Mini-LLM with {optimizer_type} ---")
    
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(data)
        # Αναδιαμόρφωση για το CrossEntropyLoss
        loss = criterion(output.view(-1, 1000), target.view(-1))
        loss.backward()

        # --- ATTENTION SPIKE INJECTION ---
        if epoch == 25:
            print(f"!!! [CRITICAL SPIKE] στο Transformer Layer 3 !!!")
            # Στοχεύουμε σκόπιμα το βαθύτερο layer του transformer
            for name, p in model.named_parameters():
                if 'transformer.layers.3' in name and p.grad is not None:
                    p.grad.data += 1000.0 # Τεράστιο gradient explosion

        optimizer.step()
        
        if epoch % 5 == 0 or epoch == 25 or epoch == 26:
            loss_val = loss.item()
            status = "STABLE" if not torch.isnan(torch.tensor(loss_val)) else "FAILED"
            print(f"Epoch {epoch:02d} | Loss: {loss_val:.4f} | Status: {status}")
            
            if torch.isnan(torch.tensor(loss_val)):
                break

if __name__ == "__main__":
    # Τρέξε και τα δύο για να δεις τη διαφορά
    run_transformer_stress_test(optimizer_type='AdamW')
    run_transformer_stress_test(optimizer_type='GMR')
