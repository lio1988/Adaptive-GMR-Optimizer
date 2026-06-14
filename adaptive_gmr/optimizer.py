import torch
from torch.optim import AdamW

class AdaptiveGMRAdamW(AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=1e-2, alpha=1.0, alpha_decay=0.999, 
                 min_alpha=0.1, theta_init=1.0, ema_decay=0.99):
        
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.min_alpha = min_alpha
        self.ema_decay = ema_decay
        self.state['running_norm'] = torch.tensor(theta_init, dtype=torch.float32)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.alpha > self.min_alpha:
            self.alpha *= self.alpha_decay
        
        # 1. Υπολογισμός Global Gradient Norm Squared (||g||^2)
        global_norm_sq = 0.0
        device = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if device is None: device = p.grad.device
                    global_norm_sq += torch.sum(p.grad.data**2)
        
        global_norm = torch.sqrt(global_norm_sq)
        
        if getattr(self.state['running_norm'], 'device', None) != device:
             self.state['running_norm'] = self.state['running_norm'].to(device)

        # 2. Ενημέρωση του Running Norm (έχει μονάδες U)
        self.state['running_norm'] = (self.ema_decay * self.state['running_norm'] + 
                                     (1 - self.ema_decay) * global_norm)
        
        # 3. Υπολογισμός Θ (έχει μονάδες U)
        theta = self.alpha * self.state['running_norm']
        
        # 4. Geman-McClure GLOBAL Scaling: theta^2 / (theta^2 + ||g||^2)
        # Και τα δύο μέρη είναι U^2, άρα το scaling είναι dimensionless [0, 1]
        theta_sq = theta**2
        scaling = theta_sq / (theta_sq + global_norm_sq + 1e-8)
        
        # 5. Εφαρμογή GLOBAL scaling (διατηρεί την κατεύθυνση του gradient)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.mul_(scaling)
                
        super().step(closure)
        return loss
