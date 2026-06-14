import torch
from torch.optim import AdamW

class AdaptiveGMRAdamW(AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=1e-2, alpha=0.5, alpha_decay=0.999, 
                 min_alpha=0.1, theta_init=1.0, ema_decay=0.99):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.min_alpha = min_alpha
        self.theta_init = theta_init
        self.ema_decay = ema_decay
        self.state['running_norm'] = torch.tensor(theta_init)
        self.step_count = 0
        
    def step(self, closure=None):
        self.step_count += 1
        if self.alpha > self.min_alpha:
            self.alpha *= self.alpha_decay
        
        total_norm = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    total_norm += torch.sum(p.grad.data**2)
        total_norm = torch.sqrt(total_norm)
        
        self.state['running_norm'] = (self.ema_decay * self.state['running_norm'] + 
                                     (1 - self.ema_decay) * total_norm.item())
        
        theta = self.alpha * self.state['running_norm']
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                norm_sq = torch.sum(grad**2)
                scaling = theta / (theta + norm_sq + 1e-6)
                p.grad.data *= scaling
                
        super().step(closure)
