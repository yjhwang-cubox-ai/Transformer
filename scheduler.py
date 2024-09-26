import matplotlib.pyplot as plt

class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps, LR_scale = 1):
        self.optimizer = optimizer
        self.current_steps = 0
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.LR_scale = LR_scale
    
    def step(self):
        self.current_steps += 1
        lrate = self.LR_scale * (self.d_model ** (-0.5) * min(self.current_steps ** (-0.5), self.current_steps * self.warmup_steps ** (-1.5)))
        self.optimizer.param_groups[0]['lr'] = lrate