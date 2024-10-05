import matplotlib.pyplot as plt

class NoamScheduler:
    # def __init__(self, optimizer, d_model, warmup_steps, LR_scale = 1):
    #     self.optimizer = optimizer
    #     self.current_steps = 0
    #     self.d_model = d_model
    #     self.warmup_steps = warmup_steps
    #     self.LR_scale = LR_scale
    
    # def step(self):
    #     self.current_steps += 1
    #     lrate = self.LR_scale * (self.d_model ** (-0.5) * min(self.current_steps ** (-0.5), self.current_steps * self.warmup_steps ** (-1.5)))
    #     self.optimizer.param_groups[0]['lr'] = lrate
    def __init__(self, optimizer, d_model, warmup_steps, LR_scale=1):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.LR_scale = LR_scale
        self._step = 0
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.d_model ** -0.5) * min(step ** -0.5, step * self.warmup_steps ** -1.5)

    def state_dict(self):
        return {
            'step': self._step,
            'rate': self._rate
        }

    def load_state_dict(self, state_dict):
        self._step = state_dict['step']
        self._rate = state_dict['rate']