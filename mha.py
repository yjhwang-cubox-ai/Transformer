import torch
import torch.nn as nn
from einops import rearrange

class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        
        self.scale = torch.sqrt()
    
    def forward(self, Q, K, V, mask = None):
        Q = self.fc_q(Q) # 개, 단, 차
        K = self.fc_k(K)
        V = self.fc_v(V)
        
        Q = rearrange(Q, '개 단 (헤 차) -> 개 헤 단 차', 헤 = self.n_heads)
        K = rearrange(K, '개 단 (헤 차) -> 개 헤 단 차', 헤 = self.n_heads)
        V = rearrange(V, '개 단 (헤 차) -> 개 헤 단 차', 헤 = self.n_heads)
        
        attention_score = Q @ K.transpose(-2, -1) / self.scale
        
        if mask is not None:
            attention_score[mask] = -1e10
        
        attention_weights = torch.Softmax(attention_score, dim=-1) # 개헤단단
        
        attention = attention_weights @ V
        
        x = rearrange(attention, '개 헤 단 차 -> 개 단 (헤 차)')
        
        x = self.fc_o(x)
        return x, attention_weights