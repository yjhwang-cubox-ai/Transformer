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
        
        self.scale = torch.sqrt(torch.tensor(d_model/n_heads))
    
    def forward(self, Q, K, V, mask = None):
        # 레이어의 디바이스를 가져옵니다.
        device = next(self.fc_q.parameters()).device

        # 입력 텐서를 레이어의 디바이스로 이동시킵니다.
        Q = Q.to(device)
        K = K.to(device)
        V = V.to(device)    


        Q = self.fc_q(Q) # 개, 단, 차
        K = self.fc_k(K)
        V = self.fc_v(V)
        
        Q = rearrange(Q, '개 단 (헤 차) -> 개 헤 단 차', 헤 = self.n_heads)
        K = rearrange(K, '개 단 (헤 차) -> 개 헤 단 차', 헤 = self.n_heads)
        V = rearrange(V, '개 단 (헤 차) -> 개 헤 단 차', 헤 = self.n_heads)
        
        attention_score = Q @ K.transpose(-2, -1) / self.scale
        
        if mask is not None:
            attention_score[mask] = -1e10
        
        attention_weights = torch.softmax(attention_score, dim=-1) # 개헤단단
        
        attention = attention_weights @ V
        
        x = rearrange(attention, '개 헤 단 차 -> 개 단 (헤 차)')
        
        x = self.fc_o(x)
        return x, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_p):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        return self.linear(x)