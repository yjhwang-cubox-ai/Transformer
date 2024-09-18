import torch
import torch.nn as nn
from einops import rearrange
from mha import MHA, FeedForward

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class EncodeLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, drop_p):
        super().__init__()
        self.self_attn = MHA(d_model=d_model, n_heads=n_heads)
        self.self_attn_LN = nn.LayerNorm(d_model)
        
        self.FF = FeedForward(d_model=d_model, d_ff=d_ff, drop_p=drop_p)
        self.FF_LN = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(drop_p)
    
    def forward(self, x, enc_mask):
        residual, atten_enc = self.self_attn(x, x, x, mask=enc_mask)
        x = x + self.dropout(residual)
        x = self.self_attn_LN(x)
        
        residual = self.FF(x)
        x = x + self.dropout(residual)
        x = self.FF_LN(x)
        
        return x, atten_enc
    
class Encoder(nn.Module):
    def __init__(self, input_embedding, max_len, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()
        
        self.scale = torch.sqrt(torch.tensor(d_model))
        self.input_embedding = input_embedding
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        self.dropout = nn.Dropout(drop_p)
        
        self.layers = nn.ModuleList([EncodeLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, drop_p=drop_p) for _ in range(n_layers)])

    def forward(self, src, mask, atten_map_save = False):
        pos = torch.arange(src.shape[1]).expand_as(src).to(DEVICE)
        
        x = self.scale*self.input_embedding(src) + self.pos_embedding(pos)
        x = self.dropout(x)
        
        atten_encs = torch.tensor([]).to(DEVICE)
        for layer in self.layers:
            x, atten_enc = layer(x, mask)
            if atten_map_save is True:
                atten_encs = torch.cat([atten_encs, atten_enc[0].unsqueeze(0)], dim=0)
        
        return x, atten_encs
            