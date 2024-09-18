import torch
import torch.nn as nn
from einops import rearrange
from mha import MHA, FeedForward

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, drop_p):
        super().__init__()
        
        self.self_attn = MHA(d_model=d_model, n_heads=n_heads)
        self.self_attn_LN = nn.LayerNorm(d_model)
        
        self.enc_dec_atten = MHA(d_model=d_model, n_heads=n_heads)
        self.enc_dec_atten_LN = nn.LayerNorm(d_model)
        
        self.FF = FeedForward(d_model=d_model, d_ff=d_ff, drop_p=drop_p)
        self.FF_LN = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(drop_p)
        
    def forward(self, x, enc_out, dec_mask, enc_dec_mask):
        residual, atten_dec = self.self_attn(x, x, x, mask=dec_mask)
        x = x + self.dropout(residual)
        x = self.self_attn_LN(x)
        
        residual, atten_enc_dec = self.enc_dec_atten(x, enc_out, enc_out, enc_dec_mask)
        x = x + self.dropout(residual)
        x = self.enc_dec_atten_LN(x)
        
        residual = self.FF(x)
        x = x + self.dropout(residual)
        x = self.FF_LN(x)
        
        return x, atten_dec, atten_enc_dec

class Decoder(nn.Module):
    def __init__(self, input_embedding, max_len, vocab_size, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()
        
        self.scale = torch.sqrt(torch.tensor(d_model))
        self.input_embedding = input_embedding
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        self.dropout = nn.Dropout(drop_p)
        
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, d_ff=d_ff, n_heads=n_heads, drop_p=drop_p) for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, trg, enc_out, dec_mask, enc_dec_mask, atten_map_save = False):
        pos = torch.arange(trg.shape[1]).expand_as(trg).to(DEVICE)
        
        x = self.scale*self.input_embedding(trg) + self.pos_embedding(pos)
        x = self.dropout(x)
        
        atten_decs = torch.tensor([]).to(DEVICE)
        atten_enc_decs = torch.tensor([]).to(DEVICE)
        for layer in self.layers:
            x, atten_dec, atten_enc_dec = layer(x, enc_out, dec_mask, enc_dec_mask)
            if atten_map_save is True:
                atten_decs = torch.cat([atten_decs, atten_dec[0].unsqueeze(0)], dim=0)
                atten_enc_decs = torch.cat([atten_enc_decs, atten_enc_dec[0].unsqueeze(0)], dim=0)
                
            x = self.fc_out(x)
            
            return x, atten_decs, atten_enc_decs