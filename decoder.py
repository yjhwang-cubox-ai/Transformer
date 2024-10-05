import torch
import torch.nn as nn
from einops import rearrange
from mha import MHA, FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, drop_p):
        super().__init__()
        self.self_attn = MHA(d_model=d_model, n_heads=n_heads)
        self.self_attn_LN = nn.LayerNorm(d_model)
        
        self.enc_dec_attn = MHA(d_model=d_model, n_heads=n_heads)
        self.enc_dec_attn_LN = nn.LayerNorm(d_model)
        
        self.FF = FeedForward(d_model=d_model, d_ff=d_ff, drop_p=drop_p)
        self.FF_LN = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(drop_p)
    
    def forward(self, x, enc_out, dec_mask, enc_dec_mask):
        # 레이어의 디바이스를 가져옵니다.
        device = next(self.self_attn.parameters()).device

        # 입력 텐서를 레이어의 디바이스로 이동시킵니다.
        x = x.to(device)
        enc_out = enc_out.to(device)
        dec_mask = dec_mask.to(device)
        enc_dec_mask = enc_dec_mask.to(device)

        residual, atten_dec = self.self_attn(x, x, x, mask=dec_mask)
        x = x + self.dropout(residual)
        x = self.self_attn_LN(x)
        
        residual, atten_enc_dec = self.enc_dec_attn(x, enc_out, enc_out, mask=enc_dec_mask)
        x = x + self.dropout(residual)
        x = self.enc_dec_attn_LN(x)
        
        residual = self.FF(x)
        x = x + self.dropout(residual)
        x = self.FF_LN(x)
        
        return x, atten_dec, atten_enc_dec

class Decoder(nn.Module):
    def __init__(self, input_embedding, max_len, vocab_size, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()
        self.input_embedding = input_embedding
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        self.dropout = nn.Dropout(drop_p)
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, d_ff=d_ff, n_heads=n_heads, drop_p=drop_p) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, trg, enc_out, dec_mask, enc_dec_mask, atten_map_save=False):
        # trg 텐서의 디바이스를 가져옵니다.
        device = trg.device

        pos = torch.arange(trg.shape[1], device=device).expand_as(trg)
        
        x = self.scale * self.input_embedding(trg) + self.pos_embedding(pos)
        x = self.dropout(x)
        
        atten_decs = torch.tensor([], device=device)
        atten_enc_decs = torch.tensor([], device=device)
        
        for layer in self.layers:
            x, atten_dec, atten_enc_dec = layer(x, enc_out, dec_mask, enc_dec_mask)
            atten_decs = torch.cat((atten_decs, atten_dec.unsqueeze(0)), dim=0)
            atten_enc_decs = torch.cat((atten_enc_decs, atten_enc_dec.unsqueeze(0)), dim=0)
        
        out = self.fc_out(x)
        
        return out, atten_decs, atten_enc_decs