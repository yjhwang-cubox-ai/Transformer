import torch
import torch.nn as nn
from einops import rearrange
from mha import MHA, FeedForward

class EncodeLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, drop_p):
        super().__init__()
        self.self_attn = MHA(d_model=d_model, n_heads=n_heads)
        self.self_attn_LN = nn.LayerNorm(d_model)
        
        self.FF = FeedForward(d_model=d_model, d_ff=d_ff, drop_p=drop_p)
        self.FF_LN = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(drop_p)
    
    def forward(self, x, enc_mask):
        # 레이어의 디바이스를 가져옵니다.
        device = next(self.self_attn.parameters()).device
        # print("encoder device: ", device)
        # 입력 텐서를 레이어의 디바이스로 이동시킵니다.
        x = x.to(device)
        enc_mask = enc_mask.to(device)

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
        pos = torch.arange(src.shape[1]).expand_as(src)

        # src와 pos가 동일한 디바이스에 있는지 확인
        src_device = src.device
        pos_device = pos.device

        # input_embedding과 pos_embedding의 가중치가 동일한 디바이스에 있는지 확인
        input_embedding_device = next(self.input_embedding.parameters()).device
        pos_embedding_device = next(self.pos_embedding.parameters()).device

        # 동일한 디바이스로 이동
        if src_device != input_embedding_device:
            self.input_embedding = self.input_embedding.to(src_device)
        if pos_device != pos_embedding_device:
            self.pos_embedding = self.pos_embedding.to(pos_device)

        # input_embedding과 pos_embedding을 동일한 디바이스로 이동
        input_emb = self.input_embedding(src).to(src_device)
        pos_emb = self.pos_embedding(pos).to(src_device)

        # x 계산
        x = self.scale * input_emb + pos_emb
        
        # x = self.scale*self.input_embedding(src) + self.pos_embedding(pos)
        x = self.dropout(x)
        
        atten_encs = torch.tensor([])
        for layer in self.layers:
            x, atten_enc = layer(x, mask)
            if atten_map_save is True:
                atten_encs = torch.cat([atten_encs, atten_enc[0].unsqueeze(0)], dim=0)
        
        return x, atten_encs