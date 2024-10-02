import lightning as L
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.profilers import SimpleProfiler

import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from scheduler import NoamScheduler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Transformer(nn.Module):
    def __init__(self, vocab_size, pad_idx, max_len, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()
        
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = Encoder(self.input_embedding, max_len, n_layers, d_model, d_ff, n_heads, drop_p)
        self.decoder = Decoder(self.input_embedding, max_len, vocab_size, n_layers, d_model, d_ff, n_heads, drop_p)
        
        self.n_heads = n_heads
        self.pad_idx = pad_idx
        
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)
        
    def make_enc_mask(self, src): # src.shape: 개단
        enc_mask = (src == self.pad_idx).unsqueeze(1).unsqueeze(2) #개11단
        enc_mask = enc_mask.expand(src.shape[0], self.n_heads, src.shape[1], src.shape[1]) # 개헤단단
        
        return enc_mask
                
    def make_dec_mask(self, trg):
        trg_pad_mask = (trg == self.pad_idx).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = trg_pad_mask.expand(trg.shape[0], self.n_heads, trg.shape[1], trg.shape[1])
        
        trg_future_mask = torch.tril(torch.ones(trg.shape[0], self.n_heads, trg.shape[1], trg.shape[1]))==0 # 개헤단단
        trg_future_mask = trg_future_mask.to(DEVICE)
        
        dec_mask = trg_pad_mask | trg_future_mask # dec_mask.shape = 개헤단단
        
        return dec_mask

    def make_enc_dec_mask(self, src, trg):
        enc_dec_mask = (src == self.pad_idx).unsqueeze(1).unsqueeze(2) # 개11단
        enc_dec_mask = enc_dec_mask.expand(trg.shape[0], self.n_heads, trg.shape[1], src.shape[1]) # 개헤단단
        
        return enc_dec_mask
    
    def forward(self, src, trg):
        enc_mask = self.make_enc_mask(src)
        dec_mask = self.make_dec_mask(trg)
        enc_dec_mask = self.make_enc_dec_mask(src, trg)
        
        enc_out, atten_encs = self.encoder(src, enc_mask)
        out, atten_decs, atten_enc_decs = self.decoder(trg, enc_out, dec_mask, enc_dec_mask)
        
        return out, atten_encs, atten_decs, atten_enc_decs   


class LitTransformer(L.LightningModule):
    def __init__(self, vocab_size, pad_idx, max_len, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()
        
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = Encoder(self.input_embedding, max_len, n_layers, d_model, d_ff, n_heads, drop_p)
        self.decoder = Decoder(self.input_embedding, max_len, vocab_size, n_layers, d_model, d_ff, n_heads, drop_p)
        
        self.n_heads = n_heads
        self.pad_idx = pad_idx
        
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)
        
    def make_enc_mask(self, src): # src.shape: 개단
        enc_mask = (src == self.pad_idx).unsqueeze(1).unsqueeze(2) #개11단
        enc_mask = enc_mask.expand(src.shape[0], self.n_heads, src.shape[1], src.shape[1]) # 개헤단단
        
        return enc_mask
                
    def make_dec_mask(self, trg):
        trg_pad_mask = (trg == self.pad_idx).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = trg_pad_mask.expand(trg.shape[0], self.n_heads, trg.shape[1], trg.shape[1])
        
        trg_future_mask = torch.tril(torch.ones(trg.shape[0], self.n_heads, trg.shape[1], trg.shape[1]))==0 # 개헤단단
        trg_future_mask = trg_future_mask.to(DEVICE)
        
        dec_mask = trg_pad_mask | trg_future_mask # dec_mask.shape = 개헤단단
        
        return dec_mask

    def make_enc_dec_mask(self, src, trg):
        enc_dec_mask = (src == self.pad_idx).unsqueeze(1).unsqueeze(2) # 개11단
        enc_dec_mask = enc_dec_mask.expand(trg.shape[0], self.n_heads, trg.shape[1], src.shape[1]) # 개헤단단
        
        return enc_dec_mask
    
    def forward(self, src, trg):
        enc_mask = self.make_enc_mask(src)
        dec_mask = self.make_dec_mask(trg)
        enc_dec_mask = self.make_enc_dec_mask(src, trg)
        
        enc_out, atten_encs = self.encoder(src, enc_mask)
        out, atten_decs, atten_enc_decs = self.decoder(trg, enc_out, dec_mask, enc_dec_mask)
        
        return out, atten_encs, atten_decs, atten_enc_decs

    def configure_optimizers(self):
        LR_scale = 0.5
        LR_init = 5e-4
        LAMBDA = 0 # l2-Regularization를 위한 hyperparam. # 저장된 모델   
        warmup_steps = 4000
        d_model = 512

        optimizer = torch.optim.Adam(self.parameters(), lr=LR_init, betas=(0.9, 0.98), eps=1e-9, weight_decay=LAMBDA)
        scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps, LR_scale=LR_scale)

        # 옵티마이저와 스케줄러를 함께 반환
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'  # Optional: validation loss 기준으로 스케줄러 모니터링 가능
        }
    
    def training_step(self, src, trg):
        enc_mask = self.make_enc_mask(src)
        dec_mask = self.make_dec_mask(trg)
        enc_dec_mask = self.make_enc_dec_mask(src, trg)
        
        enc_out, atten_encs = self.encoder(src, enc_mask)
        out, atten_decs, atten_enc_decs = self.decoder(trg, enc_out, dec_mask, enc_dec_mask)