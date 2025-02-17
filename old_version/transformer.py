import lightning as L

import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from scheduler import NoamScheduler
# from torchtext.data import metrics

from transformers import MarianTokenizer # MT: Machine Translation

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
        # trg 텐서의 디바이스를 가져옵니다.
        device = trg.device

        trg_pad_mask = (trg == self.pad_idx).unsqueeze(1).unsqueeze(2).to(device)
        trg_pad_mask = trg_pad_mask.expand(trg.shape[0], self.n_heads, trg.shape[1], trg.shape[1])
        
        trg_future_mask = torch.tril(torch.ones(trg.shape[0], self.n_heads, trg.shape[1], trg.shape[1]))==0 # 개헤단단
        trg_future_mask = trg_future_mask.to(device)
        
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

class LitTransformerModule(L.LightningModule):
    def __init__(self, max_len = 512, n_layers = 6, d_model = 512, d_ff = 2048, n_heads = 8, drop_p = 0.1):
        super().__init__()
        torch.set_float32_matmul_precision('high')
        self.tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
        self.tokenizer.clean_up_tokenization_spaces = False
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_idx = self.tokenizer.pad_token_id
        self.max_len = max_len
        self.tokenizer_max_len = 100
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.drop_p = drop_p
        self.model = Transformer(self.vocab_size, self.pad_idx, self.max_len, self.n_layers, self.d_model, self.d_ff, self.n_heads, self.drop_p)
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.pad_idx)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-9, weight_decay=0)
        scheduler = NoamScheduler(optimizer, d_model=self.d_model, warmup_steps=4000, LR_scale=0.5)
        return [optimizer], [scheduler]
    
    def lr_scheduler_step(self, scheduler):
        scheduler.step()
    
    def training_step(self, batch):
        src_texts, trg_texts = batch
        src = self.tokenizer(src_texts, padding=True, truncation=True, max_length= self.tokenizer_max_len, return_tensors='pt', add_special_tokens = False).input_ids
        trg_texts = ['</s> ' + s for s in trg_texts]
        trg = self.tokenizer(trg_texts, padding=True, truncation=True, max_length = self.tokenizer_max_len, return_tensors='pt').input_ids

        # 데이터와 모델을 동일한 디바이스로 이동
        device = self.device
        src = src.to(device)
        trg = trg.to(device)
        self.model = self.model.to(device)

        y_hat = self.model(src, trg[:,:-1])[0] # 모델 통과 시킬 땐 trg의 마지막 토큰은 제외!
        loss = self.criterion(y_hat.permute(0,2,1), trg[:,1:])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch):
        src_texts, trg_texts = batch
        src = self.tokenizer(src_texts, padding=True, truncation=True, max_length= self.tokenizer_max_len, return_tensors='pt', add_special_tokens = False).input_ids
        trg_texts = ['</s> ' + s for s in trg_texts]
        trg = self.tokenizer(trg_texts, padding=True, truncation=True, max_length = self.tokenizer_max_len, return_tensors='pt').input_ids
        
        device = self.device
        src = src.to(device)
        trg = trg.to(device)
        self.model = self.model.to(device)
        
        y_hat = self.model(src, trg[:,:-1])[0] # 모델 통과 시킬 땐 trg의 마지막 토큰은 제외!
        loss = self.criterion(y_hat.permute(0,2,1), trg[:,1:])
        self.log('val_loss', loss, batch_size=len(batch[0]))
        return loss
    
    def test_step(self, batch):
        '''==================
        추후 BLUE score 구하는 로직 추가
        =================='''
        src_texts, trg_texts = batch
        src = self.tokenizer(src_texts, padding=True, truncation=True, max_length= self.tokenizer_max_len, return_tensors='pt', add_special_tokens = False).input_ids
        trg_texts = ['</s> ' + s for s in trg_texts]
        trg = self.tokenizer(trg_texts, padding=True, truncation=True, max_length = self.tokenizer_max_len, return_tensors='pt').input_ids
        y_hat = self.model(src, trg[:,:-1])[0] # 모델 통과 시킬 땐 trg의 마지막 토큰은 제외!
        loss = self.criterion(y_hat.permute(0,2,1), trg[:,1:])
        self.log('val_loss', loss)
        return loss