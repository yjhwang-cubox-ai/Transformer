import torch
import math
from tqdm import tqdm

class Task:
    def __init__(self, model, tokenizer, train_DL, val_DL, test_DL, criterion, optimizer, scheduler, DEVICE):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_max_len = 100
        self.train_DL = train_DL
        self.val_DL = val_DL
        self.test_DL = test_DL
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.DEVICE = DEVICE
        self.save_model_path = 'Transformer_small.pt'
        self.save_history_path = 'Transformer_small_history.pt'
    
    def train(self, epoch):        
        loss_history = {"train": [], "val": []}
        best_loss = 9999
        for ep in range(epoch):
            self.model.train() # train mode로 전환
            train_loss = self.loss_epoch(self.train_DL)
            loss_history["train"].append(train_loss)

            self.model.eval() # test mode로 전환
            with torch.no_grad():
                val_loss = self.loss_epoch(self.val_DL)
                loss_history["val"] += [val_loss]
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save({"model": self.model,
                                "ep": ep,
                                "optimizer": self.optimizer,
                                "scheduler": self.scheduler,}, self.save_model_path)
            
            # print loss
            print(f"Epoch {ep+1}: train loss: {train_loss:.5f}   val loss: {val_loss:.5f}   current_LR: {self.optimizer.param_groups[0]['lr']:.8f}")
            print("-" * 20)

        torch.save({"loss_history": loss_history,
                "EPOCH": epoch,
                "BATCH_SIZE": self.train_DL.batch_size}, self.save_history_path)
    
    def test(self):
        self.model.eval() # test mode로 전환
        with torch.no_grad():
            test_loss = self.loss_epoch(self.test_DL)
        print(f"Test loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}")
    
    def loss_epoch(self, DL):
        N = len(DL.dataset)
        rloss = 0
        for src_texts, trg_texts in tqdm(DL, leave=False):
            src = self.tokenizer(src_texts, padding=True, truncation=True, max_length= self.tokenizer_max_len, return_tensors='pt', add_special_tokens = False).input_ids.to(self.DEVICE)
            trg_texts = ['</s> ' + s for s in trg_texts]
            trg = self.tokenizer(trg_texts, padding=True, truncation=True, max_length = self.tokenizer_max_len, return_tensors='pt').input_ids.to(self.DEVICE)
            # inference
            y_hat = self.model(src, trg[:,:-1])[0] # 모델 통과 시킬 땐 trg의 마지막 토큰은 제외!
            # y_hat.shape = 개단차 즉, 훈련 땐 문장이 한번에 튀어나옴
            # loss
            loss = self.criterion(y_hat.permute(0,2,1), trg[:,1:]) # loss 계산 시엔 <sos> 는 제외!
            """
            개단차 -> 개차단으로 바꿔줌 (1D segmentation으로 생각)
            개채행열(예측), 개행열(정답)으로 주거나 개채1열, 개1열로 주거나 개채열, 개열로 줘야하도록 함수를 만들어놔서
            우리 상황에서는 개차단, 개단 으로 줘야 한다.
            이렇게 함수를 만들어놔야 1D, 2D segmentation 등등으로 확장가능하기 때문
            다 필요없고, 그냥 y_hat=개차단, trg=개단으로 줘야만 계산 제대로 된다고 생각하시면 됩니다!
            """
            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            # loss accumulation
            loss_b = loss.item() * src.shape[0]
            rloss += loss_b
        
        loss_e = rloss / N
        return loss_e