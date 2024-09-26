import torch
import torch.nn as nn
from transformers import MarianTokenizer # MT: Machine Translation
from transformer import Transformer
from scheduler import NoamScheduler
from task import Task
from dataset import CustomDataset
import random
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # tokenizer
    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ko-en')
    eos_idx = tokenizer.eos_token_id
    pad_idx = tokenizer.pad_token_id
    # scheduler 설정
    scheduler_name = 'Noam'
    warmup_steps = 1000
    LR_scale = 0.5
    LR_init = 5e-4
    T0 = 1500
    T_mult = 2
    # 모델 파라미터 설정
    n_layers = 3
    d_model = 256
    d_ff = 512
    n_heads = 8
    drop_p = 0.1
    # 하이퍼파라미터 설정
    BATCH_SIZE = 64 # 논문에선 2.5만 token이 한 batch에 담기게 했다고 함.
    LAMBDA = 0 # l2-Regularization를 위한 hyperparam. # 저장된 모델
    EPOCH = 15 # 저장된 모델
    max_len = 100 # 너무 긴거 같아서 자름 (GPU 부담도 많이 덜어짐)
    criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

    # dataset
    data = pd.read_excel('대화체.xlsx')
    custom_DS = CustomDataset(data)
    train_DS, val_DS, test_DS = torch.utils.data.random_split(custom_DS, [97000, 2000, 1000])
    train_DL = torch.utils.data.DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True)
    val_DL = torch.utils.data.DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=True)
    test_DL = torch.utils.data.DataLoader(test_DS, batch_size=BATCH_SIZE, shuffle=True)

    # model    
    model = Transformer(vocab_size=tokenizer.vocab_size, pad_idx=tokenizer.pad_token_id, max_len=max_len, n_layers=n_layers, d_model=d_model, d_ff=d_ff, n_heads=n_heads, drop_p=drop_p).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    # scheduler
    if scheduler_name == 'Noam':
        optimizer = torch.optim.Adam(params, lr=LR_init, betas=(0.9, 0.98), eps=1e-9, weight_decay=LAMBDA)
        scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps, LR_scale=LR_scale)
    
    # task
    task = Task(model, tokenizer, train_DL, val_DL, test_DL, criterion, optimizer, scheduler, DEVICE)
    task.train(EPOCH)

if __name__ == "__main__":
    main()