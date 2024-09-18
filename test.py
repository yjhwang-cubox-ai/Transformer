import torch
import torch.nn as nn
import random
from transformers import MarianMTModel, MarianTokenizer # MT: Machine Translation
from transformer import Transformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    model = Transformer(vocab_size=vocab_size, pad_idx=pad_idx, max_len=max_len, n_layers=n_layers, d_model=d_model, d_ff=d_ff, n_heads=n_heads, drop_p=drop_p).to(DEVICE)
    
    src = torch.tensor([[4,6,5,1,1,1],[7,7,1,1,1,1]]).to(DEVICE)
    trg = torch.tensor([[2,5,4,4,3,1,1],[2,9,6,7,3,1,1]]).to(DEVICE)
    
    print(model(src, trg)[0].shape)



if __name__ == "__main__":
    """=================================================================================================================
    hyperparameter 설정
    ================================================================================================================="""
    random_seed = 0
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(random_seed)

    # Load the tokenizer & model
    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ko-en')
    model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ko-en').to(DEVICE) # MT: Machine Translation

    eos_idx = tokenizer.eos_token_id
    pad_idx = tokenizer.pad_token_id

    BATCH_SIZE = 64 # 논문에선 2.5만 token이 한 batch에 담기게 했다고 함.
    LAMBDA = 0 # l2-Regularization를 위한 hyperparam. # 저장된 모델
    EPOCH = 15 # 저장된 모델
    max_len = 100 # 너무 긴거 같아서 자름 (GPU 부담도 많이 덜어짐)
    criterion = nn.CrossEntropyLoss(ignore_index = pad_idx) # pad token 이 출력 나와야하는 시점의 loss는 무시 (즉, label이 <pad> 일 때는 무시) # 저장된 모델

    scheduler_name = 'Noam'
    # scheduler_name = 'Cos'
    #### Noam ####
    # warmup_steps = 4000 # 이건 논문에서 제시한 값 (총 10만 step의 4%)
    warmup_steps = 1000 # 데이터 수 * EPOCH / BS = 총 step 수 인것 고려 # 저장된 모델
    LR_scale = 0.5 # Noam scheduler에 peak LR 값 조절을 위해 곱해질 녀석 # 저장된 모델
    #### Cos ####
    LR_init = 5e-4
    T0 = 1500 # 첫 주기
    T_mult = 2 # 배 만큼 주기가 길어짐 (1보다 큰 정수여야 함)
        
    save_model_path = 'Transformer_small.pt'
    save_history_path = 'Transformer_small_history.pt'

    vocab_size = tokenizer.vocab_size

    # 좀 사이즈 줄인 모델 (훈련된 input_embedding, fc_out 사용하면 사용 불가)
    n_layers = 3
    d_model = 256
    d_ff = 512
    n_heads = 8
    drop_p = 0.1
    
    main()