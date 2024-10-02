import torch
from transformers import MarianTokenizer
import pandas as pd
from dataset import CustomDataset
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

max_len = 100
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ko-en')
loaded = torch.load('Transformer_small.pt')
model = loaded['model']


def translation(model, src_text, atten_map_save = False):
    model.eval()
    with torch.no_grad():
        src = tokenizer.encode(src_text, return_tensors='pt', add_special_tokens=False).to(DEVICE)
        enc_mask = model.make_enc_mask(src)
        enc_out, atten_encs = model.encoder(src, enc_mask, atten_map_save)

        pred = tokenizer.encode('</s>', return_tensors='pt', add_special_tokens=False).to(DEVICE)
        for _ in range(max_len - 1):
            dec_mask = model.make_dec_mask(pred)
            enc_dec_mask = model.make_enc_dec_mask(src, pred)
            out, atten_decs, atten_enc_decs = model.decoder(pred, enc_out, dec_mask, enc_dec_mask, atten_map_save)

            pred_word = out[:, -1, :].argmax(dim=1).unsqueeze(0)
            pred = torch.cat([pred, pred_word], dim=1)

            if tokenizer.decode(pred_word.item()) == '</s>':
                break
        translated_text = tokenizer.decode(pred[0])
    return translated_text, atten_encs, atten_decs, atten_enc_decs

data = pd.read_excel('대화체.xlsx')
custom_DS = CustomDataset(data)
train_DS, val_DS, test_DS = torch.utils.data.random_split(custom_DS, [97000, 2000, 1000])
train_DL = torch.utils.data.DataLoader(train_DS, batch_size=64, shuffle=True)
val_DL = torch.utils.data.DataLoader(val_DS, batch_size=64, shuffle=True)
test_DL = torch.utils.data.DataLoader(test_DS, batch_size=64, shuffle=True)

src_text, trg_text = test_DS[3]
print(f"입력: {src_text}")
print(f"정답: {trg_text}")

translated_text, atten_encs, atten_decs, atten_enc_decs = translation(model, src_text, atten_map_save = True)
print(f"AI의 번역: {translated_text}")