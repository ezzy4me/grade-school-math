import torch as th
from dataset import get_examples, GSMDataset
from transformers import BertTokenizer, BertForMaskedLM
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import string

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


def main():
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_examples = get_examples("train")
    train_dset = GSMDataset(bert_tokenizer, train_examples)

    device = th.device("cuda")
    config = GPT2Config.from_pretrained("gpt2")
    bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    bert_model.to(device)
    # model = th.nn.DataParallel(model).to(device) #lack of GPU
    bert_model.train()

    #original batch_size=16
    train_loader = DataLoader(train_dset, batch_size=8, shuffle=True)
    optim = AdamW(bert_model.parameters(), lr=1e-5)

    num_epochs = 2 #20
    num_training_steps = num_epochs * len(train_loader) #========
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    pbar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        for batch in train_loader:
            optim.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs[0]
            loss.backward()
            optim.step()
            lr_scheduler.step()
            pbar.update(1)
            pbar.set_description(f"train_loss: {loss.item():.5f}")

    model.save_pretrained("model_ckpts/")


"""
gsm8k데이터셋은 문제 + 정답풀이(풀이+정답) 으로 구성되어있다.
해당 과업을 bert를 통해 해결하고자 문제+풀이를 보여주고 정답을 예측하는 모델링을 했다.
하지만 이는 버트의 토큰 임베딩을 활용하지 않는다 판단되어 qa를 통해 해결하고자 다시 짠다.

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

top_k = 10

def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])

def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token) 
    # Replace the last correct answer with a <mask> token on GSM8K dataset.
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    # if tokenizer.mask_token == text_sentence.split()[-1]:
    #     text_sentence += ' .'

    input_ids = th.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = th.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx    

def get_all_predictions(text_sentence, top_clean=1):
    print(text_sentence)
    input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
    with th.no_grad():
        predict = bert_model(input_ids)[0]
    bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    return {'bert': bert}
"""