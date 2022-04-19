import torch as th
from dataset import get_examples, GSMDataset
from transformers import BertTokenizer, BertForMaskedLM
from transformers import AdamW #GPT2Config
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import string

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()

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
    # ========================= BERT =================================
    print(text_sentence)
    input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
    with th.no_grad():
        predict = bert_model(input_ids)[0]
    bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    return {'bert': bert}
