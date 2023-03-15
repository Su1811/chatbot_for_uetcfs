import re
import numpy as np
import pandas as pd
import py_vncorenlp
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer
from underthesea import word_tokenize, text_normalize


rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"])
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
MAX_LEN = 256
BATCH_SIZE = 32


def word_segment(text):
    return "".join(["".join(sen) for sen in rdrsegmenter.word_segment(text_normalize(text))])


def preprocess_text(text):
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), text, flags=re.IGNORECASE)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\w*\d\w*', ' ', text).strip()
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = text.lower()
    text = word_segment(text)
    return text


def encoding(data, tokenizer, max_token_len=128):
    contents = data.content
    input_ids = []
    attention_masks = []

    for index, content in enumerate(contents):
        encoded = tokenizer.encode_plus(
            content,
            truncation=True,
            add_special_tokens=True,
            max_length=max_token_len,
            padding="max_length",
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks


def data_preprocessing(content):
    content = [preprocess_text(content)]
    data = pd.DataFrame(content, columns=['content'])
    data_input_ids, data_attention_masks = encoding(data, tokenizer, max_token_len=MAX_LEN)
    return data_input_ids, data_attention_masks
