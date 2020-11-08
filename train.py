import json
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
from sympy import Integer
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer

from bert4torch.models import AutoregressionModel
from bert4torch.evaluator import Evaluator
from bert4torch.trainer import Seq2SeqTrainer
from common.data_helper import load_data, to_infix


# 基本参数
maxlen = 192
batch_size = 16
epochs = 50

pretrained_model = 'hfl/chinese-bert-wwm'

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained(pretrained_model)

class MathDataset(Dataset):

    def __init__(self, file_path):
        self._source = load_data(file_path)
    
    def __getitem__(self, idx):
        return self._source[idx]
    
    def __len__(self):
        return len(self._source)


def collate_fn(batch):
    batch_token_ids, batch_segment_ids = [], []
    for is_end, (question, equation, answer) in batch:
        token_dict = tokenizer(question, equation, max_length=maxlen)
        batch_token_ids.append(torch.LongTensor(token_dict['input_ids']))
        batch_segment_ids.append(torch.LongTensor(token_dict['token_type_ids']))
    batch_token_ids = pad_sequence(batch_token_ids, batch_first=True)
    batch_segment_ids = pad_sequence(batch_segment_ids, batch_first=True)
    return batch_token_ids, batch_segment_ids


train_dataset = MathDataset('dataset/ape210k/train.ape.json')
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
valid_loader = load_data('dataset/ape210k/valid.ape.json')


class MathEvaluator(Evaluator):

    def evaluate(self, valid_loader, model):
        total, right = 0, 0
        for question, equation, answer in valid_loader:
            total += 1
            pred_equation = model.generate(question)
            try:
                pred_equation = to_infix(pred_equation)
                pred_equation = pred_equation.replace('^', '**')
                right += int(is_equal(eval(pred_equation), eval(answer)))
            except:
                pass
        print('acc: {}'.format(right / total))
        # return {'acc': right / total}


model = AutoregressionModel(language_model='unilm', pretrained=pretrained_model)

trainer = Seq2SeqTrainer(
        model=model, 
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        evaluator=MathEvaluator())

trainer.train(max_epoch=10, lr=2e-5)
