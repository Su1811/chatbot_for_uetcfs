import torch
import torch.nn as nn
from transformers import *


class BertLSTM(nn.Module):

    def __init__(self, n_classes, drop_out=0.1):
        super(BertLSTM, self).__init__()
        self.n_classes = n_classes
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")
        self.lstm = nn.LSTM(768, 128, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(256, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = output[0]
        enc_hiddens, (last_hidden, last_cell) = self.lstm(output)
        output = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        output = self.dropout(output)
        output = self.fc(output)
        if self.n_classes == 1:
            return self.sigmoid(output)
        else:
            return output
