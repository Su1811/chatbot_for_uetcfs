import torch.nn as nn
from transformers import AutoModel


class BertBase(nn.Module):

    def __init__(self, n_classes, drop_out=0.1):
        super(BertBase, self).__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")
        self.classifier = nn.Sequential(
            nn.Dropout(drop_out),
            nn.Linear(self.bert.config.hidden_size, n_classes)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        return self.classifier(output.pooler_output)
