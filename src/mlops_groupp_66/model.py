import torch.nn as nn
from transformers import DistilBertForSequenceClassification


class FraudNN(nn.Module):
    def __init__(self, input_size):
        super(FraudNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class FraudTransformer(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        super(FraudTransformer, self).__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

