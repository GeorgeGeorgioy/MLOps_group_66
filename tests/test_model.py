import unittest
from unittest.mock import patch, MagicMock
import torch
from mlops_groupp_66.model import FraudTransformer
from transformers import DistilBertForSequenceClassification

class TestFraudTransformer(unittest.TestCase):
    @patch("src.mlops_groupp_66.model.DistilBertForSequenceClassification")
    def test_forward(self, mock_distilbert):
        # Mock the model's output
        mock_model = MagicMock()
        mock_model.return_value = {"loss": torch.tensor(0.5), "logits": torch.tensor([[0.1, 0.9]])}
        mock_distilbert.from_pretrained.return_value = mock_model
        bert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        model = FraudTransformer(bert_model)
        input_ids = torch.tensor([[101, 2054, 2003, 1996, 102]])  # Example input
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])  # Example attention mask
        labels = torch.tensor([1])  # Example label

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        self.assertIn("loss", outputs)
        self.assertIn("logits", outputs)
