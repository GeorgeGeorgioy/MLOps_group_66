import os
from pathlib import Path
import torch
import pytorch_lightning as pl
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import pandas as pd
from data import get_transformer_dataloaders
from model import FraudTransformer  # Your model file
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_transformer():
    # Load preprocessed data
    load_dotenv()
    processed_data_path = Path(os.getenv("PROCESSED_DATA")).resolve()
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Load data
    data = pd.read_csv(processed_data_path)
    _, test_loader = get_transformer_dataloaders(data, tokenizer, max_len=128, batch_size=16)
    
    # Load the trained model
    model_checkpoint_path = Path(os.getenv("TRAINED_MODEL")).resolve()
    bert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model = FraudTransformer(bert_model).to(device)
    model.load_state_dict(torch.load(model_checkpoint_path))

    # Create PyTorch Lightning Trainer
    trainer = pl.Trainer(
                devices=1,  # Use 1 CPU or GPU
                accelerator="cpu",  # Explicitly set to use CPU or GPU if available
                max_epochs=1)

    # Evaluate the model on the test data
    result = trainer.test(model, dataloaders=test_loader)

    # Collect predictions and labels
    all_preds = []
    all_labels = []
    for res in result:
        all_preds.extend(res["predictions"].cpu().numpy())
        all_labels.extend(res["labels"].cpu().numpy())

    # Calculate accuracy and classification report
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Transformer Accuracy: {accuracy}")
    print(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    evaluate_transformer()
