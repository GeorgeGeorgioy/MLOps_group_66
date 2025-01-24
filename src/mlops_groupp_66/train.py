import torch
import pytorch_lightning as pl
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from model import FraudTransformer
from pathlib import Path
from dotenv import load_dotenv
from .data import get_transformer_dataloaders
import pandas as pd
import os
import wandb
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_transformer():
    # Load preprocessed data
    load_dotenv()
    processed_data_path = Path(os.getenv("PROCESSED_DATA")).resolve()
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Load the data
    data = pd.read_csv(processed_data_path)
    train_loader, _ = get_transformer_dataloaders(data, tokenizer, max_len=128, batch_size=16)
    
    # Initialize the model
    bert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model = FraudTransformer(bert_model).to(device)
    
    # Create PyTorch Lightning Trainer
    trainer = pl.Trainer(
                devices=1,  # Use 1 CPU or GPU
                accelerator="auto",  # Explicitly set to use CPU or GPU if available
                max_epochs=1)

    # Train the model
    trainer.fit(model, train_loader)

    # Save the trained model
    model_checkpoint_path = Path(os.getenv("TRAINED_MODEL")).resolve()
    torch.save(model.state_dict(), model_checkpoint_path)
    print(f"Model saved to {model_checkpoint_path}")

if __name__ == "__main__":
    train_transformer()



