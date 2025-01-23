import torch
import pytorch_lightning as pl
from transformers import DistilBertTokenizer
from model import FraudTransformer
from pathlib import Path
from dotenv import load_dotenv
from data import get_transformer_dataloaders
import pandas as pd
import os
import wandb
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Define PyTorch Lightning's callback for logging metrics to wandb
class WandbLogger(pl.loggers.WandbLogger):
    def __init__(self, *args, **kwargs):
        super(WandbLogger, self).__init__(*args, **kwargs)

    def log_metrics(self, metrics, step):
        # Override the log_metrics function to handle custom metrics
        super(WandbLogger, self).log_metrics(metrics, step)

def train_transformer_model(model, train_loader, val_loader, num_epochs=1, lr=5e-5):
    load_dotenv()
    save_model_path = Path(os.getenv("SAVE_MODEL")).resolve()
    save_model_path.mkdir(parents=True, exist_ok=True)

    # Initialize the PyTorch Lightning trainer
    wandb_logger = WandbLogger(project="MLOps_team_66", name="Transformer_Training")
    
    trainer = pl.Trainer(
            devices=1,  # Use 1 CPU
            accelerator="cpu",  # Explicitly set to use CPU
            max_epochs=3,  # Example other arguments
            log_every_n_steps=1,  # Log frequency
        )

    # Train the model
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

    # Save the model after training
    torch.save(model.state_dict(), save_model_path / 'fraud_transformer_model.pth')
    print(f"Model saved to {save_model_path}")

if __name__ == "__main__":
    load_dotenv()
    processed_data_path = Path(os.getenv("PROCESSED_DATA")).resolve()
    save_model_path = Path(os.getenv("SAVE_MODEL")).resolve()
    save_model_path.mkdir(parents=True, exist_ok=True)

    model_file = save_model_path / "fraud_transformer_model.pth"
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    data = pd.read_csv(processed_data_path)
    train_loader_tf, test_loader_tf = get_transformer_dataloaders(data, tokenizer, max_len=128, batch_size=16)

    # Initialize the model and move it to the correct device
    transformer_model = FraudTransformer().to("cuda" if torch.cuda.is_available() else "cpu")

    # Start the training process
    train_transformer_model(transformer_model, train_loader_tf, test_loader_tf, num_epochs=1, lr=5e-5)

    print(f"Transformer model saved to {model_file}")


