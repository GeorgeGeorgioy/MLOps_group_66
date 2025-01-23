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


def evaluate_transformer():
        
        # Load environment variables
        load_dotenv()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load preprocessed data
        processed_data_path = Path(os.getenv("PROCESSED_DATA")).resolve()
        data = pd.read_csv(processed_data_path)
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        _, test_loader = get_transformer_dataloaders(data, tokenizer, max_len=128, batch_size=16)

        # Load the trained model
        model_checkpoint_path = Path(os.getenv("TRAINED_MODEL")).resolve()
        bert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        model = FraudTransformer(bert_model).to(device)
        model.load_state_dict(torch.load(model_checkpoint_path))

        # Set up PyTorch Lightning Trainer
        trainer = pl.Trainer(
            devices=1,  # Use 1 GPU or CPU
            accelerator="auto",  # Automatically choose "gpu" or "cpu"
            logger=False,  # Disable logging for simplicity
            enable_progress_bar=True  # Show progress bar
        )

        # Run evaluation
        evaluation_result = trainer.validate(model, dataloaders=test_loader)  # Use validate for consistency

        # Display results
        print("Validation Results:", evaluation_result)


if __name__ == "__main__":
    evaluate_transformer()
