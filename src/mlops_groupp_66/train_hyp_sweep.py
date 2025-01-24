############################################################

# Under constraction

###########################################################
import wandb
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification
import torch
import pytorch_lightning as pl
from .model import FraudTransformer
from pathlib import Path
from dotenv import load_dotenv
from data import get_transformer_dataloaders
import pandas as pd
import os
import logging
import time
import warnings
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.DEBUG)

# Initialize the DistilBert model
bert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Initialize device and pass model to FraudTransformer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FraudTransformer(bert_model).to(device)


def train_and_evaluate(config=None):
    with wandb.init(config=config):
        config = wandb.config
        start_time = time.time()

        try:
            logging.info("Loading data...")
            # Load data and tokenizer
            load_dotenv()
            processed_data_path = Path(os.getenv("PROCESSED_DATA")).resolve()
            data = pd.read_csv(processed_data_path)
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            train_loader, test_loader = get_transformer_dataloaders(
                data, tokenizer, max_len=config.max_len, batch_size=config.batch_size
            )

            logging.info("Data loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

        data_loading_time = time.time() - start_time
        logging.info(f"Data loading took: {data_loading_time:.2f} seconds.")

        try:
            logging.info("Initializing WandB logger...")
            wandb_logger = pl.loggers.WandbLogger(project='transformer_hyperparameter_sweep')

            # Initialize model
            bert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
            model = FraudTransformer(bert_model).to(device)


            trainer = pl.Trainer(
                devices=1,  # Use 1 CPU or GPU
                accelerator="auto",  # Explicitly set to use CPU or GPU if available
                max_epochs=1,  # Example other arguments
                log_every_n_steps=1,  # Log frequency
                logger=wandb_logger,
            )

            logging.info("Training model...")
            # Train the model
            trainer.fit(model, train_dataloaders=train_loader)
            logging.info("Model trained successfully.")

            logging.info("Evaluating model...")
            # Evaluate the model
            trainer.validate(model, dataloaders=test_loader)
            logging.info("Model evaluated successfully.")

                    # Collect predictions and true labels for the confusion matrix
            all_preds = []
            all_labels = []

            model.eval()  # Ensure the model is in evaluation mode
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    # Forward pass
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=-1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Compute confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            logging.info(f"Confusion Matrix:\n{cm}")

            # Log confusion matrix to WandB
            model.log_confusion_matrix(cm)

            logging.info("Model evaluated successfully.")

            logging.info("Saving model...")
            # Save the best model
            torch.save(model.state_dict(), os.getenv("SAVE_MODEL") + "/best_model.pth")
            wandb.log_artifact(os.getenv("SAVE_MODEL") + "/best_model.pth")
        except Exception as e:
            logging.error(f"Error during training/evaluation: {e}")
            raise

def main():
    # Define sweep configuration
    sweep_config = {
        "method": "grid",  # Can be "random" or "bayes"
        "metric": {"name": "val_accuracy", "goal": "maximize"},
        "parameters": {
            "lr": {"values": [1e-5]},  # Learning rate sweep
            "batch_size": {"values": [8]},  # Batch size sweep
            "epochs": {"value": 1},  # Fixed number of epochs
            "max_len": {"values": [128]},  # Input sequence length
        },
    }

    # Initialize WandB sweep
    sweep_id = wandb.sweep(sweep_config, project="transformer_hyperparameter_sweep")

    # Run the sweep
    wandb.agent(sweep_id, train_and_evaluate)

if __name__ == "__main__":
    main()
