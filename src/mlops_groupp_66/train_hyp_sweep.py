############################################################

# Under constraction

###########################################################
import wandb
from transformers import AdamW, DistilBertTokenizer
import torch
from model import FraudTransformer
from pathlib import Path
from dotenv import load_dotenv
from data import get_transformer_dataloaders
import pandas as pd
import os
from sklearn.metrics import accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_evaluate(config=None):
    """Train and evaluate the model for a given hyperparameter configuration."""
    with wandb.init(config=config):  # Initialize a new wandb run
        config = wandb.config

        # Load data and model
        load_dotenv()
        processed_data_path = Path(os.getenv("PROCESSED_DATA")).resolve()
        data = pd.read_csv(processed_data_path)
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        train_loader, test_loader = get_transformer_dataloaders(
            data, tokenizer, max_len=config.max_len, batch_size=config.batch_size
        )

        model = FraudTransformer().to(device)
        optimizer = AdamW(model.parameters(), lr=config.lr)

        # Training loop
        for epoch in range(config.epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device)
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            # Log epoch loss
            wandb.log({"epoch": epoch + 1, "loss": loss.item()})

        # Evaluation loop
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device)
                )
                predictions = torch.argmax(outputs.logits, axis=-1).cpu().numpy()
                all_preds.extend(predictions)
                all_labels.extend(batch['labels'].cpu().numpy())

        # Calculate and log metrics
        accuracy = accuracy_score(all_labels, all_preds)
        wandb.log({"accuracy": accuracy})

        # Save the model if it's the best
        if wandb.run.resumed or accuracy > wandb.run.summary.get("best_accuracy", 0):
            wandb.run.summary["best_accuracy"] = accuracy
            torch.save(model.state_dict(), os.getenv("SAVE_MODEL") + "/best_model.pth")
            wandb.log_artifact(os.getenv("SAVE_MODEL") + "/best_model.pth")

def main():
    # Define sweep configuration
    sweep_config = {
        "method": "grid",  # Can be "random" or "bayes"
        "metric": {"name": "accuracy", "goal": "maximize"},
        "parameters": {
            "lr": {"values": [1e-5,1e-4]},  # Learning rate sweep
            "batch_size": {"values": [16,32,64]},    # Batch size sweep
            "epochs": {"value": 1},                # Fixed value
            "max_len": {"values": [128, 256]}      # Input sequence length
        },
    }

    # Initialize WandB sweep
    sweep_id = wandb.sweep(sweep_config, project="transformer_hyperparameter_sweep")

    # Run the sweep
    wandb.agent(sweep_id, train_and_evaluate)

if __name__ == "__main__":
    main()
