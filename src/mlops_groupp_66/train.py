import torch
from transformers import AdamW
from transformers import DistilBertTokenizer
from .model import FraudTransformer
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

warnings.filterwarnings("ignore")



def train_transformer_model(model, train_loader, num_epochs=1, lr=5e-5):
    load_dotenv()
    save_model_path = Path(os.getenv("SAVE_MODEL")).resolve()
    save_model_path.mkdir(parents=True, exist_ok=True)




    wandb.init(
        project="MLOps_team_66",  # Change the project name if needed
        name="Transformer_Training",  # Optional: unique name for this run
        config={"learning_rate": lr, "epochs": num_epochs}
    )
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()


            wandb.log({"batch_loss": loss.item()}) # Log the loss for each batch
            epoch_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()    # Store predictions and true labels for metrics
            labels = batch['labels'].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

        # Log the average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch + 1})

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')  # For binary classification
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
        auc = roc_auc_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)

        # Log metrics to wandb
        wandb.log({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc
        })


        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(16, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')


        wandb.log({"confusion_matrix": wandb.Image(fig)})
        plt.close(fig)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss}")



    torch.save(model.state_dict(), save_model_path / 'fraud_transformer_model.pth')
    print(f"Model saved to {save_model_path}")

    # model_artifact = wandb.Artifact("trained_model", type="model")
    # model_artifact.add_file(str(save_model_path / 'fraud_transformer_model.pth'))

    # wandb.log({"model": model_artifact})

    wandb.finish()



if __name__ == "__main__":
    load_dotenv()
    processed_data_path = Path(os.getenv("PROCESSED_DATA")).resolve()
    save_model_path = Path(os.getenv("SAVE_MODEL")).resolve()
    save_model_path.mkdir(parents=True, exist_ok=True)


    model_file = save_model_path / "fraud_transformer_model.pth"
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    data = pd.read_csv(processed_data_path)
    train_loader_tf, test_loader_tf = get_transformer_dataloaders(data, tokenizer, max_len=128, batch_size=16)
    transformer_model = FraudTransformer().to("cuda" if torch.cuda.is_available() else "cpu")
    train_transformer_model(transformer_model, train_loader_tf, num_epochs=1, lr=5e-5)
    torch.save(transformer_model.state_dict(), save_model_path / 'fraud_transformer_model.pth')
    print(f"Transformer model saved to {model_file}")
