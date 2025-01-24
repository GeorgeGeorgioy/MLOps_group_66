import torch
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score
from transformers import DistilBertForSequenceClassification # noqa: F401
import matplotlib.pyplot as plt
import wandb
from transformers import AdamW
import seaborn as sns

# # class FraudNN(nn.Module):
# #     def __init__(self, input_size):
# #         super(FraudNN, self).__init__()
# #         self.model = nn.Sequential(
# #             nn.Linear(input_size, 64),
# #             nn.ReLU(),
# #             nn.Dropout(0.3),
# #             nn.Linear(64, 32),
# #             nn.ReLU(),
# #             nn.Dropout(0.3),
# #             nn.Linear(32, 1),
# #             nn.Sigmoid()
# #         )

# #     def forward(self, x):
# #         return self.model(x)


# class FraudTransformer(nn.Module):
#     def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
#         super(FraudTransformer, self).__init__()
#         self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

#     def forward(self, input_ids, attention_mask, labels=None):
#         return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)





class FraudTransformer(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model  # Use the passed model directly

    def forward(self, input_ids, attention_mask, labels=None):
        # Ensure that labels are passed if available, otherwise set to None
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        # Forward pass with labels for loss calculation
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Extract loss
        loss = outputs.loss

        # Log loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        accuracy = torch.tensor(accuracy_score(labels.cpu().numpy(), preds.cpu().numpy()))
        f1 = torch.tensor(f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted'))

        # Debug: Log the output
        print(f"Validation Step - Batch {batch_idx}: loss={loss}, accuracy={accuracy}, f1={f1}")

        return {"val_loss": loss, "accuracy": accuracy, "f1": f1}

    # def on_validation_epoch_end(self, outputs):
    #     pass

    def log_confusion_matrix(self, cm):
        fig = plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        wandb.log({"confusion_matrix": plt})
        plt.close(fig)

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=1e-5)
