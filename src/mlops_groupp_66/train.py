from transformers import AdamW
from transformers import DistilBertTokenizer
import torch
from model import FraudTransformer
from pathlib import Path
from dotenv import load_dotenv
from data import get_transformer_dataloaders
import pandas as pd
import os

# def train_nn_model(model, train_loader, num_epochs=10, lr=0.001):
#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     for epoch in range(num_epochs):
#         model.train()
#         for batch, inputs, labels in enumerate(train_loader):
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

# def train_nn_model(model, train_loader, num_epochs=10, lr=0.001):
#     criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     model.to(device)
#     print(f"Training on {device}...")

#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for batch_idx, (inputs, labels) in enumerate(train_loader):
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)

#             # Check output shape
#             print(f"Batch {batch_idx+1}, Output Shape: {outputs.shape}")

#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#             if (batch_idx + 1) % 10 == 0:  # Print loss every 10 batches
#                 print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

#         print(f"Epoch {epoch+1}, Avg Loss: {running_loss / len(train_loader):.4f}")


def train_transformer_model(model, train_loader, num_epochs=3, lr=5e-5):
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
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


if __name__ == "__main__":
    load_dotenv()
    processed_data_path = Path(os.getenv("PROCESSED_DATA")).resolve()
    save_model_path = Path(os.getenv("SAVE_MODEL")).resolve()
    save_model_path.mkdir(parents=True, exist_ok=True)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    data = pd.read_csv(processed_data_path)
    train_loader_tf, test_loader_tf = get_transformer_dataloaders(data, tokenizer, max_len=128, batch_size=16)
    transformer_model = FraudTransformer().to("cuda" if torch.cuda.is_available() else "cpu")
    train_transformer_model(transformer_model, train_loader_tf, num_epochs=3, lr=5e-5)
    torch.save(transformer_model.state_dict(), save_model_path / "fraud_transformer_model.pth")
    print(f"Transformer model saved to {save_model_path / 'fraud_nn_model.pth'}")