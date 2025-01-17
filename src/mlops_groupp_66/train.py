import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AdamW


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

def train_nn_model(model, train_loader, num_epochs=10, lr=0.001):
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    print(f"Training on {device}...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # Check output shape
            print(f"Batch {batch_idx+1}, Output Shape: {outputs.shape}")

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:  # Print loss every 10 batches
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1}, Avg Loss: {running_loss / len(train_loader):.4f}")


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
