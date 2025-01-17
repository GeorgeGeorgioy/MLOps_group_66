from sklearn.metrics import accuracy_score, classification_report
import torch

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_transformer(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            # The transformer model expects input_ids and attention_mask
            outputs = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            predictions = torch.argmax(outputs.logits, axis=-1).cpu().numpy()
            all_preds.extend(predictions)
            all_labels.extend(batch['labels'].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Transformer Accuracy: {accuracy}")
    print(classification_report(all_labels, all_preds))

def evaluate_nn(model, data_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()  # Set the model to evaluation mode
    all_preds, all_labels = [], []

    with torch.no_grad():  # No gradients needed for evaluation
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get the model's predictions (probabilities)
            outputs = model(inputs)

            # Convert probabilities to binary predictions (0 or 1)
            predictions = (outputs > 0.5).float().cpu().numpy()  # Apply threshold
            all_preds.extend(predictions)
            all_labels.extend(labels.cpu().numpy().squeeze())  # Remove extra dimension for comparison

    # Calculate and print accuracy and classification report
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"NN Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(all_labels, all_preds))
