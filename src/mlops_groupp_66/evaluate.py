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
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:  # Unpack tuple
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            predictions = torch.argmax(outputs, axis=-1).cpu().numpy()
            all_preds.extend(predictions)
            all_labels.extend(labels.cpu().numpy().squeeze())  # Remove extra dimension for comparison

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"NN Accuracy: {accuracy}")
    print(classification_report(all_labels, all_preds))

