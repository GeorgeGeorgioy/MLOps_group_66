from sklearn.metrics import accuracy_score, classification_report
import torch


def evaluate_model(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch['input_ids'], batch['attention_mask'])
            predictions = torch.argmax(outputs.logits, axis=-1).cpu().numpy()
            all_preds.extend(predictions)
            all_labels.extend(batch['labels'].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy}")
    print(classification_report(all_labels, all_preds))
