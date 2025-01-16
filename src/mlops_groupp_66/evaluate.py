from sklearn.metrics import accuracy_score, classification_report
import torch


def test_model(model, test_loader):
    model.eval()
    for batch in test_loader:
        with torch.no_grad():
            outputs = model(batch['input_ids'], batch['attention_mask'])
            print(outputs.logits)

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
