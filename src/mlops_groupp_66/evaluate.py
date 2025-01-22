from sklearn.metrics import accuracy_score, classification_report
import torch
import os
from pathlib import Path
from model import FraudTransformer
from dotenv import load_dotenv
from data import get_transformer_dataloaders
import pandas as pd
from transformers import DistilBertTokenizer

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_transformer(data_loader):

    load_dotenv()
    model_checkpoint = Path(os.getenv("TRAINED_MODEL")).resolve()

    model = FraudTransformer().to(device)
    model.load_state_dict(torch.load(model_checkpoint))
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


if __name__ == "__main__":
    load_dotenv()
    processed_data_path = Path(os.getenv("PROCESSED_DATA")).resolve()
    data = pd.read_csv(processed_data_path)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    _, test_loader = get_transformer_dataloaders(data, tokenizer, max_len=128, batch_size=16)
    evaluate_transformer(test_loader)
