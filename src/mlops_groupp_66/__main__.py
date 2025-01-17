import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data import MyDataset, get_transformer_dataloaders
from model import FraudTransformer
from train import train_transformer_model
from evaluate import evaluate_transformer
from transformers import DistilBertTokenizer
import torch
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Resolve paths
    raw_data_path = Path(os.getenv("RAW_DATA")).resolve()
    processed_data_path = Path(os.getenv("PROCESSED_DATA")).resolve()
    output_folder = Path(os.getenv("OUTPUT_FOLDER")).resolve()
    save_model_path = Path(os.getenv("SAVE_MODEL")).resolve()
    save_model_path.mkdir(parents=True, exist_ok=True)

    # Preprocessing data
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)
    data = pd.read_csv(processed_data_path)

    # # NN Model Workflow
    # train_loader_nn, test_loader_nn = get_nn_dataloaders(data, batch_size=32)
    # input_size = data.shape[1] - 1  # Exclude target column

    # nn_model = FraudNN(input_size).to("cuda" if torch.cuda.is_available() else "cpu")

    # print("Training Neural Network...")
    # train_nn_model(nn_model, train_loader_nn, num_epochs=10, lr=0.001)

    # print("Testing Neural Network...")
    # evaluate_nn(nn_model, test_loader_nn)

    # torch.save(nn_model.state_dict(), save_model_path / "fraud_nn_model.pth")
    # print(f"Neural Network model saved to {save_model_path / 'fraud_nn_model.pth'}")

    # Transformer Model Workflow
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_loader_tf, test_loader_tf = get_transformer_dataloaders(data, tokenizer, max_len=128, batch_size=16)
    transformer_model = FraudTransformer().to("cuda" if torch.cuda.is_available() else "cpu")

    print("Training Transformer...")
    train_transformer_model(transformer_model, train_loader_tf, num_epochs=3, lr=5e-5)

    print("Testing Transformer...")
    evaluate_transformer(transformer_model, test_loader_tf)

    torch.save(transformer_model.state_dict(), save_model_path / "fraud_nn_model.pth")
    print(f"Transformer model saved to {save_model_path / 'fraud_nn_model.pth'}")

if __name__ == "__main__":
    main()
