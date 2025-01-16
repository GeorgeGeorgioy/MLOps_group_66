import torch

def test_model(model, test_loader):
    model.eval()
    for batch in test_loader:
        with torch.no_grad():
            outputs = model(batch['input_ids'], batch['attention_mask'])
            print(outputs.logits)
