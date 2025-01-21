import sys
import os
from pathlib import Path
import torch
import pandas as pd
from transformers import DistilBertTokenizer
from dotenv import load_dotenv
import hydra
from omegaconf import OmegaConf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data import MyDataset, get_transformer_dataloaders
from model import FraudTransformer
from train import train_transformer_model
from evaluate import evaluate_transformer
from loguru import logger

load_dotenv()

@hydra.main(version_base="1.1",config_path="../../configs", config_name="config")
def main(cfg):


    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    logger.add(os.path.join(hydra_path, "logs.log"))
    logger.info(cfg)

    logger.debug("Used for debugging your code.")
    logger.info("Informative messages from your code.")
    logger.warning("Everything works but there is something to be aware of.")
    logger.error("There's been a mistake with the process.")
    logger.critical("There is something terribly wrong and process may terminate.")


    logger.info("Starting the pipeline...")
    logger.debug(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    try:
        raw_data_path = Path(os.getenv("RAW_DATA")).resolve()
        processed_data_path = Path(os.getenv("PROCESSED_DATA")).resolve()
        output_folder = Path(os.getenv("OUTPUT_FOLDER")).resolve()
        save_model_path = Path(os.getenv("SAVE_MODEL")).resolve()
        save_model_path.mkdir(parents=True, exist_ok=True)
        model_path = save_model_path / "fraud_transformer_model.pth"
        logger.info(f"Resolved paths successfully: raw_data_path={raw_data_path}, processed_data_path={processed_data_path}")
    except Exception as e:
        logger.critical("Error resolving paths from environment variables", exc_info=True)
        sys.exit(1)
   
    try:
        logger.info("Preprocessing dataset...")
        dataset = MyDataset(raw_data_path)
        dataset.preprocess(output_folder)
        data = pd.read_csv(processed_data_path)
        logger.info("Data preprocessing completed successfully.")
    except Exception as e:
        logger.error("Error during data preprocessing", exc_info=True)
        sys.exit(1)
    
    
    # try:
    #     tokenizer = DistilBertTokenizer.from_pretrained(cfg.tokenizer.name)
    #     train_loader_tf, test_loader_tf = get_transformer_dataloaders(
    #         data, tokenizer, max_len=cfg.tokenizer.max_len, batch_size=cfg.training.batch_size
    #     )
    #     transformer_model = FraudTransformer().to("cuda" if torch.cuda.is_available() else "cpu")

        
    #     logger.info("Training Transformer model...")
    #     train_transformer_model(
    #         transformer_model, train_loader_tf, num_epochs=cfg.training.num_epochs, lr=cfg.training.lr
    #     )
    #     logger.info("Training Transformer finshed...")

   
    #     logger.info("Evaluating Transformer model...")
    #     evaluate_transformer(transformer_model, test_loader_tf)

        
    #     model_path = save_model_path / "fraud_transformer_model.pth"
    #     torch.save(transformer_model.state_dict(), model_path)
    #     logger.info(f"Transformer model saved to {model_path}")
    # except Exception as e:
    #     logger.error("Error in Transformer model workflow", exc_info=True)
    #     sys.exit(1)


    tokenizer = DistilBertTokenizer.from_pretrained(cfg.tokenizer.name)
    train_loader_tf, test_loader_tf = get_transformer_dataloaders(
            data, tokenizer, max_len=cfg.tokenizer.max_len, batch_size=cfg.training.batch_size
        )
    transformer_model = FraudTransformer().to("cuda" if torch.cuda.is_available() else "cpu")
  
        
        
    train_transformer_model(
            transformer_model, train_loader_tf, num_epochs=cfg.training.num_epochs, lr=cfg.training.lr
        )
 
    #evaluate_transformer(transformer_model, test_loader_tf)
    evaluate_transformer(test_loader_tf)
        
    
    # torch.save(transformer_model.state_dict(), model_path)
    # logger.info(f"Transformer model saved to {model_path}")



    
    
    logger.info("Pipeline completed successfully.")
    
    
if __name__ == "__main__":
    main()
