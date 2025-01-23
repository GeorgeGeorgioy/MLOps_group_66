from fastapi import FastAPI
from pydantic import BaseModel
import yaml
import os
import sys
import os
import subprocess
from dotenv import load_dotenv
from pathlib import Path
app = FastAPI()
load_dotenv()
# Define the path to the config file
CONFIG_PATH = "../../configs/config.yaml"

#train_scrip = Path(os.getenv("TRAIN_SCRIPT_PATH")).resolve()
EVALUATION_SCRIPT = Path(os.getenv("EVALUATION_SCRIPT_PATH")).resolve()
TRAIN_SCRIPT_PATH = Path(os.getenv("TRAIN_SCRIPT_PATH")).resolve()
#TRAIN_SCRIPT = Path("D:/VScode/New_MLOps_project/MLOps_group_66/src/train.py").resolve()
#TRAIN_SCRIPT_PATH = "D:/VScode/New_MLOps_project/MLOps_group_66/src/mlops_groupp_66/train.py"

# Model for the hyperparameters
class Hyperparameters(BaseModel):
    lr: float
    batch_size: int
    epochs: int
    max_len: int

@app.get("/")
def read_root():
    """ Health check."""

    return {"Hello": "World"}

# Endpoint to get the current hyperparameters
@app.get("/hyperparameters/")
def get_hyperparameters():
    try:
        with open(CONFIG_PATH, "r") as file:
            config = yaml.safe_load(file)
        return {"hyperparameters": config}
    except Exception as e:
        return {"error": str(e)}

# Endpoint to update the hyperparameters
@app.post("/hyperparameters/")
def update_hyperparameters(params: Hyperparameters):
    try:
        # Load the current config
        with open(CONFIG_PATH, "r") as file:
            config = yaml.safe_load(file)

        # Update the config with the new hyperparameters
        config['training']['lr'] = params.lr
        config['training']['batch_size'] = params.batch_size
        config['training']['epochs'] = params.epochs
        config['tokenizer']['max_len'] = params.max_len

        # Save the updated config back to the file
        with open(CONFIG_PATH, "w") as file:
            yaml.dump(config, file)

        return {"message": "Hyperparameters updated successfully", "new_hyperparameters": config}
    except Exception as e:
        return {"error": str(e)}
    
# Endpoint to trigger training
@app.get("/train/")
def trigger_training():
    try:
        # Run the train.py script
        result = subprocess.run(
            ["python", TRAIN_SCRIPT_PATH],
            capture_output=True,  # Capture stdout and stderr
            text=True  # Interpret output as text
        )

        # Check for errors
        if result.returncode == 0:
            return {
                "message": "Training script executed successfully",
                "output": result.stdout
            }
        else:
            return {
                "message": "Error during training script execution",
                "error": result.stderr
            }
    except Exception as e:
        return {"error": str(e)}
    
# Endpoint to trigger training
@app.get("/evaluate/")
def trigger_evaluation():
    try:
        # Run the train.py script
        result = subprocess.run(
            ["python", str(EVALUATION_SCRIPT)],
            capture_output=True,  # Capture stdout and stderr
            text=True  # Interpret output as text
        )

        # Check for errors
        if result.returncode == 0:
            return {
                "message": "Evaluation script executed successfully",
                "output": result.stdout
            }
        else:
            return {
                "message": "Error during evaluatio script execution",
                "error": result.stderr
            }
    except Exception as e:
        return {"error": str(e)}