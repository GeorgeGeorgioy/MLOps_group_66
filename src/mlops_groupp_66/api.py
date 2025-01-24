from fastapi import FastAPI
from pydantic import BaseModel
import yaml
import os
import subprocess
from dotenv import load_dotenv
from pathlib import Path
from fastapi.responses import HTMLResponse

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

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Model Interface</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            label, input, button { display: block; margin-bottom: 10px; }
            button { padding: 10px 20px; cursor: pointer; }
            #response { margin-top: 20px; padding: 10px; border: 1px solid #ddd; background: #f9f9f9; }
        </style>
    </head>
    <body>
        <h1>Machine Learning Model Interface</h1>

        <h2>Train Model</h2>
        <form id="train-form">
            <label for="learning_rate">Learning Rate:</label>
            <input type="number" id="learning_rate" name="learning_rate" step="0.0001" value="0.001" required>

            <label for="batch_size">Batch Size:</label>
            <input type="number" id="batch_size" name="batch_size" value="32" required>

            <label for="epochs">Epochs:</label>
            <input type="number" id="epochs" name="epochs" value="10" required>

            <button type="button" id="train-btn">Train Model</button>
        </form>

        <h2>Evaluate Model</h2>
        <button id="evaluate-btn">Evaluate Model</button>

        <div id="response">Response will appear here...</div>

        <script>
            async function postTrain() {
                const formData = new FormData(document.getElementById("train-form"));
                const payload = Object.fromEntries(formData.entries());
                payload.learning_rate = parseFloat(payload.learning_rate);
                payload.batch_size = parseInt(payload.batch_size);
                payload.epochs = parseInt(payload.epochs);

                const response = await fetch('/train', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                document.getElementById('response').innerText = 'Train Response: ' + JSON.stringify(data);
            }

            async function getEvaluate() {
                const response = await fetch('/evaluate');
                const data = await response.json();
                document.getElementById('response').innerText = 'Evaluate Response: ' + JSON.stringify(data);
            }

            document.getElementById('train-btn').addEventListener('click', postTrain);
            document.getElementById('evaluate-btn').addEventListener('click', getEvaluate);
        </script>
    </body>
    </html>
    """
    return html_content


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
