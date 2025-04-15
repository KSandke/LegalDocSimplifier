import torch
import json
import os
import yaml # For loading config
from transformers import AutoModel, AutoTokenizer, AutoConfig # Added AutoConfig
from torch import nn

"""
Multitask Inference Script

This script loads the trained multi-task classification model and provides a 
function to perform inference on new text inputs for the supported tasks 
(scotus, ledgar, unfair_tos).

Prerequisites:
1. A trained multi-task model saved by `train_multitask_classifier.py` 
   (Expected location configured in `config/config.yaml`, typically 
    `models/classification/multitask_legal_model_standardized/`).
2. The `config/config.yaml` file correctly pointing to the model and paths.
3. The necessary libraries installed (see requirements.txt).

How to Run Directly:
   python src/classification/multitask_inference.py

This will run the example usage section at the bottom, classifying sample texts.

How to Use in Other Code:
1. Import the `predict` function: 
   `from src.classification.multitask_inference import predict` 
   (Ensure model loading happens only once if importing).
2. Call `predict(your_text, desired_task_name)` where `desired_task_name` is 
   one of "scotus", "ledgar", or "unfair_tos".
"""

# --- Helper Function to Load Config --- 
# (Copied from training script - ideally import from a shared utils file)
def load_config(config_path="config/config.yaml"):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        return None

# --- LegalMultiTaskModel Definition ---
# (Copied from training script - ensure it's identical)
class LegalMultiTaskModel(nn.Module):
    def __init__(self, encoder_name, task_labels):
        super().__init__()
        # Load pre-trained encoder and its config
        self.encoder = AutoModel.from_pretrained(encoder_name)
        # Task-specific classification heads
        self.task_classifiers = nn.ModuleDict({
            # Ensure num_labels is an int
            task_name: nn.Linear(self.encoder.config.hidden_size, int(num_labels))
            for task_name, num_labels in task_labels.items()
        })
        self.task_labels = task_labels # Store num labels per task

    def forward(self, input_ids, attention_mask, task_name=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        if task_name is not None:
            if task_name not in self.task_classifiers:
                raise ValueError(f"Task '{task_name}' not found in model classifiers: {list(self.task_classifiers.keys())}")
            return self.task_classifiers[task_name](pooled_output)
        return {task: classifier(pooled_output) for task, classifier in self.task_classifiers.items()}

# --- Load Configuration ---
config = load_config()
if not config:
    print("Exiting: Could not load config.")
    exit()

paths_cfg = config.get('paths', {})
model_cfg = config.get('model', {}).get('multi_task_classification', {})

# Get model details from config
MODEL_SAVE_NAME = model_cfg.get('name', 'multitask_legal_model_standardized') # Use the correct name
BASE_MODEL_NAME = model_cfg.get('base_model', 'nlpaueb/legal-bert-base-uncased')
OUTPUT_DIR_TEMPLATE = paths_cfg.get('output_dir_template', 'models/classification/{model_name}')

# Construct the actual model directory
model_dir = OUTPUT_DIR_TEMPLATE.format(model_name=MODEL_SAVE_NAME)
task_labels_path = os.path.join(model_dir, "task_labels.json")
model_weights_path = os.path.join(model_dir, "model.pt")

print(f"Attempting to load model from: {model_dir}")

# --- Load Task Labels and Model Config ---
if not os.path.exists(model_dir) or not os.path.exists(task_labels_path):
     print(f"Error: Model directory or task_labels.json not found at {model_dir}")
     exit()

try:
    with open(task_labels_path, "r") as f:
        # This contains the number of labels per task, e.g., {'scotus': 13, ...}
        task_labels = json.load(f) 
    print(f"Loaded task label counts: {task_labels}")

    # Load the model's configuration file (config.json) saved by tokenizer/model.save_pretrained
    # This often contains the id2label mapping if saved correctly
    model_config = AutoConfig.from_pretrained(model_dir)
    print("Loaded model config.")

except Exception as e:
     print(f"Error loading task labels or model config: {e}")
     exit()

# --- Load Tokenizer ---
try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("Tokenizer loaded.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()

# --- Initialize Model ---
# Pass the loaded task label counts (number of classes per task)
model = LegalMultiTaskModel(BASE_MODEL_NAME, task_labels) 
print("Model structure initialized.")

# --- Load Model Weights ---
try:
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu'))) # Load to CPU first
    print("Model weights loaded.")
except Exception as e:
    print(f"Error loading model weights from {model_weights_path}: {e}")
    exit()

# --- Setup Device and Eval Mode ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # Set model to evaluation mode (disables dropout, etc.)
print(f"Model moved to {device} and set to eval mode.")

# --- Prediction Function ---
def predict(text, task_name):
    """
    Classifies the input text for the specified task using the loaded multi-task model.

    Args:
        text (str): The input text to classify.
        task_name (str): The target task (e.g., "scotus", "ledgar", "unfair_tos"). 
                         Must match a task the model was trained on.

    Returns:
        dict: A dictionary containing the prediction results:
              {
                  "task": task_name,
                  "predicted_label_id": int, 
                  "predicted_label_name": str, 
                  "confidence": float 
              }
              or {"error": str} if the task is not supported or an error occurs.
    """
    if task_name not in model.task_classifiers:
         return {"error": f"Task '{task_name}' is not supported by this model. Supported tasks: {list(model.task_classifiers.keys())}"}
         
    # Tokenize
    # Ensure tokenizer handles padding/truncation appropriately for single inputs
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    # Move tensors to the correct device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Get predictions
    label_name = "N/A" # Default label name
    pred_class_id = -1
    confidence_score = 0.0

    with torch.no_grad():
        logits = model(input_ids, attention_mask, task_name)
        
        # Check if logits is None or has unexpected shape
        if logits is None or logits.shape[0] != 1:
             print(f"Warning: Unexpected logits output shape for task {task_name}. Logits: {logits}")
             return {"error": "Model returned unexpected output."}

        probs = torch.softmax(logits, dim=1)
        confidence_score, pred_class_id_tensor = torch.max(probs, dim=1)
        
        pred_class_id = pred_class_id_tensor.item()
        confidence_score = confidence_score.item()

        # Attempt to get label name from model config's id2label mapping
        try:
            # Model config often stores mappings as strings, convert predicted ID to string key
            label_name = model_config.id2label.get(pred_class_id, "Unknown Label") 
            # If the keys themselves are integers, direct access might work:
            # label_name = model_config.id2label[pred_class_id]
        except AttributeError:
            print("Warning: Model config does not contain id2label mapping.")
        except KeyError:
            print(f"Warning: Predicted class ID {pred_class_id} not found in id2label mapping.")
        except Exception as e:
            print(f"Warning: Error accessing id2label mapping: {e}")
            
    return {
        "task": task_name,
        "predicted_label_id": pred_class_id,
        "predicted_label_name": label_name, # Added label name
        "confidence": confidence_score
    }

# --- Example Usage ---
if __name__ == "__main__":
    """
    Demonstrates how to use the predict function. 
    Loads the model and runs sample predictions when the script is executed directly.
    """
    sample_text_scotus = "The petitioner argues that the search violated the Fourth Amendment's protection against unreasonable searches and seizures."
    sample_text_ledgar = "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of laws principles."
    sample_text_unfair = "By using this service, you grant us a perpetual, irrevocable, worldwide, royalty-free license to use, modify, and distribute your content."
    
    print("\n--- Inference Examples ---")
    
    # Predict for each available task (should be 3 now)
    available_tasks = list(task_labels.keys()) # Get tasks from loaded config
    print(f"Available tasks for prediction: {available_tasks}")

    if "scotus" in available_tasks:
        result_scotus = predict(sample_text_scotus, "scotus")
        print(f"\nInput (SCOTUS): {sample_text_scotus[:100]}...")
        print(f"Prediction: {result_scotus}")

    if "ledgar" in available_tasks:
        result_ledgar = predict(sample_text_ledgar, "ledgar")
        print(f"\nInput (LEDGAR): {sample_text_ledgar[:100]}...")
        print(f"Prediction: {result_ledgar}")
        
    if "unfair_tos" in available_tasks:
        result_unfair = predict(sample_text_unfair, "unfair_tos")
        print(f"\nInput (Unfair-ToS): {sample_text_unfair[:100]}...")
        print(f"Prediction: {result_unfair}")

    # Example of predicting for a task the model wasn't trained on (or has no head for)
    # result_invalid = predict(sample_text_scotus, "casehold") 
    # print(f"\nPrediction (Invalid Task): {result_invalid}") 