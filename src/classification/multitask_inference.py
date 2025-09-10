import torch
import json
import os
import yaml
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch import nn
from datasets import load_from_disk
import datasets
"""
Multitask Inference Script

This script loads the trained multi-task classification model and provides a 
function to perform inference on new text inputs for the supported tasks 
(scotus, ledgar, unfair_tos).

Prerequisites:
1. A trained multi-task model saved by `train_multitask_classifier.py` 
   (Expected location configured in `config/config.yaml`).
2. The `config/config.yaml` file correctly pointing to the model and paths.
3. The necessary libraries installed (see requirements.txt).

How to Run Directly:
   python src/classification/multitask_inference.py

This will run the example usage section at the bottom, classifying sample texts.

How to Use in Other Code:
1. Import the `predict` function: 
   `from src.classification.multitask_inference import predict` 
2. Call `predict(your_text, desired_task_name)` where `desired_task_name` is 
   one of "scotus", "ledgar", or "unfair_tos".
"""

# Import centralized configuration manager
try:
    from ..config_manager import ConfigManager
except ImportError:
    # Fallback for when running as script or in tests
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_manager import ConfigManager

# LegalMultiTaskModel Definition
class LegalMultiTaskModel(nn.Module):
    def __init__(self, encoder_name, task_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.task_classifiers = nn.ModuleDict({
            task_name: nn.Linear(self.encoder.config.hidden_size, int(num_labels))
            for task_name, num_labels in task_labels.items()
        })
        self.task_labels = task_labels

    def forward(self, input_ids, attention_mask, task_name=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        if task_name is not None:
            if task_name not in self.task_classifiers:
                raise ValueError(f"Task '{task_name}' not found in model classifiers: {list(self.task_classifiers.keys())}")
            return self.task_classifiers[task_name](pooled_output)
        return {task: classifier(pooled_output) for task, classifier in self.task_classifiers.items()}

# Model Manager Class for Lazy Loading
class ModelManager:
    """Manages the multi-task model with lazy loading to avoid import-time execution."""
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._task_labels = None
        self._task_to_id2label = None
        self._device = None
        self._initialized = False
        self._config = None
        self._model_dir = None
        self._base_model_name = None
    
    def _ensure_initialized(self):
        """Lazy initialization - only loads model when first needed."""
        if not self._initialized:
            self._load_model()
            self._initialized = True
    
    def _load_model(self):
        """Load the model and all required components."""
        try:
            # Load configuration using centralized manager
            self._config = ConfigManager.load_default_config('classification')
            
            paths_cfg = self._config.get('paths', {})
            model_cfg = self._config.get('model', {}).get('multi_task_classification', {})
            
            # Get model details from config
            model_save_name = model_cfg.get('name', 'multitask_legal_model_standardized')
            self._base_model_name = model_cfg.get('base_model', 'nlpaueb/legal-bert-base-uncased')
            output_dir_template = paths_cfg.get('output_dir_template', 'models/classification/{model_name}')
            
            # Construct the actual model directory
            self._model_dir = output_dir_template.format(model_name=model_save_name)
            task_labels_path = os.path.join(self._model_dir, "task_labels.json")
            model_weights_path = os.path.join(self._model_dir, "model.pt")
            
            print(f"Attempting to load model from: {self._model_dir}")
            
            # Load Task Label Counts (Number of Classes)
            try:
                with open(task_labels_path, "r") as f:
                    self._task_labels = json.load(f) 
                print(f"Loaded task label counts: {self._task_labels}")
            except Exception as e:
                raise RuntimeError(f"Error loading task labels JSON: {e}")
            
            # Build id2label mapping by loading original dataset features
            print("Building id2label mappings from original datasets...")
            self._task_to_id2label = {}
            raw_data_dir = paths_cfg.get('raw_data_dir', 'data/processed')
            
            for task_name in self._task_labels.keys():
                try:
                    original_dataset_path = os.path.join(raw_data_dir, f"{task_name}_dataset")
                    if not os.path.exists(original_dataset_path):
                        print(f"  Warning: Original dataset not found for task '{task_name}' at {original_dataset_path}. Cannot get label names.")
                        continue
                        
                    temp_dataset = load_from_disk(original_dataset_path)
                    features = temp_dataset['train'].features
                    
                    label_feature_name = None
                    if 'label' in features:
                        label_feature_name = 'label'
                    elif 'labels' in features:
                        label_feature_name = 'labels'
                        
                    if label_feature_name:
                        label_feature = features[label_feature_name]
                        if isinstance(label_feature, datasets.Sequence):
                            inner_feature = label_feature.feature 
                            if hasattr(inner_feature, 'names'):
                                self._task_to_id2label[task_name] = {i: name for i, name in enumerate(inner_feature.names)}
                                print(f"  Loaded {len(inner_feature.names)} labels for task '{task_name}' (from Sequence)")
                        elif hasattr(label_feature, 'names'):
                            self._task_to_id2label[task_name] = {i: name for i, name in enumerate(label_feature.names)}
                            print(f"  Loaded {len(label_feature.names)} labels for task '{task_name}'")
                        else:
                            print(f"  Warning: Label feature found for '{task_name}', but it has no 'names' attribute.")
                    else:
                        print(f"  Warning: Could not find 'label' or 'labels' feature for task '{task_name}'.")
            
                except Exception as e:
                    print(f"  Error loading features or building mapping for task '{task_name}': {e}")
            
            print(f"Finished building mappings. Found names for tasks: {list(self._task_to_id2label.keys())}")
            
            # Load Tokenizer
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_dir)
                print("Tokenizer loaded.")
            except Exception as e:
                raise RuntimeError(f"Error loading tokenizer: {e}")
            
            # Initialize Model
            self._model = LegalMultiTaskModel(self._base_model_name, self._task_labels) 
            print("Model structure initialized.")
            
            # Load Model Weights
            try:
                self._model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
                print("Model weights loaded.")
            except Exception as e:
                raise RuntimeError(f"Error loading model weights from {model_weights_path}: {e}")
            
            # Setup Device and Eval Mode
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
            self._model.eval()
            print(f"Model moved to {self._device} and set to eval mode.")
            
        except Exception as e:
            print(f"Error during model initialization: {e}")
            raise
    
    def predict(self, text, task_name):
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
        self._ensure_initialized()
        
        if task_name not in self._model.task_classifiers:
            return {"error": f"Task '{task_name}' is not supported by this model. Supported tasks: {list(self._model.task_classifiers.keys())}"}
            
        # Tokenize
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)
        
        # Get predictions
        label_name = "N/A" 
        pred_class_id = -1
        confidence_score = 0.0

        with torch.no_grad():
            logits = self._model(input_ids, attention_mask, task_name)
            if logits is None or logits.shape[0] != 1: 
                return {"error": "Model returned unexpected output."}

            probs = torch.softmax(logits, dim=1)
            confidence_score, pred_class_id_tensor = torch.max(probs, dim=1)
            pred_class_id = pred_class_id_tensor.item()
            confidence_score = confidence_score.item()

            if task_name in self._task_to_id2label:
                label_name = self._task_to_id2label[task_name].get(pred_class_id, f"ID_{pred_class_id}_NotInMap")
            else:
                label_name = f"ID_{pred_class_id}_NoMapForTask"
                
        return {
            "task": task_name,
            "predicted_label_id": pred_class_id,
            "predicted_label_name": label_name,
            "confidence": confidence_score
        }
    
    def get_available_tasks(self):
        """Get list of available tasks without loading the model."""
        if not self._initialized:
            # Try to load just the config to get task labels
            try:
                config = ConfigManager.load_default_config('classification')
                model_cfg = config.get('model', {}).get('multi_task_classification', {})
                model_save_name = model_cfg.get('name', 'multitask_legal_model_standardized')
                output_dir_template = config.get('paths', {}).get('output_dir_template', 'models/classification/{model_name}')
                model_dir = output_dir_template.format(model_name=model_save_name)
                task_labels_path = os.path.join(model_dir, "task_labels.json")
                
                if os.path.exists(task_labels_path):
                    with open(task_labels_path, "r") as f:
                        task_labels = json.load(f)
                    return list(task_labels.keys())
            except Exception:
                pass
            return []
        return list(self._task_labels.keys())
    
    def is_model_loaded(self):
        """Check if the model is currently loaded."""
        return self._initialized
    
    def get_model_info(self):
        """Get information about the loaded model."""
        if not self._initialized:
            return {"loaded": False, "message": "Model not loaded"}
        
        return {
            "loaded": True,
            "model_dir": self._model_dir,
            "base_model": self._base_model_name,
            "device": str(self._device),
            "available_tasks": list(self._task_labels.keys()),
            "task_labels": self._task_labels
        }

# Global model manager instance
_model_manager = ModelManager()

# Public API Functions
def predict(text, task_name):
    """
    Classifies the input text for the specified task using the loaded multi-task model.
    
    This is the main public interface for the module.
    """
    return _model_manager.predict(text, task_name)

def get_available_tasks():
    """Get list of available tasks without loading the model."""
    return _model_manager.get_available_tasks()

def is_model_loaded():
    """Check if the model is currently loaded."""
    return _model_manager.is_model_loaded()

def get_model_info():
    """Get information about the loaded model."""
    return _model_manager.get_model_info()

# Example Usage
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
    available_tasks = list(task_num_labels.keys())
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