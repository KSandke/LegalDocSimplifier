import torch
import json
from transformers import AutoModel, AutoTokenizer
from torch import nn

class LegalMultiTaskModel(nn.Module):
    def __init__(self, encoder_name, task_labels):
        super().__init__()
        # Load pre-trained encoder
        self.encoder = AutoModel.from_pretrained(encoder_name)
        
        # Task-specific classification heads
        self.task_classifiers = nn.ModuleDict({
            task_name: nn.Linear(self.encoder.config.hidden_size, num_labels)
            for task_name, num_labels in task_labels.items()
        })
        
        self.task_labels = task_labels
    
    def forward(self, input_ids, attention_mask, task_name=None):
        # Get encoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # If task specified, only return that task's logits
        if task_name is not None:
            return self.task_classifiers[task_name](pooled_output)
        
        # Otherwise return all task logits
        return {
            task: classifier(pooled_output)
            for task, classifier in self.task_classifiers.items()
        }

# Load model and task information
model_dir = "models/classification/multitask_legal_model"
with open(f"{model_dir}/task_labels.json", "r") as f:
    task_labels = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = LegalMultiTaskModel("nlpaueb/legal-bert-base-uncased", task_labels)
model.load_state_dict(torch.load(f"{model_dir}/model.pt"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict(text, task_name):
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Get predictions
    with torch.no_grad():
        logits = model(input_ids, attention_mask, task_name)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    
    return {
        "class_id": pred_class,
        "confidence": probs[0][pred_class].item(),
        "task": task_name
    }

# Example usage
if __name__ == "__main__":
    sample_text = "The Court has long held that the Federal Government has the authority..."
    
    # Try prediction with each task
    for task in task_labels.keys():
        result = predict(sample_text, task)
        print(f"Task: {task}, Prediction: {result}") 