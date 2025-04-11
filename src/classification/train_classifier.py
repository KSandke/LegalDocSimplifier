import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
dataset_name = "ledgar"  # Choose from: scotus, ledgar, unfair_tos, casehold
dataset = load_from_disk(f'data/processed/{dataset_name}_dataset')

# Prepare model and tokenizer
model_name = "nlpaueb/legal-bert-base-uncased"  # Legal domain-specific BERT
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Determine number of labels from dataset (each dataset has different classes)
num_labels = len(dataset["train"].features["label"].names) if "label" in dataset["train"].features else 2

# Initialize model with correct number of labels
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model.to(device)  # Move model to GPU if available

# Preprocess function (adapt based on your dataset structure)
def preprocess_function(examples):
    # Different datasets have different text fields
    if dataset_name == "scotus":
        texts = examples["text"]
    elif dataset_name == "ledgar":
        texts = examples["text"]
    elif dataset_name == "unfair_tos":
        texts = examples["text"]
    elif dataset_name == "casehold":
        texts = examples["text"]  # Adjust as needed
    
    return tokenizer(texts, padding="max_length", truncation=True, max_length=512)

# Tokenize datasets
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define metrics for evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir=f"models/classification/{dataset_name}_checkpoints",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    save_steps=500,
    load_best_model_at_end=True,
    push_to_hub=False,
    fp16=True,                  # Only enable GPU acceleration if your GPU is detected
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"] if "validation" in tokenized_datasets else tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate on test set
results = trainer.evaluate(tokenized_datasets["test"])
print(f"Test results: {results}")

# Save model
model_path = f"models/classification/{dataset_name}_model"
trainer.save_model(model_path)
print(f"Model saved to {model_path}")
