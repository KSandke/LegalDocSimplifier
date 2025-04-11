import torch
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from datasets import load_from_disk, concatenate_datasets
import datasets
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import json
from tqdm.auto import tqdm

# Define a multi-task classification model
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
            if task_name not in self.task_classifiers:
                raise ValueError(f"Task '{task_name}' not found in model classifiers: {list(self.task_classifiers.keys())}")
            return self.task_classifiers[task_name](pooled_output)
        
        # Otherwise return all task logits
        return {
            task: classifier(pooled_output)
            for task, classifier in self.task_classifiers.items()
        }

# SIMPLIFIED Custom dataset class (Now uses standardized columns)
class MultiTaskDataset(Dataset):
    def __init__(self, datasets_dict, tokenizer):
        # Combine all dataset splits into one list of examples and track task names
        self.all_examples = []
        self.dataset_sizes = {} # Store original sizes if needed for weighted sampling (optional)
        
        print("Combining standardized datasets for MultiTaskDataset...")
        # It's often simpler to concatenate datasets here
        # For simplicity, we'll just store references and calculate total length
        self.datasets = datasets_dict # Store the dict of loaded datasets
        self.dataset_indices = list(self.datasets.keys()) # List of task names
        
        self.total_size = 0
        for name, dataset in self.datasets.items():
            # Check for 'train' split - assuming standardization kept splits
            if 'train' in dataset:
                 size = len(dataset['train'])
                 self.dataset_sizes[name] = size
                 self.total_size += size
            else:
                 print(f"Warning: 'train' split not found for {name}. Ignoring this dataset for training.")
        
        if self.total_size == 0:
             raise ValueError("No training data found in the provided datasets.")

        self.tokenizer = tokenizer
        print(f"MultiTaskDataset initialized with {self.total_size} total examples.")

    def __len__(self):
        # Return the total size calculated from all 'train' splits
        return self.total_size

    def __getitem__(self, idx):
        # --- Simplified Logic ---
        # Need a way to map global idx to a specific dataset and index within it
        # Simple modulo sampling (might not be perfectly balanced if datasets differ greatly in size)
        current_offset = 0
        for task_name in self.dataset_indices:
             task_size = self.dataset_sizes.get(task_name, 0)
             if idx < current_offset + task_size:
                  # This idx belongs to the current task_name
                  within_dataset_idx = idx - current_offset
                  dataset = self.datasets[task_name]['train']
                  
                  # Access standardized columns directly
                  example = dataset[within_dataset_idx]
                  text = example["input_text"]
                  label = example["input_label"]
                  # Task name is already known
                  
                  # Tokenize
                  encoding = self.tokenizer(
                      text if text else "", # Handle potential None text
                      padding="max_length",
                      truncation=True,
                      max_length=512,
                      return_tensors="pt"
                  )
                  
                  return {
                      "input_ids": encoding["input_ids"].squeeze(),
                      "attention_mask": encoding["attention_mask"].squeeze(),
                      "label": torch.tensor(label, dtype=torch.long),
                      "task_name": task_name # Return the task name
                  }
             current_offset += task_size
             
        # This should not be reached if total_size is calculated correctly
        raise IndexError(f"Index {idx} out of bounds for total size {self.total_size}")

# Main training function (Simplified Loading and Setup)
def train_multitask_model():
    # --- Configuration ---
    STANDARDIZED_DATA_DIR = 'data/standardized'
    DATASET_NAMES = ["scotus", "ledgar", "unfair_tos", "casehold"] # List of datasets to load
    LABEL_COUNTS_FILE = os.path.join(STANDARDIZED_DATA_DIR, 'task_label_counts.json')

    # --- Load Standardized Datasets ---
    loaded_datasets = {}
    print("Loading standardized datasets...")
    # Use a copy of the list to avoid modification issues if we skip items
    datasets_to_load = DATASET_NAMES[:]
    
    for name in datasets_to_load:
        path = os.path.join(STANDARDIZED_DATA_DIR, f"{name}_standardized_dataset")
        print(f"Checking for dataset '{name}' at: {path}") # DEBUG
        if os.path.exists(path):
            try:
                print(f"  Loading {name}...")
                loaded_datasets[name] = load_from_disk(path)
                print(f"  Successfully loaded {name}.")
            except Exception as e:
                print(f"  ERROR loading {name} from {path}: {e}")
        else:
            print(f"  Standardized dataset not found at {path}. Skipping {name}.")
            
    # --- CRITICAL DEBUG: Check what was actually loaded ---
    print(f"\nDEBUG: Finished loading loop. Keys in loaded_datasets: {list(loaded_datasets.keys())}")
    # --- END DEBUG ---

    if not loaded_datasets:
        print("No standardized datasets found to train on!")
        return
        
    # --- Load Task Label Counts ---
    print(f"Loading task label counts from {LABEL_COUNTS_FILE}...")
    try:
        with open(LABEL_COUNTS_FILE, 'r') as f:
            task_labels = json.load(f)
        # Ensure keys are strings if needed (JSON loads them as strings)
        task_labels = {str(k): int(v) for k, v in task_labels.items()}
        print(f"Loaded Task labels: {task_labels}")
        # Verify loaded datasets match task_labels
        if set(loaded_datasets.keys()) != set(task_labels.keys()):
             print("Warning: Mismatch between loaded datasets and label counts file!")
             print(f"  Loaded datasets: {list(loaded_datasets.keys())}")
             print(f"  Label counts keys: {list(task_labels.keys())}")
             # Filter task_labels to only include loaded datasets
             task_labels = {k: v for k, v in task_labels.items() if k in loaded_datasets}
             print(f"  Using filtered task_labels: {task_labels}")

    except FileNotFoundError:
        print(f"Error: Label counts file not found at {LABEL_COUNTS_FILE}. Cannot determine model output sizes.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {LABEL_COUNTS_FILE}.")
        return

    # --- Filter Task Labels based on successfully loaded datasets ---
    print("\nFiltering task labels based on loaded datasets...")
    print(f"  Keys in loaded_datasets (before filtering): {list(loaded_datasets.keys())}") # DEBUG
    print(f"  Keys in task_labels from JSON: {list(task_labels.keys())}")
    
    filtered_task_labels = {k: v for k, v in task_labels.items() if k in loaded_datasets}
    
    print(f"  Resulting filtered_task_labels: {filtered_task_labels}") # DEBUG
    
    if not filtered_task_labels:
         print("Error: No valid tasks remaining after filtering label counts based on loaded datasets.")
         return
    # --- End Filtering ---

    # --- Prepare tokenizer and model (Uses filtered_task_labels) ---
    model_name = "nlpaueb/legal-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Pass the correctly filtered dictionary to the model
    model = LegalMultiTaskModel(model_name, filtered_task_labels)
    
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    # --- Create Simplified MultiTaskDataset and DataLoader ---
    # Pass the dictionary of loaded standardized datasets
    train_dataset = MultiTaskDataset(loaded_datasets, tokenizer) 
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # --- Optimizer, Scheduler ---
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_epochs = 3 # Or load from config
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0, # Adjust as needed
        num_training_steps=total_steps
    )
    
    # --- Training Loop (Revised for Mixed Batches) ---
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches_processed = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad() # Zero gradients at the start of the batch
            batch_processed = False
            
            try:
                 # Get common data
                 input_ids = batch["input_ids"].to(device)
                 attention_mask = batch["attention_mask"].to(device)
                 labels = batch["label"].to(device) # The labels tensor
                 task_names = batch["task_name"] # List of task names for the batch

                 # --- Verify Batch Consistency (Optional Debug) ---
                 unique_tasks_in_batch = set(task_names)
                 if len(unique_tasks_in_batch) > 1:
                      if batch_idx % 100 == 0: # Print only occasionally
                         print(f"\nINFO: Mixed batch detected at batch {batch_idx}. Tasks: {unique_tasks_in_batch}")
                 # --- End Verification ---

                 # Calculate loss per task present in the batch
                 cumulative_loss = 0.0
                 num_valid_examples = 0

                 for task_name in unique_tasks_in_batch:
                      # Find indices corresponding to the current task_name
                      task_indices = [i for i, t in enumerate(task_names) if t == task_name]
                      
                      if not task_indices: continue # Should not happen

                      # Select data for the current task
                      task_input_ids = input_ids[task_indices]
                      task_attention_mask = attention_mask[task_indices]
                      task_labels = labels[task_indices]

                      # Check label range for this task's subset
                      expected_num_classes = filtered_task_labels.get(task_name)
                      if expected_num_classes is None:
                           print(f"Critical Error: Task '{task_name}' not in filtered_task_labels! Skipping sub-batch.")
                           continue
                       
                      if torch.any(task_labels < 0) or torch.any(task_labels >= expected_num_classes):
                           # This check should ideally NOT trigger now, given the data inspection results
                           print(f"\nERROR: Out-of-bounds label detected for task {task_name} even after filtering!")
                           print(f"  Expected Range: [0, {expected_num_classes-1}], Got Labels: {task_labels}")
                           invalid_indices_task = torch.where((task_labels < 0) | (task_labels >= expected_num_classes))[0]
                           print(f"  Problematic values: {task_labels[invalid_indices_task].tolist()}")
                           continue # Skip this task's sub-batch

                      # Forward pass for this task's data
                      logits = model(task_input_ids, task_attention_mask, task_name)
                      
                      # Compute loss for this task's sub-batch
                      loss_fct = nn.CrossEntropyLoss()
                      loss = loss_fct(logits, task_labels)
                      
                      # Accumulate loss, weighted by the number of examples for this task
                      # This ensures tasks with more examples in the batch contribute proportionally
                      cumulative_loss += loss * len(task_indices) 
                      num_valid_examples += len(task_indices)

                 # Average loss across the valid examples in the batch
                 if num_valid_examples > 0:
                      average_batch_loss = cumulative_loss / num_valid_examples
                      
                      # Backward pass on the combined, averaged loss
                      average_batch_loss.backward()
                      total_loss += average_batch_loss.item() # Accumulate average loss
                      batch_processed = True
                      num_batches_processed += 1 # Count successfully processed batches
                      
                      # Update progress bar
                      progress_bar.set_postfix({'avg_loss': total_loss / num_batches_processed, 'last_batch_loss': average_batch_loss.item()})
                 else:
                      # No valid examples/tasks processed in this batch
                      progress_bar.set_postfix({'loss': 'Skipped'})

                 # Gradient step (happens once per batch after processing all tasks)
                 if batch_processed:
                     optimizer.step()
                     scheduler.step()
                     # Gradients were already zeroed at the start

            except Exception as e:
                print(f"\nError during training loop batch {batch_idx} (Epoch {epoch+1}): {e}")
                import traceback
                traceback.print_exc() # Print full traceback for unexpected errors
                print("Skipping batch due to error.")
                # Ensure gradients are cleared if an error occurred mid-step
                optimizer.zero_grad() 
                continue # Move to the next batch

        avg_epoch_loss = total_loss / num_batches_processed if num_batches_processed > 0 else 0
        print(f"\nEpoch {epoch+1}/{num_epochs} finished. Average Loss: {avg_epoch_loss:.4f}")
        progress_bar.close()

    # --- Save Model ---
    output_dir = "models/classification/multitask_legal_model_standardized" # New name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\nSaving model to {output_dir}...")
    # Save model state dict, tokenizer, and task labels used for training
    torch.save(model.state_dict(), f"{output_dir}/model.pt")
    tokenizer.save_pretrained(output_dir)
    with open(f"{output_dir}/task_labels.json", "w") as f: # Save the counts used
        json.dump(filtered_task_labels, f, indent=4) 
    print("Model saved.")

    # --- Evaluate on Test Sets (Simplified) ---
    print("\nStarting evaluation...")
    model.eval()
    evaluation_results = {}

    for name, dataset_splits in loaded_datasets.items():
        if 'test' not in dataset_splits:
            print(f"Skipping evaluation for {name}: No 'test' split found.")
            continue
            
        print(f"Evaluating on {name} test set...")
        test_set = dataset_splits["test"]
        test_dataloader = DataLoader(test_set, batch_size=16) # Simple dataloader for eval

        all_preds = []
        all_labels = []
        
        progress_bar_eval = tqdm(test_dataloader, desc=f"Evaluating {name}", leave=False)
        
        for batch in progress_bar_eval:
            # Access standardized columns
            input_ids = batch["input_text"] # Text is processed by tokenizer later
            labels = batch["input_label"] # Labels are already integers
            # task_name from batch['task_name'] - ensure it's consistent if needed

            # Tokenize
            encodings = tokenizer(input_ids, padding=True, truncation=True, max_length=512, return_tensors="pt")
            input_ids_tensor = encodings["input_ids"].to(device)
            attention_mask_tensor = encodings["attention_mask"].to(device)
            
            # Get predictions
            with torch.no_grad():
                # Pass the correct task name for the current dataset being evaluated
                logits = model(input_ids_tensor, attention_mask_tensor, task_name=name) 
                preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            # Ensure labels are tensors or lists of integers
            all_labels.extend(labels.tolist() if isinstance(labels, torch.Tensor) else labels) 
        
        progress_bar_eval.close()

        # Calculate metrics
        if all_labels:
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            evaluation_results[name] = {"accuracy": accuracy, "f1": f1}
        else:
            print(f"  No valid labels found in the {name} test set for evaluation.")
            evaluation_results[name] = {"accuracy": 0, "f1": 0}

    print("\nEvaluation complete.")
    print("Evaluation Results:", evaluation_results)

if __name__ == "__main__":
    train_multitask_model() 