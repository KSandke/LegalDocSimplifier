import torch
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from datasets import load_from_disk, concatenate_datasets
import datasets
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
import os
import json
from tqdm.auto import tqdm
import random

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

# --- Custom Task Sampler Definition ---
class TaskBalancedSampler(Sampler):
    """
    Samples batches ensuring each batch contains examples from only one task.
    Iterates through tasks and yields all examples for a task before moving to the next.
    Task order and examples within each task are shuffled each epoch.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # --- Get indices per task from the MultiTaskDataset ---
        # This requires the MultiTaskDataset to store indices per task
        # Let's assume dataset has an attribute `task_indices_map` like:
        # {'scotus': [idx1, idx2, ...], 'ledgar': [...], ...}
        if not hasattr(dataset, 'task_indices_map') or not dataset.task_indices_map:
             raise ValueError("Dataset must have a 'task_indices_map' attribute mapping task names to lists of indices.")
        self.task_indices_map = dataset.task_indices_map
        self.num_tasks = len(self.task_indices_map)
        self.tasks = list(self.task_indices_map.keys())
        
        # Calculate total size and number of batches
        self.total_size = dataset.total_size # Assumes dataset has total_size attribute
        
        # Calculate number of batches, rounding up for partial batches
        self.num_batches = 0
        for task in self.tasks:
            task_len = len(self.task_indices_map[task])
            self.num_batches += (task_len + self.batch_size - 1) // self.batch_size

        print(f"TaskBalancedSampler initialized:")
        print(f"  Tasks: {self.tasks}")
        print(f"  Total Examples: {self.total_size}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Number of Batches per Epoch: {self.num_batches}")


    def __iter__(self):
        # Shuffle task order for the epoch
        shuffled_tasks = random.sample(self.tasks, len(self.tasks))
        
        # Prepare batches for the epoch
        epoch_batches = []
        for task in shuffled_tasks:
            # Shuffle indices within the task
            indices = self.task_indices_map[task]
            random.shuffle(indices)
            
            # Create batches for this task
            for i in range(0, len(indices), self.batch_size):
                epoch_batches.append(indices[i : i + self.batch_size])
                
        # Shuffle the order of all generated batches
        random.shuffle(epoch_batches)
        
        # Yield indices one by one from the shuffled batches
        # The DataLoader's default BatchSampler will group these into final batches
        all_indices_in_order = [idx for batch in epoch_batches for idx in batch]
        
        # Sanity check length (optional)
        # print(f"Total indices yielded by sampler: {len(all_indices_in_order)}")
        
        return iter(all_indices_in_order)

    def __len__(self):
        # Corresponds to the total number of examples, not batches
        # The DataLoader uses this length.
        return self.total_size

# --- End Custom Task Sampler Definition ---

# SIMPLIFIED Custom dataset class (Now uses standardized columns)
class MultiTaskDataset(Dataset):
    def __init__(self, datasets_dict, tokenizer):
        self.datasets = datasets_dict
        self.tokenizer = tokenizer
        self.dataset_indices = list(self.datasets.keys())
        
        print("Initializing MultiTaskDataset for Sampler...")
        self.total_size = 0
        self.task_indices_map = {name: [] for name in self.dataset_indices}
        self.global_to_local_map = {} # Maps global index to (task_name, local_idx)

        current_global_idx = 0
        for name in self.dataset_indices:
            if 'train' in self.datasets[name]:
                 split_size = len(self.datasets[name]['train'])
                 print(f"  Processing {name}: {split_size} examples.")
                 # Store indices for this task
                 task_indices = list(range(current_global_idx, current_global_idx + split_size))
                 self.task_indices_map[name].extend(task_indices)
                 # Create mapping for __getitem__
                 for i in range(split_size):
                      self.global_to_local_map[current_global_idx + i] = (name, i)
                 
                 self.total_size += split_size
                 current_global_idx += split_size
            else:
                 print(f"  Warning: 'train' split not found for {name}.")
        
        if self.total_size == 0:
             raise ValueError("No training data found.")
        print(f"MultiTaskDataset initialized. Total size: {self.total_size}")
        # print(f"Task indices map snippet: {{k: v[:3] for k,v in self.task_indices_map.items()}}") # Debug

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # Use the precomputed map to find the task and local index
        if idx not in self.global_to_local_map:
             raise IndexError(f"Global index {idx} not found in map.")
             
        task_name, local_idx = self.global_to_local_map[idx]
        
        # Access the example from the correct dataset split
        example = self.datasets[task_name]['train'][local_idx]
        
        # Access standardized columns directly (no if/elif needed)
        text = example["input_text"]
        label = example["input_label"]
        # Task name is already known

        encoding = self.tokenizer(
            text if text else "",
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
            "task_name": task_name # Still return task_name for the model head selection
        }

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
    
    # --- Create Simplified MultiTaskDataset ---
    train_dataset = MultiTaskDataset(loaded_datasets, tokenizer) 
    
    # --- Create Custom Sampler ---
    batch_size = 16 # Define batch size
    sampler = TaskBalancedSampler(train_dataset, batch_size)

    # --- Create DataLoader using the Custom Sampler ---
    # shuffle=False because the sampler handles shuffling
    # batch_sampler=None because we provide a sampler
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, # DataLoader still needs batch_size
        sampler=sampler,       # Use the custom sampler
        shuffle=False,         # Sampler handles shuffling
        batch_sampler=None     # Mutually exclusive with sampler
    )
    
    # --- Optimizer, Scheduler ---
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_epochs = 3
    # total_steps now depends on the number of batches from the sampler
    total_steps = sampler.num_batches * num_epochs # Use num_batches from sampler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )
    
    # --- SIMPLIFIED Training Loop ---
    print("\nStarting training with TaskBalancedSampler...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches_processed = 0
        # Use tqdm with the dataloader directly
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            batch_processed = False
            
            try:
                 # Batch is now homogeneous, get task name from first example
                 if not batch["task_name"]: # Should not happen with sampler
                      print(f"Warning: Empty task_name list in batch {batch_idx}. Skipping.")
                      continue
                 task_name = batch["task_name"][0] 

                 # Get data
                 input_ids = batch["input_ids"].to(device)
                 attention_mask = batch["attention_mask"].to(device)
                 labels = batch["label"].to(device)
                 
                 # --- Direct Label Check (Optional Safeguard) ---
                 expected_num_classes = filtered_task_labels.get(task_name)
                 if expected_num_classes is None: 
                      print(f"Critical Error: Task '{task_name}' not in filtered_task_labels! Skipping batch {batch_idx}.")
                      continue
                 if torch.any(labels < 0) or torch.any(labels >= expected_num_classes):
                      print(f"\nERROR: Out-of-bounds label detected in homogeneous batch {batch_idx} for task {task_name}!")
                      # This should DEFINITELY not happen now if preprocessing was correct
                      # ... (print details) ...
                      continue 
                 # --- End Check ---

                 # Forward pass - batch is homogeneous for task_name
                 logits = model(input_ids, attention_mask, task_name)
                 
                 # Compute loss directly
                 loss_fct = nn.CrossEntropyLoss()
                 loss = loss_fct(logits, labels)
                 
                 # Backward pass & Optimization
                 loss.backward()
                 optimizer.step()
                 scheduler.step()
                 
                 total_loss += loss.item()
                 num_batches_processed += 1
                 batch_processed = True
                 
                 progress_bar.set_postfix({'loss': loss.item(), 'task': task_name})

            except Exception as e:
                print(f"\nError during training loop batch {batch_idx} (Epoch {epoch+1}): {e}")
                import traceback
                traceback.print_exc()
                print("Skipping batch due to error.")
                # Ensure gradients are cleared
                optimizer.zero_grad() 
                continue 

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