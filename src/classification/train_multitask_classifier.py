import torch
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from datasets import load_from_disk, concatenate_datasets
import datasets
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler, BatchSampler
import os
import json
from tqdm.auto import tqdm
import random
from torch.cuda.amp import GradScaler, autocast

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

# --- Custom Task BATCH Sampler Definition ---
class TaskBalancedBatchSampler(Sampler):
    """
    A Batch Sampler that ensures each batch contains examples from only one task.
    Iterates through tasks, creates batches for each task, shuffles the order
    of these batches, and yields them.
    """
    def __init__(self, dataset, batch_size):
        # dataset should be an instance of our MultiTaskDataset
        print("Initializing TaskBalancedBatchSampler...")
        self.dataset = dataset
        self.batch_size = batch_size

        if not hasattr(dataset, 'task_indices_map') or not dataset.task_indices_map:
             raise ValueError("Dataset must have a 'task_indices_map' attribute mapping task names to lists of global indices.")
        self.task_indices_map = dataset.task_indices_map
        self.tasks = list(self.task_indices_map.keys())
        
        # Calculate number of batches per task and total batches
        self.num_batches_per_task = {}
        self.num_batches = 0
        self.total_examples_in_sampler = 0 # Track examples covered by batches
        
        print("  Calculating batches per task:")
        for task in self.tasks:
            task_len = len(self.task_indices_map[task])
            if task_len == 0:
                print(f"    Task '{task}': 0 examples, skipping.")
                self.num_batches_per_task[task] = 0
                continue
                
            # Number of batches for this task, rounding up
            n_batches = (task_len + self.batch_size - 1) // self.batch_size
            self.num_batches_per_task[task] = n_batches
            self.num_batches += n_batches
            self.total_examples_in_sampler += task_len # Add actual examples used
            print(f"    Task '{task}': {task_len} examples -> {n_batches} batches")
            
        # Use total examples covered by sampler for dataset __len__ reference if needed,
        # but sampler len is number of batches.
        self.total_size_from_dataset = dataset.total_size 
        if self.total_examples_in_sampler != self.total_size_from_dataset:
             print(f"  Warning: Sampler covers {self.total_examples_in_sampler} examples, dataset reports {self.total_size_from_dataset}.")

        print(f"  Total Tasks: {len(self.tasks)}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Total Batches per Epoch: {self.num_batches}")
        if self.num_batches == 0:
             raise ValueError("No batches to yield. Check dataset sizes and batch size.")

    def __iter__(self):
        # This method yields lists of indices (batches)
        
        # Shuffle task order for the epoch? Optional, can just create all batches then shuffle.
        # shuffled_tasks = random.sample(self.tasks, len(self.tasks))
        
        epoch_batches = [] # List to hold all batches (each batch is a list of indices)
        
        # Create batches for each task
        for task in self.tasks:
            indices = self.task_indices_map[task]
            if not indices: continue # Skip empty tasks
            
            # Shuffle indices within the task
            random.shuffle(indices)
            
            # Create batches for this task
            for i in range(0, len(indices), self.batch_size):
                # Create a batch (list of indices)
                batch_indices = indices[i : i + self.batch_size]
                # Add the batch to our list of batches for the epoch
                epoch_batches.append(batch_indices)
                
        # Shuffle the order of all generated batches across tasks
        random.shuffle(epoch_batches)
        
        print(f"\nTaskBalancedBatchSampler: Yielding {len(epoch_batches)} batches for this epoch.") # Debug print once per epoch
        
        # Return an iterator over the list of batches
        return iter(epoch_batches)

    def __len__(self):
        # This should return the number of batches yielded per iteration (epoch)
        return self.num_batches

# --- End Custom Task BATCH Sampler Definition ---

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

# Main training function with AMP
def train_multitask_model():
    # --- Configuration ---
    STANDARDIZED_DATA_DIR = 'data/standardized'
    DATASET_NAMES = ["scotus", "ledgar", "unfair_tos", "casehold"] 
    LABEL_COUNTS_FILE = os.path.join(STANDARDIZED_DATA_DIR, 'task_label_counts.json')

    # --- Load Standardized Datasets ---
    loaded_datasets = {}
    print("Loading standardized datasets...")
    datasets_to_load = DATASET_NAMES[:] 
    for name in datasets_to_load:
        path = os.path.join(STANDARDIZED_DATA_DIR, f"{name}_standardized_dataset")
        print(f"Checking for dataset '{name}' at: {path}") 
        if os.path.exists(path):
            try:
                print(f"  Loading {name}...")
                loaded_datasets[name] = load_from_disk(path)
                print(f"  Successfully loaded {name}.")
            except Exception as e:
                print(f"  ERROR loading {name} from {path}: {e}")
        else:
            print(f"  Standardized dataset not found at {path}. Skipping {name}.")
            
    print(f"\nDEBUG: Finished loading loop. Keys in loaded_datasets: {list(loaded_datasets.keys())}")

    if not loaded_datasets:
        print("No standardized datasets found to train on!")
        return
        
    # --- Load Task Label Counts ---
    print(f"Loading task label counts from {LABEL_COUNTS_FILE}...")
    try:
        with open(LABEL_COUNTS_FILE, 'r') as f:
            task_labels = json.load(f)
        task_labels = {str(k): int(v) for k, v in task_labels.items()}
        print(f"Loaded Task labels: {task_labels}")
    except FileNotFoundError:
        print(f"Error: Label counts file not found at {LABEL_COUNTS_FILE}.")
        return
    except Exception as e: # Catch other potential errors like JSONDecodeError
        print(f"Error loading or processing label counts file {LABEL_COUNTS_FILE}: {e}")
        return

    # --- Filter Task Labels based on successfully loaded datasets ---
    print("\nFiltering task labels based on loaded datasets...")
    filtered_task_labels = {k: v for k, v in task_labels.items() if k in loaded_datasets}
    print(f"  Resulting filtered_task_labels: {filtered_task_labels}") 
    if not filtered_task_labels:
         print("Error: No valid tasks remaining after filtering label counts.")
         return

    # --- Prepare tokenizer and model ---
    model_name = "nlpaueb/legal-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LegalMultiTaskModel(model_name, filtered_task_labels)
    
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    # --- Create Dataset, Sampler, DataLoader ---
    train_dataset = MultiTaskDataset(loaded_datasets, tokenizer) 
    batch_size = 16 # Start with 16, maybe increase later
    batch_sampler = TaskBalancedBatchSampler(train_dataset, batch_size) 
    print("Creating DataLoader with TaskBalancedBatchSampler...")
    train_dataloader = DataLoader(
        train_dataset, 
        batch_sampler=batch_sampler, 
        num_workers=0,           
        pin_memory=True, # Set True for GPU       
    )
    print("DataLoader created successfully.")
    
    # --- Optimizer, Scheduler --- 
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_epochs = 3
    total_steps = len(batch_sampler) * num_epochs 
    print(f"Total training steps: {total_steps}") 
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )

    # --- Initialize GradScaler for AMP ---
    scaler = GradScaler(enabled=(device.type == 'cuda')) # Enable scaler only if using CUDA

    # --- Training Loop with AMP ---
    print("\nStarting training with TaskBalancedBatchSampler and AMP...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches_processed = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", total=len(batch_sampler), leave=False) 
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad(set_to_none=True) # Use set_to_none=True for potential efficiency gain
            
            try:
                 if not batch["task_name"]: continue
                 task_name = batch["task_name"][0] 

                 input_ids = batch["input_ids"].to(device)
                 attention_mask = batch["attention_mask"].to(device)
                 labels = batch["label"].to(device)
                 
                 # --- Optional Safeguard Label Check ---
                 expected_num_classes = filtered_task_labels.get(task_name)
                 if expected_num_classes is None: continue # Should not happen
                 if torch.any(labels < 0) or torch.any(labels >= expected_num_classes):
                      print(f"\nERROR: OOB Label! Task:{task_name}, Range:[0,{expected_num_classes-1}], Got:{labels[torch.where((labels<0)|(labels>=expected_num_classes))[0]].tolist()}")
                      continue 
                 # --- End Check ---

                 # --- Autocast context manager ---
                 # Enable only if using CUDA
                 with autocast(enabled=(device.type == 'cuda')): 
                     logits = model(input_ids, attention_mask, task_name)
                     loss_fct = nn.CrossEntropyLoss()
                     loss = loss_fct(logits, labels)
                 # --- End Autocast ---
                 
                 # --- Scaled Backward ---
                 scaler.scale(loss).backward()
                 
                 # --- Scaled Optimizer Step ---
                 scaler.step(optimizer)
                 
                 # --- Update Scaler ---
                 scaler.update()
                 
                 # --- Scheduler Step ---
                 scheduler.step() 
                 
                 total_loss += loss.item() 
                 num_batches_processed += 1
                 progress_bar.set_postfix({'loss': loss.item(), 'task': task_name})

            except Exception as e:
                print(f"\nError during training batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                optimizer.zero_grad() # Clear potentially corrupted gradients
                continue 

        avg_epoch_loss = total_loss / num_batches_processed if num_batches_processed > 0 else 0
        print(f"\nEpoch {epoch+1}/{num_epochs} finished. Average Loss: {avg_epoch_loss:.4f}")
        progress_bar.close()

    # --- Save Model ---
    output_dir = "models/classification/multitask_legal_model_standardized" 
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"\nSaving model to {output_dir}...")
    torch.save(model.state_dict(), f"{output_dir}/model.pt")
    tokenizer.save_pretrained(output_dir)
    with open(f"{output_dir}/task_labels.json", "w") as f: 
        json.dump(filtered_task_labels, f, indent=4) 
    print("Model saved.")

    # --- Evaluate on Test Sets ---
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