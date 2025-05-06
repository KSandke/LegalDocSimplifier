import torch
from torch.cuda.amp import GradScaler
from torch.amp import autocast
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
import yaml

def load_config(config_path="config/classification.yaml"):
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

class LegalMultiTaskModel(nn.Module):
    def __init__(self, encoder_name, task_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        
        self.task_classifiers = nn.ModuleDict({
            task_name: nn.Linear(self.encoder.config.hidden_size, num_labels)
            for task_name, num_labels in task_labels.items()
        })
        
        self.task_labels = task_labels
    
    def forward(self, input_ids, attention_mask, task_name=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        if task_name is not None:
            if task_name not in self.task_classifiers:
                raise ValueError(f"Task '{task_name}' not found in model classifiers: {list(self.task_classifiers.keys())}")
            return self.task_classifiers[task_name](pooled_output)
        
        return {
            task: classifier(pooled_output)
            for task, classifier in self.task_classifiers.items()
        }

class TaskBalancedBatchSampler(Sampler):
    """
    A Batch Sampler that ensures each batch contains examples from only one task.
    Iterates through tasks, creates batches for each task, shuffles the order
    of these batches, and yields them.
    """
    def __init__(self, dataset, batch_size):
        print("Initializing TaskBalancedBatchSampler...")
        self.dataset = dataset
        self.batch_size = batch_size

        if not hasattr(dataset, 'task_indices_map') or not dataset.task_indices_map:
             raise ValueError("Dataset must have a 'task_indices_map' attribute mapping task names to lists of global indices.")
        self.task_indices_map = dataset.task_indices_map
        self.tasks = list(self.task_indices_map.keys())
        
        self.num_batches_per_task = {}
        self.num_batches = 0
        self.total_examples_in_sampler = 0
        
        print("  Calculating batches per task:")
        for task in self.tasks:
            task_len = len(self.task_indices_map[task])
            if task_len == 0:
                print(f"    Task '{task}': 0 examples, skipping.")
                self.num_batches_per_task[task] = 0
                continue
                
            n_batches = (task_len + self.batch_size - 1) // self.batch_size
            self.num_batches_per_task[task] = n_batches
            self.num_batches += n_batches
            self.total_examples_in_sampler += task_len
            print(f"    Task '{task}': {task_len} examples -> {n_batches} batches")
            
        self.total_size_from_dataset = dataset.total_size 
        if self.total_examples_in_sampler != self.total_size_from_dataset:
             print(f"  Warning: Sampler covers {self.total_examples_in_sampler} examples, dataset reports {self.total_size_from_dataset}.")

        print(f"  Total Tasks: {len(self.tasks)}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Total Batches per Epoch: {self.num_batches}")
        if self.num_batches == 0:
             raise ValueError("No batches to yield. Check dataset sizes and batch size.")

    def __iter__(self):
        epoch_batches = []
        
        for task in self.tasks:
            indices = self.task_indices_map[task]
            if not indices: continue # Skip empty tasks
            
            # Shuffle indices within the task
            random.shuffle(indices)
            
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                epoch_batches.append(batch_indices)
                
        random.shuffle(epoch_batches)
        
        print(f"\nTaskBalancedBatchSampler: Yielding {len(epoch_batches)} batches for this epoch.")
        
        return iter(epoch_batches)

    def __len__(self):
        return self.num_batches

class MultiTaskDataset(Dataset):
    def __init__(self, datasets_dict, tokenizer):
        self.datasets = datasets_dict
        self.tokenizer = tokenizer
        self.dataset_indices = list(self.datasets.keys())
        
        print("Initializing MultiTaskDataset for Sampler...")
        self.total_size = 0
        self.task_indices_map = {name: [] for name in self.dataset_indices}
        self.global_to_local_map = {}

        current_global_idx = 0
        for name in self.dataset_indices:
            if 'train' in self.datasets[name]:
                 split_size = len(self.datasets[name]['train'])
                 print(f"  Processing {name}: {split_size} examples.")
                 task_indices = list(range(current_global_idx, current_global_idx + split_size))
                 self.task_indices_map[name].extend(task_indices)
                 for i in range(split_size):
                      self.global_to_local_map[current_global_idx + i] = (name, i)
                 
                 self.total_size += split_size
                 current_global_idx += split_size
            else:
                 print(f"  Warning: 'train' split not found for {name}.")
        
        if self.total_size == 0:
             raise ValueError("No training data found.")
        print(f"MultiTaskDataset initialized. Total size: {self.total_size}")

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        if idx not in self.global_to_local_map:
             raise IndexError(f"Global index {idx} not found in map.")
             
        task_name, local_idx = self.global_to_local_map[idx]
        
        example = self.datasets[task_name]['train'][local_idx]
        
        text = example["input_text"]
        label = example["input_label"]

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
            "task_name": task_name
        }

def train_multitask_model(config):
    """Trains the multi-task model based on the provided config."""
    if not config:
        print("Configuration dictionary is missing. Aborting training.")
        return

    # Extract Config Parameters
    paths_cfg = config.get('paths', {})
    datasets_cfg = config.get('datasets', {})
    model_cfg = config.get('model', {}).get('multi_task_classification', {})
    train_cfg = config.get('training', {}).get('multi_task_classification', {})
    eval_cfg = config.get('evaluation', {}).get('multi_task_classification', {})

    STANDARDIZED_DATA_DIR = paths_cfg.get('standardized_data_dir', 'data/standardized')
    DATASET_NAMES = datasets_cfg.get('multi_task_classification', [])
    LABEL_COUNTS_FILE = paths_cfg.get('label_counts_file', 'data/standardized/task_label_counts.json')
    OUTPUT_DIR_TEMPLATE = paths_cfg.get('output_dir_template', 'models/classification/{model_name}')
    
    MODEL_SAVE_NAME = model_cfg.get('name', 'multitask_model_default_name')
    BASE_MODEL_NAME = model_cfg.get('base_model', 'nlpaueb/legal-bert-base-uncased')
    
    NUM_EPOCHS = train_cfg.get('num_epochs', 3)
    BATCH_SIZE = train_cfg.get('batch_size', 16)
    LEARNING_RATE = train_cfg.get('learning_rate', 2e-5)
    NUM_WARMUP_STEPS = train_cfg.get('num_warmup_steps', 0)
    USE_AMP = train_cfg.get('use_amp', True)
    NUM_WORKERS = train_cfg.get('num_workers', 0)
    PIN_MEMORY = train_cfg.get('pin_memory', True)

    EVAL_BATCH_SIZE = eval_cfg.get('batch_size', 16)
    
    print("--- Training Configuration ---")
    print(f"Datasets: {DATASET_NAMES}")
    print(f"Base Model: {BASE_MODEL_NAME}")
    print(f"Save Name: {MODEL_SAVE_NAME}")
    print(f"Epochs: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"AMP Enabled: {USE_AMP}")
    print("----------------------------")

    if not DATASET_NAMES:
         print("Error: No datasets specified in configuration (datasets.multi_task_classification).")
         return

    # Load Standardized Datasets
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
        
    # Load Task Label Counts
    print(f"Loading task label counts from {LABEL_COUNTS_FILE}...")
    try:
        with open(LABEL_COUNTS_FILE, 'r') as f:
            task_labels = json.load(f)
        task_labels = {str(k): int(v) for k, v in task_labels.items()}
        print(f"Loaded Task labels (raw): {task_labels}")
    except FileNotFoundError:
        print(f"Error: Label counts file not found at {LABEL_COUNTS_FILE}. Cannot determine model output sizes.")
        return
    except Exception as e:
        print(f"Error loading or processing label counts file {LABEL_COUNTS_FILE}: {e}")
        return

    # Filter Task Labels based on successfully loaded datasets
    print("\nFiltering task labels based on loaded datasets...")
    filtered_task_labels = {k: v for k, v in task_labels.items() if k in loaded_datasets}
    print(f"  Resulting filtered_task_labels: {filtered_task_labels}") 
    if not filtered_task_labels or len(filtered_task_labels) != len(loaded_datasets):
         print("Error: Mismatch between loaded datasets and label counts after filtering.")
         print(f"Loaded: {list(loaded_datasets.keys())}, Filtered Counts: {list(filtered_task_labels.keys())}")
         return

    # Prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = LegalMultiTaskModel(BASE_MODEL_NAME, filtered_task_labels)
    
    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    # Create Dataset, Sampler, DataLoader
    train_dataset = MultiTaskDataset(loaded_datasets, tokenizer) 
    batch_sampler = TaskBalancedBatchSampler(train_dataset, BATCH_SIZE)
    print("Creating DataLoader...")
    train_dataloader = DataLoader(
        train_dataset, 
        batch_sampler=batch_sampler, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY and (device.type == 'cuda'),       
    )
    print("DataLoader created successfully.")
    
    # Optimizer, Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(batch_sampler) * NUM_EPOCHS
    print(f"Total training steps: {total_steps}") 
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=NUM_WARMUP_STEPS,
        num_training_steps=total_steps
    )

    # Initialize GradScaler for AMP
    scaler = GradScaler(enabled=(USE_AMP and device.type == 'cuda'))
    
    # Training Loop with AMP
    print(f"\nStarting training ({NUM_EPOCHS} Epochs) with AMP={'Enabled' if USE_AMP and device.type == 'cuda' else 'Disabled'}...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        num_batches_processed = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", total=len(batch_sampler), leave=False) 
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad(set_to_none=True) 
            
            try:
                if not batch["task_name"]: continue
                task_name = batch["task_name"][0]
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                # Safeguard Label Check
                expected_num_classes = filtered_task_labels.get(task_name)
                if expected_num_classes is None: continue
                if torch.any(labels < 0) or torch.any(labels >= expected_num_classes):
                     print(f"\nERROR: OOB Label! Task:{task_name}, Range:[0,{expected_num_classes-1}], Got:{labels[torch.where((labels<0)|(labels>=expected_num_classes))[0]].tolist()}")
                     continue 

                with autocast(device_type=device.type, enabled=(USE_AMP and device.type == 'cuda')):
                    logits = model(input_ids, attention_mask, task_name)
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits, labels)
                 
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step() 
                 
                total_loss += loss.item() 
                num_batches_processed += 1
                progress_bar.set_postfix({'loss': loss.item(), 'task': task_name})

            except Exception as e:
                print(f"\nError during training batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                optimizer.zero_grad()
                continue 

        avg_epoch_loss = total_loss / num_batches_processed if num_batches_processed > 0 else 0
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} finished. Average Loss: {avg_epoch_loss:.4f}")
        progress_bar.close()

    # Save Model
    output_dir = OUTPUT_DIR_TEMPLATE.format(model_name=MODEL_SAVE_NAME)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"\nSaving model components to {output_dir}...")

    # 1. Save Model State Dictionary (Weights)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    print("  Model state_dict saved.")

    # 2. Save Tokenizer Configuration/Vocabulary
    tokenizer.save_pretrained(output_dir)
    print("  Tokenizer saved.")
    
    # 3. Save the Base Model's Configuration (config.json)
    try:
        model.encoder.config.save_pretrained(output_dir)
        print("  Base model config.json saved.")
    except AttributeError:
        print("  Warning: Could not find model.encoder.config to save config.json.")

    # 4. Save Custom Task Labels 
    with open(os.path.join(output_dir, "task_labels.json"), "w") as f:
        json.dump(filtered_task_labels, f, indent=4) 
    print("  Task labels saved.")
    
    print("Model saving complete.")

    # Evaluate on Test Sets
    print("\nStarting evaluation...")
    model.eval()
    evaluation_results = {}

    for name, dataset_splits in loaded_datasets.items():
        if 'test' not in dataset_splits:
            print(f"Skipping evaluation for {name}: No 'test' split found.")
            continue
            
        print(f"Evaluating on {name} test set...")
        test_set = dataset_splits["test"]
        test_dataloader = DataLoader(test_set, batch_size=EVAL_BATCH_SIZE) 

        all_preds = []
        all_labels = []
        
        progress_bar_eval = tqdm(test_dataloader, desc=f"Evaluating {name}", leave=False)
        
        for batch in progress_bar_eval:
            input_ids = batch["input_text"]
            labels = batch["input_label"]

            # Tokenize
            encodings = tokenizer(input_ids, padding=True, truncation=True, max_length=512, return_tensors="pt")
            input_ids_tensor = encodings["input_ids"].to(device)
            attention_mask_tensor = encodings["attention_mask"].to(device)
            
            # Get predictions
            with torch.no_grad():
                logits = model(input_ids_tensor, attention_mask_tensor, task_name=name) 
                preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
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
    config = load_config()
    if config:
        train_multitask_model(config)
    else:
        print("Failed to load configuration. Exiting.") 