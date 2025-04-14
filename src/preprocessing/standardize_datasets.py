import os
import json
from datasets import load_from_disk, Features, Value, ClassLabel, Sequence
from tqdm.auto import tqdm # For progress bars

# --- Configuration ---
RAW_DATA_DIR = 'data/processed'
STANDARDIZED_DATA_DIR = 'data/standardized'
DATASET_NAMES = ["scotus", "ledgar", "unfair_tos"]
LABEL_COUNTS_FILE = os.path.join(STANDARDIZED_DATA_DIR, 'task_label_counts.json')

# Create output directory if it doesn't exist
os.makedirs(STANDARDIZED_DATA_DIR, exist_ok=True)

# --- Standardization Functions ---

# This dictionary will store the number of labels for each task
task_label_counts = {}

def standardize_scotus(dataset):
    """Standardizes the SCOTUS dataset."""
    print("Standardizing scotus...")
    
    def transform(batch):
        return {
            "input_text": batch["text"],
            "input_label": batch["label"],
            "task_name": ["scotus"] * len(batch["text"]) # Add task name
        }
    
    # Store label count
    task_label_counts['scotus'] = dataset['train'].features['label'].num_classes
    
    # Apply transformation and remove old columns
    standardized_dataset = dataset.map(transform, batched=True, remove_columns=["text", "label"])
    return standardized_dataset

def standardize_ledgar(dataset):
    """Standardizes the LEDGAR dataset."""
    print("Standardizing ledgar...")
    
    def transform(batch):
        return {
            "input_text": batch["text"],
            "input_label": batch["label"],
            "task_name": ["ledgar"] * len(batch["text"])
        }
        
    # Store label count
    task_label_counts['ledgar'] = dataset['train'].features['label'].num_classes
    
    standardized_dataset = dataset.map(transform, batched=True, remove_columns=["text", "label"])
    return standardized_dataset

def standardize_unfair_tos(dataset):
    """Standardizes the UNFAIR-ToS dataset."""
    print("Standardizing unfair_tos...")
    
    try:
        label_feature = dataset['train'].features['labels'].feature
        num_classes = label_feature.num_classes # Should be 8
        print(f"  Detected num_classes for unfair_tos items: {num_classes} (Valid Range [0, {num_classes-1}])")
    except Exception as e:
        print(f"  Error getting num_classes for unfair_tos: {e}. Assuming 8.")
        num_classes = 8

    # This flag helps print the warning only once if issues are found
    warning_printed = False 

    def transform(batch):
        nonlocal warning_printed # Allow modification of the outer scope flag
        processed_labels = []
        texts_out = []
        batch_task_names = []

        # Iterate through examples in the batch
        for i in range(len(batch["text"])):
            labels_list = batch["labels"][i]
            current_text = batch["text"][i]
            
            chosen_label = 0 # Default label
            if labels_list: 
                first_label = labels_list[0]
                # **CRITICAL CHECK**: Ensure the first label is within the VALID range [0, num_classes-1]
                if 0 <= first_label < num_classes:
                    chosen_label = first_label
                else:
                    # Use default 0 for invalid labels
                    chosen_label = 0 
                    if not warning_printed: # Print only once
                         print(f"  WARNING: Found out-of-bounds raw label '{first_label}' in unfair_tos list. Mapping to 0. (Further warnings suppressed)")
                         warning_printed = True
            # else: list was empty, default 0 is already set
            
            processed_labels.append(chosen_label)
            texts_out.append(current_text)
            batch_task_names.append("unfair_tos") # Ensure task name is added

        # Debug print for the first batch processed by this function
        if not hasattr(transform, "debug_printed"):
             print(f"  DEBUG (First Batch): Input labels lists sample: {batch['labels'][:5]}")
             print(f"  DEBUG (First Batch): Output processed_labels sample: {processed_labels[:5]}")
             transform.debug_printed = True # Static variable to print only once

        return {
            "input_text": texts_out,
            "input_label": processed_labels, # This is the final standardized single label
            "task_name": batch_task_names
        }

    # Store the correct label count
    task_label_counts['unfair_tos'] = num_classes
        
    # Run map - consider removing batched=True if issues persist, it might complicate debugging
    standardized_dataset = dataset.map(transform, batched=True, remove_columns=["text", "labels"]) 
    # standardized_dataset = dataset.map(transform, batched=False, remove_columns=["text", "labels"]) # Try without batching if needed
    
    return standardized_dataset

# --- Main Preprocessing Logic ---

# Define the desired final features for clarity and consistency
# Note: ClassLabel information is lost here, but we save counts separately.
# Using simple Value types is sufficient for the standardized format.
standardized_features = Features({
    'input_text': Value('string'),
    'input_label': Value('int64'),
    'task_name': Value('string'),
})


# Dictionary mapping dataset names to their standardization functions
STANDARDIZATION_MAP = {
    "scotus": standardize_scotus,
    "ledgar": standardize_ledgar,
    "unfair_tos": standardize_unfair_tos,
}

if __name__ == "__main__":
    print("Starting dataset standardization...")

    for name in tqdm(DATASET_NAMES, desc="Processing Datasets"):
        # Determine input path (no special case needed anymore)
        input_path = os.path.join(RAW_DATA_DIR, f"{name}_dataset")
            
        output_path = os.path.join(STANDARDIZED_DATA_DIR, f"{name}_standardized_dataset")

        if not os.path.exists(input_path):
            print(f"Input dataset not found at {input_path}. Skipping {name}.")
            continue
            
        if os.path.exists(output_path):
            print(f"Standardized dataset already exists at {output_path}. Skipping {name}.")
            # Optionally load and check features if needed, or force reprocessing
            continue

        try:
            print(f"\nLoading dataset: {name} from {input_path}")
            # Load the dataset
            raw_dataset = load_from_disk(input_path)

            # Apply the correct standardization function
            if name in STANDARDIZATION_MAP:
                standardize_func = STANDARDIZATION_MAP[name]
                standardized_dataset = standardize_func(raw_dataset)
                
                # Cast features to the defined standardized features
                # This ensures all datasets strictly adhere to the same final structure
                print(f"Casting features for {name}...")
                # Apply feature casting to all splits
                for split in standardized_dataset.keys():
                     standardized_dataset[split] = standardized_dataset[split].cast(standardized_features)

                # Save the standardized dataset
                print(f"Saving standardized {name} dataset to {output_path}")
                standardized_dataset.save_to_disk(output_path)
                print(f"Finished processing {name}.")
            else:
                print(f"No standardization function found for {name}. Skipping.")

        except Exception as e:
            print(f"Error processing dataset {name}: {e}")
            import traceback
            traceback.print_exc()

    # Save the collected label counts
    print(f"\nSaving task label counts to {LABEL_COUNTS_FILE}...")
    try:
        with open(LABEL_COUNTS_FILE, 'w') as f:
            json.dump(task_label_counts, f, indent=4)
        print("Label counts saved successfully.")
    except Exception as e:
        print(f"Error saving label counts: {e}")

    print("\nDataset standardization process complete.") 