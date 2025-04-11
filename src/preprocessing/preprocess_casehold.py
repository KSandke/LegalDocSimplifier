import os
from datasets import load_from_disk, ClassLabel
from tqdm.auto import tqdm

# Define the function to extract labels safely USING FEATURE ENCODING
def extract_label_robust(example, label_feature):
    """
    Safely extracts the integer label using the ClassLabel feature object.
    Returns -1 if the label cannot be determined or is missing.
    """
    # Check if the provided label_feature is actually a ClassLabel
    if not isinstance(label_feature, ClassLabel):
        print("Error: Provided label_feature is not a ClassLabel instance!")
        return -1
        
    try:
        # Access the raw label value from the example dictionary
        raw_label_value = example.get("label") 
        if raw_label_value is None:
             # If the key 'label' itself is missing
             # print(f"Warning: Missing 'label' key in example. Keys: {list(example.keys())}")
             return -1 # Indicate missing label
        
        # Use the ClassLabel feature's str2int method to convert robustly
        # Ensure the input value is converted to a string first, as str2int expects strings
        int_label = label_feature.str2int(str(raw_label_value))
        
        # Final check: Ensure the result is within the valid range
        # (str2int should handle this, but belt-and-suspenders)
        if 0 <= int_label < label_feature.num_classes:
            return int_label
        else:
            # This should only happen if the raw data contains values not in names=['0'...'4']
            print(f"Warning: Decoded label {int_label} out of range [0, {label_feature.num_classes-1}]. Raw: '{raw_label_value}'. Using -1.")
            return -1
            
    except Exception as e:
        # Catch any other error during label extraction (e.g., value not in ClassLabel names)
        print(f"Error extracting/converting label. Raw: '{raw_label_value}'. Error: {e}. Using -1.")
        # print(f"Example keys: {list(example.keys())}") # Optional: more debug info
        return -1

def preprocess_casehold():
    dataset_name = "casehold"
    raw_path = f'data/processed/{dataset_name}_dataset'
    processed_path = f'data/processed/{dataset_name}_processed_dataset' # This is the intermediate output

    if not os.path.exists(raw_path):
        print(f"Raw dataset not found at {raw_path}. Please run the loader first.")
        return

    print(f"Loading raw {dataset_name} dataset from {raw_path}...")
    raw_dataset = load_from_disk(raw_path)
    print(f"Original features: {raw_dataset['train'].features}")

    # Get the ClassLabel feature object from the 'train' split (assuming splits have same features)
    try:
        label_feature = raw_dataset['train'].features['label']
        if not isinstance(label_feature, ClassLabel):
             print("Error: 'label' feature is not a ClassLabel type!")
             return
        print(f"Label feature details: num_classes={label_feature.num_classes}, names={label_feature.names}")
    except KeyError:
        print("Error: 'label' feature not found in dataset features!")
        return
    except Exception as e:
        print(f"Error accessing label feature: {e}")
        return

    print(f"Preprocessing {dataset_name} dataset to add 'safe_label' (robust extraction)...")
    
    # Define the mapping function for the batch
    def map_func_batch(batch):
         safe_labels = []
         # Reconstruct examples from the batch dictionary to pass to extract_label_robust
         num_examples = len(batch[list(batch.keys())[0]]) # Get batch size
         for i in range(num_examples):
              # Create a dictionary for the current example in the batch
              example = {k: v[i] for k, v in batch.items()} 
              # Pass the example and the ClassLabel feature object
              safe_labels.append(extract_label_robust(example, label_feature)) 
         return {"safe_label": safe_labels}

    # Apply the map function using the batch-aware function
    try:
        processed_dataset = raw_dataset.map(map_func_batch, batched=True) 
    except Exception as e:
        print(f"Error during .map() operation: {e}")
        return

    print(f"Finished preprocessing. New features: {processed_dataset['train'].features}")
    # Optional: Check first few safe_labels
    # print(f"Sample safe_labels: {processed_dataset['train'][:5]['safe_label']}")

    # Save the processed dataset
    print(f"Saving semi-processed dataset with robust 'safe_label' to {processed_path}...")
    processed_dataset.save_to_disk(processed_path)
    print("Preprocessing complete.")

if __name__ == "__main__":
    preprocess_casehold() 