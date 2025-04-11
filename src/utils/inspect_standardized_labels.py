import os
import numpy as np
from datasets import load_from_disk
import pandas as pd # Using pandas for easier analysis

STANDARDIZED_DATA_DIR = 'data/standardized'
# Check labels for the problematic datasets
DATASET_NAMES = ["casehold", "unfair_tos", "scotus", "ledgar"] 
# Also check scotus/ledgar just in case

print("Inspecting labels in standardized datasets...")

for name in DATASET_NAMES:
    path = os.path.join(STANDARDIZED_DATA_DIR, f"{name}_standardized_dataset")
    print(f"\n--- Checking Dataset: {name} ---")
    
    if not os.path.exists(path):
        print(f"  Directory not found: {path}")
        continue
        
    try:
        # Load only the 'train' split for inspection (usually sufficient)
        dataset = load_from_disk(path)
        if 'train' not in dataset:
            print("  'train' split not found.")
            continue
            
        train_split = dataset['train']
        
        # Check if 'input_label' column exists
        if 'input_label' not in train_split.column_names:
             print(f"  ERROR: 'input_label' column missing in {name}!")
             continue

        print(f"  Number of examples in train split: {len(train_split)}")
        
        # --- Method 1: Using datasets features (if possible) ---
        try:
             label_feature = train_split.features['input_label']
             print(f"  Feature Type: {label_feature}")
             # This might not give min/max directly for Value('int64')
        except Exception as e:
             print(f"  Could not get label feature info: {e}")

        # --- Method 2: Convert to Pandas for robust analysis ---
        print("  Analyzing labels using Pandas...")
        try:
             # Load a sample or the whole dataset into pandas (watch memory for large datasets)
             # For large datasets, consider iterating or sampling: train_split.select(range(10000)).to_pandas()
             df = train_split.to_pandas() 
             
             if 'input_label' not in df.columns:
                 print("  ERROR: 'input_label' column missing in Pandas DataFrame!")
                 continue

             min_label = df['input_label'].min()
             max_label = df['input_label'].max()
             unique_labels = df['input_label'].unique()
             num_unique = len(unique_labels)
             
             print(f"  Min Label Found: {min_label}")
             print(f"  Max Label Found: {max_label}")
             print(f"  Number of Unique Labels: {num_unique}")
             
             # Print unique labels if there aren't too many
             if num_unique < 50: 
                 print(f"  Unique Labels: {sorted(unique_labels)}")
             else:
                 print(f"  Unique Labels (Sample): {sorted(unique_labels[:50])}...")
                 
             # Check against expected ranges (adjust ranges if needed)
             expected_max = -1
             if name == "casehold": expected_max = 4
             elif name == "unfair_tos": expected_max = 7
             elif name == "scotus": expected_max = 12
             elif name == "ledgar": expected_max = 99
             
             if expected_max != -1:
                 if min_label < 0 or max_label > expected_max:
                     print(f"  !!! WARNING: Labels are outside expected range [0, {expected_max}] !!!")
                     # Find specific out-of-range values
                     out_of_bounds = df[(df['input_label'] < 0) | (df['input_label'] > expected_max)]['input_label'].unique()
                     print(f"  Out-of-bounds values found: {sorted(out_of_bounds)}")
                 else:
                     print(f"  Labels seem within expected range [0, {expected_max}].")
             else:
                 print("  Could not determine expected range for comparison.")

        except Exception as e:
            print(f"  Error analyzing with Pandas: {e}")
            # Fallback: Try iterating through a sample if pandas fails
            print("  Attempting direct iteration check on sample...")
            try:
                sample_labels = [train_split[i]['input_label'] for i in range(min(1000, len(train_split)))]
                min_l, max_l = min(sample_labels), max(sample_labels)
                print(f"  Sample Min Label: {min_l}")
                print(f"  Sample Max Label: {max_l}")
                if expected_max != -1 and (min_l < 0 or max_l > expected_max):
                     print(f"  !!! WARNING: Sample labels are outside expected range [0, {expected_max}] !!!")

            except Exception as e_iter:
                print(f"  Error during direct iteration check: {e_iter}")


    except Exception as e:
        print(f"  Error loading or processing dataset {name}: {e}")

print("\nInspection complete.") 