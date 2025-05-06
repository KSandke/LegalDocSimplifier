import argparse
import yaml
import os
from datasets import load_dataset

def load_config(config_path='config/simplification.yaml'):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load configuration and get dataset name
    config = load_config()
    dataset_name = config['dataset']['name']
    num_examples = 5
    
    # Load dataset from Hugging Face
    print(f"Loading {dataset_name} dataset from Hugging Face...")
    try:
        if dataset_name == "turk":
            dataset = load_dataset("turk", "simplification")
            key_mapping = {"original": "complex", "simplifications": "simple"}
        elif dataset_name == "wikilarge":
            dataset = load_dataset("bogdancazan/wikilarge-text-simplification")
            key_mapping = {"Normal": "complex", "Simple": "simple"}
        elif dataset_name == "multisim":
            dataset = load_dataset("MichaelR207/MultiSim")
            key_mapping = {"original": "complex", "simple": "simple"}
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        print(f"Successfully loaded {dataset_name} dataset")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    # Display dataset information
    print("\nDataset structure:")
    print(f"Available splits: {list(dataset.keys())}")
    
    # Show samples from each split
    for split in dataset.keys():
        split_data = dataset[split]
        print(f"\nSplit: {split}")
        print(f"Number of examples: {len(split_data)}")
        print(f"Features: {split_data.features}")
        
        if len(split_data) == 0:
            print("  (No examples in this split)")
            continue
            
        print(f"\nFirst {num_examples} examples:")
        for i in range(min(num_examples, len(split_data))):
            example = split_data[i]
            print(f"\nExample {i+1}:")
            
            # Show complex and simple text pairs with consistent formatting
            for original_key, display_key in key_mapping.items():
                if original_key in example:
                    value = example[original_key]
                    
                    # Handle special case for simplifications list in turk dataset
                    if original_key == "simplifications" and isinstance(value, list):
                        if len(value) > 0:
                            print(f"  {display_key}: {value[0]}")
                        else:
                            print(f"  {display_key}: (empty list)")
                    elif isinstance(value, str) and len(value) > 100:
                        print(f"  {display_key}: {value[:100]}... (truncated)")
                    else:
                        print(f"  {display_key}: {value}")
    
    # Summarize dataset characteristics
    print("\n" + "="*50)
    print(f"Dataset '{dataset_name}' Information:")
    print("="*50)
    
    if dataset_name == "turk":
        print("Note: The Turk dataset doesn't have a 'train' split by default.")
        print("When used for training, all splits will be combined and re-split (80/10/10)")
    
    total_examples = sum(len(dataset[split]) for split in dataset.keys())
    print(f"Total examples across all splits: {total_examples}")
    print("="*50)

if __name__ == "__main__":
    main() 