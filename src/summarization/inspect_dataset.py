import os
import datasets
import yaml

def load_config(config_path='config/summarization.yaml'):
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['fine_tuning']

def main():
    # Load configuration
    config = load_config()
    dataset_name = config['dataset_name']
    text_column = config['text_column']
    summary_column = config['summary_column']
    
    # Load dataset
    print(f"Loading dataset '{dataset_name}'...")
    processed_path = os.path.join('data', 'processed', dataset_name)
    
    dataset = datasets.load_from_disk(processed_path)
    print(f"Dataset splits: {list(dataset.keys())}")
    
    # Look at first few examples
    print("\nExamining 'train' split:")
    train_data = dataset['train']
    print(f"Number of examples: {len(train_data)}")
    print(f"Column names: {train_data.column_names}")
    
    # Check the types of columns
    example = train_data[0]
    for column in train_data.column_names:
        value = example[column]
        print(f"\nColumn: {column}")
        print(f"Type: {type(value)}")
        print(f"Value: {value}")
        
        # Print more info for the summary column
        if column == summary_column:
            print("\nMore detail about the summary column:")
            print(f"First 5 values:")
            for i in range(min(5, len(train_data))):
                val = train_data[i][summary_column]
                print(f"  [{i}] Type: {type(val)}, Value: {val}")
    
    # Check how many unique values we have in the label column
    if summary_column in train_data.column_names:
        unique_values = set()
        for example in train_data:
            val = example[summary_column]
            if isinstance(val, (list, tuple, dict)):
                val = str(val)  # Convert containers to strings for counting
            unique_values.add(val)
        
        print(f"\nUnique values in '{summary_column}' column: {len(unique_values)}")
        
        if len(unique_values) < 20:
            print("All unique values:")
            for i, val in enumerate(unique_values):
                print(f"  [{i}] {val}")

if __name__ == "__main__":
    main() 