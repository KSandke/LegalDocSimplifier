import os
import datasets
import yaml

# Import centralized configuration manager
try:
    from ..config_manager import ConfigManager
except ImportError:
    # Fallback for when running as script or in tests
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_manager import ConfigManager

def load_config(config_path='config/summarization.yaml'):
    """Loads configuration from a YAML file."""
    try:
        # Use centralized config manager
        config = ConfigManager.load_config(config_path)
        return config.get('fine_tuning', {})
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        print("Using default configuration...")
        return {
            'dataset_name': 'scotus',
            'text_column': 'text',
            'summary_column': 'summary',
            'max_examples': 100
        }

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