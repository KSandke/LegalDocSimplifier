from datasets import load_from_disk
import pandas as pd
import os

# Check if datasets are already saved locally
DATASETS_DIR = 'data/processed'
dataset_paths = {
    'scotus': os.path.join(DATASETS_DIR, 'scotus_dataset'),
    'ledgar': os.path.join(DATASETS_DIR, 'ledgar_dataset'),
    'unfair_tos': os.path.join(DATASETS_DIR, 'unfair_tos_dataset'),
    'casehold': os.path.join(DATASETS_DIR, 'casehold_dataset')
}

# Load datasets
datasets = {}
for name, path in dataset_paths.items():
    if os.path.exists(path):
        print(f"Loading {name} dataset from disk...")
        datasets[name] = load_from_disk(path)
    else:
        print(f"Dataset {name} not found locally. Run lex_glue_loader.py first.")

if not datasets:
    print("No datasets found. Run lex_glue_loader.py first to download and save the datasets.")
    exit()

# Method 1: Convert to pandas DataFrame for easy viewing
def view_as_dataframe(dataset, split='train', num_examples=10):
    """Convert dataset to pandas DataFrame for easy viewing and analysis"""
    if split not in dataset:
        print(f"Split '{split}' not found. Available splits: {dataset.keys()}")
        return None
    
    # Convert to pandas DataFrame (for the specified split)
    df = pd.DataFrame(dataset[split][:num_examples])
    return df

# Method 2: Interactive example viewer
def interactive_viewer(dataset, split='train'):
    """Simple interactive viewer to browse through examples"""
    if split not in dataset:
        print(f"Split '{split}' not found. Available splits: {dataset.keys()}")
        return
    
    data = dataset[split]
    current_idx = 0
    total = len(data)
    
    while True:
        print(f"\n--- Example {current_idx+1}/{total} ---")
        example = data[current_idx]
        
        # Display all fields
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 500:
                print(f"{key}: {value[:500]}... (truncated)")
            else:
                print(f"{key}: {value}")
        
        # Navigation options
        print("\nOptions:")
        print("n: next example")
        print("p: previous example")
        print("j: jump to index")
        print("q: quit")
        
        choice = input("Enter choice: ").strip().lower()
        
        if choice == 'n':
            current_idx = (current_idx + 1) % total
        elif choice == 'p':
            current_idx = (current_idx - 1) % total
        elif choice == 'j':
            try:
                idx = int(input(f"Enter index (1-{total}): ")) - 1
                if 0 <= idx < total:
                    current_idx = idx
                else:
                    print(f"Index must be between 1 and {total}")
            except ValueError:
                print("Invalid index")
        elif choice == 'q':
            break
        else:
            print("Invalid choice")

# Method 3: Export to CSV
def export_to_csv(dataset, split='train', output_path=None, num_examples=None):
    """Export dataset to CSV for viewing in spreadsheet software"""
    if split not in dataset:
        print(f"Split '{split}' not found. Available splits: {dataset.keys()}")
        return
    
    data = dataset[split]
    if num_examples is not None:
        data = data.select(range(min(num_examples, len(data))))
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Set default output path if not provided
    if output_path is None:
        os.makedirs('data/exports', exist_ok=True)
        output_path = f'data/exports/{dataset_name}_{split}.csv'
    
    # Export to CSV
    df.to_csv(output_path, index=False)
    print(f"Exported to {output_path}")
    return output_path

# Main menu
def main_menu():
    while True:
        print("\n=== Dataset Viewer ===")
        print("Available datasets:")
        for i, name in enumerate(datasets.keys(), 1):
            print(f"{i}. {name}")
        print("0. Exit")
        
        try:
            choice = int(input("Select dataset (0-exit): "))
            if choice == 0:
                break
            
            if 1 <= choice <= len(datasets):
                dataset_name = list(datasets.keys())[choice-1]
                dataset = datasets[dataset_name]
                
                # Show dataset details
                print(f"\nDataset: {dataset_name}")
                print(f"Splits: {list(dataset.keys())}")
                print(f"Features: {dataset['train'].features}")
                
                # Show viewing options
                print("\nViewing options:")
                print("1. View as pandas DataFrame")
                print("2. Interactive viewer")
                print("3. Export to CSV")
                print("0. Back")
                
                view_choice = int(input("Select viewing option: "))
                
                if view_choice == 1:
                    split = input("Enter split name (default: train): ") or 'train'
                    num_examples = int(input("Number of examples to view: ") or "10")
                    df = view_as_dataframe(dataset, split, num_examples)
                    if df is not None:
                        print("\nDataFrame Preview:")
                        print(df)
                        print("\nColumn Statistics:")
                        print(df.describe(include='all'))
                
                elif view_choice == 2:
                    split = input("Enter split name (default: train): ") or 'train'
                    interactive_viewer(dataset, split)
                
                elif view_choice == 3:
                    split = input("Enter split name (default: train): ") or 'train'
                    num_examples = input("Number of examples to export (leave empty for all): ")
                    num_examples = int(num_examples) if num_examples else None
                    export_to_csv(dataset, split, num_examples=num_examples)
            else:
                print("Invalid choice")
        
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main_menu() 