import os
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from typing import Dict, List, Optional, Tuple

def load_lexsimple_dataset(
    data_path: Optional[str] = None,
    use_huggingface: bool = True,
    split_ratio: Dict[str, float] = {"train": 0.8, "validation": 0.1, "test": 0.1},
    random_state: int = 42,
) -> DatasetDict:
    """
    Load the LexSimple dataset either from local file or HuggingFace
    
    Args:
        data_path: Path to local dataset (CSV expected with 'complex' and 'simple' columns)
        use_huggingface: Whether to download from HuggingFace instead of local
        split_ratio: Dataset split ratios for train/validation/test
        random_state: Random seed for reproducibility
        
    Returns:
        DatasetDict containing the loaded dataset with splits
    """
    if use_huggingface:
        try:
            # Try to load from HuggingFace
            dataset = load_dataset("legal_simplification/lexsimple")
            print(f"Successfully loaded LexSimple dataset from HuggingFace")
            return dataset
        except Exception as e:
            print(f"Failed to load LexSimple from HuggingFace: {e}")
            if data_path is None:
                raise ValueError("Cannot load from HuggingFace and no local data path provided")
    
    # Load from local CSV file
    if data_path is None:
        data_path = os.path.join("data", "raw", "lexsimple.csv")
        
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"LexSimple dataset not found at {data_path}")
    
    # Try to read the dataset and determine its format
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded LexSimple dataset with {len(df)} examples")
        
        # Check for required columns
        required_cols = {"complex", "simple"}
        if not required_cols.issubset(set(df.columns)):
            # If columns don't match exactly, try to infer
            if len(df.columns) >= 2:
                print(f"Warning: Expected columns {required_cols} not found")
                print(f"Found columns: {df.columns}")
                print(f"Using first column as 'complex' and second as 'simple'")
                df.columns = ["complex", "simple"] + list(df.columns[2:])
            else:
                raise ValueError(f"Dataset doesn't have enough columns. Found: {df.columns}")
    except Exception as e:
        # Try alternative format (e.g., TSV or other delimiter)
        try:
            df = pd.read_csv(data_path, sep='\t')
            print(f"Loaded LexSimple dataset as TSV with {len(df)} examples")
            if len(df.columns) >= 2:
                df.columns = ["complex", "simple"] + list(df.columns[2:])
        except:
            raise ValueError(f"Failed to load dataset: {e}")
    
    # Create Dataset object
    dataset = Dataset.from_pandas(df)
    
    # Split the dataset
    split_dataset = dataset.train_test_split(
        test_size=split_ratio["validation"] + split_ratio["test"],
        seed=random_state
    )
    
    # Further split the test portion into validation and test
    test_valid_split = split_ratio["test"] / (split_ratio["validation"] + split_ratio["test"])
    test_valid_dataset = split_dataset["test"].train_test_split(
        test_size=test_valid_split,
        seed=random_state
    )
    
    # Create final DatasetDict
    final_dataset = DatasetDict({
        "train": split_dataset["train"],
        "validation": test_valid_dataset["train"],
        "test": test_valid_dataset["test"]
    })
    
    return final_dataset

if __name__ == "__main__":
    # Example usage
    try:
        dataset = load_lexsimple_dataset(use_huggingface=True)
        print(f"Dataset splits: {dataset.keys()}")
        print(f"Example from train split: {dataset['train'][0]}")
    except Exception as e:
        print(f"Could not load from HuggingFace, trying local path...")
        try:
            dataset = load_lexsimple_dataset(
                data_path="data/raw/lexsimple.csv", 
                use_huggingface=False
            )
            print(f"Dataset splits: {dataset.keys()}")
            print(f"Example from train split: {dataset['train'][0]}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Please make sure the dataset is available locally or create a sample dataset") 