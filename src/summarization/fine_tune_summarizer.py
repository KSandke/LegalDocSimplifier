import os
import yaml
import torch
import datasets
import numpy as np
import nltk
import traceback
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from rouge_score import rouge_scorer
from tqdm import tqdm

# --- Configuration Loading ---
def load_config(config_path='config/summarization.yaml'):
    """Loads fine-tuning configuration from the YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Provide defaults if section is missing
    if 'fine_tuning' not in config:
        return {
            'dataset_name': 'scotus',
            'text_column': 'text',
            'summary_column': 'summary',
            'base_model': 'google/pegasus-cnn_dailymail',
            'max_input_length': 1024,
            'max_target_length': 256,
            'batch_size': 2,
            'learning_rate': 5e-5,
            'weight_decay': 0.01,
            'num_epochs': 4,
            'output_dir': './models/pegasus-legal-summarizer',
            'logging_dir': './logs'
        }
    return config['fine_tuning']

# --- Data Preprocessing ---
def preprocess_function(examples, tokenizer, max_input_length, max_target_length, text_column, summary_column):
    """Tokenizes texts and summaries for fine-tuning."""
    # Convert inputs to strings if they're not already
    inputs = examples[text_column]
    if inputs and not isinstance(inputs[0], str):
        inputs = [str(item) if item is not None else "" for item in inputs]
    
    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

    # Handle summaries, ensuring they're strings
    summaries = examples[summary_column]
    if summaries and not isinstance(summaries[0], str):
        summaries = [str(item) if item is not None else "" for item in summaries]
    
    # Tokenize targets
    labels = tokenizer(text_target=summaries, max_length=max_target_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- Evaluation Metrics ---
def compute_metrics(eval_pred):
    """Calculates ROUGE scores for model evaluation during training."""
    predictions, labels = eval_pred
    
    # Decode generated tokens
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate scores for each pair
    rouge1_f = []
    rouge2_f = []
    rougeL_f = []
    
    for pred, label in zip(decoded_preds, decoded_labels):
        # Clean up predictions for better evaluation
        pred = post_process_summary(pred)
        
        # Calculate scores
        scores = scorer.score(label, pred)
        rouge1_f.append(scores['rouge1'].fmeasure)
        rouge2_f.append(scores['rouge2'].fmeasure)
        rougeL_f.append(scores['rougeL'].fmeasure)
    
    # Calculate mean scores
    result = {
        'rouge1': np.mean(rouge1_f) * 100,
        'rouge2': np.mean(rouge2_f) * 100,
        'rougeL': np.mean(rougeL_f) * 100
    }
    
    # Include prediction length metrics
    prediction_lens = [len(pred.split()) for pred in decoded_preds]
    result["gen_len"] = np.mean(prediction_lens)
    
    return result

# --- Post-processing for better summaries ---
def post_process_summary(text):
    """Improves summary formatting and handles truncated sentences."""
    lines = text.split('\n')
    numbered_points = []
    current_number = 1
    
    for line in lines:
        # Correct numbering for points
        if line.strip() and line.strip()[0].isdigit() and '. ' in line[:5]:
            numbered_line = f"{current_number}. {line.split('. ', 1)[1]}"
            numbered_points.append(numbered_line)
            current_number += 1
        elif line.strip():
            numbered_points.append(line)
    
    # Rejoin with proper newlines
    cleaned_text = '\n'.join(numbered_points)
    
    # Fix incomplete sentences
    if cleaned_text and not cleaned_text.rstrip()[-1] in ['.', '?', '!']:
        last_period = max(cleaned_text.rfind('.'), cleaned_text.rfind('?'), cleaned_text.rfind('!'))
        if last_period > len(cleaned_text) * 0.7:
            cleaned_text = cleaned_text[:last_period+1]
    
    return cleaned_text.strip()

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # 1. Load Configuration
        print("Loading configuration...")
        config = load_config()
        
        # Extract parameters
        dataset_name = config['dataset_name']
        text_column = config['text_column']
        summary_column = config['summary_column']
        base_model = config['base_model']
        max_input_length = config['max_input_length']
        max_target_length = config['max_target_length']
        batch_size = config['batch_size']
        learning_rate = float(config['learning_rate'])
        weight_decay = config['weight_decay']
        num_epochs = config['num_epochs']
        output_dir = config['output_dir']
        logging_dir = config['logging_dir']
        
        # Create directories if needed
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        
        print("\nVerifying dataset and columns...")
        # Check column structure in a small sample
        processed_path = os.path.join('data', 'processed', dataset_name)
        if os.path.exists(processed_path):
            try:
                sample_data = datasets.load_from_disk(processed_path)
                if 'train' in sample_data:
                    sample = sample_data['train'].select(range(min(5, len(sample_data['train']))))
                    print(f"Sample columns: {sample.column_names}")
                    if text_column in sample.column_names and summary_column in sample.column_names:
                        print(f"✓ Found text column '{text_column}' and summary column '{summary_column}'")
                        # Print sample length statistics
                        text_lengths = [len(x.split()) for x in sample[text_column]]
                        summary_lengths = [len(x.split()) for x in sample[summary_column]]
                        print(f"Average text length: {sum(text_lengths)/len(text_lengths):.1f} words")
                        print(f"Average summary length: {sum(summary_lengths)/len(summary_lengths):.1f} words")
                    else:
                        print(f"⚠️ Column mismatch! Available: {sample.column_names}. Check configuration.")
            except Exception as e:
                print(f"Error checking sample: {e}")
        
        # 2. Load Dataset
        print(f"Loading dataset '{dataset_name}'...")
        processed_path = os.path.join('data', 'processed', dataset_name)
        standardized_path = os.path.join('data', 'standardized', dataset_name)
        
        try:
            if os.path.exists(processed_path):
                print(f"Loading dataset from {processed_path}...")
                dataset = datasets.load_from_disk(processed_path)
                print(f"Dataset loaded with splits: {dataset.keys()}")
            elif os.path.exists(standardized_path):
                print(f"Loading dataset from {standardized_path}...")
                dataset = datasets.load_from_disk(standardized_path)
                print(f"Dataset loaded with splits: {dataset.keys()}")
            else:
                # Try loading from Hugging Face
                print(f"Dataset not found locally. Trying to load from Hugging Face Hub...")
                dataset = datasets.load_dataset(dataset_name)
                print(f"Dataset loaded with splits: {dataset.keys()}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Please ensure dataset exists in data/processed/ or data/standardized/ directory.")
            print("Available processed datasets:", os.listdir('data/processed') if os.path.exists('data/processed') else "None")
            print("Available standardized datasets:", os.listdir('data/standardized') if os.path.exists('data/standardized') else "None")
            traceback.print_exc()
            exit(1)
        
        # 3. Prepare Train/Validation Split
        if 'train' in dataset and 'validation' not in dataset:
            # Create validation split if it doesn't exist
            print("Creating validation split...")
            train_test_split = dataset['train'].train_test_split(test_size=0.2)
            dataset['train'] = train_test_split['train']
            dataset['validation'] = train_test_split['test'] 
            print(f"Split dataset into {len(dataset['train'])} train and {len(dataset['validation'])} validation examples")
        elif 'train' not in dataset and 'test' in dataset:
            # If only test is available, split it for training
            print("Only test split found. Creating train and validation splits...")
            splits = dataset['test'].train_test_split(test_size=0.2)
            dataset = datasets.DatasetDict({
                'train': splits['train'],
                'validation': splits['test']
            })
            print(f"Split dataset into {len(dataset['train'])} train and {len(dataset['validation'])} validation examples")
            
        # Limit training samples if requested
        max_train_samples = config.get('max_train_samples', None)
        if max_train_samples is not None:
            max_train_samples = min(len(dataset['train']), int(max_train_samples))
            dataset['train'] = dataset['train'].select(range(max_train_samples))
            print(f"Limited training data to {max_train_samples} examples for faster training")
        
        # Limit validation samples for faster evaluation
        max_val_samples = config.get('max_val_samples', 1000)
        if max_val_samples is not None and len(dataset['validation']) > max_val_samples:
            dataset['validation'] = dataset['validation'].select(range(max_val_samples))
            print(f"Limited validation data to {max_val_samples} examples for faster evaluation")
            
        # Verify columns exist
        if text_column not in dataset['train'].column_names:
            raise ValueError(f"Text column '{text_column}' not found in dataset. Available columns: {dataset['train'].column_names}")
        if summary_column not in dataset['train'].column_names:
            raise ValueError(f"Summary column '{summary_column}' not found in dataset. Available columns: {dataset['train'].column_names}")
        
        # 4. Load Tokenizer and Model
        print(f"Loading model '{base_model}'...")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        
        # 5. Preprocess Data
        print("Preprocessing data...")
        def preprocess_data(examples):
            return preprocess_function(
                examples, tokenizer, max_input_length, max_target_length, 
                text_column, summary_column
            )
        
        # Process training data
        tokenized_train = dataset['train'].map(
            preprocess_data,
            batched=True,
            remove_columns=dataset['train'].column_names
        )
        
        # Process validation data
        tokenized_valid = dataset['validation'].map(
            preprocess_data,
            batched=True,
            remove_columns=dataset['validation'].column_names
        )
        
        print(f"Processed {len(tokenized_train)} training examples and {len(tokenized_valid)} validation examples")
        
        # 6. Setup Training Arguments
        print("Setting up training configuration...")
        
        # Check GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Configure training parameters
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            evaluation_strategy=config.get('evaluation_strategy', "epoch"),
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=weight_decay,
            save_total_limit=3,
            num_train_epochs=num_epochs,
            predict_with_generate=True,
            fp16=config.get('fp16', torch.cuda.is_available()),
            logging_dir=logging_dir,
            logging_steps=100,
            save_strategy=config.get('save_strategy', "epoch"),
            save_steps=config.get('save_steps', 500),
            load_best_model_at_end=True,
            metric_for_best_model="rouge2",
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
            eval_steps=config.get('eval_steps', 500),
        )
        
        # 7. Initialize Trainer
        print("Initializing trainer...")
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_valid,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        
        # 8. Train the model
        print("Starting training...")
        trainer.train()
        
        # 9. Save the model
        print(f"Saving model to {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # 10. Evaluate on validation set
        print("Final evaluation on validation set...")
        eval_results = trainer.evaluate(max_length=max_target_length)
        
        print("\nTraining complete!")
        print("Evaluation results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value}")
        
        print(f"\nFine-tuned model saved to {output_dir}")
        print("You can now update your config/summarization.yaml to use this model:")
        print(f"  path_to_finetuned_model: '{output_dir}'")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc() 