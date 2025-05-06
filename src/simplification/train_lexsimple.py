import os
import yaml
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset, Dataset
import evaluate
import numpy as np
import pandas as pd

def load_config(config_path='config/simplification.yaml'):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_simplification_dataset(dataset_name):
    """Load dataset and create unified train/val/test splits"""
    try:
        if dataset_name == "turk":
            dataset = load_dataset("turk", "simplification")
            examples = []
            
            # Combine all splits since Turk dataset doesn't have a standard split structure
            for split in dataset:
                for item in dataset[split]:
                    if len(item["simplifications"]) > 0:
                        examples.append({
                            "complex": item["original"],
                            "simple": item["simplifications"][0]  # Use first simplification
                        })
            
            # Create combined dataset and split into train/val/test (80/10/10)
            from datasets import Dataset as HFDataset
            combined_dataset = HFDataset.from_list(examples)
            splits = combined_dataset.train_test_split(test_size=0.2, seed=42)
            train_test = splits['test'].train_test_split(test_size=0.5, seed=42)
            
            return {
                'train': splits['train'],
                'validation': train_test['train'],
                'test': train_test['test']
            }
            
        elif dataset_name == "wikilarge":
            dataset = load_dataset("bogdancazan/wikilarge-text-simplification")
            
            # Create standard splits if they don't exist
            if 'train' not in dataset or 'validation' not in dataset or 'test' not in dataset:
                all_examples = []
                for split in dataset:
                    for example in dataset[split]:
                        all_examples.append({
                            "complex": example["Normal"],
                            "simple": example["Simple"]
                        })
                
                from datasets import Dataset as HFDataset
                combined_dataset = HFDataset.from_list(all_examples)
                splits = combined_dataset.train_test_split(test_size=0.2, seed=42)
                train_test = splits['test'].train_test_split(test_size=0.5, seed=42)
                
                return {
                    'train': splits['train'],
                    'validation': train_test['train'],
                    'test': train_test['test']
                }
            else:
                # Use existing splits with unified column names
                processed_dataset = {}
                for split in dataset:
                    processed_dataset[split] = dataset[split].rename_columns({
                        "Normal": "complex", 
                        "Simple": "simple"
                    })
                
                return processed_dataset
            
        elif dataset_name == "multisim":
            dataset = load_dataset("MichaelR207/MultiSim")
            
            # Create standard splits if they don't exist
            if 'train' not in dataset or 'validation' not in dataset or 'test' not in dataset:
                all_examples = []
                for split in dataset:
                    for example in dataset[split]:
                        if "original" in example and "simple" in example:
                            all_examples.append({
                                "complex": example["original"],
                                "simple": example["simple"]
                            })
                
                from datasets import Dataset as HFDataset
                combined_dataset = HFDataset.from_list(all_examples)
                splits = combined_dataset.train_test_split(test_size=0.2, seed=42)
                train_test = splits['test'].train_test_split(test_size=0.5, seed=42)
                
                return {
                    'train': splits['train'],
                    'validation': train_test['train'],
                    'test': train_test['test']
                }
            else:
                # Use existing splits but standardize column names
                processed_dataset = {}
                for split in dataset:
                    processed_dataset[split] = dataset[split].rename_columns({
                        "original": "complex"
                    })
                
                return processed_dataset
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
    except Exception as e:
        raise ValueError(f"Failed to load dataset {dataset_name}: {e}")

def preprocess_function(examples, tokenizer, max_input_length, max_target_length):
    """Tokenize and prepare examples for training"""
    inputs = examples["complex"]
    targets = examples["simple"]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, max_length=max_input_length, padding="max_length", truncation=True
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=max_target_length, padding="max_length", truncation=True
        )
    
    model_inputs["labels"] = labels["input_ids"]
    
    # Replace padding token id with -100 for loss calculation
    model_inputs["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in label]
        for label in model_inputs["labels"]
    ]
    
    return model_inputs

def compute_metrics(eval_preds, tokenizer, generation_params):
    """Compute ROUGE and BLEU metrics for evaluation"""
    try:
        # Import required packages
        from rouge_score import rouge_scorer
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        # Decode predictions and references
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        # Calculate ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        results = {
            'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
            'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
            'rougeL': {'precision': [], 'recall': [], 'fmeasure': []},
        }
        
        for pred, label in zip(decoded_preds, decoded_labels):
            scores = scorer.score(label, pred)
            for rouge_type in results.keys():
                results[rouge_type]['precision'].append(scores[rouge_type].precision)
                results[rouge_type]['recall'].append(scores[rouge_type].recall)
                results[rouge_type]['fmeasure'].append(scores[rouge_type].fmeasure)
        
        # Calculate BLEU scores
        smoothing = SmoothingFunction().method1
        bleu_scores = []
        
        for pred, ref in zip(decoded_preds, decoded_labels):
            pred_tokens = pred.split()
            ref_tokens = [ref.split()]  # BLEU expects a list of references
            if pred_tokens:
                try:
                    bleu_scores.append(sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing))
                except Exception as e:
                    print(f"Error calculating BLEU: {e}")
        
        # Average the scores
        rouge_results = {}
        for rouge_type in results.keys():
            avg_fmeasure = np.mean(results[rouge_type]['fmeasure']) * 100
            rouge_results[rouge_type] = round(avg_fmeasure, 2)
        
        bleu_result = np.mean(bleu_scores) * 100 if bleu_scores else 0
        
        return {
            "rouge1": rouge_results['rouge1'],
            "rouge2": rouge_results['rouge2'],
            "rougeL": rouge_results['rougeL'],
            "bleu": round(bleu_result, 2)
        }
        
    except Exception as e:
        print(f"Error computing metrics: {e}")
        import traceback
        traceback.print_exc()
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu": 0.0}

def main():
    # Load configuration
    config = load_config()
    
    # Get required settings
    model_name = config['model']['base_model']
    output_model_name = config['model']['simplification_model_name']
    dataset_name = config['dataset']['name']
    output_dir = config['paths']['output_models'].format(model_name=output_model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset and model
    print(f"Loading {dataset_name} dataset...")
    dataset = load_simplification_dataset(dataset_name)
    print(f"Dataset loaded with {len(dataset['train'])} training examples")
    
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Extract training parameters
    training_config = config['training']
    max_input_length = training_config['max_input_length']
    max_target_length = training_config['max_target_length']
    
    # Preprocess data
    print("Preprocessing dataset...")
    tokenized_dataset = {}
    for split in dataset:
        tokenized_dataset[split] = dataset[split].map(
            lambda examples: preprocess_function(
                examples, tokenizer, max_input_length, max_target_length
            ),
            batched=True,
            remove_columns=dataset[split].column_names,
        )
    
    # Configure training
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if training_config['fp16'] else None,
    )
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=training_config['evaluation_strategy'],
        save_strategy="steps",
        save_steps=training_config.get('save_steps', 100),
        eval_steps=training_config.get('eval_steps', 100),
        learning_rate=float(training_config['learning_rate']),
        per_device_train_batch_size=training_config['batch_size'],
        per_device_eval_batch_size=training_config['eval_batch_size'],
        weight_decay=training_config['weight_decay'],
        save_total_limit=training_config.get('save_total_limit', 5),
        num_train_epochs=training_config['epochs'],
        predict_with_generate=True,
        fp16=training_config['fp16'],
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        warmup_steps=training_config['warmup_steps'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
    )
    
    # Set up trainer
    generation_params = config['simplification_params']['generation_params']
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer, generation_params),
    )
    
    # Train and evaluate
    print(f"Starting training with model {model_name}...")
    print(f"Training for {training_config['epochs']} epochs with batch size {training_config['batch_size']}")
    trainer.train()
    
    print("Evaluating model on test set...")
    results = trainer.evaluate(tokenized_dataset["test"])
    print(f"Test results: {results}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model and tokenizer saved to {final_model_path}")

if __name__ == "__main__":
    main() 