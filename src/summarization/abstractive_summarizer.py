import os
import yaml
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import nltk # For ROUGE score calculation
import numpy as np
import traceback
from rouge_score import rouge_scorer, scoring # Import rouge-score directly

# --- Configuration Loading ---
def load_config(config_path='config/summarization.yaml'):
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Ensure the abstractive section exists
    if 'abstractive' not in config:
        raise ValueError("'abstractive' section not found in config/summarization.yaml")
    return config['abstractive']

# --- Data Preprocessing ---
def preprocess_function(examples, tokenizer, max_input_length, max_target_length, text_column, summary_column):
    """Tokenizes the text and summary fields."""
    # T5 requires a prefix for summarization tasks
    prefix = "summarize: "
    inputs = [prefix + doc for doc in examples[text_column]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples[summary_column], max_length=max_target_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- Evaluation using rouge-score directly ---
def compute_metrics(decoded_preds, decoded_labels):
    """Computes ROUGE scores using the rouge-score package directly."""
    # Initialize the scorer. We want rouge1, rouge2, and rougeL
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Prepare results dictionary
    results = {
        'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
        'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
        'rougeL': {'precision': [], 'recall': [], 'fmeasure': []},
    }

    # Accumulate scores for each prediction/label pair
    for pred, label in zip(decoded_preds, decoded_labels):
        # Add newline splitting (optional but common for ROUGE)
        pred_proc = "\n".join(nltk.sent_tokenize(pred.strip()))
        label_proc = "\n".join(nltk.sent_tokenize(label.strip()))

        # Calculate scores for this pair
        scores = scorer.score(label_proc, pred_proc)

        # Add scores to lists
        for rouge_type in results.keys():
            results[rouge_type]['precision'].append(scores[rouge_type].precision)
            results[rouge_type]['recall'].append(scores[rouge_type].recall)
            results[rouge_type]['fmeasure'].append(scores[rouge_type].fmeasure)

    # Calculate average scores (fmeasure is often reported)
    final_scores = {}
    for rouge_type in results.keys():
        # Calculate the average F1-score (fmeasure)
        avg_fmeasure = np.mean(results[rouge_type]['fmeasure']) * 100
        final_scores[rouge_type] = round(avg_fmeasure, 4)
        # You could also average precision and recall if needed
        # avg_precision = np.mean(results[rouge_type]['precision']) * 100
        # avg_recall = np.mean(results[rouge_type]['recall']) * 100
        # final_scores[f"{rouge_type}_precision"] = round(avg_precision, 4)
        # final_scores[f"{rouge_type}_recall"] = round(avg_recall, 4)

    # Add mean generated length
    prediction_lens = [len(pred.split()) for pred in decoded_preds] # Simple word count
    final_scores["gen_len"] = round(np.mean(prediction_lens), 4)

    return final_scores

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # 1. Load Configuration
        print("Loading configuration...")
        config = load_config()

        # --- Parameters --- 
        dataset_name = config['dataset_name']
        dataset_split = config['dataset_split']
        text_column = config['text_column']
        summary_column = config['summary_column']
        base_model = config['base_model']
        max_input_length = config['max_input_length']
        max_target_length = config['max_target_length']
        batch_size = config.get('batch_size', 4) # Use .get for optional args
        num_beams = config.get('num_beams', 4)
        length_penalty = config.get('length_penalty', 1.0)
        no_repeat_ngram_size = config.get('no_repeat_ngram_size', 3)
        path_to_finetuned = config.get('path_to_finetuned_model')

        # 2. Load Dataset
        print(f"Loading dataset '{dataset_name}' split '{dataset_split}'...")
        # Note: This might download the dataset if not cached
        raw_datasets = datasets.load_dataset(dataset_name, split=dataset_split)
        print(f"Loaded {len(raw_datasets)} examples.")

        # Limit dataset size for quick testing (optional)
        # num_examples_to_process = 100 # <-- Commented out
        # raw_datasets = raw_datasets.select(range(num_examples_to_process))
        # print(f"Processing only the first {num_examples_to_process} examples for testing.") # <-- Commented out

        # 3. Load Tokenizer and Model
        model_load_path = path_to_finetuned if path_to_finetuned else base_model
        print(f"Loading tokenizer and model from '{model_load_path}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_load_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_load_path)

        # Check for GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Using device: {device}")

        # 4. Preprocess Data
        print("Preprocessing data...")
        tokenized_datasets = raw_datasets.map(
            lambda examples: preprocess_function(
                examples, tokenizer, max_input_length, max_target_length, text_column, summary_column
            ),
            batched=True,
            remove_columns=raw_datasets.column_names # Remove original columns
        )

        # 5. Prepare DataLoader
        print("Preparing DataLoader...")
        # We only need input_ids and attention_mask for generation
        tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        eval_dataloader = DataLoader(tokenized_datasets, collate_fn=data_collator, batch_size=batch_size)

        # 6. Generate Summaries
        print("Generating summaries...")
        all_preds = []
        all_labels = [] # Store labels for ROUGE calculation
        model.eval() # Set model to evaluation mode
        for batch in eval_dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
            with torch.no_grad():
                generated_tokens = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=max_target_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    # Add other generation parameters from config if needed
                )

            # Decode generated tokens
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            all_preds.extend(decoded_preds)

            # Decode labels (references)
            # Replace -100 in labels as we can't decode them
            labels = batch['labels']
            labels = labels.cpu().numpy() # Move to CPU for numpy ops
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            all_labels.extend(decoded_labels)

            # Print progress (optional)
            print(f"  Processed batch {len(all_preds)} / {len(raw_datasets)}")

        print(f"\nGenerated {len(all_preds)} summaries.")

        # 7. Evaluate (Compute ROUGE scores)
        print("\nComputing ROUGE scores...")
        if all_labels:
             rouge_scores = compute_metrics(all_preds, all_labels)
             print("ROUGE Scores:")
             for metric, score in rouge_scores.items():
                 print(f"  {metric}: {score}")
        else:
             print("Skipping ROUGE calculation as no labels were processed.")

        # 8. Print some examples
        print("\n--- Example Summaries ---")
        num_examples_to_print = 3
        for i in range(min(num_examples_to_print, len(all_preds))):
            print(f"\nExample {i+1}:")
            print(f"  Reference Summary: {all_labels[i]}")
            print(f"  Generated Summary: {all_preds[i]}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
        traceback.print_exc()
    except ImportError as e:
        print(f"Import Error: {e}. Make sure you have installed all necessary libraries.")
        print("Consider running: pip install -r requirements.txt --upgrade") # Suggest update
        print("You might also need: pip install torch torchvision torchaudio transformers datasets evaluate rouge_score nltk sentencepiece")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc() 