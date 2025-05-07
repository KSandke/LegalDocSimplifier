import os
import yaml
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import nltk
import numpy as np
import traceback
from rouge_score import rouge_scorer, scoring

# --- Configuration Loading ---
def load_config(config_path='config/summarization.yaml'):
    """Loads abstractive summarization settings from config file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Provide defaults if config section missing
    if 'abstractive' not in config:
        return {
            'dataset_name': 'ChicagoHAI/CaseSumm',
            'dataset_split': 'train',
            'text_column': 'opinion',
            'summary_column': 'syllabus',
            'base_model': 'nsi319/legal-pegasus',
            'max_input_length': 1024,
            'max_target_length': 256,
            'batch_size': 4,
            'num_beams': 5,
            'length_penalty': 2.0,
            'min_length': 100,
            'no_repeat_ngram_size': 3,
            'early_stopping': True
        }
    return config['abstractive']

# --- Data Preprocessing ---
def preprocess_function(examples, tokenizer, max_input_length, max_target_length, text_column, summary_column):
    """Tokenizes input texts and reference summaries for model processing."""
    # Handle T5 models which require a prefix
    is_t5_model = 't5' in tokenizer.name_or_path.lower()
    
    inputs = ["summarize: " + doc for doc in examples[text_column]] if is_t5_model else examples[text_column]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

    # Tokenize target summaries
    labels = tokenizer(text_target=examples[summary_column], max_length=max_target_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- Post-processing for better summaries ---
def post_process_summary(text):
    """Improves generated summaries by fixing formatting and addressing truncation issues."""
    # Replace <n> tags with newlines
    text = text.replace("<n>", "\n")
    
    # Convert to numbered points if not already formatted
    if not any(line.strip().startswith(str(i)+'.') for i in range(1,10) for line in text.split('\n')):
        sentences = nltk.sent_tokenize(text)
        if len(sentences) > 1:
            return '\n'.join([f"{i+1}. {sent}" for i, sent in enumerate(sentences)])
    
    # Process numbered points and maintain proper sequence
    lines = text.split('\n')
    numbered_points = []
    current_number = 1
    
    for line in lines:
        if line.strip() and line.strip()[0].isdigit() and '. ' in line[:5]:
            numbered_line = f"{current_number}. {line.split('. ', 1)[1]}"
            numbered_points.append(numbered_line)
            current_number += 1
        elif line.strip():
            numbered_points.append(line)
    
    # Rejoin with proper newlines
    cleaned_text = '\n'.join(numbered_points)
    
    # Handle incomplete sentences at the end
    if cleaned_text and not cleaned_text.rstrip()[-1] in ['.', '?', '!']:
        # Check if last point has at least 3 words
        last_point = numbered_points[-1] if numbered_points else ""
        words_in_last_point = len(last_point.split())
        
        if words_in_last_point < 3:
            # Remove very short fragments
            if len(numbered_points) > 1:
                cleaned_text = '\n'.join(numbered_points[:-1])
        else:
            # Find natural stopping points
            last_period = max(cleaned_text.rfind('.'), cleaned_text.rfind('?'), cleaned_text.rfind('!'))
            last_comma = cleaned_text.rfind(',')
            last_conjunction = max(
                cleaned_text.rfind(' and '), 
                cleaned_text.rfind(' but '),
                cleaned_text.rfind(' or ')
            )
            
            # Complete the sentence in the most natural way possible
            if last_period > len(cleaned_text) * 0.7:
                cleaned_text = cleaned_text[:last_period+1]
            elif last_comma > len(cleaned_text) * 0.8:
                cleaned_text = cleaned_text[:last_comma+1] + " etc."
            elif last_conjunction > len(cleaned_text) * 0.8:
                cleaned_text = cleaned_text[:last_conjunction] + "."
            else:
                cleaned_text = cleaned_text.rstrip() + "..."
    
    return cleaned_text.strip()

# --- Evaluation using rouge-score ---
def compute_metrics(decoded_preds, decoded_labels):
    """Calculates ROUGE scores between generated and reference summaries."""
    # Initialize scorer for key ROUGE metrics
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Prepare results dictionary
    results = {
        'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
        'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
        'rougeL': {'precision': [], 'recall': [], 'fmeasure': []},
    }

    # Calculate scores for each prediction/reference pair
    for pred, label in zip(decoded_preds, decoded_labels):
        # Format texts for sentence-level ROUGE evaluation
        pred_proc = "\n".join(nltk.sent_tokenize(pred.strip()))
        label_proc = "\n".join(nltk.sent_tokenize(label.strip()))

        # Calculate scores for this pair
        scores = scorer.score(label_proc, pred_proc)

        # Store all score components
        for rouge_type in results.keys():
            results[rouge_type]['precision'].append(scores[rouge_type].precision)
            results[rouge_type]['recall'].append(scores[rouge_type].recall)
            results[rouge_type]['fmeasure'].append(scores[rouge_type].fmeasure)

    # Calculate averages (focus on F1 scores)
    final_scores = {}
    for rouge_type in results.keys():
        avg_fmeasure = np.mean(results[rouge_type]['fmeasure']) * 100
        final_scores[rouge_type] = round(avg_fmeasure, 4)

    # Add average summary length
    prediction_lens = [len(pred.split()) for pred in decoded_preds]
    final_scores["gen_len"] = round(np.mean(prediction_lens), 4)

    return final_scores

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # 1. Load Configuration
        print("Loading configuration...")
        config = load_config()

        # Extract parameters 
        dataset_name = config['dataset_name']
        dataset_split = config['dataset_split']
        text_column = config['text_column']
        summary_column = config['summary_column']
        base_model = config['base_model']
        max_input_length = config['max_input_length']
        max_target_length = config['max_target_length']
        batch_size = config.get('batch_size', 4)
        num_beams = config.get('num_beams', 5)
        length_penalty = config.get('length_penalty', 2.0)
        min_length = config.get('min_length', 100)
        no_repeat_ngram_size = config.get('no_repeat_ngram_size', 3)
        early_stopping = config.get('early_stopping', True)
        path_to_finetuned = config.get('path_to_finetuned_model')

        # Print model info
        print(f"Using model: {path_to_finetuned if path_to_finetuned else base_model}")
        
        # 2. Load Dataset
        print(f"Loading dataset '{dataset_name}' split '{dataset_split}'...")
        
        # Try loading from local directories first, then Hugging Face
        processed_path = os.path.join('data', 'processed', dataset_name)
        standardized_path = os.path.join('data', 'standardized', dataset_name)
        
        try:
            # Check if dataset exists locally
            if os.path.exists(processed_path):
                print(f"Loading dataset from {processed_path}...")
                raw_datasets = datasets.load_from_disk(processed_path)
                if dataset_split in raw_datasets:
                    raw_datasets = raw_datasets[dataset_split]
                    print(f"Loaded {len(raw_datasets)} examples from processed data.")
                else:
                    print(f"Split '{dataset_split}' not found in dataset. Available splits: {list(raw_datasets.keys())}")
                    # Use first available split if requested split is not found
                    first_split = list(raw_datasets.keys())[0]
                    print(f"Using '{first_split}' split instead.")
                    raw_datasets = raw_datasets[first_split]
            elif os.path.exists(standardized_path):
                print(f"Loading dataset from {standardized_path}...")
                raw_datasets = datasets.load_from_disk(standardized_path)
                if dataset_split in raw_datasets:
                    raw_datasets = raw_datasets[dataset_split]
                    print(f"Loaded {len(raw_datasets)} examples from standardized data.")
                else:
                    print(f"Split '{dataset_split}' not found in dataset. Available splits: {list(raw_datasets.keys())}")
                    first_split = list(raw_datasets.keys())[0]
                    print(f"Using '{first_split}' split instead.")
                    raw_datasets = raw_datasets[first_split]
            else:
                # Try to load from Hugging Face
                print(f"Dataset not found locally. Trying to load from Hugging Face Hub...")
                raw_datasets = datasets.load_dataset(dataset_name, split=dataset_split)
                print(f"Loaded {len(raw_datasets)} examples from Hugging Face Hub.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Please ensure dataset exists in data/processed/ or data/standardized/ directory.")
            print("Available processed datasets:", os.listdir('data/processed') if os.path.exists('data/processed') else "None")
            print("Available standardized datasets:", os.listdir('data/standardized') if os.path.exists('data/standardized') else "None")
            raise

        print(f"Loaded {len(raw_datasets)} examples.")

        # Verify required columns exist
        if text_column not in raw_datasets.column_names:
            available_columns = raw_datasets.column_names
            print(f"Warning: Text column '{text_column}' not found in dataset. Available columns: {available_columns}")
            # Try to find a suitable text column
            text_candidates = [col for col in available_columns if any(t in col.lower() for t in ['text', 'content', 'document', 'opinion'])]
            if text_candidates:
                text_column = text_candidates[0]
                print(f"Using '{text_column}' as text column instead.")
            else:
                raise ValueError(f"Could not find a suitable text column in {available_columns}")
                
        if summary_column not in raw_datasets.column_names:
            available_columns = raw_datasets.column_names
            print(f"Warning: Summary column '{summary_column}' not found in dataset. Available columns: {available_columns}")
            # Try to find a suitable summary column
            summary_candidates = [col for col in available_columns if any(s in col.lower() for s in ['summary', 'abstract', 'syllabus'])]
            if summary_candidates:
                summary_column = summary_candidates[0]
                print(f"Using '{summary_column}' as summary column instead.")
            else:
                raise ValueError(f"Could not find a suitable summary column in {available_columns}")

        # Limit dataset size for quick testing
        num_examples_to_process = 5 
        raw_datasets = raw_datasets.select(range(num_examples_to_process))
        print(f"Processing only the first {num_examples_to_process} examples for testing.")

        # 3. Load Tokenizer and Model
        model_load_path = path_to_finetuned if path_to_finetuned else base_model
        print(f"Loading tokenizer and model from '{model_load_path}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_load_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_load_path)

        # Use GPU if available
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
            remove_columns=raw_datasets.column_names
        )

        # 5. Prepare DataLoader
        print("Preparing DataLoader...")
        tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        eval_dataloader = DataLoader(tokenized_datasets, collate_fn=data_collator, batch_size=batch_size)

        # 6. Generate Summaries
        print("Generating summaries...")
        all_preds = []
        all_labels = []
        model.eval()
        for batch in eval_dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
            with torch.no_grad():
                generated_tokens = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=max_target_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    early_stopping=early_stopping,
                    decoder_start_token_id=model.config.decoder_start_token_id
                )

            # Decode generated tokens
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            # Apply post-processing to fix formatting issues
            decoded_preds = [post_process_summary(pred) for pred in decoded_preds]
            all_preds.extend(decoded_preds)

            # Decode labels (references)
            # Replace -100 in labels as we can't decode them
            labels = batch['labels']
            labels = labels.cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            all_labels.extend(decoded_labels)

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

        # 8. Print example summaries
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
        print("Consider running: pip install -r requirements.txt --upgrade")
        print("You might also need: pip install torch torchvision torchaudio transformers datasets evaluate rouge_score nltk sentencepiece")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc() 