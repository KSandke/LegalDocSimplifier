import os
import yaml
import torch
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import nltk
import numpy as np
import traceback
from rouge_score import rouge_scorer

# --- Configuration Loading ---
def load_config(config_path='config/summarization.yaml'):
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if 'abstractive' not in config:
        raise ValueError("'abstractive' section not found in config/summarization.yaml")
    return config['abstractive']

# --- Data Preprocessing ---
def preprocess_function(examples, tokenizer, max_input_length, max_target_length, text_column, summary_column):
    """Tokenizes the text and summary fields for T5."""
    prefix = "summarize: "
    inputs = [prefix + (doc if doc is not None else "") for doc in examples[text_column]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(
        text_target=[(doc if doc is not None else "") for doc in examples[summary_column]],
        max_length=max_target_length,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- Evaluation using rouge-score ---
def compute_metrics(eval_pred):
    """Computes ROUGE scores for Seq2SeqTrainer evaluation."""
    predictions, labels = eval_pred
    # Decode generated summaries (predictions)
    # Replace -100 in predictions as we can't decode them (might occur in pad tokens)
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Decode reference summaries (labels)
    # Replace -100 in labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Initialize the scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True) # Use rougeLsum for summaries

    # Calculate scores and aggregate
    results = {
        'rouge1': [], 'rouge2': [], 'rougeLsum': []
    }
    for pred, label in zip(decoded_preds, decoded_labels):
        score = scorer.score(label, pred)
        results['rouge1'].append(score['rouge1'].fmeasure)
        results['rouge2'].append(score['rouge2'].fmeasure)
        results['rougeLsum'].append(score['rougeLsum'].fmeasure)

    # Compute averages
    final_scores = {key: np.mean(val) * 100 for key, val in results.items()}

    # Add mean generated length
    prediction_lens = [len(pred.split()) for pred in decoded_preds]
    final_scores["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in final_scores.items()}

# --- Main Fine-tuning Script ---
if __name__ == "__main__":
    tokenizer = None # Define tokenizer in outer scope for compute_metrics
    try:
        # 1. Load Configuration
        print("Loading configuration...")
        config = load_config()
        training_config = config.get('training')
        if not training_config:
            raise ValueError("'training' section not found in abstractive config")

        # --- Parameters ---
        dataset_name = config['dataset_name']
        text_column = config['text_column']
        summary_column = config['summary_column']
        base_model = config['base_model']
        max_input_length = config['max_input_length']
        max_target_length = config['max_target_length']

        # Training specific params
        output_dir = training_config['output_dir']
        num_train_epochs = training_config['num_train_epochs']
        per_device_train_batch_size = training_config['per_device_train_batch_size']
        per_device_eval_batch_size = training_config['per_device_eval_batch_size']
        learning_rate = float(training_config['learning_rate']) # Ensure learning_rate is float
        weight_decay = training_config['weight_decay']
        gradient_accumulation_steps = training_config['gradient_accumulation_steps']
        lr_scheduler_type = training_config['lr_scheduler_type']
        num_warmup_steps = training_config['num_warmup_steps']
        logging_steps = training_config['logging_steps']
        save_steps = training_config['save_steps']
        save_total_limit = training_config['save_total_limit']
        predict_with_generate = training_config['predict_with_generate']
        fp16 = training_config.get('fp16', torch.cuda.is_available()) # Default fp16 if GPU available

        # 2. Load Dataset
        # Fine-tuning should generally use the 'train' split
        print(f"Loading dataset '{dataset_name}' split 'train' for fine-tuning...")
        raw_datasets = datasets.load_dataset(dataset_name, split='train')
        print(f"Loaded {len(raw_datasets)} training examples.")

        # Optional: Create a small validation set if not provided by dataset
        # Example: split train 90/10
        # split_datasets = raw_datasets.train_test_split(test_size=0.1)
        # train_dataset = split_datasets["train"]
        # eval_dataset = split_datasets["test"]
        # print(f"Using {len(train_dataset)} for training, {len(eval_dataset)} for evaluation.")
        # For now, we train on the full train set without evaluation during training
        train_dataset = raw_datasets
        eval_dataset = None # No evaluation during training in this setup

        # 3. Load Tokenizer and Model
        print(f"Loading tokenizer and base model from '{base_model}'...")
        # Define tokenizer in outer scope so compute_metrics can access it
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

        # 4. Preprocess Data
        print("Preprocessing data...")
        # Ensure NLTK punkt is available for compute_metrics sentence tokenization
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)

        preprocess_with_tokenizer = lambda examples: preprocess_function(
            examples, tokenizer, max_input_length, max_target_length, text_column, summary_column
        )
        train_dataset_tokenized = train_dataset.map(preprocess_with_tokenizer, batched=True)
        if eval_dataset:
            eval_dataset_tokenized = eval_dataset.map(preprocess_with_tokenizer, batched=True)
        else:
            eval_dataset_tokenized = None

        # 5. Set up Training Arguments
        print("Setting up training arguments...")
        args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lr_scheduler_type=lr_scheduler_type,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            predict_with_generate=predict_with_generate,
            fp16=fp16, # Enable mixed precision
            # evaluation_strategy="steps" if eval_dataset else "no", # Evaluate periodically if eval set exists
            # eval_steps=save_steps, # Evaluate at the same frequency as saving
            evaluation_strategy="no", # No evaluation during training for this setup
            generation_max_length=max_target_length, # Ensure generation uses correct max length
            generation_num_beams=config.get('num_beams', 4) # Use beam search params from config
        )

        # 6. Initialize Trainer
        print("Initializing Trainer...")
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset_tokenized,
            eval_dataset=eval_dataset_tokenized, # Will be None in this setup
            data_collator=data_collator,
            tokenizer=tokenizer,
            # compute_metrics=compute_metrics if eval_dataset else None # Only compute if eval set exists
            compute_metrics=None # No evaluation during training
        )

        # 7. Start Training
        print("\n--- Starting Fine-Tuning ---")
        train_result = trainer.train()
        print("--- Fine-Tuning Completed ---")

        # 8. Save Final Model & Metrics
        print("Saving final model...")
        trainer.save_model() # Saves the model to output_dir
        trainer.save_state() # Saves optimizer state, etc.

        # Log metrics
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset_tokenized)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        print(f"\nFine-tuned model saved to: {output_dir}")
        print("Training metrics saved.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
        traceback.print_exc()
    except ImportError as e:
        print(f"Import Error: {e}. Make sure you have installed all necessary libraries.")
        print("Consider running: pip install -r requirements.txt --upgrade")
        print("You might also need: pip install torch torchvision torchaudio transformers datasets evaluate rouge_score nltk sentencepiece accelerate")
    except Exception as e:
        print(f"An unexpected error occurred during fine-tuning: {e}")
        traceback.print_exc() 