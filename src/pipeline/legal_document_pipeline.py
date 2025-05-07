import os
import argparse
import sys
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components from other modules
from classification.multitask_inference import predict as classify_text
from summarization.abstractive_summarizer import post_process_summary


def load_config(config_path):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def summarize_text(text, config=None):
    """Summarize text using the abstractive summarizer"""
    if config is None:
        config = load_config('config/summarization.yaml')['abstractive']
    
    # Load model and tokenizer
    model_path = config.get('path_to_finetuned_model', config['base_model'])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Prepare inputs
    is_t5_model = 't5' in model_path.lower()
    input_text = "summarize: " + text if is_t5_model else text
    inputs = tokenizer(input_text, max_length=config['max_input_length'], 
                       truncation=True, padding="max_length", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate summary
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=config['max_target_length'],
            min_length=config.get('min_length', 50),
            num_beams=config.get('num_beams', 5),
            length_penalty=config.get('length_penalty', 2.0),
            no_repeat_ngram_size=config.get('no_repeat_ngram_size', 3),
            early_stopping=config.get('early_stopping', True),
            decoder_start_token_id=model.config.decoder_start_token_id
        )
    
    # Decode output
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    summary = post_process_summary(summary)
    
    return summary


def simplify_text(text, config=None):
    """Simplify text using the trained simplification model"""
    if config is None:
        config = load_config('config/simplification.yaml')
    
    # Determine model path
    model_name = config['model']['simplification_model_name']
    output_dir = config['paths']['output_models'].format(model_name=model_name)
    model_path = os.path.join(output_dir, "final_model")
    
    # Fall back to base model if fine-tuned model doesn't exist
    if not os.path.exists(model_path):
        print(f"Warning: Fine-tuned model not found at {model_path}")
        print(f"Using base model {config['model']['base_model']} instead")
        model_path = config['model']['base_model']
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Get generation parameters
    gen_params = config['simplification_params']['generation_params']
    
    # Prepare inputs
    inputs = tokenizer(text, max_length=config['training']['max_input_length'], 
                      truncation=True, padding="max_length", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate simplified text
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=config['training']['max_target_length'],
            num_beams=gen_params.get('num_beams', 4),
            length_penalty=gen_params.get('length_penalty', 1.0),
            early_stopping=gen_params.get('early_stopping', True),
            do_sample=gen_params.get('do_sample', False),
            temperature=gen_params.get('temperature', 1.0),
        )
    
    # Decode output
    simplified_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return simplified_text


def process_document(text, classification_task="ledgar"):
    """Full pipeline: classify, summarize, and simplify the document"""
    print("=== Legal Document Processing Pipeline ===\n")
    
    # Step 1: Classification
    print("Step 1: Classifying document...")
    classification = classify_text(text, classification_task)
    if "error" in classification:
        print(f"Error in classification: {classification['error']}")
    else:
        print(f"Classification result: {classification['predicted_label_name']}")
        print(f"Confidence: {classification['confidence']:.4f}")
    
    # Step 2: Summarization
    print("\nStep 2: Generating abstractive summary...")
    summary = summarize_text(text)
    print(f"Summary length: {len(summary.split())} words")
    print(f"Summary:\n{summary}\n")
    
    # Step 3: Simplification
    print("Step 3: Simplifying the summary...")
    simplified = simplify_text(summary)
    print(f"Simplified length: {len(simplified.split())} words")
    print(f"Simplified summary:\n{simplified}\n")
    
    # Return all results
    return {
        "classification": classification,
        "summary": summary,
        "simplified_summary": simplified
    }


def main():
    parser = argparse.ArgumentParser(description="Legal Document Processing Pipeline")
    parser.add_argument("--input_file", type=str, help="Path to legal document text file")
    parser.add_argument("--input_text", type=str, help="Legal document text (alternative to input_file)")
    parser.add_argument("--output_file", type=str, help="Path to save the output")
    parser.add_argument("--classification_task", type=str, default="ledgar", 
                      choices=["scotus", "ledgar", "unfair_tos"], 
                      help="Classification task to use")
    args = parser.parse_args()
    
    # Get input text
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    elif args.input_text:
        text = args.input_text
    else:
        # Demo text if no input provided
        text = """
        This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, 
        without regard to its conflict of laws principles. The parties hereby submit to the exclusive jurisdiction
        of the courts located in New Castle County, Delaware for resolution of any disputes arising out of or relating 
        to this Agreement. In the event that any provision of this Agreement is found to be invalid or unenforceable, 
        the remaining provisions shall remain in full force and effect. This Agreement constitutes the entire 
        understanding between the parties with respect to the subject matter hereof and supersedes all prior 
        agreements, whether written or oral, between the parties.
        """
        print("No input provided. Using demo text.")
    
    # Process the document
    results = process_document(text, args.classification_task)
    
    # Save results if output file specified
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(f"Classification: {results['classification']['predicted_label_name']}\n\n")
            f.write(f"Summary:\n{results['summary']}\n\n")
            f.write(f"Simplified Summary:\n{results['simplified_summary']}")
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main() 