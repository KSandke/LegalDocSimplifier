"""
Legal Document Processing Pipeline (Hugging Face Version)
========================================================

This script provides a complete pipeline for processing legal documents through three stages,
loading all models directly from Hugging Face Hub:

1. Classification - Identifies the document type or clause category
2. Summarization - Creates an abstractive summary of the document
3. Simplification - Rewrites the summary in simplified language

Usage:
------
python huggingface_pipeline.py [arguments]

Arguments:
  --input_file TEXT          Path to a text file containing legal document
  --input_text TEXT          Direct input of legal text (alternative to input_file)
  --output_file TEXT         Path to save the processing results
  --classification_model     HF model ID for classification (default: 'KSandke/legal-classifier')
  --summarization_model      HF model ID for summarization (default: 'KSandke/legal-summarizer')
  --simplification_model     HF model ID for simplification (default: 'KSandke/legal-simplifier')
  --quiet                    Suppress progress bars and transformer warnings

Examples:
---------
# Process a file with default model IDs
python huggingface_pipeline.py --input_file=contracts/agreement.txt --output_file=results/agreement_processed.txt

# Process text directly with custom model IDs
python huggingface_pipeline.py --input_text="This Agreement shall be governed by..." --classification_model="nlpaueb/legal-bert-base-uncased"

# Suppress transformer logs and progress bars
python huggingface_pipeline.py --quiet
"""

import os
import argparse
import sys
import torch
from transformers import pipeline

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components from other modules if needed
from summarization.abstractive_summarizer import post_process_summary


def summarize_text(text, model_id="KSandke/legal-summarizer"):
    """Summarize text using the abstractive summarizer from Hugging Face Hub"""
    try:
        # Create summarization pipeline
        summarizer = pipeline(
            "summarization", 
            model=model_id,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Generate summary
        result = summarizer(
            text,
            max_length=250,
            min_length=50,
            do_sample=False
        )
        
        # Extract and post-process the summary
        summary = result[0]['summary_text']
        summary = post_process_summary(summary)
        
        return summary
    
    except Exception as e:
        print(f"Error in summarization: {str(e)}")
        print("Falling back to base model...")
        # Fallback to a public model if the custom one fails
        return summarize_text(text, "facebook/bart-large-cnn") if model_id != "facebook/bart-large-cnn" else "Error: Could not generate summary."


def simplify_text(text, model_id="KSandke/legal-simplifier"):
    """Simplify text using the simplification model from Hugging Face Hub"""
    try:
        # Create text2text pipeline
        simplifier = pipeline(
            "text2text-generation", 
            model=model_id,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Generate simplified text
        result = simplifier(
            text,
            max_length=512,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True
        )
        
        # Extract simplified text
        simplified_text = result[0]['generated_text']
        
        return simplified_text
    
    except Exception as e:
        print(f"Error in simplification: {str(e)}")
        print("Falling back to base model...")
        # Fallback to a public model if the custom one fails
        return simplify_text(text, "facebook/bart-large-xsum") if model_id != "facebook/bart-large-xsum" else "Error: Could not simplify text."


def classify_text(text, model_id="KSandke/legal-classifier"):
    """Classify the type of legal document or clause"""
    try:
        # Create classification pipeline
        classifier = pipeline(
            "text-classification", 
            model=model_id,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Generate classification
        result = classifier(text)
        
        return {
            "predicted_label_name": result[0]['label'],
            "confidence": result[0]['score']
        }
    
    except Exception as e:
        print(f"Error in classification: {str(e)}")
        # Fallback to a sentiment model as a demo if the specific model fails
        try:
            print("Falling back to sentiment classification...")
            sentiment = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)
            result = sentiment(text)
            return {
                "predicted_label_name": f"FALLBACK_{result[0]['label']}",
                "confidence": result[0]['score']
            }
        except:
            return {"error": str(e)}


def process_document(text, classification_model="KSandke/legal-classifier", 
                    summarization_model="KSandke/legal-summarizer",
                    simplification_model="KSandke/legal-simplifier"):
    """Full pipeline: classify, summarize, and simplify the document using HF models"""
    print("=== Legal Document Processing Pipeline (Hugging Face Version) ===\n")
    print("Processing document... (this may take a minute)")
    
    # Step 1: Classification
    print("Step 1: Classifying document...")
    classification = classify_text(text, classification_model)
    
    # Step 2: Summarization
    print("Step 2: Generating abstractive summary...")
    summary = summarize_text(text, summarization_model)
    
    # Step 3: Simplification
    print("Step 3: Simplifying the summary...")
    simplified = simplify_text(summary, simplification_model)
    
    # Collect results
    results = {
        "classification": classification,
        "summary": summary,
        "simplified_summary": simplified
    }
    
    # Display all results at the end
    print("\n=== Results ===\n")
    
    print("CLASSIFICATION:")
    if "error" in classification:
        print(f"  Error: {classification['error']}")
    else:
        print(f"  Result: {classification['predicted_label_name']}")
        print(f"  Confidence: {classification['confidence']:.4f}")
    
    print("\nSUMMARY:")
    print(f"  Length: {len(summary.split())} words")
    print(f"  {summary}")
    
    print("\nSIMPLIFIED SUMMARY:")
    print(f"  Length: {len(simplified.split())} words")
    print(f"  {simplified}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Legal Document Processing Pipeline (Hugging Face Version)")
    parser.add_argument("--input_file", type=str, help="Path to legal document text file")
    parser.add_argument("--input_text", type=str, help="Legal document text (alternative to input_file)")
    parser.add_argument("--output_file", type=str, help="Path to save the output")
    parser.add_argument("--classification_model", type=str, default="KSandke/legal-classifier", 
                      help="HF model ID for classification")
    parser.add_argument("--summarization_model", type=str, default="KSandke/legal-summarizer", 
                      help="HF model ID for summarization")
    parser.add_argument("--simplification_model", type=str, default="KSandke/legal-simplifier", 
                      help="HF model ID for simplification")
    parser.add_argument("--quiet", action="store_true", help="Suppress logging from transformers")
    args = parser.parse_args()
    
    # Reduce verbosity if quiet flag is set
    if args.quiet:
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Get input text
    if args.input_file:
        print(f"Reading input from {args.input_file}")
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
    results = process_document(
        text, 
        classification_model=args.classification_model,
        summarization_model=args.summarization_model,
        simplification_model=args.simplification_model
    )
    
    # Save results if output file specified
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write("=== LEGAL DOCUMENT ANALYSIS ===\n\n")
            f.write(f"CLASSIFICATION: {results['classification'].get('predicted_label_name', 'Error')}\n")
            f.write(f"Confidence: {results['classification'].get('confidence', 0):.4f}\n\n")
            f.write(f"SUMMARY:\n{results['summary']}\n\n")
            f.write(f"SIMPLIFIED SUMMARY:\n{results['simplified_summary']}")
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main() 