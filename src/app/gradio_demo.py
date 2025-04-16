import gradio as gr
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import yaml
import nltk
import sys

# --- Add src directory to path for imports ---
# This allows importing modules from src (like extractive_summarizer)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..')) # Go up two levels from src/app to project root
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# --- Try importing our backend functions ---
try:
    from src.summarization.extractive_summarizer import textrank_summarize
except ImportError as e:
    print(f"Could not import extractive_summarizer: {e}. Ensure it's in the Python path.")
    textrank_summarize = lambda text, num_sentences=5: "Extractive summarizer not available."

# --- Configuration Loading ---
def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# --- Load Models (Do this once on startup) ---
print("Loading models and tokenizers...")

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -- Summarization Model --
try:
    sum_config = load_config('config/summarization.yaml')['abstractive']
    sum_model_path = sum_config.get('base_model', 'google-t5/t5-small') # Default to base if fine-tuned not found
    if not os.path.exists(sum_model_path):
        print(f"Warning: Summarization model path not found at '{sum_model_path}'. Trying default base model.")
        sum_model_path = 'google-t5/t5-small'
    sum_tokenizer = AutoTokenizer.from_pretrained(sum_model_path)
    sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_model_path).to(device)
    sum_model.eval() # Set to eval mode
    print(f"Loaded Abstractive Summarization model from: {sum_model_path}")
except Exception as e:
    print(f"Error loading summarization model: {e}")
    sum_tokenizer = None
    sum_model = None

# -- Classification Model --
try:
    cls_config = load_config('config/classification.yaml')
    cls_model_path = cls_config['model_output_dir']
    # Adjust path relative to project root if needed
    cls_model_path = os.path.join(src_dir, cls_model_path) 

    # Load task_label_counts (needed for model init)
    label_counts_path = os.path.join(src_dir, 'data', 'standardized', 'task_label_counts.json')
    if not os.path.exists(label_counts_path):
        raise FileNotFoundError("task_label_counts.json not found")
    import json
    with open(label_counts_path, 'r') as f:
        task_label_counts = json.load(f)
    
    cls_tokenizer = AutoTokenizer.from_pretrained(cls_model_path)
    # We need the LegalMultiTaskModel class definition
    # Easiest is to import it, assuming it's defined cleanly
    try:
        from src.classification.train_multitask_classifier import LegalMultiTaskModel # Adjust import if needed
        cls_model = LegalMultiTaskModel(cls_config['base_model_name'], task_label_counts).to(device)
        # Load the saved state dict
        state_dict_path = os.path.join(cls_model_path, 'pytorch_model.bin')
        if not os.path.exists(state_dict_path):
             state_dict_path = os.path.join(cls_model_path, 'model_state_dict.pt') # Try alternate name
        if not os.path.exists(state_dict_path):
            raise FileNotFoundError(f"Model state dict not found in {cls_model_path}")
        cls_model.load_state_dict(torch.load(state_dict_path, map_location=device))
        cls_model.eval() # Set to eval mode
        print(f"Loaded Classification model from: {cls_model_path}")

        # Load label maps for displaying results (similar to multitask_inference.py)
        label_maps = {}
        for task, count_info in task_label_counts.items():
            label_maps[task] = count_info.get('label_map', {str(i): f'{task}_label_{i}' for i in range(count_info['count'])})
            # Special handling for scotus based on previous inference script? Maybe load from a file?
            if task == 'scotus':
                 label_maps['scotus'] = {str(i): f'SCOTUS_{i}' for i in range(count_info['count'])} # Placeholder
        print("Loaded label maps.")

    except ImportError as e:
        print(f"Could not import LegalMultiTaskModel: {e}. Classification may fail.")
        cls_model = None
    except FileNotFoundError as e:
         print(f"Could not load classification model components: {e}")
         cls_model = None

except Exception as e:
    print(f"Error loading classification model: {e}")
    cls_model = None
    cls_tokenizer = None
    label_maps = {}

print("Model loading complete.")

# --- Backend Functions for Gradio ---

def classify_text(text, task_name):
    if not cls_model or not cls_tokenizer:
        return "Classification model not loaded."
    if task_name not in label_maps:
        return f"Unknown task: {task_name}"
    
    inputs = cls_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = cls_model(**inputs)
        # Get logits for the specific task
        task_logits = outputs[task_name] 
        probabilities = torch.softmax(task_logits, dim=-1)
        top_prob, top_idx = torch.max(probabilities, dim=-1)
        
    pred_label_idx = top_idx.item()
    pred_label_name = label_maps[task_name].get(str(pred_label_idx), f"Unknown Label ({pred_label_idx})")
    confidence = top_prob.item()
    
    return f"Predicted Task ({task_name}): {pred_label_name} (Confidence: {confidence:.4f})"

def summarize_text(text, summary_type):
    if summary_type == "Extractive (TextRank)":
        if not textrank_summarize: # Check if function loaded
             return "Extractive summarizer function not loaded."
        try:
            # Assuming textrank_summarize takes text and num_sentences
            return textrank_summarize(text, num_sentences=5) 
        except Exception as e:
            return f"Error during extractive summarization: {e}"
    elif summary_type == "Abstractive (Fine-tuned T5)":
        if not sum_model or not sum_tokenizer:
            return "Abstractive summarization model not loaded."
        try:
            prefix = "summarize: "
            inputs = sum_tokenizer(prefix + text, return_tensors="pt", max_length=1024, truncation=True).to(device)
            with torch.no_grad():
                summary_ids = sum_model.generate(
                    inputs.input_ids,
                    num_beams=4, # Use settings consistent with prior runs
                    max_length=150,
                    min_length=40,
                    length_penalty=2.0,
                    no_repeat_ngram_size=3
                )
            summary = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            return f"Error during abstractive summarization: {e}"
    else:
        return "Invalid summary type selected."

# --- Build Gradio Interface --- 
print("Building Gradio interface...")

# Define tasks for classification dropdown
classification_tasks = list(label_maps.keys()) if label_maps else ["scotus", "ledgar", "unfair_tos"] # Fallback

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Legal Document Simplifier Demo")
    
    with gr.Tabs():
        with gr.TabItem("Summarize"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_input_sum = gr.Textbox(label="Enter Legal Text Here", lines=15)
                    summary_type = gr.Radio([
                        "Extractive (TextRank)", 
                        "Abstractive (Fine-tuned T5)"
                        ], label="Summarization Type", value="Abstractive (Fine-tuned T5)")
                    submit_btn_sum = gr.Button("Summarize")
                with gr.Column(scale=1):
                    text_output_sum = gr.Textbox(label="Generated Summary", lines=17, interactive=False)
        
        with gr.TabItem("Classify"):
            with gr.Row():
                 with gr.Column(scale=1):
                    text_input_cls = gr.Textbox(label="Enter Legal Text Here", lines=15)
                    task_name = gr.Dropdown(classification_tasks, label="Select Task", value=classification_tasks[0] if classification_tasks else None)
                    submit_btn_cls = gr.Button("Classify")
                 with gr.Column(scale=1):
                    text_output_cls = gr.Textbox(label="Classification Result", lines=2, interactive=False)

        # with gr.TabItem("Simplify"):
        #     gr.Markdown("Simplification feature coming soon!")

    # --- Event Handlers ---
    submit_btn_sum.click(summarize_text, inputs=[text_input_sum, summary_type], outputs=text_output_sum)
    submit_btn_cls.click(classify_text, inputs=[text_input_cls, task_name], outputs=text_output_cls)

print("Launching Gradio demo...")
demo.launch() # Share=True makes it accessible via public link (use with caution) 