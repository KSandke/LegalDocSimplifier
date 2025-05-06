# Legal Document Simplifier

A pipeline for processing legal documents through classification, summarization, and simplification.

## Project Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/KSandke/LegalDocSimplifier.git
    cd LegalDocSimplifier
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows PowerShell
    .\venv\Scripts\Activate.ps1 
    # macOS/Linux
    # source venv/bin/activate 
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download NLTK data (if needed):**
    The first time you run summarization or sentence tokenization, NLTK might need to download data (`punkt`). The scripts should handle this automatically, but you can also do it manually:
    ```bash
    python -m nltk.downloader punkt
    ```

## Using the Pipeline

This project provides two different pipeline implementations:

### Option 1: Using Models from Hugging Face Hub (Recommended)

Our fine-tuned models are hosted on Hugging Face Hub for easy access without dealing with large file downloads:

1. **Use the Hugging Face pipeline:**
   ```bash
   python src/pipeline/huggingface_pipeline.py --input_file=YOUR_DOCUMENT.txt
   ```

2. **Command-line options:**
   ```
   --input_file TEXT          Path to a text file containing legal document
   --input_text TEXT          Direct input of legal text (alternative to input_file)
   --output_file TEXT         Path to save the processing results
   --classification_model     HF model ID for classification (default: 'KSandke/legal-classifier')
   --summarization_model      HF model ID for summarization (default: 'KSandke/legal-summarizer')
   --simplification_model     HF model ID for simplification (default: 'KSandke/legal-simplifier')
   --quiet                    Suppress progress bars and transformer warnings
   ```

3. **Examples:**
   ```bash
   # Process a file with default model IDs
   python src/pipeline/huggingface_pipeline.py --input_file=contracts/agreement.txt --output_file=results/agreement_processed.txt

   # Process text directly with custom model IDs
   python src/pipeline/huggingface_pipeline.py --input_text="This Agreement shall be governed by..." --classification_model="nlpaueb/legal-bert-base-uncased"

   # Suppress transformer logs and progress bars
   python src/pipeline/huggingface_pipeline.py --quiet
   ```

4. **Direct API usage:**
   ```python
   from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

   # Load summarizer model
   tokenizer = AutoTokenizer.from_pretrained("KSandke/legal-summarizer")
   model = AutoModelForSeq2SeqLM.from_pretrained("KSandke/legal-summarizer")
   ```

### Option 2: Fine-tune Your Own Models

For customization or offline use, you can fine-tune your own models:

1. Follow the training instructions in the [Model Fine-tuning Guide](docs/model_fine_tuning.md)
2. Save your models to the appropriate directories
3. Run the pipeline with your custom models:
   ```bash
   python src/pipeline/legal_document_pipeline.py --input_file=YOUR_DOCUMENT.txt
   ```

## Project Structure

- `config/`: YAML configuration files for different components.
    - `classification.yaml`: Settings for classification models.
    - `summarization.yaml`: Settings for summarization models (extractive & abstractive).
    - `simplification.yaml`: Settings for simplification models.
- `data/`: Directory for datasets.
    - `raw/`: Original, unprocessed data (put downloaded datasets here).
    - `processed/`: Datasets after initial processing (e.g., by `lex_glue_loader.py`).
    - `standardized/`: Datasets processed into a uniform format for training.
- `models/`: Stored trained model files (ignored by git).
    - `classification/`: Classification models.
    - `simplification/`: Models for simplification.
    - `summarization/`: Summarization models.
- `src/`: Source code for the project.
    - `classification/`: Multi-task document classification.
        - `train_multitask_classifier.py`: Trains the model.
        - `multitask_inference.py`: Performs inference using the trained model.
    - `preprocessing/`: Data loading, cleaning, and standardization scripts.
        - `standardize_datasets.py`: Standardizes datasets for multi-task training.
    - `summarization/`: Document summarization logic.
        - `extractive_summarizer.py`: Implements TextRank extractive summarization. (Not used in the pipeline)
        - `abstractive_summarizer.py`: Runs inference using a pre-trained/fine-tuned abstractive model.
        - `finetune_abstractive.py`: Fine-tunes an abstractive summarization model.
    - `simplification/`: Text simplification logic.
        - `train_lexsimple.py`: Fine-tuning script for text simplification models.
        - `data_loader.py`: Handles loading and processing simplification datasets.
        - `inspect_lexsimple.py`: Utility for examining the simplification dataset.
    - `pipeline/`: Legal document processing pipeline.
        - `legal_document_pipeline.py`: Pipeline for using locally fine-tuned models.
        - `huggingface_pipeline.py`: Pipeline using Hugging Face Hub models that I trained myself.
    - `utils/`: Shared utility functions.
        - `inspect_standardized_labels.py`: Utility to check label ranges.
- `requirements.txt`: Project dependencies.
- `.gitignore` configured to ignore contents by default.
- `model_fine_tuning.md`: Guide for fine tuning your own models.

## Multi-Task Classification

The primary classification model is trained using a multi-task approach on several datasets simultaneously.

### Datasets Used

Uses datasets standardized by `src/preprocessing/standardize_datasets.py`:

- SCOTUS (`scotus`)
- LEDGAR (`ledgar`)
- UNFAIR-ToS (`unfair_tos`)

*(Note: CaseHOLD was excluded as its task structure is incompatible with this classification setup.)*

### Training

The multi-task model is trained using `src/classification/train_multitask_classifier.py`.

### Using the Classifier (Inference)

Run the inference script `src/classification/multitask_inference.py` to load the trained model and classify new text for the supported tasks (`scotus`, `ledgar`, `unfair_tos`).

## Extractive Summarization

Implements the TextRank algorithm for extractive summarization. Configuration is managed via the `extractive` section in `config/summarization.yaml`. This method of summarization is not used in the pipeline. 

### Running

The script `src/summarization/extractive_summarizer.py` can be run directly to demonstrate summarization on examples from the configured dataset (default: standardized SCOTUS dataset):

```bash
python src/summarization/extractive_summarizer.py
```

## Abstractive Summarization

Provides functionality to fine-tune and run inference with sequence-to-sequence models for abstractive summarization. Configuration is managed via the `abstractive` section in `config/summarization.yaml`.

### Fine-tuning

The script `src/summarization/finetune_abstractive.py` fine-tunes a specified base model (e.g., `google-t5/t5-small`) on a summarization dataset (e.g., `ChicagoHAI/CaseSumm`).

1.  **Configure:** Adjust parameters in `config/summarization.yaml` under the `abstractive` -> `training` section (e.g., `output_dir`, `num_train_epochs`, `per_device_train_batch_size`, `learning_rate`, etc.).
2.  **Run Training:** Execute the script. This will download the base model and dataset if not cached, preprocess the data, and run the fine-tuning loop using the `transformers` `Seq2SeqTrainer`, saving checkpoints and the final model to the specified `output_dir`.
    ```bash
    # Ensure dependencies are installed (incl. accelerate)
    pip install -r requirements.txt
    # Run the fine-tuning process
    python src/summarization/finetune_abstractive.py
    ```
    *Note: Fine-tuning can be computationally intensive and time-consuming, especially on large datasets. A GPU is highly recommended.* 

### Inference

The script `src/summarization/abstractive_summarizer.py` loads a pre-trained or fine-tuned model and generates summaries for a given dataset split.

1.  **Configure:** Set the `base_model` in `config/summarization.yaml` (under `abstractive`). To use a fine-tuned model, you can either set `base_model` to the path where the fine-tuned model was saved (e.g., `models/summarization/t5_small_casesumm_finetuned`) or potentially add a `path_to_finetuned_model` field in the config and modify the script to prioritize loading from there.
2.  **Run Inference:** Execute the script. It will load the specified model and dataset, generate summaries, calculate ROUGE scores against reference summaries (if available), and print examples.
    ```bash
    # Run inference using the configured model and dataset
    python src/summarization/abstractive_summarizer.py
    ```

## Text Simplification

The text simplification component transforms complex legal language into more accessible text, preserving core meaning while reducing complexity. This makes legal documents more understandable to non-specialists.

### Approach

Our simplification model is built on transformer-based sequence-to-sequence architecture, fine-tuned specifically for legal language simplification:

1. **Models**: The system can use various pre-trained models as a base:
   - `nsi319/legal-pegasus`: Legal domain-adapted PEGASUS model (default, best performance)
   - `nsi319/legal-led-base-16384`: Legal domain model for longer texts
   - `facebook/bart-base`: General text simplification model
   - `t5-small`: Lightweight alternative for constrained environments

2. **Training Datasets**: The simplification models are trained on parallel datasets of complex-simple text pairs:
   - `turk`: English simplifications created by Amazon Mechanical Turk workers
   - `wikilarge`: English Wikipedia simplifications (simple vs. standard)
   - `multisim`: Multilingual simplification dataset
   - `lexsimple`: Custom legal document simplification dataset

3. **Simplification Parameters**: The model behavior can be configured via `simplification_params` in `config/simplification.yaml`:
   - `level`: Controls simplification aggressiveness (low, medium, high)
   - `preserve_meaning_strictness`: How strictly to maintain original meaning
   - Generation parameters like beam search settings, length penalty, and sampling options

### Using the Simplifier

The simplification can be used in three ways:

1. **Through the Pipeline**: The most common approach is to use it as part of the full pipeline, after classification and summarization:
   ```bash
   python src/pipeline/huggingface_pipeline.py --input_file=your_document.txt
   ```

2. **Fine-tuning Your Own Model**: For domain-specific simplification, you can fine-tune on your own parallel data:
   ```bash
   # Prepare your data in simple/complex pairs
   # Edit config/simplification.yaml with your settings
   python src/simplification/train_lexsimple.py
   ```

3. **Direct API Usage**: For integration into other applications:
   ```python
   from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
   
   # Load simplifier model
   tokenizer = AutoTokenizer.from_pretrained("KSandke/legal-simplifier")
   model = AutoModelForSeq2SeqLM.from_pretrained("KSandke/legal-simplifier")
   
   # Simplify text
   inputs = tokenizer("This Agreement shall be governed by and construed in accordance with the laws...", 
                     return_tensors="pt", truncation=True, max_length=512)
   outputs = model.generate(**inputs, max_length=150, min_length=40, num_beams=4)
   simplified_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
   ```

### Performance Evaluation

The simplification model is evaluated using multiple metrics:
- **ROUGE scores**: Measures lexical overlap with human-created simplifications
- **BLEU scores**: Measures phrase-level accuracy
- **Readability metrics**: Flesch-Kincaid and other readability scores to verify reduction in complexity
- **Meaning preservation**: Measured by semantic similarity between original and simplified text

## Model Types

This project uses three types of models:

1. **Classification Model** - Identifies legal document types using Legal-BERT
2. **Summarization Model** - Creates concise summaries using T5 or PEGASUS
3. **Simplification Model** - Rewrites text in simpler language using legal domain-adapted models

All three models are available on Hugging Face Hub:
- [KSandke/legal-classifier](https://huggingface.co/KSandke/legal-classifier)
- [KSandke/legal-summarizer](https://huggingface.co/KSandke/legal-summarizer)
- [KSandke/legal-simplifier](https://huggingface.co/KSandke/legal-simplifier)

## Output Format

The pipeline produces results in the following format:

```
=== Results ===

CLASSIFICATION:
  Result: LABEL_0
  Confidence: 0.5955

SUMMARY:
  Length: 76 words
  This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware...

SIMPLIFIED SUMMARY:
  Length: 29 words
  The parties submit to the exclusive jurisdiction of the courts located in New Castle County...
```
