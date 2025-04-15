# Legal Document Simplifier

A project to classify, summarize, and simplify legal documents using NLP.

## Project Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
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

## Project Structure

- `config/`: YAML configuration files for different components.
    - `classification.yaml`: Settings for classification models.
    - `summarization.yaml`: Settings for summarization models (extractive & abstractive).
    - `simplification.yaml`: Settings for simplification (placeholder).
- `data/`: Directory for datasets.
    - `raw/`: Original, unprocessed data (put downloaded datasets here).
    - `processed/`: Datasets after initial processing (e.g., by `lex_glue_loader.py`).
    - `standardized/`: Datasets processed into a uniform format for training.
    - `dictionary/`: Legal term dictionary/knowledge base (optional).
    - `reference_summaries/`: Human-written summaries for evaluation (optional).
    - `.gitignore` configured to ignore contents by default.
- `models/`: Stored trained model files (ignored by git).
    - `classification/`: Classification models.
    - `simplification_ner/`: NER models for simplification.
    - `summarization/`: Summarization models.
- `notebooks/`: Jupyter notebooks for exploration and experimentation.
- `src/`: Source code for the project.
    - `classification/`: Multi-task document classification.
        - `train_multitask_classifier.py`: Trains the model.
        - `multitask_inference.py`: Performs inference using the trained model.
    - `preprocessing/`: Data loading, cleaning, and standardization scripts.
        - `standardize_datasets.py`: Standardizes datasets for multi-task training.
    - `summarization/`: Document summarization logic.
        - `extractive_summarizer.py`: Implements TextRank extractive summarization.
        - `abstractive_summarizer.py`: Runs inference using a pre-trained/fine-tuned abstractive model.
        - `finetune_abstractive.py`: Fine-tunes an abstractive summarization model.
    - `simplification/`: Text simplification logic (placeholder).
    - `evaluation/`: Model evaluation scripts (placeholder).
    - `utils/`: Shared utility functions.
        - `inspect_standardized_labels.py`: Utility to check label ranges.
- `tests/`: Unit and integration tests (placeholder).
- `requirements.txt`: Project dependencies.

## Multi-Task Classification

The primary classification model is trained using a multi-task approach on several datasets simultaneously.

### Datasets Used

Uses datasets standardized by `src/preprocessing/standardize_datasets.py`:

- SCOTUS (`scotus`)
- LEDGAR (`ledgar`)
- UNFAIR-ToS (`unfair_tos`)

*(Note: CaseHOLD was excluded as its task structure is incompatible with this classification setup.)*

### Workflow

1.  **Initial Loading:** Raw datasets are loaded using `data/raw/lex_glue_loader.py` and saved to `data/processed/`.
2.  **Standardization:** All relevant datasets (`scotus`, `ledgar`, `unfair_tos`) are converted to a uniform format (`input_text`, `input_label`, `task_name`) using `src/preprocessing/standardize_datasets.py`. This script saves the final training-ready datasets to `data/standardized/` and also creates `data/standardized/task_label_counts.json`.
3.  **Training:** The multi-task model is trained using `src/classification/train_multitask_classifier.py`. This script:
    - Loads the standardized datasets (`scotus`, `ledgar`, `unfair_tos`) from `data/standardized/`.
    - Loads the label counts from `data/standardized/task_label_counts.json`.
    - Uses the `LegalMultiTaskModel` architecture (shared Legal-BERT encoder, separate heads per task).
    - Employs the `TaskBalancedBatchSampler` to ensure each training batch contains examples from only one task.
    - Utilizes Automatic Mixed Precision (AMP) for faster training on compatible GPUs.
    - Saves the final trained model to `models/classification/multitask_legal_model_standardized/`.

### Training Configuration (`config/classification.yaml`)

- Batch Size: 16 (used in `TaskBalancedBatchSampler`)
- AMP Enabled: Yes
- Optimizer: AdamW (lr=2e-5)
- Epochs: 3 (default)

### Running

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# 1. Ensure raw/processed data exists in data/processed/
# Example: python data/raw/lex_glue_loader.py

# 2. Run standardization for classification datasets
python src/preprocessing/standardize_datasets.py

# 3. Run multi-task training (uses settings from config/classification.yaml)
python src/classification/train_multitask_classifier.py
```

### Using the Classifier (Inference)

Run the inference script `src/classification/multitask_inference.py` to load the trained model and classify new text for the supported tasks (`scotus`, `ledgar`, `unfair_tos`).

```bash
python src/classification/multitask_inference.py
```

See the script's docstring for details on importing the `predict` function.

## Extractive Summarization

Implements the TextRank algorithm for extractive summarization. Configuration is managed via the `extractive` section in `config/summarization.yaml`.

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

## Simplification

*(Placeholder - Functionality to be added)* 