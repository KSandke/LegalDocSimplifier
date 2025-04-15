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
    - `summarization.yaml`: Settings for summarization models (placeholder).
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
    - Saves the final trained model to `models/classification/multitask_legal_model_standardized/` (consider renaming if only 3 tasks now).

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

Currently implements the TextRank algorithm for extractive summarization.

### Running

The script `src/summarization/extractive_summarizer.py` can be run directly to demonstrate summarization on examples from the standardized SCOTUS dataset:

```bash
python src/summarization/extractive_summarizer.py
```

*(Further development needed to integrate with other components and add configuration via `config/summarization.yaml`)*

## Simplification

*(Placeholder - Functionality to be added)* 