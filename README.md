# Legal Document Simplifier

A project to classify, summarize, and simplify legal documents using NLP.

## Structure

- `data/`: Directory for datasets.
    - `raw/`: Original, unprocessed data.
    - `processed/`: Cleaned and preprocessed data.
    - `dictionary/`: Legal term dictionary/knowledge base.
    - `reference_summaries/`: Human-written summaries for evaluation.
- `src/`: Source code for the project.
    - `__init__.py`
    - `preprocessing/`: Data loading and cleaning modules. (`__init__.py`)
    - `classification/`: Document classification models and logic. (`__init__.py`)
    - `simplification/`: Term identification (NER) and replacement modules. (`__init__.py`)
    - `summarization/`: Document summarization models and logic. (`__init__.py`)
    - `evaluation/`: Model evaluation scripts and utilities. (`__init__.py`)
    - `utils/`: Shared utility functions. (`__init__.py`)
- `models/`: Stored trained model files.
    - `classification/`: Classification models.
    - `simplification_ner/`: NER models for simplification.
    - `summarization/`: Summarization models.
- `notebooks/`: Jupyter notebooks for exploration, experimentation, and visualization.
- `tests/`: Unit and integration tests. (`__init__.py`)
- `config/`: Configuration files (e.g., `config.yaml`).
- `requirements.txt`: Project dependencies.

## Multi-Task Classification Training

The primary classification model is trained using a multi-task approach on several datasets from the LexGLUE benchmark simultaneously.

### Datasets Used

- SCOTUS (`scotus`)
- LEDGAR (`ledgar`)
- UNFAIR-ToS (`unfair_tos`)

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

### Optimal Settings (for NVIDIA RTX 3080 10GB)

- Batch Size: 16 (used in `TaskBalancedBatchSampler`)
- AMP Enabled: Yes
- Optimizer: AdamW (lr=2e-5)
- Epochs: 3 (default)

### Running

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# 1. Run initial loader (if needed)
# python data/raw/lex_glue_loader.py

# 2. Run casehold preprocessing (No longer needed for this training script)
# python src/preprocessing/preprocess_casehold.py

# 3. Run standardization for relevant datasets
python src/preprocessing/standardize_datasets.py

# 4. Run multi-task training
python src/classification/train_multitask_classifier.py
``` 