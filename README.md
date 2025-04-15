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

## Using the Multi-Task Classifier (Inference)

Once the multi-task model has been trained and saved (to the location specified in `config.yaml`, typically `models/classification/multitask_legal_model_standardized/`), you can use it to classify new text snippets for the supported tasks.

### Inference Script

The script `src/classification/multitask_inference.py` handles loading the trained model, tokenizer, and configuration.

### Supported Tasks

Based on the training configuration (excluding CaseHOLD), the model supports inference for:

- `scotus`
- `ledgar`
- `unfair_tos`

### Running Inference

1.  **Direct Execution (Examples):**
    You can run the script directly to see sample predictions on predefined text snippets:
    ```bash
    # Activate virtual environment
    .\venv\Scripts\Activate.ps1
    
    python src/classification/multitask_inference.py
    ```

2.  **Importing the `predict` Function:**
    To use the classifier in other parts of your application, import and use the `predict` function:
    ```python
    from src.classification.multitask_inference import predict
    
    my_text = "Some legal text snippet..."
    task = "scotus" # Or "ledgar", "unfair_tos"
    
    prediction_result = predict(my_text, task)
    
    if "error" in prediction_result:
        print(f"Error: {prediction_result['error']}")
    else:
        print(f"Task: {prediction_result['task']}")
        print(f"Predicted Label ID: {prediction_result['predicted_label_id']}")
        print(f"Predicted Label Name: {prediction_result['predicted_label_name']}")
        print(f"Confidence: {prediction_result['confidence']:.4f}")
    ```
    *Note: Ensure the model loading within `multitask_inference.py` happens only once if you import it repeatedly (e.g., in a web server context).* 