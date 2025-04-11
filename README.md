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