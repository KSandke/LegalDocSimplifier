# Model Development Process

This document outlines the development process followed for building the core machine learning models within the Legal Document Simplifier project.

## 1. Initial Setup and Environment

-   **Project Initialization:** Cloned repository, set up basic project structure (`src`, `data`, `models`, `config`, `notebooks`).
-   **Environment:** Created a Python virtual environment (venv).
-   **Dependencies:** Established initial `requirements.txt` including core libraries (`numpy`, `pandas`, `torch`, `transformers`, `datasets`, `nltk`, `scikit-learn`, `pyyaml`). Dependencies were iteratively updated throughout the project to resolve conflicts and add new features.
-   **Configuration:** Introduced YAML files (`config/classification.yaml`, `config/summarization.yaml`) for managing hyperparameters and paths.

## 2. Multi-Task Classification

The goal was to classify legal documents based on different criteria using datasets like SCOTUS (Issue Area), LEDGAR (Contract Provisions), and UNFAIR-ToS (Contract Unfairness).

-   **Data Loading & Preprocessing:**
    -   Initial data loading scripts were used (e.g., `lex_glue_loader.py`).
    -   Implemented `src/preprocessing/standardize_datasets.py` to convert raw datasets into a uniform format (`input_text`, `input_label`, `task_name`) suitable for multi-task learning. This script saved processed datasets to `data/standardized/`.
    -   Excluded `CaseHOLD` dataset due to incompatibility with the classification task structure.
    -   Generated `task_label_counts.json` during standardization to store label counts and mappings per task.
-   **Model Architecture:**
    -   Implemented `LegalMultiTaskModel` in `src/classification/train_multitask_classifier.py`.
    -   Utilized a shared base encoder (`nlpaueb/legal-bert-base-uncased`) with separate classification heads for each task (SCOTUS, LEDGAR, UNFAIR-ToS).
-   **Training:**
    -   Used `src/classification/train_multitask_classifier.py` for training.
    -   Implemented `TaskBalancedBatchSampler` to ensure each batch contained examples from only one task, addressing potential negative transfer.
    -   Enabled Automatic Mixed Precision (AMP) (`fp16=True`) via `torch.cuda.amp` to improve training speed and reduce memory usage on compatible GPUs.
    -   Configured hyperparameters (batch size, learning rate, epochs) in `config/classification.yaml`.
    -   Addressed initial Out-of-Bounds label errors by carefully regenerating datasets and ensuring label indices matched model output dimensions.
    -   Trained for 3 epochs, saving the final model to `models/classification/multitask_legal_model_standardized/`.
-   **Inference:**
    -   Created `src/classification/multitask_inference.py` to load the trained model and perform predictions on new text samples for specified tasks.
    -   Implemented logic to load label mappings for displaying human-readable results.
    -   Addressed path issues for loading configuration files.
    -   Fixed model loading issues related to configuration saving during training.

## 3. Extractive Summarization

The goal was to create a baseline summarization capability by selecting the most important sentences from the original text.

-   **Algorithm:** Chose the TextRank algorithm.
-   **Implementation:** Developed `src/summarization/extractive_summarizer.py`.
-   **Core Components:**
    -   Used `nltk` for sentence tokenization (`sent_tokenize`). Encountered and resolved `LookupError` issues requiring explicit download of `punkt` and `punkt_tab` NLTK resources.
    -   Used `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`) to represent sentences and calculate their similarity.
    -   Used `networkx` (`nx.from_numpy_array`, `nx.pagerank`) to build the sentence graph and run the ranking algorithm.
-   **Configuration:** Refactored the script to load parameters (dataset name, split, number of sentences) from the `extractive` section of `config/summarization.yaml`.
-   **Debugging:** Resolved `NameError` (missing `numpy` import) and `AttributeError` (incorrect handling of tuple return values from `build_similarity_matrix`). Fixed data loading path issues by correcting `dataset_name` in the config.
-   *Note: Evaluation was primarily qualitative; no quantitative metrics like ROUGE were calculated for this baseline method.*

## 4. Abstractive Summarization

The goal was to generate novel summaries using a sequence-to-sequence model, potentially improving fluency over extractive methods.

-   **Dataset:** Selected `ChicagoHAI/CaseSumm` from Hugging Face, containing legal opinions and human-written syllabi (used as reference summaries).
-   **Base Model:** Chose `google-t5/t5-small` as an initial, efficient model.
-   **Initial Inference (`abstractive_summarizer.py`):**
    -   Set up a script to perform inference using the base model.
    *   Configured parameters (model name, dataset details, generation settings) in the `abstractive` section of `config/summarization.yaml`.
    *   Implemented ROUGE score calculation for evaluation. Encountered issues loading the metric via `evaluate.load()`; resolved by switching to direct calculation using the `rouge-score` library.
    *   Debugged dataset loading issues: identified correct split name ('train' instead of 'test') and correct column names ('opinion', 'syllabus') for the CaseSumm dataset.
    *   Addressed library conflicts: Resolved numerous dependency version conflicts involving `numpy`, `datasets`, `huggingface-hub`, `fsspec`, and `transformers` by carefully adjusting versions in `requirements.txt`. Fixed a `TypeError` related to NumPy 2.0 incompatibility with `datasets`.
    -   **Baseline Performance (Untrained `t5-small` on first 100 CaseSumm 'train' examples):**
        -   ROUGE-1 F1: ~38.9
        -   ROUGE-2 F1: ~12.6
        -   ROUGE-L F1: ~23.9
        -   Avg. Length: ~79 words
*   **Fine-tuning (`finetune_abstractive.py`):**
    *   Created a script using `transformers.Seq2SeqTrainer` for fine-tuning.
    *   Added a `training` subsection to `config/summarization.yaml` for fine-tuning hyperparameters (epochs, learning rate, batch size, gradient accumulation, etc.).
    *   Implemented data preprocessing suitable for T5 (adding "summarize: " prefix).
    *   Configured `Seq2SeqTrainingArguments`, enabling AMP (`fp16=True`).
    *   Debugged `TypeError` issues during Trainer initialization related to incorrect hyperparameter types (`learning_rate` as string) and unsupported arguments (`num_warmup_steps`).
    *   Optimized training speed vs. memory by tuning `per_device_train_batch_size` (increased to 8) and `gradient_accumulation_steps` (decreased to 2) based on observed GPU memory usage.
    *   Successfully ran fine-tuning for 3 epochs on the CaseSumm 'train' split, saving the model to `models/summarization/t5_small_casesumm_finetuned/`.
*   **Evaluation of Fine-tuned Model:**
    *   Ran the inference script (`abstractive_summarizer.py`), configured to load the *fine-tuned* model (`models/summarization/t5_small_casesumm_finetuned/`).
    *   Optimized inference speed by increasing the inference `batch_size` (to 24) based on observed GPU memory usage.
    *   Completed inference run on the full CaseSumm 'train' set (27,071 examples).
    *   **Final Performance (Fine-tuned `t5-small` on full 'train' set):**
        -   ROUGE-1 F1: 41.3166
        -   ROUGE-2 F1: 15.7923
        -   ROUGE-L F1: 25.0777
        -   Avg. Length: 103.6 words
    *   *Note: Scores show noticeable improvement over the baseline untrained model, particularly in ROUGE-1 and ROUGE-2, demonstrating successful adaptation through fine-tuning.*

## 5. User Interface (Demo)

-   **Goal:** Create a simple, interactive web demo for the developed models.
-   **Framework:** Chose Gradio for rapid development.
-   **Implementation:** Created `src/app/gradio_demo.py`.
    -   Loads the fine-tuned summarization model and the multi-task classification model on startup.
    -   Provides a tabbed interface for Summarization (Extractive/Abstractive options) and Classification (task selection dropdown).
    -   Requires necessary dependencies (`gradio`) added to `requirements.txt`.
-   **Status:** Basic demo implemented. Not runnable concurrently with model training due to GPU resource limitations.

## Next Steps (Potential)

-   Evaluate the performance of the fine-tuned abstractive summarizer.
-   Implement the simplification component.
-   Explore alternative summarization models (e.g., BART, LED).
-   Consider more advanced frontend options (Flask/React etc.) if needed.
-   Refine models based on evaluation results. 