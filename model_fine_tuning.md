# Model Fine-tuning Guide

This document provides detailed instructions for fine-tuning the three models required for the Legal Document Simplifier pipeline:

1. **Document Classifier**
2. **Abstractive Summarizer**
3. **Text Simplifier**

## Setup

Before starting any fine-tuning, ensure your environment is set up:

```bash
# Create and activate virtual environment
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# Linux/Mac
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: If you have a CUDA-compatible GPU
pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
```

## 1. Document Classifier

The classifier categorizes legal documents by type, using a multi-task learning approach.

### Base Model

Start with Legal-BERT:
```python
from transformers import AutoModel, AutoTokenizer

base_model_name = "nlpaueb/legal-bert-base-uncased"
```

### Datasets

We use three datasets from the LexGLUE benchmark:
- SCOTUS (Supreme Court opinions)
- LEDGAR (contract provisions)
- UNFAIR-ToS (unfair terms of service)

### Fine-tuning Instructions

1. **Prepare datasets**:
```bash
# Run data preparation scripts
python src/preprocessing/standardize_datasets.py
```

2. **Configure training**:
   - Edit `config/classification.yaml` with these recommended settings:
     ```yaml
     batch_size: 16
     learning_rate: 2e-5
     num_epochs: 3
     max_length: 512
     ```

3. **Run training**:
```bash
python src/classification/train_multitask_classifier.py
```

4. **Expected output**:
   - Training time: 2-6 hours on GPU
   - Expected metrics:
     - SCOTUS: ~70% accuracy
     - LEDGAR: ~85% macro-F1
     - UNFAIR-ToS: ~90% macro-F1

5. **Save location**: `models/classification/multitask_legal_model_standardized/`

## 2. Abstractive Summarizer

The summarizer creates concise summaries of legal documents.

### Base Model

Use T5 or PEGASUS:
```python
from transformers import AutoModelForSeq2SeqLM

# Option 1: T5
base_model_name = "google-t5/t5-base"

# Option 2: PEGASUS (specialized for summarization)
# base_model_name = "google/pegasus-large"
```

### Datasets

We use two legal summarization datasets:
- CaseSumm (case summarization)
- BillSum (legislation summarization)

### Fine-tuning Instructions

1. **Configure training**:
   - Edit `config/summarization.yaml` with these settings:
     ```yaml
     abstractive:
       base_model: "google-t5/t5-base"  # or "google/pegasus-large"
       dataset_name: "ChicagoHAI/CaseSumm"  # or "billsum"
       training:
         max_input_length: 1024
         max_target_length: 256
         learning_rate: 5e-5
         num_train_epochs: 3
         per_device_train_batch_size: 4
         gradient_accumulation_steps: 4
         warmup_ratio: 0.1
     ```

2. **Run training**:
```bash
python src/summarization/finetune_abstractive.py
```

3. **Expected output**:
   - Training time: 8-24 hours on GPU
   - Expected metrics:
     - ROUGE-1: ~40-45
     - ROUGE-2: ~20-25
     - ROUGE-L: ~35-40

4. **Save location**: `models/pipeline/abstractive_summarizer/`

## 3. Text Simplifier

The simplifier converts complex legal language into more accessible text.

### Base Model

Use T5 (text-to-text):
```python
from transformers import AutoModelForSeq2SeqLM

base_model_name = "google-t5/t5-base"
```

### Datasets

We use two text simplification datasets:
- ASSET (general English simplification)
- Legal-ASSET (synthetic legal simplification)

### Fine-tuning Instructions

1. **Download datasets**:
```bash
# Download and prepare datasets
python src/simplification/prepare_simplification_data.py
```

2. **Configure training**:
   - Edit `config/simplification.yaml` with these settings:
     ```yaml
     model:
       base_model: "google-t5/t5-base"
       simplification_model_name: "legal-t5-simplifier"
     training:
       max_input_length: 512
       max_target_length: 512
       num_train_epochs: 5
       learning_rate: 3e-5
       per_device_train_batch_size: 8
     ```

3. **Run training**:
```bash
python src/simplification/finetune_simplifier.py
```

4. **Expected output**:
   - Training time: 6-12 hours on GPU
   - Expected metrics:
     - SARI: ~38-42
     - BLEU: ~75-80
     - Flesch-Kincaid Reading Ease: Improved by ~10-15 points

5. **Save location**: `models/pipeline/simplifier/`

## Using the Pipeline with Your Fine-tuned Models

After fine-tuning, update the pipeline code with your model paths:

```python
# In src/pipeline/legal_document_pipeline.py
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                         "models/pipeline/abstractive_summarizer")
```

## Alternative: Loading Models from Hugging Face Hub

If you publish your fine-tuned models to Hugging Face Hub, you can load them directly:

```python
# Example for the summarizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "your-username/legal-summarizer"  # Replace with your model ID
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

## Troubleshooting

- **Out of memory errors**: Reduce batch size or use gradient accumulation
- **Poor performance**: Try longer training, different learning rates
- **Slow training**: Enable mixed precision training with `fp16=True` 