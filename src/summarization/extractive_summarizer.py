import nltk
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datasets
import traceback
import yaml
import os
import re

# --- Configuration Loading ---
def load_config(config_path='config/summarization.yaml'):
    """Loads extractive summarization settings from config file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Download NLTK dependencies if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab') 
except LookupError: 
    print("Downloading NLTK sentence tokenizer data...")
    try:
        nltk.download('punkt', quiet=True) 
        nltk.download('punkt_tab', quiet=True) 
        print("NLTK data downloaded.")
    except Exception as e_download:
        print(f"Error downloading NLTK data: {e_download}")
        print("Please try downloading manually: python -m nltk.downloader punkt punkt_tab")

# Constants (fallbacks if config is unavailable)
STANDARDIZED_DATA_DIR = 'data/standardized'
DEFAULT_DATASET_NAME = "scotus"

def clean_text(text):
    """Removes extra whitespace and unwanted characters."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_sentences(text, min_length=5):
    """Splits text into sentences and filters out very short ones."""
    try:
        sentences = nltk.sent_tokenize(text)
        # Keep only sentences with sufficient words
        sentences = [s for s in sentences if len(s.split()) > min_length]
        if not sentences:
            print("Warning: No sentences found after filtering. Check min_length or input text.")
        return sentences
    except Exception as e:
        print(f"Error during sentence tokenization: {e}")
        traceback.print_exc()
        return []

def build_similarity_matrix(sentences):
    """Creates a matrix of sentence similarities using TF-IDF and cosine similarity."""
    if not sentences:
        return None, None

    try:
        # Convert sentences to TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate similarities between all sentence pairs
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Handle potential NaN values from empty sentences
        if np.isnan(similarity_matrix).any():
             print("Warning: NaN values found in similarity matrix. Replacing with 0.")
             similarity_matrix = np.nan_to_num(similarity_matrix)

        return similarity_matrix, vectorizer
    except ValueError as ve:
         print(f"ValueError during TF-IDF calculation: {ve}")
         print(f"  Number of sentences: {len(sentences)}")
         return None, None
    except Exception as e:
        print(f"Unexpected error during similarity calculation: {e}")
        return None, None

def textrank_summarize(text, num_sentences=5, min_sentence_length=5):
    """Generates an extractive summary using TextRank algorithm."""
    # 1. Preprocess text
    cleaned_text = clean_text(text)
    sentences = get_sentences(cleaned_text, min_length=min_sentence_length)
    if not sentences or len(sentences) < num_sentences:
        print("Warning: Not enough sentences to generate the requested summary length.")
        return ' '.join(sentences) if sentences else cleaned_text[:500] + "... (trimmed)"

    # 2. Calculate sentence similarities
    similarity_matrix, _ = build_similarity_matrix(sentences)
    if similarity_matrix is None or similarity_matrix.shape[0] == 0:
         print("Warning: Could not build similarity matrix.")
         return ' '.join(sentences[:num_sentences])

    # 3. Apply TextRank (PageRank algorithm)
    try:
        nx_graph = nx.from_numpy_array(similarity_matrix)
        # Verify graph has content before running PageRank
        if nx_graph.number_of_nodes() == 0 or nx_graph.number_of_edges() == 0:
            print("Warning: Similarity graph has no nodes or edges. Using sequential order.")
            ranked_sentences = list(enumerate(sentences))
            ranked_sentences.sort(key=lambda x: x[0])
        else:
            scores = nx.pagerank(nx_graph, weight='weight')
            # Score sentences by PageRank results
            ranked_sentences = [(scores[i], s, i) for i, s in enumerate(sentences)]
            ranked_sentences.sort(reverse=True, key=lambda x: x[0])

    except Exception as e:
        print(f"Error during PageRank calculation: {e}")
        traceback.print_exc()
        # Fallback to original sentence order
        ranked_sentences = list(enumerate(sentences))
        ranked_sentences.sort(key=lambda x: x[0])

    # 4. Select top sentences while preserving original document order
    if len(ranked_sentences) > 0:
        num_to_select = min(num_sentences, len(ranked_sentences))
        # First select the best sentences by importance
        top_sentences = ranked_sentences[:num_to_select]
        
        # Then sort by original position to maintain document flow
        if len(top_sentences[0]) == 3:  # (score, sentence, index) format
            top_sentences_ordered = sorted(top_sentences, key=lambda x: x[2])
            selected_sentences = [s[1] for s in top_sentences_ordered]
        else:  # (index, sentence) format from fallback
            top_sentences_ordered = sorted(top_sentences, key=lambda x: x[0])
            selected_sentences = [s[1] for s in top_sentences_ordered]
    else:
        selected_sentences = []

    # 5. Join sentences into summary
    summary = ' '.join(selected_sentences)
    return summary

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # Load configuration
        config = load_config()
        extractive_config = config.get('extractive', {})
        if not extractive_config:
            print("Warning: 'extractive' section not found in config. Using defaults.")

        # Get parameters with fallbacks
        dataset_name = extractive_config.get('dataset_name', 'scotus_processor')
        dataset_split = extractive_config.get('dataset_split', 'test')
        num_summary_sentences = extractive_config.get('num_summary_sentences', 5)
        min_sent_len = extractive_config.get('min_sentence_length', 5)

        print(f"Loading dataset: {dataset_name}, split: {dataset_split}")
        
        # Load the dataset
        try:
            # Try loading from processed data directory first
            processed_data_path = os.path.join('data', 'processed', dataset_name)
            if os.path.exists(processed_data_path):
                 dataset = datasets.load_from_disk(processed_data_path)[dataset_split]
                 print(f"Loaded {len(dataset)} examples from disk.")
            else:
                 # Fallback to Hugging Face datasets
                 print(f"Dataset not found at {processed_data_path}. Attempting to load '{dataset_name}' directly.")
                 dataset = datasets.load_dataset(dataset_name, split=dataset_split)
                 print(f"Loaded {len(dataset)} examples directly.")
        except Exception as load_e:
            print(f"Error loading dataset '{dataset_name}': {load_e}")
            print("Please ensure the dataset name in config/summarization.yaml is correct and the data exists.")
            exit(1)

        # Determine text column name
        text_column = 'text'
        if text_column not in dataset.column_names:
            print(f"Error: Text column '{text_column}' not found in dataset. Available columns: {dataset.column_names}")
            # Try to guess appropriate text column
            potential_cols = [col for col in dataset.column_names if 'text' in col or 'content' in col or 'opinion' in col]
            if potential_cols:
                text_column = potential_cols[0]
                print(f"Using column '{text_column}' as text input.")
            else:
                print("Cannot determine text column. Exiting.")
                exit(1)

        print(f"\nGenerating summaries for the first 3 examples (max {num_summary_sentences} sentences each, min sentence length {min_sent_len})...")

        # Generate and display summaries
        for i in range(min(3, len(dataset))):
            example_text = dataset[i][text_column]
            print(f"\n--- Example {i+1} ---")
            summary = textrank_summarize(example_text, num_sentences=num_summary_sentences, min_sentence_length=min_sent_len)
            print(f"\nGenerated Summary:")
            print(summary)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in the main execution block: {e}")
        traceback.print_exc() 