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
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Ensure necessary NLTK data is downloaded (run once)
try:
    nltk.data.find('tokenizers/punkt')
    # We also need to ensure punkt_tab is available if punkt is
    nltk.data.find('tokenizers/punkt_tab') 
except LookupError: 
    print("Downloading NLTK sentence tokenizer data (punkt and punkt_tab)...")
    # Download both resources
    try:
        nltk.download('punkt', quiet=True) 
        nltk.download('punkt_tab', quiet=True) 
        print("NLTK data downloaded.")
    except Exception as e_download:
        print(f"Error downloading NLTK data: {e_download}")
        print("Please try downloading manually: python -m nltk.downloader punkt punkt_tab")

# --- Configuration ---
# Ideally, load from config/summarization.yaml, but hardcode for now
STANDARDIZED_DATA_DIR = 'data/standardized'
DEFAULT_DATASET_NAME = "scotus" # Start with SCOTUS

def clean_text(text):
    """Basic text cleaning (can be expanded)."""
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    # Add more cleaning steps if needed (e.g., removing specific headers/footers)
    return text

def get_sentences(text, min_length=5):
    """Tokenize text into sentences using NLTK and filter short ones."""
    try:
        sentences = nltk.sent_tokenize(text)
        # Filter out very short sentences
        sentences = [s for s in sentences if len(s.split()) > min_length]
        if not sentences:
            print("Warning: No sentences found after filtering. Check min_length or input text.")
        return sentences
    except Exception as e:
        print(f"Error during sentence tokenization: {e}")
        traceback.print_exc()
        return []

def build_similarity_matrix(sentences):
    """Builds sentence similarity matrix using TF-IDF and Cosine Similarity."""
    if not sentences:
        return None, None # Return None if no sentences

    try:
        # Use TF-IDF to represent sentences as vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate cosine similarity between sentence vectors
        # Result is a square matrix where similarity_matrix[i, j] is the similarity
        # between sentence i and sentence j.
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Check for NaN values which can occur if a sentence has no TF-IDF features
        # (e.g., only stop words after filtering)
        if np.isnan(similarity_matrix).any():
             print("Warning: NaN values found in similarity matrix. Replacing with 0.")
             similarity_matrix = np.nan_to_num(similarity_matrix) # Replace NaN with 0

        return similarity_matrix, vectorizer # Return vectorizer if needed later
    except ValueError as ve:
         print(f"ValueError during TF-IDF/Similarity calculation: {ve}")
         print(f"  Number of sentences: {len(sentences)}")
         # If error is due to empty vocabulary, likely sentences were only stop words
         return None, None
    except Exception as e:
        print(f"Unexpected error during similarity calculation: {e}")
        return None, None

def textrank_summarize(text, num_sentences=5, min_sentence_length=5):
    """Generates an extractive summary using TextRank."""
    # 1. Clean and get sentences
    cleaned_text = clean_text(text)
    sentences = get_sentences(cleaned_text, min_length=min_sentence_length) # Pass min_length
    if not sentences or len(sentences) < num_sentences:
        print("Warning: Not enough sentences to generate the requested summary length.")
        # Return first few sentences or original text if too short
        return ' '.join(sentences) if sentences else cleaned_text[:500] + "... (trimmed)"

    # 2. Build Similarity Matrix
    similarity_matrix, _ = build_similarity_matrix(sentences) # Unpack the tuple
    if similarity_matrix is None or similarity_matrix.shape[0] == 0:
         print("Warning: Could not build similarity matrix.")
         return ' '.join(sentences[:num_sentences]) # Fallback

    # 3. Apply TextRank (PageRank)
    try:
        nx_graph = nx.from_numpy_array(similarity_matrix)
        # Check if graph is empty or has no edges before calculating pagerank
        if nx_graph.number_of_nodes() == 0 or nx_graph.number_of_edges() == 0:
            print("Warning: Similarity graph has no nodes or edges. Returning top sentences sequentially.")
            ranked_sentences = list(enumerate(sentences))
            # Sort by original index to keep order
            ranked_sentences.sort(key=lambda x: x[0])
        else:
            scores = nx.pagerank(nx_graph, weight='weight')
            # Get scored sentences with their original indices
            ranked_sentences = [(scores[i], s, i) for i, s in enumerate(sentences)]
            # Sort sentences by score (descending)
            ranked_sentences.sort(reverse=True, key=lambda x: x[0])

    except Exception as e:
        print(f"Error during PageRank calculation: {e}")
        traceback.print_exc()
        # Fallback: return the first few sentences
        ranked_sentences = list(enumerate(sentences))
        ranked_sentences.sort(key=lambda x: x[0]) # Sort by index

    # 4. Select top N sentences based on importance, but preserve original document order
    if len(ranked_sentences) > 0:
        num_to_select = min(num_sentences, len(ranked_sentences))
        # First select the top sentences by importance
        top_sentences = ranked_sentences[:num_to_select]
        
        # Then sort these top sentences by their original position in the document
        if len(top_sentences[0]) == 3:  # (score, sentence, index) format
            top_sentences_ordered = sorted(top_sentences, key=lambda x: x[2])
            selected_sentences = [s[1] for s in top_sentences_ordered]
        else:  # (index, sentence) format from fallback
            top_sentences_ordered = sorted(top_sentences, key=lambda x: x[0])
            selected_sentences = [s[1] for s in top_sentences_ordered]
    else:
        selected_sentences = []

    # 5. Combine into summary
    summary = ' '.join(selected_sentences)
    return summary

# --- Main Execution --- (Example Usage)
if __name__ == "__main__":
    try:
        # Load configuration
        config = load_config() # Load from default path
        extractive_config = config.get('extractive', {})
        if not extractive_config:
            print("Warning: 'extractive' section not found in config/summarization.yaml. Using defaults.")

        # Get parameters from config, with fallbacks
        dataset_name = extractive_config.get('dataset_name', 'scotus_processor')
        dataset_split = extractive_config.get('dataset_split', 'test')
        num_summary_sentences = extractive_config.get('num_summary_sentences', 5)
        min_sent_len = extractive_config.get('min_sentence_length', 5)

        print(f"Loading dataset: {dataset_name}, split: {dataset_split}")
        # Load the specified dataset split
        # Adjust path/loading based on how standardized datasets are saved
        # Assuming they are saved in a way `datasets.load_from_disk` can handle
        # or loaded via Hugging Face datasets library directly if applicable.
        # THIS MIGHT NEED ADJUSTMENT BASED ON YOUR ACTUAL DATA SAVING STRUCTURE
        try:
            # Try loading from disk first (common pattern in this project)
            processed_data_path = os.path.join('data', 'processed', dataset_name)
            if os.path.exists(processed_data_path):
                 dataset = datasets.load_from_disk(processed_data_path)[dataset_split]
                 print(f"Loaded {len(dataset)} examples from disk.")
            else:
                 # Fallback: attempt to load directly if it's a HF dataset name
                 print(f"Dataset not found at {processed_data_path}. Attempting to load '{dataset_name}' directly.")
                 dataset = datasets.load_dataset(dataset_name, split=dataset_split)
                 print(f"Loaded {len(dataset)} examples directly.")
        except Exception as load_e:
            print(f"Error loading dataset '{dataset_name}': {load_e}")
            print("Please ensure the dataset name in config/summarization.yaml is correct and the data exists.")
            exit(1)


        # Determine the text column (assuming 'text' based on previous scripts)
        text_column = 'text' # Adjust if your standardized dataset uses a different column name
        if text_column not in dataset.column_names:
            print(f"Error: Text column '{text_column}' not found in dataset. Available columns: {dataset.column_names}")
            # Attempt to guess a text-like column if 'text' isn't present
            potential_cols = [col for col in dataset.column_names if 'text' in col or 'content' in col or 'opinion' in col]
            if potential_cols:
                text_column = potential_cols[0]
                print(f"Using column '{text_column}' as text input.")
            else:
                print("Cannot determine text column. Exiting.")
                exit(1)


        print(f"\nGenerating summaries for the first 3 examples (max {num_summary_sentences} sentences each, min sentence length {min_sent_len})...")

        # Summarize a few examples
        for i in range(min(3, len(dataset))):
            example_text = dataset[i][text_column]
            print(f"\n--- Example {i+1} ---")
            # print(f"Original Text (first 500 chars): {example_text[:500]}...")
            summary = textrank_summarize(example_text, num_sentences=num_summary_sentences, min_sentence_length=min_sent_len)
            print(f"\nGenerated Summary:")
            print(summary)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in the main execution block: {e}")
        traceback.print_exc() 