import nltk
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_from_disk
import os
import re

# Ensure necessary NLTK data is downloaded (run once)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK sentence tokenizer (punkt)...")
    nltk.download('punkt')

# --- Configuration ---
# Ideally, load from config/summarization.yaml, but hardcode for now
STANDARDIZED_DATA_DIR = 'data/standardized'
DEFAULT_DATASET_NAME = "scotus" # Start with SCOTUS

def clean_text(text):
    """Basic text cleaning (can be expanded)."""
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    # Add more cleaning steps if needed (e.g., removing specific headers/footers)
    return text

def get_sentences(text):
    """Split text into sentences using NLTK."""
    if not isinstance(text, str):
         print(f"Warning: Input to get_sentences is not a string (type: {type(text)}). Returning empty list.")
         return []
    try:
        # Use NLTK's recommended sentence tokenizer
        sentences = nltk.sent_tokenize(text)
        # Filter out very short sentences (likely noise or headings)
        sentences = [s.strip() for s in sentences if len(s.split()) > 5] 
        return sentences
    except Exception as e:
        print(f"Error during sentence tokenization: {e}")
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


def textrank_summarize(text, num_sentences=5):
    """
    Generates an extractive summary using the TextRank algorithm.

    Args:
        text (str): The input document text.
        num_sentences (int): The desired number of sentences in the summary.

    Returns:
        str: The generated summary, or an empty string if summarization fails.
    """
    print(f"\nStarting TextRank summarization for text (first 100 chars): {text[:100]}...")
    
    # 1. Clean and Sentence Tokenize
    cleaned_text = clean_text(text)
    sentences = get_sentences(cleaned_text)
    
    if not sentences or len(sentences) <= num_sentences:
        print("  Not enough sentences to summarize meaningfully.")
        return " ".join(sentences) # Return original text if too short

    print(f"  Found {len(sentences)} sentences.")

    # 2. Build Similarity Matrix
    similarity_matrix, _ = build_similarity_matrix(sentences)
    
    if similarity_matrix is None:
        print("  Failed to build similarity matrix. Cannot summarize.")
        return ""

    # 3. Create Graph and Run PageRank
    try:
        # Create a graph representation of the sentences
        # Nodes are sentences, edges are weighted by similarity
        sentence_graph = nx.from_numpy_array(similarity_matrix)
        
        # Apply the PageRank algorithm (TextRank is based on PageRank)
        # Scores represent the importance/centrality of each sentence
        scores = nx.pagerank(sentence_graph, weight='weight') # Use edge weights (similarity)
        
    except Exception as e:
        print(f"Error during graph creation or PageRank: {e}")
        return ""

    # 4. Rank Sentences and Extract Top N
    # Sort sentences by their TextRank score in descending order
    ranked_sentences = sorted(
        ((scores[i], s, i) for i, s in enumerate(sentences)), 
        reverse=True,
        key=lambda x: x[0] # Sort by score (index 0)
    )
    
    # Select the top N sentences based on rank
    top_sentence_indices = sorted([item[2] for item in ranked_sentences[:num_sentences]])
    
    # 5. Combine into Summary (preserving original order)
    summary_sentences = [sentences[i] for i in top_sentence_indices]
    summary = " ".join(summary_sentences)
    
    print(f"  Generated summary with {len(summary_sentences)} sentences.")
    return summary

# --- Example Usage ---
if __name__ == "__main__":
    print("--- Extractive Summarizer Example ---")
    
    # Load the standardized SCOTUS dataset
    dataset_name = DEFAULT_DATASET_NAME
    dataset_path = os.path.join(STANDARDIZED_DATA_DIR, f"{dataset_name}_standardized_dataset")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Standardized dataset not found at {dataset_path}")
        exit()
        
    try:
        print(f"Loading {dataset_name} dataset...")
        scotus_dataset = load_from_disk(dataset_path)
        # Use the 'test' split for demonstration
        if 'test' in scotus_dataset:
             test_data = scotus_dataset['test']
             print(f"Loaded {len(test_data)} examples from test split.")
        elif 'train' in scotus_dataset:
             print("Warning: 'test' split not found, using 'train' split instead.")
             test_data = scotus_dataset['train']
        else:
             print("Error: No 'test' or 'train' split found in the dataset.")
             exit()
             
        # Summarize the first few examples
        num_examples_to_summarize = 3
        desired_summary_length = 5 # Number of sentences

        for i in range(min(num_examples_to_summarize, len(test_data))):
            print(f"\n--- Summarizing Example {i+1} ---")
            original_text = test_data[i]['input_text']
            original_label = test_data[i]['input_label'] # Just for context
            
            print(f"Original Text Length: {len(original_text)} chars")
            print(f"Original Label: {original_label}")
            
            summary = textrank_summarize(original_text, num_sentences=desired_summary_length)
            
            print("\nGenerated Summary:")
            print(summary)
            print("-" * 30)

    except Exception as e:
        print(f"An error occurred during example usage: {e}")
        import traceback
        traceback.print_exc() 