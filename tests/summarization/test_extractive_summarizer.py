"""
Unit tests for extractive summarization functionality.
"""
import pytest
import torch
import numpy as np
import os
import tempfile
import yaml
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the functions we're testing
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))


class TestExtractiveSummarizerForwardPass:
    """Test class for extractive summarizer functionality."""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning functionality."""
        from summarization.extractive_summarizer import clean_text
        
        # Test with normal text
        input_text = "This   is   a   test   with   extra   spaces."
        result = clean_text(input_text)
        
        assert isinstance(result, str)
        assert "   " not in result  # Should remove extra spaces
        assert result.strip() == result  # Should be trimmed
    
    def test_clean_text_with_newlines(self):
        """Test text cleaning with newlines and tabs."""
        from summarization.extractive_summarizer import clean_text
        
        # Test with newlines and tabs
        input_text = "This is a test.\n\n\tWith newlines and tabs.\n\n"
        result = clean_text(input_text)
        
        assert isinstance(result, str)
        assert result.strip() == result  # Should be trimmed
    
    def test_get_sentences_basic(self):
        """Test sentence tokenization functionality."""
        from summarization.extractive_summarizer import get_sentences
        
        # Test with normal text (each sentence has more than 5 words)
        input_text = "This is the first sentence here. This is the second sentence here. This is the third sentence here."
        result = get_sentences(input_text)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(s, str) for s in result)
        assert "first sentence" in result[0]
        assert "second sentence" in result[1]
        assert "third sentence" in result[2]
    
    def test_get_sentences_short_sentences(self):
        """Test sentence tokenization with short sentences (filtered out)."""
        from summarization.extractive_summarizer import get_sentences
        
        # Test with short sentences
        input_text = "Hi. This is a longer sentence here. OK. Another much longer sentence here."
        result = get_sentences(input_text, min_length=5)
        
        assert isinstance(result, list)
        assert len(result) == 2  # Only longer sentences should remain
        assert "longer sentence" in result[0]
        assert "Another much longer sentence" in result[1]
    
    def test_get_sentences_empty_text(self):
        """Test sentence tokenization with empty text."""
        from summarization.extractive_summarizer import get_sentences
        
        # Test with empty text
        result = get_sentences("")
        
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_build_similarity_matrix_basic(self):
        """Test similarity matrix building functionality."""
        from summarization.extractive_summarizer import build_similarity_matrix
        
        # Test with normal sentences
        sentences = [
            "This is the first sentence about legal matters.",
            "This is the second sentence about legal matters.",
            "This is a completely different sentence about technology."
        ]
        
        similarity_matrix, vectorizer = build_similarity_matrix(sentences)
        
        assert similarity_matrix is not None
        assert vectorizer is not None
        assert similarity_matrix.shape == (3, 3)  # 3x3 matrix for 3 sentences
        assert np.allclose(similarity_matrix, similarity_matrix.T)  # Should be symmetric
    
    def test_build_similarity_matrix_empty_sentences(self):
        """Test similarity matrix building with empty sentences."""
        from summarization.extractive_summarizer import build_similarity_matrix
        
        # Test with empty list
        similarity_matrix, vectorizer = build_similarity_matrix([])
        
        assert similarity_matrix is None
        assert vectorizer is None
    
    def test_textrank_summarize_basic(self, sample_legal_document):
        """Test basic TextRank summarization functionality."""
        from summarization.extractive_summarizer import textrank_summarize
        
        # Test with sample document
        result = textrank_summarize(sample_legal_document, num_sentences=3)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert result.strip() == result  # Should be trimmed
    
    def test_textrank_summarize_short_text(self):
        """Test TextRank summarization with short text."""
        from summarization.extractive_summarizer import textrank_summarize
        
        # Test with very short text
        short_text = "This is a short document."
        result = textrank_summarize(short_text, num_sentences=5)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_textrank_summarize_empty_text(self):
        """Test TextRank summarization with empty text."""
        from summarization.extractive_summarizer import textrank_summarize
        
        # Test with empty text
        result = textrank_summarize("", num_sentences=3)
        
        assert isinstance(result, str)
        assert "trimmed" in result  # Should return trimmed message
    
    def test_textrank_summarize_different_parameters(self, sample_legal_document):
        """Test TextRank summarization with different parameters."""
        from summarization.extractive_summarizer import textrank_summarize
        
        # Test with different number of sentences
        result_1 = textrank_summarize(sample_legal_document, num_sentences=1)
        result_3 = textrank_summarize(sample_legal_document, num_sentences=3)
        result_5 = textrank_summarize(sample_legal_document, num_sentences=5)
        
        assert isinstance(result_1, str)
        assert isinstance(result_3, str)
        assert isinstance(result_5, str)
        
        # More sentences should generally produce longer summaries
        assert len(result_5.split('.')) >= len(result_3.split('.'))
        assert len(result_3.split('.')) >= len(result_1.split('.'))
    
    def test_textrank_summarize_min_sentence_length(self, sample_legal_document):
        """Test TextRank summarization with different minimum sentence lengths."""
        from summarization.extractive_summarizer import textrank_summarize
        
        # Test with different minimum sentence lengths
        result_short = textrank_summarize(sample_legal_document, min_sentence_length=5)
        result_long = textrank_summarize(sample_legal_document, min_sentence_length=20)
        
        assert isinstance(result_short, str)
        assert isinstance(result_long, str)
        assert len(result_short) > 0
        assert len(result_long) > 0


class TestExtractiveSummarizerIntegration:
    """Test class for extractive summarizer integration functionality."""
    
    def test_end_to_end_summarization(self, sample_legal_document):
        """Test end-to-end summarization workflow."""
        from summarization.extractive_summarizer import textrank_summarize
        
        # Test complete workflow
        summary = textrank_summarize(sample_legal_document, num_sentences=3)
        
        # Verify summary quality
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert len(summary) < len(sample_legal_document)  # Summary should be shorter
        
        # Verify summary contains sentences
        sentences = summary.split('.')
        assert len(sentences) >= 1
    
    def test_summarization_with_legal_terms(self):
        """Test summarization with legal terminology."""
        from summarization.extractive_summarizer import textrank_summarize
        
        legal_text = """
        The court hereby finds that the defendant's motion to dismiss is without merit. 
        The plaintiff has established a prima facie case for breach of contract. 
        The defendant's argument regarding the statute of limitations is rejected. 
        The court will proceed with the trial as scheduled.
        """
        
        summary = textrank_summarize(legal_text, num_sentences=2)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert any(term in summary.lower() for term in ['court', 'defendant', 'plaintiff', 'motion'])
    
    def test_summarization_error_handling(self):
        """Test summarization error handling with problematic input."""
        from summarization.extractive_summarizer import textrank_summarize
        
        # Test with text that might cause issues
        problematic_text = "a" * 10000  # Very long single word
        
        # Should not crash and should return some result
        result = textrank_summarize(problematic_text, num_sentences=3)
        assert isinstance(result, str)


class TestExtractiveSummarizer:
    """Test class for extractive summarization functionality."""
    
    def test_placeholder(self):
        """Placeholder test to ensure test structure is working."""
        assert True
