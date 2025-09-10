"""
Unit tests for abstractive summarization functionality.
"""
import pytest
import torch
import numpy as np
import os
import tempfile
import yaml
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the function we're testing
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from summarization.abstractive_summarizer import load_config


class TestSummarizationConfigLoading:
    """Test class for summarization configuration loading functionality."""
    
    def test_load_config_valid_file(self, temp_dir):
        """Test loading a valid summarization configuration file."""
        config_data = {
            'abstractive': {
                'dataset_name': 'ChicagoHAI/CaseSumm',
                'dataset_split': 'train',
                'text_column': 'opinion',
                'summary_column': 'syllabus',
                'base_model': 'nsi319/legal-pegasus',
                'max_input_length': 1024,
                'max_target_length': 256,
                'batch_size': 4,
                'num_beams': 8,
                'length_penalty': 2.0,
                'min_length': 50,
                'no_repeat_ngram_size': 2,
                'early_stopping': True
            },
            'extractive': {
                'dataset_name': 'scotus_dataset',
                'dataset_split': 'test',
                'num_summary_sentences': 5,
                'min_sentence_length': 5
            }
        }
        
        config_path = os.path.join(temp_dir, 'summarization_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        result = load_config(config_path)
        
        assert result is not None
        # The function returns only the abstractive section, not the full config
        assert result == config_data['abstractive']
        assert result['base_model'] == 'nsi319/legal-pegasus'
        assert result['dataset_name'] == 'ChicagoHAI/CaseSumm'
    
    def test_load_config_missing_file(self):
        """Test loading a non-existent summarization configuration file."""
        non_existent_path = "non_existent_summarization_config.yaml"
        
        with pytest.raises(FileNotFoundError):
            load_config(non_existent_path)
    
    def test_load_config_malformed_yaml(self, temp_dir):
        """Test loading a malformed YAML file for summarization."""
        malformed_yaml = """
        abstractive:
            dataset_name: ChicagoHAI/CaseSumm
            base_model: nsi319/legal-pegasus
            # Missing colon after this line
            max_input_length 1024
        extractive:
            dataset_name: scotus_dataset
        """
        
        config_path = os.path.join(temp_dir, 'malformed_summarization_config.yaml')
        with open(config_path, 'w') as f:
            f.write(malformed_yaml)
        
        with pytest.raises(yaml.YAMLError):
            load_config(config_path)
    
    def test_load_config_empty_file(self, temp_dir):
        """Test loading an empty summarization configuration file."""
        config_path = os.path.join(temp_dir, 'empty_summarization_config.yaml')
        with open(config_path, 'w') as f:
            pass  # Create empty file
        
        # Empty YAML returns None, which causes TypeError in the function
        with pytest.raises(TypeError):
            load_config(config_path)
    
    def test_load_config_missing_sections(self, temp_dir):
        """Test loading a config file missing required sections."""
        incomplete_config = {
            'abstractive': {
                'base_model': 'nsi319/legal-pegasus'
                # Missing other required keys
            }
            # Missing extractive section
        }
        
        config_path = os.path.join(temp_dir, 'incomplete_summarization_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(incomplete_config, f)
        
        result = load_config(config_path)
        
        # Should return only the abstractive section
        assert result is not None
        assert result == incomplete_config['abstractive']
        assert result['base_model'] == 'nsi319/legal-pegasus'
    
    def test_load_config_default_path(self, temp_dir):
        """Test loading config with default path when file doesn't exist."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            with pytest.raises(FileNotFoundError):
                load_config()  # Should use default path and raise error
        finally:
            os.chdir(original_cwd)
    
    def test_load_config_unicode_content(self, temp_dir):
        """Test loading config file with unicode content for summarization."""
        config_data = {
            'abstractive': {
                'dataset_name': 'ChicagoHAI/CaseSumm',
                'base_model': 'nsi319/legal-pegasus',
                'description': 'Legal summarization model: 法律摘要模型'
            },
            'extractive': {
                'dataset_name': 'scotus_dataset',
                'description': 'Extractive summarization: 提取式摘要'
            }
        }
        
        config_path = os.path.join(temp_dir, 'unicode_summarization_config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, allow_unicode=True)
        
        result = load_config(config_path)
        
        assert result is not None
        # Function returns only the abstractive section
        assert result == config_data['abstractive']
        assert '法律摘要模型' in result['description']


class TestAbstractiveSummarizerForwardPass:
    """Test class for abstractive summarizer model forward pass functionality."""
    
    def test_preprocess_function_basic(self):
        """Test basic preprocessing function with valid inputs."""
        from summarization.abstractive_summarizer import preprocess_function
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.name_or_path = "facebook/bart-base"
        
        def mock_tokenize(*args, **kwargs):
            return {
                'input_ids': [[1, 2, 3, 4, 5]],
                'attention_mask': [[1, 1, 1, 1, 1]]
            }
        
        mock_tokenizer.side_effect = mock_tokenize
        
        # Test data
        examples = {
            'text': ['This is a test document for summarization.'],
            'summary': ['Test summary.']
        }
        
        result = preprocess_function(
            examples, mock_tokenizer, 512, 128, 'text', 'summary'
        )
        
        # Verify result structure
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert 'labels' in result
        assert len(result['input_ids']) == 1
        assert len(result['labels']) == 1
    
    def test_preprocess_function_t5_model(self):
        """Test preprocessing function with T5 model (requires prefix)."""
        from summarization.abstractive_summarizer import preprocess_function
        
        # Mock T5 tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.name_or_path = "t5-small"
        
        def mock_tokenize(*args, **kwargs):
            return {
                'input_ids': [[1, 2, 3, 4, 5]],
                'attention_mask': [[1, 1, 1, 1, 1]]
            }
        
        mock_tokenizer.side_effect = mock_tokenize
        
        # Test data
        examples = {
            'text': ['This is a test document for summarization.'],
            'summary': ['Test summary.']
        }
        
        result = preprocess_function(
            examples, mock_tokenizer, 512, 128, 'text', 'summary'
        )
        
        # Verify result structure
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert 'labels' in result
        assert len(result['input_ids']) == 1
        assert len(result['labels']) == 1
    
    def test_preprocess_function_batch_processing(self):
        """Test preprocessing function with batch of examples."""
        from summarization.abstractive_summarizer import preprocess_function
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.name_or_path = "facebook/bart-base"
        
        def mock_tokenize(*args, **kwargs):
            return {
                'input_ids': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                'attention_mask': [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
            }
        
        mock_tokenizer.side_effect = mock_tokenize
        
        # Test data
        examples = {
            'text': ['First document for summarization.', 'Second document for summarization.'],
            'summary': ['First summary.', 'Second summary.']
        }
        
        result = preprocess_function(
            examples, mock_tokenizer, 512, 128, 'text', 'summary'
        )
        
        # Verify result structure
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert 'labels' in result
        assert len(result['input_ids']) == 2
        assert len(result['labels']) == 2


class TestPostProcessSummary:
    """Test class for post-processing summary functionality."""
    
    def test_post_process_summary_basic(self):
        """Test basic post-processing of summary text."""
        from summarization.abstractive_summarizer import post_process_summary
        
        # Test basic text
        input_text = "This is a test summary. It has multiple sentences. Each sentence should be processed correctly."
        result = post_process_summary(input_text)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert result.strip() == result  # Should be trimmed
    
    def test_post_process_summary_numbered_points(self):
        """Test post-processing with numbered points."""
        from summarization.abstractive_summarizer import post_process_summary
        
        # Test with numbered points
        input_text = "1. First point about the case. 2. Second point about the ruling. 3. Third point about the implications."
        result = post_process_summary(input_text)
        
        assert isinstance(result, str)
        assert "1." in result
        assert "2." in result
        assert "3." in result
    
    def test_post_process_summary_newline_tags(self):
        """Test post-processing with newline tags."""
        from summarization.abstractive_summarizer import post_process_summary
        
        # Test with newline tags
        input_text = "First point<n>Second point<n>Third point"
        result = post_process_summary(input_text)
        
        assert isinstance(result, str)
        assert "<n>" not in result  # Should be replaced with actual newlines
        assert "\n" in result
    
    def test_post_process_summary_incomplete_sentences(self):
        """Test post-processing with incomplete sentences."""
        from summarization.abstractive_summarizer import post_process_summary
        
        # Test with incomplete sentence
        input_text = "This is a complete sentence. This is an incomplete sentence"
        result = post_process_summary(input_text)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_post_process_summary_empty_text(self):
        """Test post-processing with empty text."""
        from summarization.abstractive_summarizer import post_process_summary
        
        # Test with empty text
        result = post_process_summary("")
        assert isinstance(result, str)
        assert result.strip() == ""


class TestROUGEMetrics:
    """Test class for ROUGE metrics computation."""
    
    def test_compute_metrics_basic(self):
        """Test basic ROUGE metrics computation."""
        from summarization.abstractive_summarizer import compute_metrics
        
        # Test data
        decoded_preds = ["This is a test summary.", "Another test summary."]
        decoded_labels = ["This is a reference summary.", "Another reference summary."]
        
        result = compute_metrics(decoded_preds, decoded_labels)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'rouge1' in result
        assert 'rouge2' in result
        assert 'rougeL' in result
        assert 'gen_len' in result
        
        # Verify score types
        assert isinstance(result['rouge1'], (int, float))
        assert isinstance(result['rouge2'], (int, float))
        assert isinstance(result['rougeL'], (int, float))
        assert isinstance(result['gen_len'], (int, float))
    
    def test_compute_metrics_empty_inputs(self):
        """Test ROUGE metrics with empty inputs."""
        from summarization.abstractive_summarizer import compute_metrics
        
        # Test with empty lists
        result = compute_metrics([], [])
        
        assert isinstance(result, dict)
        assert 'rouge1' in result
        assert 'rouge2' in result
        assert 'rougeL' in result
        assert 'gen_len' in result
    
    def test_compute_metrics_different_lengths(self):
        """Test ROUGE metrics with different prediction and label lengths."""
        from summarization.abstractive_summarizer import compute_metrics
        
        # Test with different lengths
        decoded_preds = ["Short summary."]
        decoded_labels = ["This is a much longer reference summary with more details."]
        
        result = compute_metrics(decoded_preds, decoded_labels)
        
        assert isinstance(result, dict)
        assert 'rouge1' in result
        assert 'rouge2' in result
        assert 'rougeL' in result
        assert 'gen_len' in result


class TestAbstractiveSummarizer:
    """Test class for abstractive summarization functionality."""
    
    def test_placeholder(self):
        """Placeholder test to ensure test structure is working."""
        assert True
