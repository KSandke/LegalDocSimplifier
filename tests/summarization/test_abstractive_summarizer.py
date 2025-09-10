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


class TestAbstractiveSummarizer:
    """Test class for abstractive summarization functionality."""
    
    def test_placeholder(self):
        """Placeholder test to ensure test structure is working."""
        assert True
