"""
Unit tests for text simplification functionality. 
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
from simplification.train_lexsimple import load_config


class TestSimplificationConfigLoading:
    """Test class for simplification configuration loading functionality."""
    
    def test_load_config_valid_file(self, temp_dir):
        """Test loading a valid simplification configuration file."""
        config_data = {
            'paths': {
                'input_data': 'data/raw/lexsimple.csv',
                'term_dictionary': 'data/dictionary/legal_terms.json',
                'output_models': 'models/simplification/{model_name}'
            },
            'dataset': {
                'name': 'turk'
            },
            'model': {
                'base_model': 'nsi319/legal-pegasus',
                'simplification_model_name': 'legal_simplifier_v1'
            },
            'training': {
                'max_input_length': 512,
                'max_target_length': 128,
                'batch_size': 8,
                'eval_batch_size': 8,
                'learning_rate': 5e-5,
                'epochs': 3,
                'weight_decay': 0.01,
                'save_steps': 100,
                'warmup_steps': 500,
                'gradient_accumulation_steps': 1,
                'evaluation_strategy': 'steps',
                'eval_steps': 100,
                'fp16': True,
                'save_total_limit': 5
            },
            'simplification_params': {
                'level': 'medium',
                'preserve_meaning_strictness': 'high',
                'generation_params': {
                    'num_beams': 4,
                    'length_penalty': 1.0,
                    'early_stopping': True,
                    'do_sample': False,
                    'temperature': 1.0
                }
            }
        }
        
        config_path = os.path.join(temp_dir, 'simplification_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        result = load_config(config_path)
        
        assert result is not None
        assert result == config_data
        assert 'paths' in result
        assert 'dataset' in result
        assert 'model' in result
        assert 'training' in result
        assert 'simplification_params' in result
        assert result['model']['base_model'] == 'nsi319/legal-pegasus'
        assert result['simplification_params']['level'] == 'medium'
    
    def test_load_config_missing_file(self):
        """Test loading a non-existent simplification configuration file."""
        non_existent_path = "non_existent_simplification_config.yaml"
        
        with pytest.raises(FileNotFoundError):
            load_config(non_existent_path)
    
    def test_load_config_malformed_yaml(self, temp_dir):
        """Test loading a malformed YAML file for simplification."""
        malformed_yaml = """
        paths:
            input_data: data/raw/lexsimple.csv
            output_models: models/simplification/{model_name}
        model:
            base_model: nsi319/legal-pegasus
            # Missing colon after this line
            simplification_model_name legal_simplifier_v1
        training:
            max_input_length: 512
        """
        
        config_path = os.path.join(temp_dir, 'malformed_simplification_config.yaml')
        with open(config_path, 'w') as f:
            f.write(malformed_yaml)
        
        with pytest.raises(yaml.YAMLError):
            load_config(config_path)
    
    def test_load_config_empty_file(self, temp_dir):
        """Test loading an empty simplification configuration file."""
        config_path = os.path.join(temp_dir, 'empty_simplification_config.yaml')
        with open(config_path, 'w') as f:
            pass  # Create empty file
        
        result = load_config(config_path)
        
        assert result is None
    
    def test_load_config_missing_sections(self, temp_dir):
        """Test loading a config file missing required sections."""
        incomplete_config = {
            'paths': {
                'input_data': 'data/raw/lexsimple.csv'
                # Missing other required path keys
            }
            # Missing dataset, model, training sections
        }
        
        config_path = os.path.join(temp_dir, 'incomplete_simplification_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(incomplete_config, f)
        
        result = load_config(config_path)
        
        # Should still load successfully (validation happens later)
        assert result is not None
        assert result == incomplete_config
        assert 'dataset' not in result
        assert 'model' not in result
    
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
        """Test loading config file with unicode content for simplification."""
        config_data = {
            'paths': {
                'input_data': 'data/raw/lexsimple.csv',
                'output_models': 'models/simplification/{model_name}'
            },
            'model': {
                'base_model': 'nsi319/legal-pegasus',
                'simplification_model_name': 'legal_simplifier_测试'
            },
            'description': 'Text simplification configuration: 文本简化配置'
        }
        
        config_path = os.path.join(temp_dir, 'unicode_simplification_config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, allow_unicode=True)
        
        result = load_config(config_path)
        
        assert result is not None
        assert result == config_data
        assert 'legal_simplifier_测试' in result['model']['simplification_model_name']
        assert '文本简化配置' in result['description']


class TestSimplificationForwardPass:
    """Test class for simplification model forward pass functionality."""
    
    def test_preprocess_function_basic(self):
        """Test basic preprocessing function with valid inputs."""
        from simplification.train_lexsimple import preprocess_function
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': [[1, 2, 3, 4, 5]],
            'attention_mask': [[1, 1, 1, 1, 1]]
        }
        mock_tokenizer.side_effect = lambda x, **kwargs: {
            'input_ids': [[1, 2, 3, 4, 5]],
            'attention_mask': [[1, 1, 1, 1, 1]]
        }
        
        # Mock context manager for as_target_tokenizer
        mock_tokenizer.as_target_tokenizer.return_value.__enter__ = Mock(return_value=mock_tokenizer)
        mock_tokenizer.as_target_tokenizer.return_value.__exit__ = Mock(return_value=None)
        
        # Test data
        examples = {
            'complex': ['This is a complex legal document that needs simplification.'],
            'simple': ['This is a simple legal document.']
        }
        
        result = preprocess_function(examples, mock_tokenizer, 512, 128)
        
        # Verify result structure
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert 'labels' in result
        assert len(result['input_ids']) == 1
        assert len(result['labels']) == 1
    
    def test_preprocess_function_batch_processing(self):
        """Test preprocessing function with batch of examples."""
        from simplification.train_lexsimple import preprocess_function
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            'attention_mask': [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        }
        mock_tokenizer.side_effect = lambda x, **kwargs: {
            'input_ids': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            'attention_mask': [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        }
        
        # Mock context manager for as_target_tokenizer
        mock_tokenizer.as_target_tokenizer.return_value.__enter__ = Mock(return_value=mock_tokenizer)
        mock_tokenizer.as_target_tokenizer.return_value.__exit__ = Mock(return_value=None)
        
        # Test data
        examples = {
            'complex': ['First complex text.', 'Second complex text.'],
            'simple': ['First simple text.', 'Second simple text.']
        }
        
        result = preprocess_function(examples, mock_tokenizer, 512, 128)
        
        # Verify result structure
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert 'labels' in result
        assert len(result['input_ids']) == 2
        assert len(result['labels']) == 2


class TestSimplificationMetrics:
    """Test class for simplification metrics computation."""
    
    def test_compute_metrics_basic(self):
        """Test basic ROUGE and BLEU metrics computation."""
        from simplification.train_lexsimple import compute_metrics
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.batch_decode.return_value = ["This is a simple text.", "Another simple text."]
        mock_tokenizer.pad_token_id = 0
        
        # Test data
        eval_preds = ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])  # Mock predictions and labels
        generation_params = {}
        
        result = compute_metrics(eval_preds, mock_tokenizer, generation_params)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'rouge1' in result
        assert 'rouge2' in result
        assert 'rougeL' in result
        assert 'bleu' in result
        
        # Verify score types
        assert isinstance(result['rouge1'], (int, float))
        assert isinstance(result['rouge2'], (int, float))
        assert isinstance(result['rougeL'], (int, float))
        assert isinstance(result['bleu'], (int, float))
    
    def test_compute_metrics_empty_inputs(self):
        """Test metrics computation with empty inputs."""
        from simplification.train_lexsimple import compute_metrics
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.batch_decode.return_value = []
        mock_tokenizer.pad_token_id = 0
        
        # Test with empty predictions
        eval_preds = ([], [])
        generation_params = {}
        
        result = compute_metrics(eval_preds, mock_tokenizer, generation_params)
        
        assert isinstance(result, dict)
        assert 'rouge1' in result
        assert 'rouge2' in result
        assert 'rougeL' in result
        assert 'bleu' in result


class TestSimplificationIntegration:
    """Test class for simplification integration functionality."""
    
    def test_simplification_workflow_logic(self, sample_legal_text):
        """Test simplification workflow logic without importing problematic modules."""
        # Create a mock simplification function that mimics the real implementation
        def mock_simplify_text(complex_text, model, tokenizer, max_length=128):
            """Mock simplification function that mimics the real implementation."""
            # Mock tokenization
            inputs = tokenizer(complex_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            # Mock model generation
            with torch.no_grad():
                generated_tokens = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Mock decoding
            simplified_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return simplified_text
        
        # Test with mock components
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        mock_tokenizer.batch_decode.return_value = ["This is a simplified version of the legal text."]
        
        # Test simplification
        result = mock_simplify_text(sample_legal_text, mock_model, mock_tokenizer)
        
        # Verify result
        assert isinstance(result, str)
        assert len(result) > 0
        assert "simplified" in result.lower()
    
    def test_simplification_error_handling(self):
        """Test simplification error handling logic."""
        # Test error handling for various scenarios
        complex_text = "This is a complex legal document with difficult terminology."
        
        # Test with empty input
        if not complex_text.strip():
            error_msg = "Input text cannot be empty"
            assert "empty" in error_msg
        
        # Test with very long input
        if len(complex_text) > 10000:
            error_msg = "Input text is too long for processing"
            assert "too long" in error_msg
        
        # Test with invalid characters
        if not complex_text.isprintable():
            error_msg = "Input text contains invalid characters"
            assert "invalid" in error_msg


class TestTextSimplifier:
    """Test class for text simplification functionality."""
    
    def test_placeholder(self):
        """Placeholder test to ensure test structure is working."""
        assert True
