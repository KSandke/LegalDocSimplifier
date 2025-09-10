"""
Unit tests for classification models and training functionality.
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
from classification.train_multitask_classifier import load_config


class TestConfigurationLoading:
    """Test class for configuration loading functionality."""
    
    def test_load_config_valid_file(self, temp_dir):
        """Test loading a valid configuration file."""
        # Create a valid config file
        config_data = {
            'paths': {
                'raw_data_dir': 'data/processed',
                'standardized_data_dir': 'data/standardized',
                'label_counts_file': 'data/standardized/task_label_counts.json',
                'output_dir_template': 'models/classification/{model_name}'
            },
            'datasets': {
                'multi_task_classification': ['scotus', 'ledgar', 'unfair_tos']
            },
            'model': {
                'multi_task_classification': {
                    'name': 'test_model',
                    'base_model': 'nlpaueb/legal-bert-base-uncased'
                }
            },
            'training': {
                'multi_task_classification': {
                    'num_epochs': 3,
                    'batch_size': 16,
                    'learning_rate': 2e-5
                }
            }
        }
        
        config_path = os.path.join(temp_dir, 'test_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Test loading
        result = load_config(config_path)
        
        # Assertions
        assert result is not None
        assert result == config_data
        assert 'paths' in result
        assert 'datasets' in result
        assert 'model' in result
        assert 'training' in result
        assert result['datasets']['multi_task_classification'] == ['scotus', 'ledgar', 'unfair_tos']
    
    def test_load_config_missing_file(self):
        """Test loading a non-existent configuration file."""
        non_existent_path = "non_existent_config.yaml"
        
        result = load_config(non_existent_path)
        
        assert result is None
    
    def test_load_config_malformed_yaml(self, temp_dir):
        """Test loading a malformed YAML file."""
        # Create truly malformed YAML with syntax errors
        malformed_yaml = """
        paths:
            raw_data_dir: data/processed
            standardized_data_dir: data/standardized
        model:
            multi_task_classification:
                name: test_model
                base_model: nlpaueb/legal-bert-base-uncased
                # Missing colon after this line
                num_epochs 3
        training:
            multi_task_classification:
                num_epochs: 3
        """
        
        config_path = os.path.join(temp_dir, 'malformed_config.yaml')
        with open(config_path, 'w') as f:
            f.write(malformed_yaml)
        
        result = load_config(config_path)
        
        assert result is None
    
    def test_load_config_empty_file(self, temp_dir):
        """Test loading an empty configuration file."""
        config_path = os.path.join(temp_dir, 'empty_config.yaml')
        with open(config_path, 'w') as f:
            pass  # Create empty file
        
        result = load_config(config_path)
        
        assert result is None
    
    def test_load_config_missing_required_keys(self, temp_dir):
        """Test loading a config file missing required keys."""
        # Create config missing required keys
        incomplete_config = {
            'paths': {
                'raw_data_dir': 'data/processed'
                # Missing other required path keys
            }
            # Missing datasets, model, training sections
        }
        
        config_path = os.path.join(temp_dir, 'incomplete_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(incomplete_config, f)
        
        result = load_config(config_path)
        
        # Should still load successfully (validation happens later)
        assert result is not None
        assert result == incomplete_config
        assert 'datasets' not in result
        assert 'model' not in result
    
    def test_load_config_default_path(self, temp_dir):
        """Test loading config with default path when file doesn't exist."""
        # Change to temp directory to avoid loading real config
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            result = load_config()  # Should use default path
            assert result is None  # Should return None since file doesn't exist
        finally:
            os.chdir(original_cwd)
    
    def test_load_config_permission_error(self, temp_dir):
        """Test loading config file with permission error."""
        config_path = os.path.join(temp_dir, 'no_permission.yaml')
        
        # Create file and remove read permission
        with open(config_path, 'w') as f:
            yaml.dump({'test': 'data'}, f)
        
        # On Unix systems, remove read permission
        if os.name != 'nt':  # Not Windows
            os.chmod(config_path, 0o000)  # No permissions
            
            try:
                result = load_config(config_path)
                assert result is None
            finally:
                # Restore permissions for cleanup
                os.chmod(config_path, 0o644)
    
    def test_load_config_unicode_content(self, temp_dir):
        """Test loading config file with unicode content."""
        config_data = {
            'paths': {
                'raw_data_dir': 'data/processed',
                'standardized_data_dir': 'data/standardized',
                'label_counts_file': 'data/standardized/task_label_counts.json',
                'output_dir_template': 'models/classification/{model_name}'
            },
            'datasets': {
                'multi_task_classification': ['scotus', 'ledgar', 'unfair_tos']
            },
            'model': {
                'multi_task_classification': {
                    'name': 'test_model_unicode_测试',
                    'base_model': 'nlpaueb/legal-bert-base-uncased'
                }
            },
            'description': 'Test configuration with unicode: 测试配置'
        }
        
        config_path = os.path.join(temp_dir, 'unicode_config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, allow_unicode=True)
        
        result = load_config(config_path)
        
        assert result is not None
        assert result == config_data
        assert '测试配置' in result['description']


class TestLegalMultiTaskModel:
    """Test class for LegalMultiTaskModel forward pass functionality."""
    
    def test_model_initialization(self):
        """Test that the model initializes correctly with given parameters."""
        from classification.train_multitask_classifier import LegalMultiTaskModel
        
        # Mock task labels
        task_labels = {
            'scotus': 3,
            'ledgar': 5,
            'unfair_tos': 2
        }
        
        # Create model with mocked encoder
        with patch('transformers.AutoModel.from_pretrained') as mock_encoder:
            # Mock encoder config
            mock_encoder.return_value.config.hidden_size = 768
            
            model = LegalMultiTaskModel('nlpaueb/legal-bert-base-uncased', task_labels)
            
            # Verify model structure
            assert hasattr(model, 'encoder')
            assert hasattr(model, 'task_classifiers')
            assert hasattr(model, 'task_labels')
            assert model.task_labels == task_labels
            assert len(model.task_classifiers) == 3
            assert 'scotus' in model.task_classifiers
            assert 'ledgar' in model.task_classifiers
            assert 'unfair_tos' in model.task_classifiers
    
    def test_forward_single_task(self, mock_model, mock_tokenizer):
        """Test forward pass with a specific task name."""
        from classification.train_multitask_classifier import LegalMultiTaskModel
        
        # Create model with mocked components
        task_labels = {'scotus': 3, 'ledgar': 5}
        
        with patch('transformers.AutoModel.from_pretrained') as mock_encoder:
            mock_encoder.return_value.config.hidden_size = 768
            model = LegalMultiTaskModel('nlpaueb/legal-bert-base-uncased', task_labels)
            
            # Mock the encoder forward pass
            mock_encoder_output = Mock()
            mock_encoder_output.last_hidden_state = torch.randn(2, 10, 768)  # [batch_size, seq_len, hidden_size]
            mock_encoder.return_value.return_value = mock_encoder_output
            
            # Create input tensors
            input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
            attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
            
            # Test forward pass for specific task
            output = model(input_ids, attention_mask, task_name='scotus')
            
            # Verify output shape
            assert output.shape == (2, 3)  # [batch_size, num_labels]
            assert isinstance(output, torch.Tensor)
    
    def test_forward_all_tasks(self, mock_model, mock_tokenizer):
        """Test forward pass without specifying task (returns all tasks)."""
        from classification.train_multitask_classifier import LegalMultiTaskModel
        
        task_labels = {'scotus': 3, 'ledgar': 5, 'unfair_tos': 2}
        
        with patch('transformers.AutoModel.from_pretrained') as mock_encoder:
            mock_encoder.return_value.config.hidden_size = 768
            model = LegalMultiTaskModel('nlpaueb/legal-bert-base-uncased', task_labels)
            
            # Mock the encoder forward pass
            mock_encoder_output = Mock()
            mock_encoder_output.last_hidden_state = torch.randn(2, 10, 768)
            mock_encoder.return_value.return_value = mock_encoder_output
            
            # Create input tensors
            input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
            attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
            
            # Test forward pass for all tasks
            output = model(input_ids, attention_mask)
            
            # Verify output structure
            assert isinstance(output, dict)
            assert len(output) == 3
            assert 'scotus' in output
            assert 'ledgar' in output
            assert 'unfair_tos' in output
            
            # Verify each task output shape
            assert output['scotus'].shape == (2, 3)
            assert output['ledgar'].shape == (2, 5)
            assert output['unfair_tos'].shape == (2, 2)
    
    def test_forward_invalid_task(self, mock_model, mock_tokenizer):
        """Test forward pass with invalid task name raises error."""
        from classification.train_multitask_classifier import LegalMultiTaskModel
        
        task_labels = {'scotus': 3, 'ledgar': 5}
        
        with patch('transformers.AutoModel.from_pretrained') as mock_encoder:
            mock_encoder.return_value.config.hidden_size = 768
            model = LegalMultiTaskModel('nlpaueb/legal-bert-base-uncased', task_labels)
            
            # Mock the encoder forward pass
            mock_encoder_output = Mock()
            mock_encoder_output.last_hidden_state = torch.randn(2, 10, 768)
            mock_encoder.return_value.return_value = mock_encoder_output
            
            # Create input tensors
            input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
            attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
            
            # Test with invalid task name
            with pytest.raises(ValueError, match="Task 'invalid_task' not found"):
                model(input_ids, attention_mask, task_name='invalid_task')
    
    def test_forward_different_batch_sizes(self, mock_model, mock_tokenizer):
        """Test forward pass with different batch sizes."""
        from classification.train_multitask_classifier import LegalMultiTaskModel
        
        task_labels = {'scotus': 3}
        
        with patch('transformers.AutoModel.from_pretrained') as mock_encoder:
            mock_encoder.return_value.config.hidden_size = 768
            model = LegalMultiTaskModel('nlpaueb/legal-bert-base-uncased', task_labels)
            
            # Mock the encoder forward pass
            mock_encoder_output = Mock()
            mock_encoder.return_value.return_value = mock_encoder_output
            
            # Test batch size 1
            mock_encoder_output.last_hidden_state = torch.randn(1, 10, 768)
            input_ids = torch.tensor([[1, 2, 3, 4, 5]])
            attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
            output = model(input_ids, attention_mask, task_name='scotus')
            assert output.shape == (1, 3)
            
            # Test batch size 4
            mock_encoder_output.last_hidden_state = torch.randn(4, 10, 768)
            input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]])
            attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
            output = model(input_ids, attention_mask, task_name='scotus')
            assert output.shape == (4, 3)
    
    def test_forward_different_sequence_lengths(self, mock_model, mock_tokenizer):
        """Test forward pass with different sequence lengths."""
        from classification.train_multitask_classifier import LegalMultiTaskModel
        
        task_labels = {'scotus': 3}
        
        with patch('transformers.AutoModel.from_pretrained') as mock_encoder:
            mock_encoder.return_value.config.hidden_size = 768
            model = LegalMultiTaskModel('nlpaueb/legal-bert-base-uncased', task_labels)
            
            # Mock the encoder forward pass
            mock_encoder_output = Mock()
            mock_encoder.return_value.return_value = mock_encoder_output
            
            # Test short sequence
            mock_encoder_output.last_hidden_state = torch.randn(2, 5, 768)
            input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
            attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
            output = model(input_ids, attention_mask, task_name='scotus')
            assert output.shape == (2, 3)
            
            # Test long sequence
            mock_encoder_output.last_hidden_state = torch.randn(2, 512, 768)
            input_ids = torch.tensor([[1] * 512, [2] * 512])
            attention_mask = torch.tensor([[1] * 512, [1] * 512])
            output = model(input_ids, attention_mask, task_name='scotus')
            assert output.shape == (2, 3)
    
    def test_forward_attention_mask_handling(self, mock_model, mock_tokenizer):
        """Test forward pass with different attention mask patterns."""
        from classification.train_multitask_classifier import LegalMultiTaskModel
        
        task_labels = {'scotus': 3}
        
        with patch('transformers.AutoModel.from_pretrained') as mock_encoder:
            mock_encoder.return_value.config.hidden_size = 768
            model = LegalMultiTaskModel('nlpaueb/legal-bert-base-uncased', task_labels)
            
            # Mock the encoder forward pass
            mock_encoder_output = Mock()
            mock_encoder_output.last_hidden_state = torch.randn(2, 10, 768)
            mock_encoder.return_value.return_value = mock_encoder_output
            
            # Test with padding (attention mask with zeros)
            input_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0], [6, 7, 8, 9, 10, 11, 12, 0, 0, 0]])
            attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
            output = model(input_ids, attention_mask, task_name='scotus')
            assert output.shape == (2, 3)
    
    def test_forward_gradient_flow(self, mock_model, mock_tokenizer):
        """Test that gradients flow correctly through the model."""
        from classification.train_multitask_classifier import LegalMultiTaskModel
        
        task_labels = {'scotus': 3}
        
        with patch('transformers.AutoModel.from_pretrained') as mock_encoder:
            mock_encoder.return_value.config.hidden_size = 768
            model = LegalMultiTaskModel('nlpaueb/legal-bert-base-uncased', task_labels)
            
            # Mock the encoder forward pass
            mock_encoder_output = Mock()
            mock_encoder_output.last_hidden_state = torch.randn(2, 10, 768, requires_grad=True)
            mock_encoder.return_value.return_value = mock_encoder_output
            
            # Create input tensors (input_ids are integers, no gradients needed)
            input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
            attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
            
            # Forward pass
            output = model(input_ids, attention_mask, task_name='scotus')
            
            # Test gradient computation
            loss = output.sum()
            loss.backward()
            
            # Verify gradients exist in the model parameters
            assert output.requires_grad
            # Check that model parameters have gradients
            for param in model.parameters():
                if param.requires_grad:
                    assert param.grad is not None


class TestClassificationModels:
    """Test class for classification model functionality."""
    
    def test_placeholder(self):
        """Placeholder test to ensure test structure is working."""
        assert True
