"""
Unit tests for classification inference functionality.
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


class TestLegalMultiTaskModelForwardPass:
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


class TestClassificationInference:
    """Test class for classification inference functionality."""
    
    def test_placeholder(self):
        """Placeholder test to ensure test structure is working."""
        assert True
