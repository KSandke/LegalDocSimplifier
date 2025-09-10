"""
Test utilities and helper functions for Legal Document Simplifier tests.

This module contains common utilities used across different test modules.
"""
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch


def create_mock_dataset(num_samples=10, num_classes=3, task_name="test_task"):
    """Create a mock dataset for testing."""
    return {
        'train': [
            {
                'input_text': f'Sample legal text {i}',
                'input_label': i % num_classes,
                'task_name': task_name
            }
            for i in range(num_samples)
        ],
        'test': [
            {
                'input_text': f'Test legal text {i}',
                'input_label': i % num_classes,
                'task_name': task_name
            }
            for i in range(3)
        ]
    }


def create_mock_model_output(logits_shape=(1, 3)):
    """Create mock model output for testing."""
    return [[0.1, 0.8, 0.1] for _ in range(logits_shape[0])]


def create_mock_tokenizer_output(sequence_length=10):
    """Create mock tokenizer output for testing."""
    return {
        'input_ids': torch.randint(1, 1000, (1, sequence_length)),
        'attention_mask': torch.ones(1, sequence_length)
    }


def assert_tensor_close(actual, expected, rtol=1e-5, atol=1e-8):
    """Assert that two tensors are close within tolerance."""
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        assert torch.allclose(actual, expected, rtol=rtol, atol=atol)
    else:
        assert actual == expected


def create_temp_config_file(config_dict, filename="test_config.yaml"):
    """Create a temporary configuration file for testing."""
    import yaml
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        return f.name


def cleanup_temp_file(filepath):
    """Clean up a temporary file."""
    try:
        os.unlink(filepath)
    except OSError:
        pass


class MockModel:
    """Mock model class for testing."""
    
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.training = False
    
    def forward(self, input_ids, attention_mask, task_name=None):
        """Mock forward pass."""
        batch_size = input_ids.shape[0]
        return torch.randn(batch_size, self.num_classes)
    
    def eval(self):
        """Mock eval mode."""
        self.training = False
        return self
    
    def train(self):
        """Mock train mode."""
        self.training = True
        return self
    
    def to(self, device):
        """Mock device transfer."""
        return self
    
    def cpu(self):
        """Mock CPU transfer."""
        return self


class MockTokenizer:
    """Mock tokenizer class for testing."""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 1
        self.name_or_path = "test-tokenizer"
    
    def __call__(self, text, **kwargs):
        """Mock tokenizer call."""
        if isinstance(text, str):
            text = [text]
        
        max_length = kwargs.get('max_length', 512)
        padding = kwargs.get('padding', False)
        
        # Create mock token IDs
        input_ids = []
        attention_mask = []
        
        for t in text:
            # Simple tokenization simulation
            tokens = t.split()[:max_length]
            token_ids = [hash(token) % self.vocab_size for token in tokens]
            
            if padding and len(token_ids) < max_length:
                token_ids.extend([self.pad_token_id] * (max_length - len(token_ids)))
            
            input_ids.append(token_ids)
            attention_mask.append([1] * len(tokens) + [0] * (max_length - len(tokens)))
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask)
        }
    
    def save_pretrained(self, path):
        """Mock save pretrained."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'tokenizer_config.json'), 'w') as f:
            json.dump({'vocab_size': self.vocab_size}, f)
