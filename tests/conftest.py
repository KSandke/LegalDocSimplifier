"""
Simplified pytest configuration and shared fixtures for Legal Document Simplifier tests.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import torch


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_legal_text():
    """Sample legal text for testing."""
    return "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of law principles."


@pytest.fixture
def sample_legal_document():
    """Sample legal document for testing."""
    return """
    WHEREAS, the Company desires to enter into this Agreement to provide services;
    WHEREAS, the Client desires to engage the Company for such services;
    NOW, THEREFORE, in consideration of the mutual covenants contained herein, the parties agree as follows:
    1. The Company shall provide consulting services as described in Schedule A.
    2. The Client shall pay the Company according to the terms set forth in Schedule B.
    3. This Agreement shall be governed by the laws of the State of Delaware.
    """


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    return {
        'train': [
            {'input_text': 'Sample legal text 1', 'input_label': 0, 'task_name': 'scotus'},
            {'input_text': 'Sample legal text 2', 'input_label': 1, 'task_name': 'ledgar'}
        ],
        'test': [
            {'input_text': 'Test legal text 1', 'input_label': 0, 'task_name': 'scotus'}
        ]
    }


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
    }
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.bos_token_id = 1
    tokenizer.name_or_path = "test-tokenizer"
    return tokenizer


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = Mock()
    model.forward.return_value = torch.tensor([[0.1, 0.9, 0.0]])  # Mock logits
    model.eval.return_value = model
    model.train.return_value = model
    model.to.return_value = model
    model.cpu.return_value = model
    return model


@pytest.fixture
def mock_pipeline():
    """Mock Hugging Face pipeline for testing."""
    pipeline = Mock()
    pipeline.return_value = [{'summary_text': 'This is a test summary.'}]
    return pipeline


@pytest.fixture(autouse=True)
def mock_cuda_availability():
    """Mock CUDA availability to ensure consistent testing."""
    with patch('torch.cuda.is_available', return_value=False):
        with patch('torch.cuda.device_count', return_value=0):
            yield