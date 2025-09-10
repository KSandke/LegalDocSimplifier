"""
Unit tests for legal document pipeline functionality.
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
from pipeline.legal_document_pipeline import load_config


class TestPipelineConfigLoading:
    """Test class for pipeline configuration loading functionality."""
    
    def test_load_config_valid_file(self, temp_dir):
        """Test loading a valid pipeline configuration file."""
        config_data = {
            'classification': {
                'model_path': 'models/classification/multitask_model',
                'task_name': 'scotus'
            },
            'summarization': {
                'model_path': 'models/summarization/legal_summarizer',
                'max_length': 250,
                'min_length': 50
            },
            'simplification': {
                'model_path': 'models/simplification/legal_simplifier',
                'level': 'medium',
                'preserve_meaning': True
            },
            'pipeline': {
                'batch_size': 1,
                'device': 'cpu',
                'verbose': True
            }
        }
        
        config_path = os.path.join(temp_dir, 'pipeline_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        result = load_config(config_path)
        
        assert result is not None
        assert result == config_data
        assert 'classification' in result
        assert 'summarization' in result
        assert 'simplification' in result
        assert 'pipeline' in result
        assert result['classification']['task_name'] == 'scotus'
        assert result['summarization']['max_length'] == 250
    
    def test_load_config_missing_file(self):
        """Test loading a non-existent pipeline configuration file."""
        non_existent_path = "non_existent_pipeline_config.yaml"
        
        with pytest.raises(FileNotFoundError):
            load_config(non_existent_path)
    
    def test_load_config_malformed_yaml(self, temp_dir):
        """Test loading a malformed YAML file for pipeline."""
        malformed_yaml = """
        classification:
            model_path: models/classification/multitask_model
            task_name: scotus
        summarization:
            model_path: models/summarization/legal_summarizer
            # Missing colon after this line
            max_length 250
        pipeline:
            batch_size: 1
        """
        
        config_path = os.path.join(temp_dir, 'malformed_pipeline_config.yaml')
        with open(config_path, 'w') as f:
            f.write(malformed_yaml)
        
        with pytest.raises(yaml.YAMLError):
            load_config(config_path)
    
    def test_load_config_empty_file(self, temp_dir):
        """Test loading an empty pipeline configuration file."""
        config_path = os.path.join(temp_dir, 'empty_pipeline_config.yaml')
        with open(config_path, 'w') as f:
            pass  # Create empty file
        
        result = load_config(config_path)
        
        # Empty YAML should return None
        assert result is None
    
    def test_load_config_missing_sections(self, temp_dir):
        """Test loading a config file missing required sections."""
        incomplete_config = {
            'classification': {
                'model_path': 'models/classification/multitask_model'
                # Missing other required keys
            }
            # Missing summarization, simplification, pipeline sections
        }
        
        config_path = os.path.join(temp_dir, 'incomplete_pipeline_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(incomplete_config, f)
        
        result = load_config(config_path)
        
        # Should still load successfully (validation happens later)
        assert result is not None
        assert result == incomplete_config
        assert 'summarization' not in result
        assert 'simplification' not in result
        assert 'pipeline' not in result
    
    def test_load_config_unicode_content(self, temp_dir):
        """Test loading config file with unicode content for pipeline."""
        config_data = {
            'classification': {
                'model_path': 'models/classification/multitask_model',
                'task_name': 'scotus',
                'description': 'Classification model: 分类模型'
            },
            'summarization': {
                'model_path': 'models/summarization/legal_summarizer',
                'description': 'Summarization model: 摘要模型'
            },
            'pipeline': {
                'description': 'Legal document pipeline: 法律文档管道'
            }
        }
        
        config_path = os.path.join(temp_dir, 'unicode_pipeline_config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, allow_unicode=True)
        
        result = load_config(config_path)
        
        assert result is not None
        assert result == config_data
        assert '分类模型' in result['classification']['description']
        assert '摘要模型' in result['summarization']['description']
        assert '法律文档管道' in result['pipeline']['description']
    
    def test_load_config_file_permission_error(self, temp_dir):
        """Test loading config file with permission error."""
        config_path = os.path.join(temp_dir, 'no_permission_pipeline.yaml')
        
        # Create file and remove read permission
        with open(config_path, 'w') as f:
            yaml.dump({'test': 'data'}, f)
        
        # On Unix systems, remove read permission
        if os.name != 'nt':  # Not Windows
            os.chmod(config_path, 0o000)  # No permissions
            
            try:
                with pytest.raises(PermissionError):
                    load_config(config_path)
            finally:
                # Restore permissions for cleanup
                os.chmod(config_path, 0o644)


class TestLegalDocumentPipeline:
    """Test class for legal document pipeline functionality."""
    
    def test_placeholder(self):
        """Placeholder test to ensure test structure is working."""
        assert True
