"""
Basic tests to verify the testing infrastructure is working.

These tests don't require any external dependencies and can be run
to verify the test setup is correct.
"""
import pytest
import os
import sys
from pathlib import Path


class TestBasicSetup:
    """Test basic testing infrastructure setup."""
    
    def test_project_structure(self):
        """Test that the project structure is correct."""
        project_root = Path(__file__).parent.parent
        assert project_root.exists()
        assert (project_root / "src").exists()
        assert (project_root / "tests").exists()
        assert (project_root / "requirements.txt").exists()
        assert (project_root / "pytest.ini").exists()
    
    def test_test_structure(self):
        """Test that the test directory structure is correct."""
        tests_dir = Path(__file__).parent
        assert tests_dir.exists()
        assert (tests_dir / "conftest.py").exists()
        assert (tests_dir / "test_utils.py").exists()
        assert (tests_dir / "classification").exists()
        assert (tests_dir / "summarization").exists()
        assert (tests_dir / "simplification").exists()
        assert (tests_dir / "pipeline").exists()
    
    def test_test_files_exist(self):
        """Test that all test files exist."""
        tests_dir = Path(__file__).parent
        
        expected_files = [
            "classification/test_classification_models.py",
            "classification/test_classification_inference.py",
            "summarization/test_abstractive_summarizer.py",
            "summarization/test_extractive_summarizer.py",
            "simplification/test_text_simplifier.py",
            "pipeline/test_legal_document_pipeline.py"
        ]
        
        for file_path in expected_files:
            assert (tests_dir / file_path).exists(), f"Missing test file: {file_path}"
    
    def test_pytest_config(self):
        """Test that pytest configuration is valid."""
        pytest_ini = Path(__file__).parent.parent / "pytest.ini"
        assert pytest_ini.exists()
        
        with open(pytest_ini, 'r') as f:
            content = f.read()
            assert "[tool:pytest]" in content
            assert "testpaths = tests" in content
            assert "addopts" in content
    
    def test_requirements_updated(self):
        """Test that requirements.txt includes testing dependencies."""
        requirements_file = Path(__file__).parent.parent / "requirements.txt"
        assert requirements_file.exists()
        
        with open(requirements_file, 'r') as f:
            content = f.read()
            assert "pytest" in content
            assert "pytest-cov" in content
            assert "pytest-mock" in content
    
    def test_run_tests_script(self):
        """Test that the test runner script exists and is executable."""
        run_tests_script = Path(__file__).parent.parent / "run_tests.py"
        assert run_tests_script.exists()
        assert os.access(run_tests_script, os.X_OK)
    
    def test_placeholder_tests(self):
        """Test that placeholder tests are working."""
        # This is a simple test to verify pytest is working
        assert True
        assert 1 + 1 == 2
        assert "test" in "testing"
    
    @pytest.mark.parametrize("test_input,expected", [
        (1, 1),
        (2, 2),
        (3, 3),
    ])
    def test_parametrized(self, test_input, expected):
        """Test parametrized testing works."""
        assert test_input == expected
