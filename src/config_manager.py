"""
Centralized Configuration Management for Legal Document Simplifier

This module provides a unified interface for loading and validating configuration
across all modules in the project. It standardizes error handling, logging, and
provides consistent configuration access patterns.

Usage:
    from src.config_manager import ConfigManager
    
    # Load a specific config file
    config = ConfigManager.load_config('config/classification.yaml')
    
    # Load with validation
    config = ConfigManager.load_config('config/classification.yaml', validate=True)
    
    # Load a specific section
    model_config = ConfigManager.load_section('config/classification.yaml', 'model')
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Custom exception for configuration-related errors."""
    pass


class ConfigManager:
    """Centralized configuration management for the Legal Document Simplifier project."""
    
    # Cache for loaded configurations to avoid repeated file reads
    _config_cache: Dict[str, Dict[str, Any]] = {}
    
    # Default configuration paths
    DEFAULT_CONFIGS = {
        'classification': 'config/classification.yaml',
        'summarization': 'config/summarization.yaml', 
        'simplification': 'config/simplification.yaml',
        'pipeline': 'config/summarization.yaml'  # Pipeline uses summarization config
    }
    
    @classmethod
    def load_config(cls, config_path: str, validate: bool = True, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load configuration from a YAML file with standardized error handling.
        
        Args:
            config_path: Path to the configuration file
            validate: Whether to validate the configuration structure
            use_cache: Whether to use cached configuration if available
            
        Returns:
            Dictionary containing the configuration
            
        Raises:
            ConfigError: If configuration cannot be loaded or is invalid
        """
        # Normalize path
        config_path = os.path.abspath(config_path)
        
        # Check cache first
        if use_cache and config_path in cls._config_cache:
            logger.debug(f"Using cached configuration from {config_path}")
            return cls._config_cache[config_path]
        
        # Check if file exists
        if not os.path.exists(config_path):
            error_msg = f"Configuration file not found: {config_path}"
            logger.error(error_msg)
            raise ConfigError(error_msg)
        
        try:
            # Load YAML file
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                error_msg = f"Configuration file is empty or invalid: {config_path}"
                logger.error(error_msg)
                raise ConfigError(error_msg)
            
            # Validate configuration if requested
            if validate:
                cls._validate_config(config, config_path)
            
            # Cache the configuration
            if use_cache:
                cls._config_cache[config_path] = config
            
            logger.info(f"Configuration loaded successfully from {config_path}")
            return config
            
        except yaml.YAMLError as e:
            error_msg = f"Invalid YAML syntax in {config_path}: {e}"
            logger.error(error_msg)
            raise ConfigError(error_msg)
        except Exception as e:
            error_msg = f"Error loading configuration from {config_path}: {e}"
            logger.error(error_msg)
            raise ConfigError(error_msg)
    
    @classmethod
    def load_section(cls, config_path: str, section: str, validate: bool = True) -> Dict[str, Any]:
        """
        Load a specific section from a configuration file.
        
        Args:
            config_path: Path to the configuration file
            section: Name of the section to load
            validate: Whether to validate the configuration
            
        Returns:
            Dictionary containing the section configuration
            
        Raises:
            ConfigError: If section is not found or configuration cannot be loaded
        """
        config = cls.load_config(config_path, validate=validate)
        
        if section not in config:
            error_msg = f"Section '{section}' not found in {config_path}. Available sections: {list(config.keys())}"
            logger.error(error_msg)
            raise ConfigError(error_msg)
        
        section_config = config[section]
        if not isinstance(section_config, dict):
            error_msg = f"Section '{section}' is not a dictionary in {config_path}"
            logger.error(error_msg)
            raise ConfigError(error_msg)
        
        logger.debug(f"Loaded section '{section}' from {config_path}")
        return section_config
    
    @classmethod
    def get_default_config_path(cls, config_type: str) -> str:
        """
        Get the default configuration path for a given type.
        
        Args:
            config_type: Type of configuration (classification, summarization, etc.)
            
        Returns:
            Default path for the configuration type
            
        Raises:
            ConfigError: If config_type is not recognized
        """
        if config_type not in cls.DEFAULT_CONFIGS:
            available_types = list(cls.DEFAULT_CONFIGS.keys())
            error_msg = f"Unknown config type '{config_type}'. Available types: {available_types}"
            logger.error(error_msg)
            raise ConfigError(error_msg)
        
        return cls.DEFAULT_CONFIGS[config_type]
    
    @classmethod
    def load_default_config(cls, config_type: str, validate: bool = True) -> Dict[str, Any]:
        """
        Load the default configuration for a given type.
        
        Args:
            config_type: Type of configuration to load
            validate: Whether to validate the configuration
            
        Returns:
            Dictionary containing the configuration
        """
        config_path = cls.get_default_config_path(config_type)
        return cls.load_config(config_path, validate=validate)
    
    @classmethod
    def _validate_config(cls, config: Dict[str, Any], config_path: str) -> None:
        """
        Validate configuration structure and required fields.
        
        Args:
            config: Configuration dictionary to validate
            config_path: Path to the configuration file (for error messages)
            
        Raises:
            ConfigError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ConfigError(f"Configuration must be a dictionary in {config_path}")
        
        # Basic validation - check for common required sections
        required_sections = ['paths', 'model']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            logger.warning(f"Missing recommended sections in {config_path}: {missing_sections}")
        
        # Validate paths section if present
        if 'paths' in config:
            paths = config['paths']
            if not isinstance(paths, dict):
                raise ConfigError(f"'paths' section must be a dictionary in {config_path}")
        
        # Validate model section if present
        if 'model' in config:
            model = config['model']
            if not isinstance(model, dict):
                raise ConfigError(f"'model' section must be a dictionary in {config_path}")
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the configuration cache."""
        cls._config_cache.clear()
        logger.debug("Configuration cache cleared")
    
    @classmethod
    def get_cache_info(cls) -> Dict[str, int]:
        """Get information about the configuration cache."""
        return {
            'cached_configs': len(cls._config_cache),
            'cache_keys': list(cls._config_cache.keys())
        }


# Convenience functions for backward compatibility
def load_config(config_path: str, validate: bool = True) -> Dict[str, Any]:
    """
    Convenience function for loading configuration.
    
    This function provides backward compatibility with existing code.
    """
    return ConfigManager.load_config(config_path, validate=validate)


def load_section(config_path: str, section: str, validate: bool = True) -> Dict[str, Any]:
    """
    Convenience function for loading a configuration section.
    
    This function provides backward compatibility with existing code.
    """
    return ConfigManager.load_section(config_path, section, validate=validate)
