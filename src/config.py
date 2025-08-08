"""
Configuration management for the multilingual app reviews analysis system.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Data processing configuration."""
    input_file_path: str = "multilingual_mobile_app_reviews_2025.csv"
    output_dir: str = "output"
    cache_dir: str = "cache"
    encoding: str = "utf-8"
    chunk_size: int = 1000
    max_text_length: int = 1000
    min_text_length: int = 5


@dataclass
class MLConfig:
    """Machine learning configuration."""
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    cv_folds: int = 5
    max_features: int = 10000
    n_jobs: int = -1
    
    # Model-specific parameters
    rating_prediction_models: List[str] = None
    sentiment_models: List[str] = None
    classification_models: List[str] = None
    
    def __post_init__(self):
        if self.rating_prediction_models is None:
            self.rating_prediction_models = ["linear_regression", "random_forest", "xgboost"]
        if self.sentiment_models is None:
            self.sentiment_models = ["textblob", "vader", "transformers"]
        if self.classification_models is None:
            self.classification_models = ["naive_bayes", "svm", "random_forest"]


@dataclass
class NLPConfig:
    """Natural language processing configuration."""
    supported_languages: List[str] = None
    default_language: str = "en"
    language_detection_threshold: float = 0.8
    sentiment_confidence_threshold: float = 0.6
    max_tokens: int = 512
    
    # Model names for different NLP tasks
    multilingual_model: str = "distilbert-base-multilingual-cased"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = [
                "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko",
                "ar", "hi", "th", "vi", "nl", "sv", "da", "no", "fi", "pl",
                "tr", "ms", "tl", "id"
            ]


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    figure_width: int = 1200
    figure_height: int = 800
    color_palette: List[str] = None
    theme: str = "plotly_white"
    font_family: str = "Arial"
    font_size: int = 12
    
    # Geographic visualization
    map_style: str = "open-street-map"
    map_zoom: int = 2
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
            ]


@dataclass
class ProcessingConfig:
    """Data processing configuration."""
    batch_size: int = 500
    parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    cache_results: bool = True
    log_level: str = "INFO"
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 1.0
    continue_on_error: bool = True


@dataclass
class OutputConfig:
    """Output configuration."""
    save_intermediate_results: bool = True
    export_formats: List[str] = None
    report_template: str = "default"
    dashboard_port: int = 8050
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["html", "pdf", "json"]


class Config:
    """Main configuration class that combines all configuration sections."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.data = DataConfig()
        self.ml = MLConfig()
        self.nlp = NLPConfig()
        self.visualization = VisualizationConfig()
        self.processing = ProcessingConfig()
        self.output = OutputConfig()
        
        # Set up directories
        self.project_root = Path.cwd()
        self.data_dir = self.project_root / "data"
        self.output_dir = self.project_root / self.data.output_dir
        self.cache_dir = self.project_root / self.data.cache_dir
        self.models_dir = self.project_root / "models"
        self.reports_dir = self.output_dir / "reports"
        self.visualizations_dir = self.output_dir / "visualizations"
        
        # Create directories if they don't exist
        self._create_directories()
        
        # Load custom configuration if provided
        if config_file:
            self.load_from_file(config_file)
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.data_dir,
            self.output_dir,
            self.cache_dir,
            self.models_dir,
            self.reports_dir,
            self.visualizations_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_from_file(self, config_file: str):
        """Load configuration from a file (JSON/YAML)."""
        # TODO: Implement configuration file loading
        pass
    
    def save_to_file(self, config_file: str):
        """Save current configuration to a file."""
        # TODO: Implement configuration file saving
        pass
    
    def get_data_file_path(self) -> Path:
        """Get the full path to the input data file."""
        return self.project_root / self.data.input_file_path
    
    def get_output_path(self, filename: str) -> Path:
        """Get the full path for an output file."""
        return self.output_dir / filename
    
    def get_cache_path(self, filename: str) -> Path:
        """Get the full path for a cache file."""
        return self.cache_dir / filename
    
    def get_model_path(self, model_name: str) -> Path:
        """Get the full path for a model file."""
        return self.models_dir / f"{model_name}.pkl"


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs):
    """Update configuration parameters."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # Try to find the attribute in sub-configurations
            for attr_name in ['data', 'ml', 'nlp', 'visualization', 'processing', 'output']:
                attr = getattr(config, attr_name)
                if hasattr(attr, key):
                    setattr(attr, key, value)
                    break