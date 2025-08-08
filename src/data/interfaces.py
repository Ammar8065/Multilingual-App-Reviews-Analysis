"""
Base interfaces and abstract classes for the data processing components.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from src.data.models import (
    ReviewRecord, ValidationResult, ValidationError, ProcessingResult,
    SentimentResult, TimeSeriesResult, ForecastResult, GeographicVisualization,
    ComparisonResult, ClassificationModel, RegressionModel, ModelPerformance
)


class DataLoaderInterface(ABC):
    """Interface for data loading operations."""
    
    @abstractmethod
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV file and return DataFrame."""
        pass
    
    @abstractmethod
    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding."""
        pass
    
    @abstractmethod
    def validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """Validate DataFrame schema."""
        pass


class DataValidatorInterface(ABC):
    """Interface for data validation operations."""
    
    @abstractmethod
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """Check for missing values in each column."""
        pass
    
    @abstractmethod
    def identify_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify duplicate records."""
        pass
    
    @abstractmethod
    def validate_data_types(self, df: pd.DataFrame) -> List[ValidationError]:
        """Validate data types for each column."""
        pass


class TextPreprocessorInterface(ABC):
    """Interface for text preprocessing operations."""
    
    @abstractmethod
    def clean_text(self, text: str, language: str) -> str:
        """Clean and normalize text."""
        pass
    
    @abstractmethod
    def normalize_encoding(self, text: str) -> str:
        """Normalize text encoding."""
        pass
    
    @abstractmethod
    def tokenize_multilingual(self, text: str, language: str) -> List[str]:
        """Tokenize text for specific language."""
        pass


class LanguageDetectorInterface(ABC):
    """Interface for language detection operations."""
    
    @abstractmethod
    def detect_language(self, text: str) -> str:
        """Detect language of text."""
        pass
    
    @abstractmethod
    def validate_language_code(self, lang_code: str) -> bool:
        """Validate language code format."""
        pass
    
    @abstractmethod
    def standardize_language_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize language codes in DataFrame."""
        pass


class EDAAnalyzerInterface(ABC):
    """Interface for exploratory data analysis operations."""
    
    @abstractmethod
    def generate_dataset_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive dataset overview."""
        pass
    
    @abstractmethod
    def analyze_language_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distribution of languages."""
        pass
    
    @abstractmethod
    def analyze_rating_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze rating patterns and distributions."""
        pass


class SentimentAnalyzerInterface(ABC):
    """Interface for sentiment analysis operations."""
    
    @abstractmethod
    def analyze_sentiment(self, text: str, language: str) -> SentimentResult:
        """Analyze sentiment of single text."""
        pass
    
    @abstractmethod
    def batch_sentiment_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform batch sentiment analysis."""
        pass
    
    @abstractmethod
    def compare_sentiment_by_language(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare sentiment across languages."""
        pass


class TextClassifierInterface(ABC):
    """Interface for text classification operations."""
    
    @abstractmethod
    def classify_review_topics(self, text: str, language: str) -> List[str]:
        """Classify review into topics."""
        pass
    
    @abstractmethod
    def train_multilingual_classifier(self, df: pd.DataFrame) -> ClassificationModel:
        """Train multilingual classification model."""
        pass


class RatingPredictorInterface(ABC):
    """Interface for rating prediction operations."""
    
    @abstractmethod
    def train_rating_model(self, df: pd.DataFrame) -> RegressionModel:
        """Train rating prediction model."""
        pass
    
    @abstractmethod
    def predict_ratings(self, texts: List[str], languages: List[str]) -> List[float]:
        """Predict ratings for texts."""
        pass
    
    @abstractmethod
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> ModelPerformance:
        """Evaluate model performance."""
        pass


class ModelEvaluatorInterface(ABC):
    """Interface for model evaluation operations."""
    
    @abstractmethod
    def evaluate_regression(self, y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """Evaluate regression model performance."""
        pass
    
    @abstractmethod
    def evaluate_classification(self, y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        """Evaluate classification model performance."""
        pass
    
    @abstractmethod
    def cross_validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform cross-validation on model."""
        pass


class TimeSeriesAnalyzerInterface(ABC):
    """Interface for time series analysis operations."""
    
    @abstractmethod
    def analyze_review_trends(self, df: pd.DataFrame) -> TimeSeriesResult:
        """Analyze trends in review data."""
        pass
    
    @abstractmethod
    def detect_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect seasonal patterns in data."""
        pass
    
    @abstractmethod
    def forecast_future_trends(self, df: pd.DataFrame, periods: int) -> ForecastResult:
        """Forecast future trends."""
        pass


class GeographicAnalyzerInterface(ABC):
    """Interface for geographic analysis operations."""
    
    @abstractmethod
    def analyze_sentiment_by_country(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sentiment by country."""
        pass
    
    @abstractmethod
    def create_sentiment_map(self, df: pd.DataFrame) -> GeographicVisualization:
        """Create geographic sentiment visualization."""
        pass
    
    @abstractmethod
    def compare_regional_patterns(self, df: pd.DataFrame) -> ComparisonResult:
        """Compare patterns across regions."""
        pass


class VisualizationEngineInterface(ABC):
    """Interface for visualization operations."""
    
    @abstractmethod
    def create_distribution_plots(self, df: pd.DataFrame) -> List[Any]:
        """Create distribution plots."""
        pass
    
    @abstractmethod
    def create_time_series_plots(self, df: pd.DataFrame) -> List[Any]:
        """Create time series plots."""
        pass
    
    @abstractmethod
    def create_geographic_maps(self, df: pd.DataFrame) -> List[Any]:
        """Create geographic maps."""
        pass
    
    @abstractmethod
    def create_correlation_heatmaps(self, df: pd.DataFrame) -> Any:
        """Create correlation heatmaps."""
        pass