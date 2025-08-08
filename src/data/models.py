"""
Core data models for the multilingual app reviews analysis system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np
from plotly.graph_objects import Figure


@dataclass
class ReviewRecord:
    """Core data structure representing a single app review."""
    review_id: int
    user_id: int
    app_name: str
    app_category: str
    review_text: str
    review_language: str
    rating: float
    review_date: datetime
    verified_purchase: bool
    device_type: str
    num_helpful_votes: int
    user_age: Optional[float]
    user_country: str
    user_gender: Optional[str]
    app_version: str


@dataclass
class SentimentResult:
    """Result of sentiment analysis on a review."""
    sentiment: str  # 'positive', 'negative', 'neutral'
    confidence: float
    language: str
    processing_method: str
    review_id: Optional[int] = None


@dataclass
class EDAReport:
    """Comprehensive exploratory data analysis report."""
    dataset_overview: Dict[str, Any]
    language_analysis: Dict[str, Any]
    rating_analysis: Dict[str, Any]
    temporal_analysis: Dict[str, Any]
    geographic_analysis: Dict[str, Any]
    data_quality_report: Dict[str, Any]
    visualizations: List[Figure]
    generated_at: datetime


@dataclass
class ModelPerformance:
    """Model performance metrics and evaluation results."""
    model_type: str
    accuracy_metrics: Dict[str, float]
    confusion_matrix: Optional[np.ndarray]
    feature_importance: Optional[Dict[str, float]]
    cross_validation_scores: List[float]
    training_time: Optional[float]
    prediction_time: Optional[float]


@dataclass
class ValidationResult:
    """Result of data validation operations."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]


@dataclass
class ValidationError:
    """Specific validation error details."""
    column: str
    error_type: str
    message: str
    affected_rows: List[int]


@dataclass
class ProcessingResult:
    """Generic result for data processing operations."""
    success: bool
    data: Optional[Any]
    warnings: List[str]
    errors: List[str]
    fallback_applied: bool
    processing_time: Optional[float]


@dataclass
class TimeSeriesResult:
    """Result of time series analysis."""
    trends: Dict[str, Any]
    seasonality: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    forecasts: Optional[Dict[str, Any]]
    analysis_period: Dict[str, datetime]


@dataclass
class ForecastResult:
    """Time series forecasting results."""
    predictions: List[float]
    confidence_intervals: List[tuple]
    forecast_dates: List[datetime]
    model_type: str
    accuracy_metrics: Dict[str, float]


@dataclass
class GeographicVisualization:
    """Geographic visualization data and metadata."""
    map_figure: Figure
    data_summary: Dict[str, Any]
    geographic_insights: List[str]
    coverage_stats: Dict[str, int]


@dataclass
class ComparisonResult:
    """Result of comparative analysis between groups."""
    group_comparisons: Dict[str, Dict[str, Any]]
    statistical_tests: Dict[str, Dict[str, float]]
    significant_differences: List[str]
    recommendations: List[str]


@dataclass
class ClassificationModel:
    """Wrapper for trained classification models."""
    model: Any  # The actual trained model object
    model_type: str
    training_languages: List[str]
    feature_names: List[str]
    performance: ModelPerformance
    created_at: datetime


@dataclass
class RegressionModel:
    """Wrapper for trained regression models."""
    model: Any  # The actual trained model object
    model_type: str
    feature_names: List[str]
    performance: ModelPerformance
    created_at: datetime


@dataclass
class Dashboard:
    """Dashboard configuration and components."""
    title: str
    components: List[Dict[str, Any]]
    layout_config: Dict[str, Any]
    data_sources: List[str]
    last_updated: datetime


# Enums for standardized values
class SentimentType:
    """Standardized sentiment categories."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class DeviceType:
    """Standardized device types."""
    ANDROID = "Android"
    ANDROID_TABLET = "Android Tablet"
    IOS = "iOS"
    IPAD = "iPad"
    WINDOWS_PHONE = "Windows Phone"


class Gender:
    """Standardized gender categories."""
    MALE = "Male"
    FEMALE = "Female"
    NON_BINARY = "Non-binary"
    PREFER_NOT_TO_SAY = "Prefer not to say"


class ProcessingStatus:
    """Processing status indicators."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"