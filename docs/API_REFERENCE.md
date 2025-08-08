# API Reference - Multilingual Mobile App Reviews Analysis System

## Overview

This document provides comprehensive API reference for all components of the Multilingual Mobile App Reviews Analysis System.

## Core Data Models

### ReviewRecord
```python
@dataclass
class ReviewRecord:
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
```

### SentimentResult
```python
@dataclass
class SentimentResult:
    sentiment: str  # 'positive', 'negative', 'neutral'
    confidence: float
    language: str
    processing_method: str
    review_id: Optional[int] = None
```

### EDAReport
```python
@dataclass
class EDAReport:
    dataset_overview: Dict[str, Any]
    language_analysis: Dict[str, Any]
    rating_analysis: Dict[str, Any]
    temporal_analysis: Dict[str, Any]
    geographic_analysis: Dict[str, Any]
    data_quality_report: Dict[str, Any]
    visualizations: List[Figure]
    generated_at: datetime
```

## Data Processing Components

### DataLoader

#### Methods

##### `load_csv(file_path: str) -> pd.DataFrame`
Load CSV file and return DataFrame.

**Parameters:**
- `file_path`: Path to the CSV file

**Returns:**
- `pd.DataFrame`: Loaded data

**Example:**
```python
from src.data.loader import DataLoader

loader = DataLoader()
df = loader.load_csv("data.csv")
```

##### `detect_encoding(file_path: str) -> str`
Detect file encoding.

**Parameters:**
- `file_path`: Path to the file

**Returns:**
- `str`: Detected encoding

##### `validate_schema(df: pd.DataFrame) -> ValidationResult`
Validate DataFrame schema.

**Parameters:**
- `df`: DataFrame to validate

**Returns:**
- `ValidationResult`: Validation results

### DataValidator

#### Methods

##### `check_missing_values(df: pd.DataFrame) -> Dict[str, float]`
Check for missing values in each column.

**Parameters:**
- `df`: DataFrame to check

**Returns:**
- `Dict[str, float]`: Missing value percentages by column

##### `identify_duplicates(df: pd.DataFrame) -> pd.DataFrame`
Identify duplicate records.

**Parameters:**
- `df`: DataFrame to check

**Returns:**
- `pd.DataFrame`: Duplicate records

##### `validate_data_types(df: pd.DataFrame) -> List[ValidationError]`
Validate data types for each column.

**Parameters:**
- `df`: DataFrame to validate

**Returns:**
- `List[ValidationError]`: List of validation errors

### TextPreprocessor

#### Methods

##### `clean_text(text: str, language: str) -> str`
Clean and normalize text.

**Parameters:**
- `text`: Text to clean
- `language`: Language code

**Returns:**
- `str`: Cleaned text

##### `preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame`
Preprocess entire DataFrame.

**Parameters:**
- `df`: DataFrame to preprocess

**Returns:**
- `pd.DataFrame`: Preprocessed DataFrame

## Analysis Components

### EDAAnalyzer

#### Methods

##### `generate_comprehensive_eda(df: pd.DataFrame) -> EDAReport`
Generate comprehensive EDA report.

**Parameters:**
- `df`: DataFrame to analyze

**Returns:**
- `EDAReport`: Comprehensive analysis report

**Example:**
```python
from src.analysis.eda_analyzer import EDAAnalyzer

analyzer = EDAAnalyzer()
report = analyzer.generate_comprehensive_eda(df)
```

##### `generate_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]`
Generate dataset overview.

**Parameters:**
- `df`: DataFrame to analyze

**Returns:**
- `Dict[str, Any]`: Dataset overview

##### `analyze_language_distribution(df: pd.DataFrame) -> Dict[str, Any]`
Analyze language distribution.

**Parameters:**
- `df`: DataFrame to analyze

**Returns:**
- `Dict[str, Any]`: Language analysis

### MultilingualSentimentAnalyzer

#### Methods

##### `analyze_sentiment(text: str, language: str) -> SentimentResult`
Analyze sentiment of single text.

**Parameters:**
- `text`: Text to analyze
- `language`: Language code

**Returns:**
- `SentimentResult`: Sentiment analysis result

**Example:**
```python
from src.analysis.sentiment_analyzer import MultilingualSentimentAnalyzer

analyzer = MultilingualSentimentAnalyzer()
result = analyzer.analyze_sentiment("Great app!", "en")
print(result.sentiment)  # 'positive'
```

##### `batch_sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame`
Perform batch sentiment analysis.

**Parameters:**
- `df`: DataFrame with review text

**Returns:**
- `pd.DataFrame`: DataFrame with sentiment column added

##### `compare_sentiment_by_language(df: pd.DataFrame) -> Dict[str, Any]`
Compare sentiment across languages.

**Parameters:**
- `df`: DataFrame with sentiment data

**Returns:**
- `Dict[str, Any]`: Language comparison results

### TimeSeriesAnalyzer

#### Methods

##### `analyze_review_trends(df: pd.DataFrame) -> TimeSeriesResult`
Analyze trends in review data over time.

**Parameters:**
- `df`: DataFrame with review data

**Returns:**
- `TimeSeriesResult`: Comprehensive trend analysis

##### `detect_seasonal_patterns(df: pd.DataFrame) -> Dict[str, Any]`
Detect seasonal patterns in review data.

**Parameters:**
- `df`: DataFrame with review data

**Returns:**
- `Dict[str, Any]`: Seasonal pattern analysis

##### `forecast_future_trends(df: pd.DataFrame, periods: int) -> ForecastResult`
Forecast future trends in review data.

**Parameters:**
- `df`: DataFrame with review data
- `periods`: Number of periods to forecast

**Returns:**
- `ForecastResult`: Forecasting results

## Machine Learning Components

### RatingPredictor

#### Methods

##### `train_rating_model(df: pd.DataFrame) -> RegressionModel`
Train rating prediction model.

**Parameters:**
- `df`: Training data

**Returns:**
- `RegressionModel`: Trained model

**Example:**
```python
from src.ml.rating_predictor import RatingPredictor

predictor = RatingPredictor()
model = predictor.train_rating_model(df)
```

##### `predict_ratings(texts: List[str], languages: List[str]) -> List[float]`
Predict ratings for texts.

**Parameters:**
- `texts`: List of review texts
- `languages`: List of language codes

**Returns:**
- `List[float]`: Predicted ratings

##### `evaluate_model(X_test: pd.DataFrame, y_test: pd.Series) -> ModelPerformance`
Evaluate model performance.

**Parameters:**
- `X_test`: Test features
- `y_test`: Test targets

**Returns:**
- `ModelPerformance`: Performance metrics

### ModelEvaluator

#### Methods

##### `evaluate_regression(y_true: List[float], y_pred: List[float]) -> Dict[str, float]`
Evaluate regression model performance.

**Parameters:**
- `y_true`: True values
- `y_pred`: Predicted values

**Returns:**
- `Dict[str, float]`: Performance metrics

##### `evaluate_classification(y_true: List[str], y_pred: List[str]) -> Dict[str, float]`
Evaluate classification model performance.

**Parameters:**
- `y_true`: True labels
- `y_pred`: Predicted labels

**Returns:**
- `Dict[str, float]`: Performance metrics

## Visualization Components

### VisualizationEngine

#### Methods

##### `create_distribution_plots(df: pd.DataFrame) -> List[go.Figure]`
Create distribution plots for ratings, languages, and countries.

**Parameters:**
- `df`: DataFrame with review data

**Returns:**
- `List[go.Figure]`: List of Plotly figures

**Example:**
```python
from src.visualization.visualization_engine import VisualizationEngine

viz_engine = VisualizationEngine()
plots = viz_engine.create_distribution_plots(df)
```

##### `create_time_series_plots(df: pd.DataFrame) -> List[go.Figure]`
Create time series visualizations.

**Parameters:**
- `df`: DataFrame with review data

**Returns:**
- `List[go.Figure]`: List of time series plots

##### `create_geographic_maps(df: pd.DataFrame) -> List[go.Figure]`
Create geographic maps.

**Parameters:**
- `df`: DataFrame with review data

**Returns:**
- `List[go.Figure]`: List of geographic visualizations

##### `create_correlation_heatmaps(df: pd.DataFrame) -> go.Figure`
Create correlation heatmaps.

**Parameters:**
- `df`: DataFrame with review data

**Returns:**
- `go.Figure`: Correlation heatmap

### DashboardGenerator

#### Methods

##### `create_eda_dashboard(df: pd.DataFrame) -> Dashboard`
Create comprehensive EDA dashboard.

**Parameters:**
- `df`: DataFrame with review data

**Returns:**
- `Dashboard`: Dashboard configuration

##### `create_ml_results_dashboard(results: Dict[str, Any]) -> Dashboard`
Create ML results dashboard.

**Parameters:**
- `results`: ML model results

**Returns:**
- `Dashboard`: ML dashboard configuration

##### `generate_streamlit_app(df: pd.DataFrame) -> str`
Generate Streamlit application code.

**Parameters:**
- `df`: DataFrame with review data

**Returns:**
- `str`: Streamlit app code

## Pipeline Components

### PipelineOrchestrator

#### Methods

##### `run_complete_pipeline(data_file_path: str, analysis_types: List[str] = None) -> Dict[str, Any]`
Run the complete analysis pipeline.

**Parameters:**
- `data_file_path`: Path to input data file
- `analysis_types`: List of analysis types to run

**Returns:**
- `Dict[str, Any]`: Complete analysis results

**Example:**
```python
from src.pipeline.orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator()
results = orchestrator.run_complete_pipeline(
    "data.csv", 
    ["eda", "sentiment", "prediction"]
)
```

##### `run_specific_analysis(df: pd.DataFrame, analysis_type: str) -> Any`
Run a specific analysis type.

**Parameters:**
- `df`: DataFrame with review data
- `analysis_type`: Type of analysis to run

**Returns:**
- `Any`: Analysis results

##### `get_pipeline_status() -> Dict[str, Any]`
Get current pipeline status.

**Returns:**
- `Dict[str, Any]`: Pipeline status information

## Configuration Management

### Config

#### Methods

##### `get_config() -> Config`
Get the global configuration instance.

**Returns:**
- `Config`: Configuration instance

##### `update_config(**kwargs)`
Update configuration parameters.

**Parameters:**
- `**kwargs`: Configuration parameters to update

**Example:**
```python
from src.config import get_config, update_config

# Get current config
config = get_config()

# Update configuration
update_config(batch_size=1000, max_workers=4)
```

## Error Handling

### Common Exceptions

#### `ValidationError`
Raised when data validation fails.

#### `ProcessingError`
Raised when data processing encounters errors.

#### `ModelError`
Raised when ML model operations fail.

#### `LanguageError`
Raised when language processing fails.

## Usage Examples

### Basic Pipeline Execution

```python
from src.pipeline.orchestrator import PipelineOrchestrator

# Initialize orchestrator
orchestrator = PipelineOrchestrator()

# Run complete pipeline
results = orchestrator.run_complete_pipeline(
    data_file_path="multilingual_mobile_app_reviews_2025.csv",
    analysis_types=["eda", "sentiment", "prediction", "time_series", "geographic"]
)

# Access results
eda_results = results['analysis_results']['eda']
sentiment_results = results['analysis_results']['sentiment']
```

### Custom Analysis Workflow

```python
import pandas as pd
from src.data.loader import DataLoader
from src.analysis.sentiment_analyzer import MultilingualSentimentAnalyzer
from src.visualization.visualization_engine import VisualizationEngine

# Load data
loader = DataLoader()
df = loader.load_csv("data.csv")

# Perform sentiment analysis
sentiment_analyzer = MultilingualSentimentAnalyzer()
df_with_sentiment = sentiment_analyzer.batch_sentiment_analysis(df)

# Create visualizations
viz_engine = VisualizationEngine()
sentiment_plots = viz_engine.create_sentiment_visualizations(df_with_sentiment)
```

### Dashboard Creation

```python
from src.visualization.dashboard_generator import DashboardGenerator

# Create dashboard
dashboard_gen = DashboardGenerator()
eda_dashboard = dashboard_gen.create_eda_dashboard(df)

# Generate Streamlit app
streamlit_code = dashboard_gen.generate_streamlit_app(df)

# Save to file
with open("dashboard_app.py", "w") as f:
    f.write(streamlit_code)
```

## Performance Considerations

### Memory Management
- Use batch processing for large datasets
- Configure batch size based on available memory
- Monitor memory usage during processing

### Processing Speed
- Utilize parallel processing where available
- Cache expensive computations
- Use efficient data structures

### Scalability
- Pipeline supports distributed processing
- Components can be deployed independently
- Results can be stored in databases for historical analysis

## Testing

### Unit Tests
```bash
python -m pytest tests/test_pipeline.py -v
```

### Integration Tests
```bash
python run_integration_tests.py
```

### Performance Tests
```bash
python main.py --test-pipeline --profile
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or use streaming processing
2. **Language Detection Failures**: Check text encoding and language codes
3. **Model Training Failures**: Ensure sufficient data and proper preprocessing
4. **Visualization Errors**: Check data types and missing values

### Debug Mode
```bash
python main.py --verbose --log-file debug.log
```

### Performance Profiling
```bash
python main.py --profile --analysis-type eda
```