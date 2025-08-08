# Usage Examples - Multilingual Mobile App Reviews Analysis System

## Table of Contents
1. [Quick Start](#quick-start)
2. [Command Line Interface](#command-line-interface)
3. [Python API Usage](#python-api-usage)
4. [Analysis Workflows](#analysis-workflows)
5. [Visualization Examples](#visualization-examples)
6. [Dashboard Creation](#dashboard-creation)
7. [Advanced Usage](#advanced-usage)
8. [Integration Examples](#integration-examples)

## Quick Start

### 1. Basic Analysis
Run complete analysis on your dataset:

```bash
python main.py --data-file multilingual_mobile_app_reviews_2025.csv --analysis-type all
```

### 2. Specific Analysis Types
Run only sentiment analysis:

```bash
python main.py --analysis-type sentiment --verbose
```

### 3. Launch Interactive Dashboard
Create and launch an interactive dashboard:

```bash
python main.py --dashboard --port 8050
```

## Command Line Interface

### Basic Commands

#### Run Complete Pipeline
```bash
# Run all analysis types
python main.py --analysis-type all

# Run specific analysis types
python main.py --analysis-type eda
python main.py --analysis-type sentiment
python main.py --analysis-type prediction
python main.py --analysis-type time-series
python main.py --analysis-type geographic
```

#### Multiple Analysis Types
```bash
# Run multiple specific analyses
python main.py --analysis-list eda sentiment prediction
```

#### Custom Configuration
```bash
# Use custom data file and output directory
python main.py \
  --data-file /path/to/your/data.csv \
  --output-dir /path/to/output \
  --batch-size 1000 \
  --max-workers 8
```

#### Validation and Testing
```bash
# Validate data without running analysis
python main.py --validate-data --data-file data.csv

# Run pipeline tests
python main.py --test-pipeline

# Run with performance profiling
python main.py --profile --analysis-type eda
```

#### Dashboard Options
```bash
# Launch dashboard on custom host/port
python main.py --dashboard --host 0.0.0.0 --port 8080

# Generate dashboard code only
python main.py --dashboard --output-dir dashboards/
```

#### Logging Options
```bash
# Verbose logging
python main.py --verbose --analysis-type all

# Save logs to file
python main.py --log-file analysis.log --analysis-type sentiment

# Quiet mode (errors only)
python main.py --quiet --analysis-type eda
```

## Python API Usage

### Basic Data Loading and Analysis

```python
import pandas as pd
from src.pipeline.orchestrator import PipelineOrchestrator

# Initialize orchestrator
orchestrator = PipelineOrchestrator()

# Run complete pipeline
results = orchestrator.run_complete_pipeline(
    data_file_path="multilingual_mobile_app_reviews_2025.csv",
    analysis_types=["eda", "sentiment", "prediction"]
)

# Access results
print(f"Processed {results['pipeline_metadata']['total_records_processed']} records")
print(f"Execution time: {results['pipeline_metadata']['execution_time']:.2f} seconds")
```

### Individual Component Usage

#### Data Loading and Preprocessing
```python
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.data.preprocessor import TextPreprocessor

# Load data
loader = DataLoader()
df = loader.load_csv("data.csv")
print(f"Loaded {len(df)} records")

# Validate data
validator = DataValidator()
validation_result = validator.validate_schema(df)
if validation_result.is_valid:
    print("✅ Data validation passed")
else:
    print("❌ Data validation failed:", validation_result.errors)

# Preprocess text
preprocessor = TextPreprocessor()
df_processed = preprocessor.preprocess_dataframe(df)
print("Text preprocessing completed")
```

#### Sentiment Analysis
```python
from src.analysis.sentiment_analyzer import MultilingualSentimentAnalyzer

# Initialize analyzer
sentiment_analyzer = MultilingualSentimentAnalyzer()

# Analyze single text
result = sentiment_analyzer.analyze_sentiment("This app is amazing!", "en")
print(f"Sentiment: {result.sentiment} (confidence: {result.confidence:.3f})")

# Batch analysis
df_with_sentiment = sentiment_analyzer.batch_sentiment_analysis(df)
print(f"Added sentiment to {len(df_with_sentiment)} reviews")

# Compare sentiment by language
comparison = sentiment_analyzer.compare_sentiment_by_language(df_with_sentiment)
for lang, stats in comparison.items():
    print(f"{lang}: {stats['positive_percentage']:.1f}% positive")
```

#### Rating Prediction
```python
from src.ml.rating_predictor import RatingPredictor
from src.ml.model_evaluator import ModelEvaluator

# Train model
predictor = RatingPredictor()
model = predictor.train_rating_model(df)
print("Rating prediction model trained")

# Make predictions
test_texts = [
    "Great app, love the interface!",
    "Terrible experience, crashes constantly",
    "Average app, nothing special"
]
test_languages = ["en", "en", "en"]

predictions = predictor.predict_ratings(test_texts, test_languages)
for text, pred in zip(test_texts, predictions):
    print(f"'{text}' -> Predicted rating: {pred:.2f}")

# Evaluate model
evaluator = ModelEvaluator()
performance = evaluator.evaluate_regression_model(model, df.sample(100))
print(f"Model RMSE: {performance.accuracy_metrics['rmse']:.3f}")
```

## Analysis Workflows

### Comprehensive EDA Workflow

```python
from src.analysis.eda_analyzer import EDAAnalyzer
from src.visualization.visualization_engine import VisualizationEngine

# Initialize components
eda_analyzer = EDAAnalyzer()
viz_engine = VisualizationEngine()

# Generate comprehensive EDA
eda_report = eda_analyzer.generate_comprehensive_eda(df)

# Print key insights
print("Dataset Overview:")
print(f"  Shape: {eda_report.dataset_overview['shape']}")
print(f"  Languages: {len(eda_report.language_analysis['language_counts'])}")
print(f"  Countries: {len(eda_report.geographic_analysis['country_counts'])}")

# Create visualizations
distribution_plots = viz_engine.create_distribution_plots(df)
print(f"Created {len(distribution_plots)} distribution plots")

# Save plots
for i, plot in enumerate(distribution_plots):
    plot.write_html(f"output/distribution_plot_{i}.html")
```

### Time Series Analysis Workflow

```python
from src.analysis.time_series_analyzer import TimeSeriesAnalyzer

# Initialize analyzer
ts_analyzer = TimeSeriesAnalyzer()

# Analyze trends
trends_result = ts_analyzer.analyze_review_trends(df)
print("Time Series Analysis Results:")
print(f"  Analysis period: {trends_result.analysis_period['start_date']} to {trends_result.analysis_period['end_date']}")
print(f"  Anomalies detected: {len(trends_result.anomalies)}")

# Detect seasonal patterns
seasonal_patterns = ts_analyzer.detect_seasonal_patterns(df)
print("Seasonal Patterns:")
print(f"  Peak month: {seasonal_patterns['monthly']['peak_month']}")
print(f"  Peak day: {seasonal_patterns['weekly']['peak_day']}")

# Forecast future trends
forecast = ts_analyzer.forecast_future_trends(df, periods=30)
print(f"30-day forecast generated with {forecast.model_type}")
print(f"Forecast accuracy (MAE): {forecast.accuracy_metrics['mae']:.2f}")
```

### Geographic Analysis Workflow

```python
from src.analysis.sentiment_mapper import SentimentMapper, RegionalComparator

# Initialize components
sentiment_mapper = SentimentMapper()
regional_comparator = RegionalComparator()

# Ensure sentiment data is available
if 'sentiment' not in df.columns:
    df = sentiment_analyzer.batch_sentiment_analysis(df)

# Create sentiment map
geo_viz = sentiment_mapper.create_sentiment_map(df)
print("Geographic Sentiment Analysis:")
for insight in geo_viz.geographic_insights:
    print(f"  - {insight}")

# Regional comparison
comparison_result = regional_comparator.compare_regional_patterns(df)
print("Regional Comparisons:")
for region, data in comparison_result.group_comparisons.items():
    print(f"  {region}: {data['total_reviews']} reviews, avg rating {data['avg_rating']:.2f}")

# Print recommendations
print("Recommendations:")
for rec in comparison_result.recommendations:
    print(f"  - {rec}")
```

## Visualization Examples

### Creating Custom Visualizations

```python
import plotly.express as px
import plotly.graph_objects as go
from src.visualization.visualization_engine import VisualizationEngine

viz_engine = VisualizationEngine()

# Distribution plots
distribution_plots = viz_engine.create_distribution_plots(df)
rating_plot = distribution_plots[0]  # Rating distribution
rating_plot.show()

# Time series plots
time_series_plots = viz_engine.create_time_series_plots(df)
volume_plot = time_series_plots[0]  # Review volume over time
volume_plot.show()

# Geographic maps
geographic_maps = viz_engine.create_geographic_maps(df)
country_map = geographic_maps[0]  # Reviews by country
country_map.show()

# Correlation heatmap
correlation_heatmap = viz_engine.create_correlation_heatmaps(df)
correlation_heatmap.show()
```

### Sentiment Visualizations

```python
# Ensure sentiment data exists
if 'sentiment' not in df.columns:
    df = sentiment_analyzer.batch_sentiment_analysis(df)

# Create sentiment-specific visualizations
sentiment_plots = viz_engine.create_sentiment_visualizations(df, 'sentiment')

# Display sentiment distribution
sentiment_dist_plot = sentiment_plots[0]
sentiment_dist_plot.show()

# Sentiment by language
sentiment_lang_plot = sentiment_plots[1]
sentiment_lang_plot.show()

# Sentiment vs rating comparison
sentiment_rating_plot = sentiment_plots[3]
sentiment_rating_plot.show()
```

### Word Clouds

```python
# Create word clouds
word_clouds = viz_engine.create_word_clouds(df, 'review_text')

# Display overall word cloud
import matplotlib.pyplot as plt

if 'overall' in word_clouds:
    plt.figure(figsize=(10, 6))
    plt.imshow(word_clouds['overall'], interpolation='bilinear')
    plt.axis('off')
    plt.title('Overall Word Cloud')
    plt.show()

# Display sentiment-specific word clouds
for sentiment in ['positive', 'negative', 'neutral']:
    key = f'sentiment_{sentiment}'
    if key in word_clouds:
        plt.figure(figsize=(8, 5))
        plt.imshow(word_clouds[key], interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{sentiment.title()} Sentiment Word Cloud')
        plt.show()
```

## Dashboard Creation

### Streamlit Dashboard

```python
from src.visualization.dashboard_generator import DashboardGenerator

# Create dashboard generator
dashboard_gen = DashboardGenerator()

# Generate Streamlit app code
streamlit_code = dashboard_gen.generate_streamlit_app(df)

# Save to file
with open("multilingual_reviews_dashboard.py", "w") as f:
    f.write(streamlit_code)

print("Streamlit dashboard saved to: multilingual_reviews_dashboard.py")
print("Run with: streamlit run multilingual_reviews_dashboard.py")
```

### Custom Dashboard Components

```python
# Create EDA dashboard
eda_dashboard = dashboard_gen.create_eda_dashboard(df)
print(f"EDA Dashboard created with {len(eda_dashboard.components)} components")

# Create ML results dashboard (if ML analysis was performed)
if 'prediction' in results['analysis_results']:
    ml_results = results['analysis_results']['prediction']
    ml_dashboard = dashboard_gen.create_ml_results_dashboard(ml_results)
    print(f"ML Dashboard created with {len(ml_dashboard.components)} components")

# Create geographic dashboard
if 'sentiment' in df.columns:
    geo_dashboard = dashboard_gen.create_geographic_dashboard(df, 'sentiment')
    print(f"Geographic Dashboard created with {len(geo_dashboard.components)} components")
```

### Report Generation

```python
from src.visualization.dashboard_generator import ReportGenerator

# Generate comprehensive report
report_gen = ReportGenerator()
html_report = report_gen.generate_comprehensive_report(df, results['analysis_results'])

# Save report
with open("output/comprehensive_report.html", "w") as f:
    f.write(html_report)

print("Comprehensive report saved to: output/comprehensive_report.html")
```

## Advanced Usage

### Custom Configuration

```python
from src.config import get_config, update_config

# Get current configuration
config = get_config()
print(f"Current batch size: {config.processing.batch_size}")

# Update configuration
update_config(
    batch_size=2000,
    max_workers=8,
    cache_results=True,
    log_level="DEBUG"
)

# Verify changes
updated_config = get_config()
print(f"Updated batch size: {updated_config.processing.batch_size}")
```

### Pipeline Monitoring

```python
from src.pipeline.orchestrator import PipelineMonitor

# Initialize monitor
monitor = PipelineMonitor()
monitor.start_monitoring()

# Run analysis with monitoring
orchestrator = PipelineOrchestrator()
results = orchestrator.run_complete_pipeline("data.csv", ["eda", "sentiment"])

# Log completion
monitor.log_stage_completion("complete_analysis", 120.5, 512.0)

# Get performance report
performance_report = monitor.get_performance_report()
print(f"Total duration: {performance_report['total_duration']:.2f} seconds")
print(f"Stages completed: {performance_report['stages_completed']}")
print(f"Average stage duration: {performance_report['average_stage_duration']:.2f} seconds")
```

### Error Handling

```python
import logging
from src.utils.logger import setup_logger

# Setup detailed logging
logger = setup_logger("DEBUG", "detailed_analysis.log")

try:
    # Run analysis with error handling
    orchestrator = PipelineOrchestrator()
    results = orchestrator.run_complete_pipeline("data.csv", ["all"])
    
except Exception as e:
    logger.error(f"Analysis failed: {e}")
    
    # Get pipeline status for debugging
    status = orchestrator.get_pipeline_status()
    logger.error(f"Pipeline status: {status}")
    
    # Continue with partial results if available
    if orchestrator.results:
        logger.info("Using partial results for visualization")
        # Create visualizations with available data
```

### Batch Processing

```python
import glob
from pathlib import Path

# Process multiple files
data_files = glob.glob("data/*.csv")
all_results = {}

for data_file in data_files:
    file_name = Path(data_file).stem
    print(f"Processing {file_name}...")
    
    try:
        orchestrator = PipelineOrchestrator()
        results = orchestrator.run_complete_pipeline(
            data_file, 
            ["eda", "sentiment"]
        )
        all_results[file_name] = results
        print(f"✅ {file_name} completed")
        
    except Exception as e:
        print(f"❌ {file_name} failed: {e}")
        all_results[file_name] = {"error": str(e)}

# Generate combined report
print(f"Processed {len(all_results)} files")
for file_name, result in all_results.items():
    if "error" in result:
        print(f"  {file_name}: ERROR - {result['error']}")
    else:
        records = result['pipeline_metadata']['total_records_processed']
        time_taken = result['pipeline_metadata']['execution_time']
        print(f"  {file_name}: {records} records in {time_taken:.2f}s")
```

## Integration Examples

### Jupyter Notebook Integration

```python
# In Jupyter Notebook
import pandas as pd
import plotly.io as pio
from src.pipeline.orchestrator import PipelineOrchestrator

# Set Plotly renderer for Jupyter
pio.renderers.default = "notebook"

# Load and analyze data
orchestrator = PipelineOrchestrator()
results = orchestrator.run_complete_pipeline(
    "multilingual_mobile_app_reviews_2025.csv",
    ["eda", "sentiment", "time_series"]
)

# Display results inline
if 'eda' in results['analysis_results']:
    eda_report = results['analysis_results']['eda']
    print("Dataset Overview:")
    display(pd.DataFrame([eda_report.dataset_overview]).T)

# Show visualizations
if 'visualization_results' in results:
    viz_results = results['visualization_results']
    if 'distribution_plots' in viz_results:
        for plot in viz_results['distribution_plots'][:3]:  # Show first 3 plots
            plot.show()
```

### Flask Web Application

```python
from flask import Flask, render_template, request, jsonify
from src.pipeline.orchestrator import PipelineOrchestrator
import json

app = Flask(__name__)
orchestrator = PipelineOrchestrator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data_file = request.files['data_file']
    analysis_types = request.form.getlist('analysis_types')
    
    # Save uploaded file
    file_path = f"uploads/{data_file.filename}"
    data_file.save(file_path)
    
    try:
        # Run analysis
        results = orchestrator.run_complete_pipeline(file_path, analysis_types)
        
        # Return summary
        return jsonify({
            'status': 'success',
            'records_processed': results['pipeline_metadata']['total_records_processed'],
            'execution_time': results['pipeline_metadata']['execution_time'],
            'analysis_types': analysis_types
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
```

### Database Integration

```python
import sqlite3
import json
from datetime import datetime

# Save results to database
def save_results_to_db(results, db_path="analysis_results.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            records_processed INTEGER,
            execution_time REAL,
            analysis_types TEXT,
            results TEXT
        )
    ''')
    
    # Insert results
    cursor.execute('''
        INSERT INTO analysis_runs 
        (timestamp, records_processed, execution_time, analysis_types, results)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        results['pipeline_metadata']['total_records_processed'],
        results['pipeline_metadata']['execution_time'],
        json.dumps(results['pipeline_metadata']['analysis_types_completed']),
        json.dumps(results, default=str)
    ))
    
    conn.commit()
    conn.close()
    print("Results saved to database")

# Use after running analysis
results = orchestrator.run_complete_pipeline("data.csv", ["eda", "sentiment"])
save_results_to_db(results)
```

### API Integration

```python
import requests
import json

# Send results to external API
def send_results_to_api(results, api_endpoint):
    # Prepare payload
    payload = {
        'timestamp': results['pipeline_metadata']['completion_timestamp'],
        'records_processed': results['pipeline_metadata']['total_records_processed'],
        'execution_time': results['pipeline_metadata']['execution_time'],
        'summary': {
            'total_languages': len(results['analysis_results']['eda'].language_analysis['language_counts']),
            'total_countries': len(results['analysis_results']['eda'].geographic_analysis['country_counts']),
            'avg_rating': results['analysis_results']['eda'].rating_analysis['mean_rating']
        }
    }
    
    # Send to API
    response = requests.post(
        api_endpoint,
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 200:
        print("Results successfully sent to API")
    else:
        print(f"API request failed: {response.status_code}")

# Use after analysis
results = orchestrator.run_complete_pipeline("data.csv", ["eda"])
send_results_to_api(results, "https://api.example.com/analysis-results")
```

These examples demonstrate the flexibility and power of the Multilingual Mobile App Reviews Analysis System. You can adapt these patterns to fit your specific use cases and integration requirements.