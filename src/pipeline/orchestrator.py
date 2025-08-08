"""
Pipeline orchestrator for end-to-end automation of multilingual app reviews analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import time
import traceback
from pathlib import Path
import json
import pickle

from src.config import get_config
from src.utils.logger import get_logger
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.data.preprocessor import TextPreprocessor
from src.data.language_detector import LanguageDetector
from src.data.cleaner import DataCleaner
from src.analysis.eda_analyzer import EDAAnalyzer
from src.analysis.sentiment_analyzer import MultilingualSentimentAnalyzer
from src.analysis.text_classifier import TextClassifier
from src.analysis.cross_cultural_analyzer import CrossCulturalAnalyzer
from src.analysis.time_series_analyzer import TimeSeriesAnalyzer
from src.analysis.geographic_analyzer import GeographicAnalyzer
from src.analysis.sentiment_mapper import SentimentMapper, RegionalComparator
from src.ml.rating_predictor import RatingPredictor
from src.ml.model_evaluator import ModelEvaluator
from src.ml.model_manager import ModelManager
from src.visualization.visualization_engine import VisualizationEngine
from src.visualization.dashboard_generator import DashboardGenerator, ReportGenerator


class PipelineOrchestrator:
    """
    Main orchestrator for the complete analysis pipeline.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.results = {}
        self.pipeline_state = {}
        self.error_count = 0
        self.max_errors = 5
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        self.logger.info("Initializing pipeline components")
        
        try:
            # Data components
            self.data_loader = DataLoader()
            self.data_validator = DataValidator()
            self.text_preprocessor = TextPreprocessor()
            self.language_detector = LanguageDetector()
            self.data_cleaner = DataCleaner()
            
            # Analysis components
            self.eda_analyzer = EDAAnalyzer()
            self.sentiment_analyzer = MultilingualSentimentAnalyzer()
            self.text_classifier = TextClassifier()
            self.cross_cultural_analyzer = CrossCulturalAnalyzer()
            self.time_series_analyzer = TimeSeriesAnalyzer()
            self.geographic_analyzer = GeographicAnalyzer()
            self.sentiment_mapper = SentimentMapper()
            self.regional_comparator = RegionalComparator()
            
            # ML components
            self.rating_predictor = RatingPredictor()
            self.model_evaluator = ModelEvaluator()
            self.model_manager = ModelManager()
            
            # Visualization components
            self.viz_engine = VisualizationEngine()
            self.dashboard_generator = DashboardGenerator()
            self.report_generator = ReportGenerator()
            
            self.logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def run_complete_pipeline(self, data_file_path: str, analysis_types: List[str] = None) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        Args:
            data_file_path: Path to the input data file
            analysis_types: List of analysis types to run (default: all)
            
        Returns:
            Dictionary with all analysis results
        """
        self.logger.info("Starting complete pipeline execution")
        
        if analysis_types is None:
            analysis_types = ['eda', 'sentiment', 'prediction', 'time_series', 'geographic', 'visualization']
        
        pipeline_start_time = time.time()
        
        try:
            # Stage 1: Data Loading and Validation
            self.logger.info("Stage 1: Data Loading and Validation")
            df = self._run_data_loading_stage(data_file_path)
            
            # Stage 2: Data Preprocessing
            self.logger.info("Stage 2: Data Preprocessing")
            df_processed = self._run_preprocessing_stage(df)
            
            # Stage 3: Analysis Execution
            self.logger.info("Stage 3: Analysis Execution")
            analysis_results = self._run_analysis_stage(df_processed, analysis_types)
            
            # Stage 4: Visualization and Reporting
            self.logger.info("Stage 4: Visualization and Reporting")
            visualization_results = self._run_visualization_stage(df_processed, analysis_results)
            
            # Stage 5: Results Compilation
            self.logger.info("Stage 5: Results Compilation")
            final_results = self._compile_final_results(df_processed, analysis_results, visualization_results)
            
            pipeline_end_time = time.time()
            execution_time = pipeline_end_time - pipeline_start_time
            
            final_results['pipeline_metadata'] = {
                'execution_time': execution_time,
                'total_records_processed': len(df_processed),
                'analysis_types_completed': analysis_types,
                'completion_timestamp': datetime.now().isoformat(),
                'error_count': self.error_count
            }
            
            self.logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _run_data_loading_stage(self, data_file_path: str) -> pd.DataFrame:
        """Run data loading and validation stage."""
        stage_results = {}
        
        try:
            # Load data
            self.logger.info("Loading data from file")
            df = self.data_loader.load_csv(data_file_path)
            stage_results['data_loaded'] = True
            stage_results['original_shape'] = df.shape
            
            # Validate data
            self.logger.info("Validating data quality")
            validation_result = self.data_validator.validate_schema(df)
            stage_results['validation_result'] = validation_result
            
            if not validation_result.is_valid:
                self.logger.warning(f"Data validation issues found: {validation_result.errors}")
                self.error_count += len(validation_result.errors)
            
            # Data quality checks
            missing_values = self.data_validator.check_missing_values(df)
            duplicates = self.data_validator.identify_duplicates(df)
            
            stage_results['missing_values'] = missing_values
            stage_results['duplicate_count'] = len(duplicates)
            
            self.results['data_loading_stage'] = stage_results
            self.logger.info(f"Data loading stage completed. Shape: {df.shape}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Data loading stage failed: {str(e)}")
            self._handle_stage_error('data_loading', e)
            raise
    
    def _run_preprocessing_stage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run data preprocessing stage."""
        stage_results = {}
        
        try:
            df_processed = df.copy()
            
            # Text preprocessing
            self.logger.info("Preprocessing text data")
            df_processed = self.text_preprocessor.preprocess_dataframe(df_processed)
            stage_results['text_preprocessing'] = True
            
            # Language detection and validation
            self.logger.info("Validating and standardizing languages")
            df_processed = self.language_detector.standardize_language_codes(df_processed)
            stage_results['language_standardization'] = True
            
            # Data cleaning
            self.logger.info("Cleaning data")
            df_processed = self.data_cleaner.clean_dataframe(df_processed)
            stage_results['data_cleaning'] = True
            stage_results['final_shape'] = df_processed.shape
            
            self.results['preprocessing_stage'] = stage_results
            self.logger.info(f"Preprocessing stage completed. Final shape: {df_processed.shape}")
            
            return df_processed
            
        except Exception as e:
            self.logger.error(f"Preprocessing stage failed: {str(e)}")
            self._handle_stage_error('preprocessing', e)
            raise
    
    def _run_analysis_stage(self, df: pd.DataFrame, analysis_types: List[str]) -> Dict[str, Any]:
        """Run analysis stage."""
        analysis_results = {}
        
        try:
            # EDA Analysis
            if 'eda' in analysis_types:
                self.logger.info("Running EDA analysis")
                eda_result = self.eda_analyzer.generate_comprehensive_eda(df)
                analysis_results['eda'] = eda_result
            
            # Sentiment Analysis
            if 'sentiment' in analysis_types:
                self.logger.info("Running sentiment analysis")
                df_with_sentiment = self.sentiment_analyzer.batch_sentiment_analysis(df)
                sentiment_comparison = self.sentiment_analyzer.compare_sentiment_by_language(df_with_sentiment)
                analysis_results['sentiment'] = {
                    'dataframe': df_with_sentiment,
                    'language_comparison': sentiment_comparison
                }
            
            # Text Classification
            if 'classification' in analysis_types:
                self.logger.info("Running text classification")
                classification_model = self.text_classifier.train_multilingual_classifier(df)
                analysis_results['classification'] = classification_model
            
            # Cross-Cultural Analysis
            if 'cross_cultural' in analysis_types:
                self.logger.info("Running cross-cultural analysis")
                cultural_analysis = self.cross_cultural_analyzer.analyze_cultural_patterns(df)
                analysis_results['cross_cultural'] = cultural_analysis
            
            # Time Series Analysis
            if 'time_series' in analysis_types:
                self.logger.info("Running time series analysis")
                time_series_result = self.time_series_analyzer.analyze_review_trends(df)
                forecast_result = self.time_series_analyzer.forecast_future_trends(df, periods=30)
                analysis_results['time_series'] = {
                    'trends': time_series_result,
                    'forecast': forecast_result
                }
            
            # Geographic Analysis
            if 'geographic' in analysis_types:
                self.logger.info("Running geographic analysis")
                if 'sentiment' in analysis_results:
                    df_with_sentiment = analysis_results['sentiment']['dataframe']
                    geo_sentiment = self.sentiment_mapper.create_sentiment_map(df_with_sentiment)
                    regional_comparison = self.regional_comparator.compare_regional_patterns(df_with_sentiment)
                    analysis_results['geographic'] = {
                        'sentiment_map': geo_sentiment,
                        'regional_comparison': regional_comparison
                    }
                else:
                    self.logger.warning("Geographic analysis requires sentiment analysis. Running sentiment analysis first.")
                    df_with_sentiment = self.sentiment_analyzer.batch_sentiment_analysis(df)
                    geo_sentiment = self.sentiment_mapper.create_sentiment_map(df_with_sentiment)
                    analysis_results['geographic'] = {'sentiment_map': geo_sentiment}
            
            # Rating Prediction
            if 'prediction' in analysis_types:
                self.logger.info("Running rating prediction")
                rating_model = self.rating_predictor.train_rating_model(df)
                model_performance = self.model_evaluator.evaluate_regression_model(rating_model, df)
                analysis_results['prediction'] = {
                    'model': rating_model,
                    'performance': model_performance
                }
            
            self.results['analysis_stage'] = analysis_results
            self.logger.info("Analysis stage completed successfully")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Analysis stage failed: {str(e)}")
            self._handle_stage_error('analysis', e)
            raise
    
    def _run_visualization_stage(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run visualization and reporting stage."""
        viz_results = {}
        
        try:
            # Create visualizations
            self.logger.info("Creating visualizations")
            
            # Distribution plots
            distribution_plots = self.viz_engine.create_distribution_plots(df)
            viz_results['distribution_plots'] = distribution_plots
            
            # Time series plots
            time_series_plots = self.viz_engine.create_time_series_plots(df)
            viz_results['time_series_plots'] = time_series_plots
            
            # Geographic maps
            geographic_maps = self.viz_engine.create_geographic_maps(df)
            viz_results['geographic_maps'] = geographic_maps
            
            # Correlation heatmap
            correlation_heatmap = self.viz_engine.create_correlation_heatmaps(df)
            viz_results['correlation_heatmap'] = correlation_heatmap
            
            # Sentiment visualizations (if available)
            if 'sentiment' in analysis_results:
                df_with_sentiment = analysis_results['sentiment']['dataframe']
                sentiment_plots = self.viz_engine.create_sentiment_visualizations(df_with_sentiment)
                viz_results['sentiment_plots'] = sentiment_plots
            
            # Create dashboards
            self.logger.info("Creating dashboards")
            eda_dashboard = self.dashboard_generator.create_eda_dashboard(df)
            viz_results['eda_dashboard'] = eda_dashboard
            
            if 'prediction' in analysis_results:
                ml_dashboard = self.dashboard_generator.create_ml_results_dashboard(analysis_results['prediction'])
                viz_results['ml_dashboard'] = ml_dashboard
            
            if 'geographic' in analysis_results:
                geo_dashboard = self.dashboard_generator.create_geographic_dashboard(df)
                viz_results['geographic_dashboard'] = geo_dashboard
            
            # Generate reports
            self.logger.info("Generating reports")
            comprehensive_report = self.report_generator.generate_comprehensive_report(df, analysis_results)
            viz_results['comprehensive_report'] = comprehensive_report
            
            self.results['visualization_stage'] = viz_results
            self.logger.info("Visualization stage completed successfully")
            
            return viz_results
            
        except Exception as e:
            self.logger.error(f"Visualization stage failed: {str(e)}")
            self._handle_stage_error('visualization', e)
            raise
    
    def _compile_final_results(self, df: pd.DataFrame, analysis_results: Dict[str, Any], 
                             visualization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile final results."""
        final_results = {
            'dataset_info': {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'data_types': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum()
            },
            'analysis_results': analysis_results,
            'visualization_results': visualization_results,
            'pipeline_results': self.results
        }
        
        # Save results to files
        self._save_results(final_results)
        
        return final_results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to files."""
        try:
            output_dir = self.config.get_output_path('')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save JSON summary
            json_path = output_dir / f'analysis_results_{timestamp}.json'
            with open(json_path, 'w') as f:
                # Convert non-serializable objects to strings
                serializable_results = self._make_json_serializable(results)
                json.dump(serializable_results, f, indent=2, default=str)
            
            # Save pickle file with full results
            pickle_path = output_dir / f'full_results_{timestamp}.pkl'
            with open(pickle_path, 'wb') as f:
                pickle.dump(results, f)
            
            self.logger.info(f"Results saved to {json_path} and {pickle_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return f"<{type(obj).__name__} shape={getattr(obj, 'shape', 'N/A')}>"
        elif hasattr(obj, '__dict__'):
            return f"<{type(obj).__name__} object>"
        else:
            return obj
    
    def _handle_stage_error(self, stage_name: str, error: Exception):
        """Handle stage-specific errors."""
        self.error_count += 1
        self.pipeline_state[f'{stage_name}_error'] = {
            'error_message': str(error),
            'error_type': type(error).__name__,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.error_count >= self.max_errors:
            self.logger.error(f"Maximum error count ({self.max_errors}) reached. Stopping pipeline.")
            raise RuntimeError("Pipeline stopped due to excessive errors")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'pipeline_state': self.pipeline_state,
            'error_count': self.error_count,
            'results_available': list(self.results.keys()),
            'timestamp': datetime.now().isoformat()
        }
    
    def run_specific_analysis(self, df: pd.DataFrame, analysis_type: str) -> Any:
        """Run a specific analysis type."""
        self.logger.info(f"Running specific analysis: {analysis_type}")
        
        try:
            if analysis_type == 'eda':
                return self.eda_analyzer.generate_comprehensive_eda(df)
            elif analysis_type == 'sentiment':
                return self.sentiment_analyzer.batch_sentiment_analysis(df)
            elif analysis_type == 'classification':
                return self.text_classifier.train_multilingual_classifier(df)
            elif analysis_type == 'cross_cultural':
                return self.cross_cultural_analyzer.analyze_cultural_patterns(df)
            elif analysis_type == 'time_series':
                return self.time_series_analyzer.analyze_review_trends(df)
            elif analysis_type == 'geographic':
                df_with_sentiment = self.sentiment_analyzer.batch_sentiment_analysis(df)
                return self.sentiment_mapper.create_sentiment_map(df_with_sentiment)
            elif analysis_type == 'prediction':
                return self.rating_predictor.train_rating_model(df)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
                
        except Exception as e:
            self.logger.error(f"Error running {analysis_type} analysis: {str(e)}")
            raise


class PipelineMonitor:
    """
    Pipeline monitoring and performance tracking component.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.metrics = {}
        self.start_time = None
    
    def start_monitoring(self):
        """Start pipeline monitoring."""
        self.start_time = time.time()
        self.metrics = {
            'start_time': self.start_time,
            'stages_completed': [],
            'stage_durations': {},
            'memory_usage': [],
            'error_log': []
        }
        self.logger.info("Pipeline monitoring started")
    
    def log_stage_completion(self, stage_name: str, duration: float, memory_usage: float = None):
        """Log completion of a pipeline stage."""
        self.metrics['stages_completed'].append(stage_name)
        self.metrics['stage_durations'][stage_name] = duration
        
        if memory_usage:
            self.metrics['memory_usage'].append({
                'stage': stage_name,
                'memory_mb': memory_usage
            })
        
        self.logger.info(f"Stage '{stage_name}' completed in {duration:.2f} seconds")
    
    def log_error(self, stage_name: str, error: Exception):
        """Log pipeline error."""
        error_info = {
            'stage': stage_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat()
        }
        self.metrics['error_log'].append(error_info)
        self.logger.error(f"Error in stage '{stage_name}': {str(error)}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        if self.start_time:
            total_duration = time.time() - self.start_time
        else:
            total_duration = 0
        
        return {
            'total_duration': total_duration,
            'stages_completed': len(self.metrics['stages_completed']),
            'stage_durations': self.metrics['stage_durations'],
            'average_stage_duration': np.mean(list(self.metrics['stage_durations'].values())) if self.metrics['stage_durations'] else 0,
            'total_errors': len(self.metrics['error_log']),
            'memory_usage': self.metrics['memory_usage'],
            'completion_rate': len(self.metrics['stages_completed']) / 5 * 100  # Assuming 5 main stages
        }