"""
Comprehensive testing framework for the multilingual app reviews analysis pipeline.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path
import json

from src.pipeline.orchestrator import PipelineOrchestrator, PipelineMonitor
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.data.preprocessor import TextPreprocessor
from src.analysis.eda_analyzer import EDAAnalyzer
from src.analysis.sentiment_analyzer import MultilingualSentimentAnalyzer
from src.ml.rating_predictor import RatingPredictor
from src.config import get_config


class TestDataGeneration:
    """Generate test data for pipeline testing."""
    
    @staticmethod
    def create_sample_dataframe(n_rows: int = 100) -> pd.DataFrame:
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        
        apps = ['TestApp1', 'TestApp2', 'TestApp3', 'TestApp4', 'TestApp5']
        categories = ['Social', 'Games', 'Productivity', 'Entertainment', 'Education']
        languages = ['en', 'es', 'fr', 'de', 'zh']
        countries = ['USA', 'Spain', 'France', 'Germany', 'China']
        devices = ['iOS', 'Android', 'iPad', 'Android Tablet']
        genders = ['Male', 'Female', 'Non-binary', 'Prefer not to say']
        
        data = {
            'review_id': range(1, n_rows + 1),
            'user_id': np.random.randint(1000, 9999, n_rows),
            'app_name': np.random.choice(apps, n_rows),
            'app_category': np.random.choice(categories, n_rows),
            'review_text': [f"This is a test review {i} with some content." for i in range(n_rows)],
            'review_language': np.random.choice(languages, n_rows),
            'rating': np.random.uniform(1.0, 5.0, n_rows),
            'review_date': pd.date_range('2023-01-01', periods=n_rows, freq='D'),
            'verified_purchase': np.random.choice([True, False], n_rows),
            'device_type': np.random.choice(devices, n_rows),
            'num_helpful_votes': np.random.randint(0, 100, n_rows),
            'user_age': np.random.uniform(18, 70, n_rows),
            'user_country': np.random.choice(countries, n_rows),
            'user_gender': np.random.choice(genders, n_rows),
            'app_version': [f"1.{np.random.randint(0, 10)}.{np.random.randint(0, 10)}" for _ in range(n_rows)]
        }
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_test_csv(file_path: str, n_rows: int = 100):
        """Create test CSV file."""
        df = TestDataGeneration.create_sample_dataframe(n_rows)
        df.to_csv(file_path, index=False)
        return df


class TestDataComponents(unittest.TestCase):
    """Test data processing components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_df = TestDataGeneration.create_sample_dataframe(50)
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv_path = os.path.join(self.temp_dir, 'test_data.csv')
        TestDataGeneration.create_test_csv(self.test_csv_path, 50)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)
        os.rmdir(self.temp_dir)
    
    def test_data_loader(self):
        """Test DataLoader functionality."""
        loader = DataLoader()
        
        # Test CSV loading
        df = loader.load_csv(self.test_csv_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 50)
        self.assertIn('review_text', df.columns)
        
        # Test encoding detection
        encoding = loader.detect_encoding(self.test_csv_path)
        self.assertIsInstance(encoding, str)
        
        # Test schema validation
        validation_result = loader.validate_schema(df)
        self.assertTrue(hasattr(validation_result, 'is_valid'))
    
    def test_data_validator(self):
        """Test DataValidator functionality."""
        validator = DataValidator()
        
        # Test missing values check
        missing_values = validator.check_missing_values(self.test_df)
        self.assertIsInstance(missing_values, dict)
        
        # Test duplicate identification
        duplicates = validator.identify_duplicates(self.test_df)
        self.assertIsInstance(duplicates, pd.DataFrame)
        
        # Test data type validation
        validation_errors = validator.validate_data_types(self.test_df)
        self.assertIsInstance(validation_errors, list)
    
    def test_text_preprocessor(self):
        """Test TextPreprocessor functionality."""
        preprocessor = TextPreprocessor()
        
        # Test text cleaning
        test_text = "This is a TEST text with SPECIAL characters!!! @#$"
        cleaned_text = preprocessor.clean_text(test_text, 'en')
        self.assertIsInstance(cleaned_text, str)
        self.assertNotEqual(cleaned_text, test_text)
        
        # Test encoding normalization
        normalized_text = preprocessor.normalize_encoding(test_text)
        self.assertIsInstance(normalized_text, str)
        
        # Test tokenization
        tokens = preprocessor.tokenize_multilingual(test_text, 'en')
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)


class TestAnalysisComponents(unittest.TestCase):
    """Test analysis components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_df = TestDataGeneration.create_sample_dataframe(100)
    
    def test_eda_analyzer(self):
        """Test EDAAnalyzer functionality."""
        analyzer = EDAAnalyzer()
        
        # Test dataset overview
        overview = analyzer.generate_dataset_overview(self.test_df)
        self.assertIsInstance(overview, dict)
        self.assertIn('shape', overview)
        self.assertIn('columns', overview)
        
        # Test language distribution analysis
        lang_analysis = analyzer.analyze_language_distribution(self.test_df)
        self.assertIsInstance(lang_analysis, dict)
        
        # Test rating patterns analysis
        rating_analysis = analyzer.analyze_rating_patterns(self.test_df)
        self.assertIsInstance(rating_analysis, dict)
    
    def test_sentiment_analyzer(self):
        """Test MultilingualSentimentAnalyzer functionality."""
        analyzer = MultilingualSentimentAnalyzer()
        
        # Test single text sentiment analysis
        test_text = "This is a great app, I love it!"
        sentiment_result = analyzer.analyze_sentiment(test_text, 'en')
        self.assertIsInstance(sentiment_result.sentiment, str)
        self.assertIn(sentiment_result.sentiment, ['positive', 'negative', 'neutral'])
        self.assertIsInstance(sentiment_result.confidence, float)
        
        # Test batch sentiment analysis
        df_with_sentiment = analyzer.batch_sentiment_analysis(self.test_df.head(10))
        self.assertIn('sentiment', df_with_sentiment.columns)
        self.assertEqual(len(df_with_sentiment), 10)
    
    def test_rating_predictor(self):
        """Test RatingPredictor functionality."""
        predictor = RatingPredictor()
        
        # Test model training
        model = predictor.train_rating_model(self.test_df)
        self.assertIsNotNone(model)
        
        # Test prediction
        test_texts = ["Great app!", "Terrible experience"]
        test_languages = ["en", "en"]
        predictions = predictor.predict_ratings(test_texts, test_languages)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(isinstance(pred, (int, float)) for pred in predictions))


class TestPipelineOrchestrator(unittest.TestCase):
    """Test pipeline orchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv_path = os.path.join(self.temp_dir, 'test_data.csv')
        TestDataGeneration.create_test_csv(self.test_csv_path, 100)
        self.orchestrator = PipelineOrchestrator()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)
        os.rmdir(self.temp_dir)
    
    def test_component_initialization(self):
        """Test that all components are properly initialized."""
        self.assertIsNotNone(self.orchestrator.data_loader)
        self.assertIsNotNone(self.orchestrator.data_validator)
        self.assertIsNotNone(self.orchestrator.eda_analyzer)
        self.assertIsNotNone(self.orchestrator.sentiment_analyzer)
        self.assertIsNotNone(self.orchestrator.rating_predictor)
    
    def test_data_loading_stage(self):
        """Test data loading stage."""
        df = self.orchestrator._run_data_loading_stage(self.test_csv_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('data_loading_stage', self.orchestrator.results)
    
    def test_preprocessing_stage(self):
        """Test preprocessing stage."""
        df = TestDataGeneration.create_sample_dataframe(50)
        df_processed = self.orchestrator._run_preprocessing_stage(df)
        self.assertIsInstance(df_processed, pd.DataFrame)
        self.assertIn('preprocessing_stage', self.orchestrator.results)
    
    def test_specific_analysis(self):
        """Test running specific analysis types."""
        df = TestDataGeneration.create_sample_dataframe(50)
        
        # Test EDA analysis
        eda_result = self.orchestrator.run_specific_analysis(df, 'eda')
        self.assertIsNotNone(eda_result)
        
        # Test sentiment analysis
        sentiment_result = self.orchestrator.run_specific_analysis(df, 'sentiment')
        self.assertIsInstance(sentiment_result, pd.DataFrame)
        self.assertIn('sentiment', sentiment_result.columns)
    
    def test_pipeline_status(self):
        """Test pipeline status tracking."""
        status = self.orchestrator.get_pipeline_status()
        self.assertIsInstance(status, dict)
        self.assertIn('pipeline_state', status)
        self.assertIn('error_count', status)
        self.assertIn('timestamp', status)


class TestPipelineMonitor(unittest.TestCase):
    """Test pipeline monitoring."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PipelineMonitor()
    
    def test_monitoring_initialization(self):
        """Test monitor initialization."""
        self.monitor.start_monitoring()
        self.assertIsNotNone(self.monitor.start_time)
        self.assertIsInstance(self.monitor.metrics, dict)
        self.assertIn('stages_completed', self.monitor.metrics)
    
    def test_stage_logging(self):
        """Test stage completion logging."""
        self.monitor.start_monitoring()
        self.monitor.log_stage_completion('test_stage', 1.5, 100.0)
        
        self.assertIn('test_stage', self.monitor.metrics['stages_completed'])
        self.assertEqual(self.monitor.metrics['stage_durations']['test_stage'], 1.5)
    
    def test_error_logging(self):
        """Test error logging."""
        self.monitor.start_monitoring()
        test_error = ValueError("Test error")
        self.monitor.log_error('test_stage', test_error)
        
        self.assertEqual(len(self.monitor.metrics['error_log']), 1)
        self.assertEqual(self.monitor.metrics['error_log'][0]['error_type'], 'ValueError')
    
    def test_performance_report(self):
        """Test performance report generation."""
        self.monitor.start_monitoring()
        self.monitor.log_stage_completion('stage1', 1.0)
        self.monitor.log_stage_completion('stage2', 2.0)
        
        report = self.monitor.get_performance_report()
        self.assertIsInstance(report, dict)
        self.assertIn('total_duration', report)
        self.assertIn('stages_completed', report)
        self.assertEqual(report['stages_completed'], 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv_path = os.path.join(self.temp_dir, 'test_data.csv')
        TestDataGeneration.create_test_csv(self.test_csv_path, 200)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)
        os.rmdir(self.temp_dir)
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline execution."""
        orchestrator = PipelineOrchestrator()
        
        # Run a minimal pipeline with basic analysis types
        analysis_types = ['eda', 'sentiment']
        
        try:
            results = orchestrator.run_complete_pipeline(self.test_csv_path, analysis_types)
            
            # Verify results structure
            self.assertIsInstance(results, dict)
            self.assertIn('dataset_info', results)
            self.assertIn('analysis_results', results)
            self.assertIn('pipeline_metadata', results)
            
            # Verify analysis results
            self.assertIn('eda', results['analysis_results'])
            self.assertIn('sentiment', results['analysis_results'])
            
            # Verify pipeline metadata
            metadata = results['pipeline_metadata']
            self.assertIn('execution_time', metadata)
            self.assertIn('total_records_processed', metadata)
            self.assertEqual(metadata['total_records_processed'], 200)
            
        except Exception as e:
            self.fail(f"End-to-end pipeline test failed: {str(e)}")
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling."""
        orchestrator = PipelineOrchestrator()
        
        # Test with non-existent file
        with self.assertRaises(Exception):
            orchestrator.run_complete_pipeline('non_existent_file.csv')
    
    def test_pipeline_with_monitoring(self):
        """Test pipeline execution with monitoring."""
        orchestrator = PipelineOrchestrator()
        monitor = PipelineMonitor()
        
        monitor.start_monitoring()
        
        try:
            # Run minimal pipeline
            results = orchestrator.run_complete_pipeline(self.test_csv_path, ['eda'])
            
            # Log completion
            monitor.log_stage_completion('complete_pipeline', 10.0)
            
            # Get performance report
            report = monitor.get_performance_report()
            self.assertGreater(report['total_duration'], 0)
            
        except Exception as e:
            monitor.log_error('pipeline_execution', e)
            self.fail(f"Monitored pipeline test failed: {str(e)}")


class TestDataQuality(unittest.TestCase):
    """Test data quality validation and alerts."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_df = TestDataGeneration.create_sample_dataframe(100)
    
    def test_missing_data_detection(self):
        """Test missing data detection."""
        # Introduce missing values
        df_with_missing = self.test_df.copy()
        df_with_missing.loc[0:10, 'user_age'] = np.nan
        df_with_missing.loc[5:15, 'review_text'] = np.nan
        
        validator = DataValidator()
        missing_values = validator.check_missing_values(df_with_missing)
        
        self.assertGreater(missing_values['user_age'], 0)
        self.assertGreater(missing_values['review_text'], 0)
    
    def test_duplicate_detection(self):
        """Test duplicate detection."""
        # Introduce duplicates
        df_with_duplicates = pd.concat([self.test_df, self.test_df.head(10)], ignore_index=True)
        
        validator = DataValidator()
        duplicates = validator.identify_duplicates(df_with_duplicates)
        
        self.assertGreater(len(duplicates), 0)
    
    def test_data_type_validation(self):
        """Test data type validation."""
        # Introduce type issues
        df_with_type_issues = self.test_df.copy()
        df_with_type_issues.loc[0:5, 'rating'] = 'invalid_rating'
        
        validator = DataValidator()
        validation_errors = validator.validate_data_types(df_with_type_issues)
        
        # Should detect type issues
        self.assertGreater(len(validation_errors), 0)


def run_all_tests():
    """Run all tests and generate coverage report."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDataComponents,
        TestAnalysisComponents,
        TestPipelineOrchestrator,
        TestPipelineMonitor,
        TestIntegration,
        TestDataQuality
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Generate summary
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = ((total_tests - failures - errors) / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"{'='*50}")
    
    return result


if __name__ == '__main__':
    run_all_tests()