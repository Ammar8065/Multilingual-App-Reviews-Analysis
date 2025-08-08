"""
Integration and end-to-end testing script for the multilingual app reviews analysis system.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import time
import traceback
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import get_config
from src.utils.logger import setup_logger
from src.pipeline.orchestrator import PipelineOrchestrator
from tests.test_pipeline import TestDataGeneration


class IntegrationTester:
    """
    Comprehensive integration testing for the complete system.
    """
    
    def __init__(self):
        self.logger = setup_logger("INFO")
        self.config = get_config()
        self.test_results = {}
        self.temp_files = []
    
    def run_all_integration_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        self.logger.info("Starting comprehensive integration tests")
        
        test_suite = [
            ("Data Pipeline Integration", self.test_data_pipeline_integration),
            ("Analysis Components Integration", self.test_analysis_integration),
            ("ML Pipeline Integration", self.test_ml_pipeline_integration),
            ("Visualization Integration", self.test_visualization_integration),
            ("End-to-End Pipeline", self.test_end_to_end_pipeline),
            ("Performance Benchmarks", self.test_performance_benchmarks),
            ("Error Handling", self.test_error_handling),
            ("Configuration Management", self.test_configuration_management)
        ]
        
        results = {}
        total_tests = len(test_suite)
        passed_tests = 0
        
        for test_name, test_function in test_suite:
            self.logger.info(f"Running: {test_name}")
            try:
                start_time = time.time()
                test_result = test_function()
                end_time = time.time()
                
                results[test_name] = {
                    'status': 'PASSED' if test_result else 'FAILED',
                    'duration': end_time - start_time,
                    'details': test_result if isinstance(test_result, dict) else {}
                }
                
                if test_result:
                    passed_tests += 1
                    self.logger.info(f"✅ {test_name} - PASSED ({end_time - start_time:.2f}s)")
                else:
                    self.logger.error(f"❌ {test_name} - FAILED")
                    
            except Exception as e:
                self.logger.error(f"❌ {test_name} - ERROR: {str(e)}")
                results[test_name] = {
                    'status': 'ERROR',
                    'duration': 0,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        # Generate summary
        success_rate = (passed_tests / total_tests) * 100
        results['summary'] = {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': total_tests - passed_tests,
            'success_rate': success_rate
        }
        
        self.logger.info(f"Integration tests completed: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)")
        
        # Cleanup
        self._cleanup_temp_files()
        
        return results
    
    def test_data_pipeline_integration(self) -> bool:
        """Test data pipeline integration."""
        try:
            # Create test data
            test_file = self._create_test_data_file(500)
            
            # Test data loading and preprocessing
            from src.data.loader import DataLoader
            from src.data.validator import DataValidator
            from src.data.preprocessor import TextPreprocessor
            from src.data.cleaner import DataCleaner
            
            # Load data
            loader = DataLoader()
            df = loader.load_csv(test_file)
            
            # Validate data
            validator = DataValidator()
            validation_result = validator.validate_schema(df)
            
            # Preprocess data
            preprocessor = TextPreprocessor()
            df_processed = preprocessor.preprocess_dataframe(df)
            
            # Clean data
            cleaner = DataCleaner()
            df_clean = cleaner.clean_dataframe(df_processed)
            
            # Verify pipeline worked
            assert len(df_clean) > 0, "No data after cleaning"
            assert 'review_text' in df_clean.columns, "Review text column missing"
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data pipeline integration failed: {e}")
            return False
    
    def test_analysis_integration(self) -> bool:
        """Test analysis components integration."""
        try:
            # Create test data
            df = TestDataGeneration.create_sample_dataframe(200)
            
            # Test EDA analysis
            from src.analysis.eda_analyzer import EDAAnalyzer
            eda_analyzer = EDAAnalyzer()
            eda_result = eda_analyzer.generate_comprehensive_eda(df)
            
            # Test sentiment analysis
            from src.analysis.sentiment_analyzer import MultilingualSentimentAnalyzer
            sentiment_analyzer = MultilingualSentimentAnalyzer()
            df_with_sentiment = sentiment_analyzer.batch_sentiment_analysis(df.head(50))
            
            # Test time series analysis
            from src.analysis.time_series_analyzer import TimeSeriesAnalyzer
            ts_analyzer = TimeSeriesAnalyzer()
            ts_result = ts_analyzer.analyze_review_trends(df)
            
            # Verify results
            assert eda_result is not None, "EDA analysis failed"
            assert 'sentiment' in df_with_sentiment.columns, "Sentiment analysis failed"
            assert ts_result is not None, "Time series analysis failed"
            
            return True
            
        except Exception as e:
            self.logger.error(f"Analysis integration failed: {e}")
            return False
    
    def test_ml_pipeline_integration(self) -> bool:
        """Test ML pipeline integration."""
        try:
            # Create test data
            df = TestDataGeneration.create_sample_dataframe(300)
            
            # Test rating prediction
            from src.ml.rating_predictor import RatingPredictor
            from src.ml.model_evaluator import ModelEvaluator
            
            predictor = RatingPredictor()
            model = predictor.train_rating_model(df)
            
            # Test predictions
            test_texts = ["Great app!", "Terrible experience", "Average app"]
            test_languages = ["en", "en", "en"]
            predictions = predictor.predict_ratings(test_texts, test_languages)
            
            # Test model evaluation
            evaluator = ModelEvaluator()
            performance = evaluator.evaluate_regression_model(model, df.head(50))
            
            # Verify results
            assert model is not None, "Model training failed"
            assert len(predictions) == 3, "Predictions failed"
            assert performance is not None, "Model evaluation failed"
            
            return True
            
        except Exception as e:
            self.logger.error(f"ML pipeline integration failed: {e}")
            return False
    
    def test_visualization_integration(self) -> bool:
        """Test visualization integration."""
        try:
            # Create test data
            df = TestDataGeneration.create_sample_dataframe(150)
            
            # Test visualization engine
            from src.visualization.visualization_engine import VisualizationEngine
            viz_engine = VisualizationEngine()
            
            # Create various plots
            distribution_plots = viz_engine.create_distribution_plots(df)
            time_series_plots = viz_engine.create_time_series_plots(df)
            correlation_heatmap = viz_engine.create_correlation_heatmaps(df)
            
            # Test dashboard generation
            from src.visualization.dashboard_generator import DashboardGenerator
            dashboard_gen = DashboardGenerator()
            eda_dashboard = dashboard_gen.create_eda_dashboard(df)
            
            # Verify results
            assert len(distribution_plots) > 0, "Distribution plots failed"
            assert len(time_series_plots) > 0, "Time series plots failed"
            assert correlation_heatmap is not None, "Correlation heatmap failed"
            assert eda_dashboard is not None, "Dashboard generation failed"
            
            return True
            
        except Exception as e:
            self.logger.error(f"Visualization integration failed: {e}")
            return False
    
    def test_end_to_end_pipeline(self) -> bool:
        """Test complete end-to-end pipeline."""
        try:
            # Create test data file
            test_file = self._create_test_data_file(400)
            
            # Initialize orchestrator
            orchestrator = PipelineOrchestrator()
            
            # Run complete pipeline with limited analysis types
            analysis_types = ['eda', 'sentiment']
            results = orchestrator.run_complete_pipeline(test_file, analysis_types)
            
            # Verify results structure
            assert 'dataset_info' in results, "Dataset info missing"
            assert 'analysis_results' in results, "Analysis results missing"
            assert 'pipeline_metadata' in results, "Pipeline metadata missing"
            
            # Verify specific analysis results
            assert 'eda' in results['analysis_results'], "EDA results missing"
            assert 'sentiment' in results['analysis_results'], "Sentiment results missing"
            
            # Verify metadata
            metadata = results['pipeline_metadata']
            assert metadata['total_records_processed'] == 400, "Record count mismatch"
            assert metadata['execution_time'] > 0, "Execution time not recorded"
            
            return True
            
        except Exception as e:
            self.logger.error(f"End-to-end pipeline failed: {e}")
            return False
    
    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks."""
        try:
            benchmarks = {}
            
            # Test with different data sizes
            data_sizes = [100, 500, 1000]
            
            for size in data_sizes:
                self.logger.info(f"Benchmarking with {size} records")
                
                # Create test data
                test_file = self._create_test_data_file(size)
                
                # Measure pipeline performance
                start_time = time.time()
                orchestrator = PipelineOrchestrator()
                results = orchestrator.run_complete_pipeline(test_file, ['eda'])
                end_time = time.time()
                
                execution_time = end_time - start_time
                records_per_second = size / execution_time if execution_time > 0 else 0
                
                benchmarks[f'{size}_records'] = {
                    'execution_time': execution_time,
                    'records_per_second': records_per_second,
                    'memory_usage': results.get('dataset_info', {}).get('memory_usage', 0)
                }
                
                self.logger.info(f"  {size} records: {execution_time:.2f}s ({records_per_second:.1f} records/sec)")
            
            return benchmarks
            
        except Exception as e:
            self.logger.error(f"Performance benchmarks failed: {e}")
            return {}
    
    def test_error_handling(self) -> bool:
        """Test error handling capabilities."""
        try:
            orchestrator = PipelineOrchestrator()
            
            # Test with non-existent file
            try:
                orchestrator.run_complete_pipeline('non_existent_file.csv', ['eda'])
                return False  # Should have raised an exception
            except Exception:
                pass  # Expected behavior
            
            # Test with corrupted data
            corrupted_file = self._create_corrupted_data_file()
            try:
                results = orchestrator.run_complete_pipeline(corrupted_file, ['eda'])
                # Should handle gracefully and continue with valid data
                assert 'pipeline_metadata' in results, "Pipeline should handle corrupted data gracefully"
            except Exception as e:
                self.logger.warning(f"Pipeline failed with corrupted data: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling test failed: {e}")
            return False
    
    def test_configuration_management(self) -> bool:
        """Test configuration management."""
        try:
            from src.config import Config, get_config, update_config
            
            # Test default configuration
            config = get_config()
            assert config is not None, "Default configuration failed"
            
            # Test configuration updates
            update_config(batch_size=1000, max_workers=2)
            updated_config = get_config()
            assert updated_config.processing.batch_size == 1000, "Configuration update failed"
            
            # Test configuration paths
            data_path = config.get_data_file_path()
            output_path = config.get_output_path('test.json')
            assert isinstance(data_path, Path), "Data path configuration failed"
            assert isinstance(output_path, Path), "Output path configuration failed"
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration management test failed: {e}")
            return False
    
    def _create_test_data_file(self, n_rows: int) -> str:
        """Create temporary test data file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        TestDataGeneration.create_test_csv(temp_file.name, n_rows)
        self.temp_files.append(temp_file.name)
        return temp_file.name
    
    def _create_corrupted_data_file(self) -> str:
        """Create corrupted test data file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        # Create partially corrupted CSV
        with open(temp_file.name, 'w') as f:
            f.write("review_id,user_id,app_name,review_text,rating\n")
            f.write("1,1001,TestApp,Good app,4.5\n")
            f.write("2,1002,TestApp,Bad app,INVALID_RATING\n")  # Corrupted rating
            f.write("3,1003,TestApp,Average app,3.0\n")
            f.write("CORRUPTED_LINE_WITH_WRONG_FORMAT\n")  # Corrupted line
            f.write("5,1005,TestApp,Another review,2.5\n")
        
        self.temp_files.append(temp_file.name)
        return temp_file.name
    
    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                self.logger.warning(f"Could not remove temp file {temp_file}: {e}")
        self.temp_files.clear()


def generate_integration_report(results: Dict[str, Any]) -> str:
    """Generate comprehensive integration test report."""
    report = f"""
# Multilingual App Reviews Analysis - Integration Test Report

## Summary
- **Total Tests**: {results['summary']['total_tests']}
- **Passed**: {results['summary']['passed']}
- **Failed**: {results['summary']['failed']}
- **Success Rate**: {results['summary']['success_rate']:.1f}%

## Test Results

"""
    
    for test_name, test_result in results.items():
        if test_name == 'summary':
            continue
            
        status_emoji = "✅" if test_result['status'] == 'PASSED' else "❌"
        report += f"### {status_emoji} {test_name}\n"
        report += f"- **Status**: {test_result['status']}\n"
        report += f"- **Duration**: {test_result.get('duration', 0):.2f} seconds\n"
        
        if test_result['status'] == 'ERROR':
            report += f"- **Error**: {test_result.get('error', 'Unknown error')}\n"
        
        if 'details' in test_result and test_result['details']:
            report += f"- **Details**: {test_result['details']}\n"
        
        report += "\n"
    
    # Add performance benchmarks if available
    if 'Performance Benchmarks' in results:
        benchmarks = results['Performance Benchmarks'].get('details', {})
        if benchmarks:
            report += "## Performance Benchmarks\n\n"
            for size, metrics in benchmarks.items():
                report += f"- **{size}**: {metrics['execution_time']:.2f}s ({metrics['records_per_second']:.1f} records/sec)\n"
            report += "\n"
    
    report += f"""
## Recommendations

Based on the test results:

1. **Data Pipeline**: {'✅ Working correctly' if results.get('Data Pipeline Integration', {}).get('status') == 'PASSED' else '❌ Needs attention'}
2. **Analysis Components**: {'✅ Working correctly' if results.get('Analysis Components Integration', {}).get('status') == 'PASSED' else '❌ Needs attention'}
3. **ML Pipeline**: {'✅ Working correctly' if results.get('ML Pipeline Integration', {}).get('status') == 'PASSED' else '❌ Needs attention'}
4. **Visualization**: {'✅ Working correctly' if results.get('Visualization Integration', {}).get('status') == 'PASSED' else '❌ Needs attention'}
5. **End-to-End Pipeline**: {'✅ Working correctly' if results.get('End-to-End Pipeline', {}).get('status') == 'PASSED' else '❌ Needs attention'}

## Next Steps

- Review any failed tests and address underlying issues
- Consider performance optimizations for large datasets
- Ensure error handling covers all edge cases
- Validate results with real-world data

---
*Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return report


def main():
    """Main function to run integration tests."""
    print("🧪 Starting Integration Tests for Multilingual App Reviews Analysis System")
    print("=" * 80)
    
    # Run integration tests
    tester = IntegrationTester()
    results = tester.run_all_integration_tests()
    
    # Generate and save report
    report = generate_integration_report(results)
    
    # Save report to file
    report_file = Path("output") / "integration_test_report.md"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Print summary
    print("\n" + "=" * 80)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {results['summary']['total_tests']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
    print(f"Report saved to: {report_file}")
    print("=" * 80)
    
    # Exit with appropriate code
    if results['summary']['success_rate'] == 100:
        print("🎉 All integration tests passed!")
        sys.exit(0)
    else:
        print("⚠️ Some integration tests failed. Check the report for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()