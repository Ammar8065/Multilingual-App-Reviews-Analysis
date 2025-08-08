"""
Main entry point for the Multilingual Mobile App Reviews Analysis System.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional
import traceback

from src.config import get_config, update_config
from src.utils.logger import setup_logger
from src.pipeline.orchestrator import PipelineOrchestrator, PipelineMonitor
from src.visualization.dashboard_generator import DashboardGenerator


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Multilingual Mobile App Reviews Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --data-file data.csv --analysis-type all
  python main.py --analysis-type eda --output-dir results/
  python main.py --analysis-type sentiment --verbose
  python main.py --dashboard --port 8050
  python main.py --test-pipeline
        """
    )
    
    # Data and configuration arguments
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (JSON/YAML)"
    )
    
    parser.add_argument(
        "--data-file",
        type=str,
        default="multilingual_mobile_app_reviews_2025.csv",
        help="Path to input data file (default: multilingual_mobile_app_reviews_2025.csv)"
    )
    
    # Analysis type arguments
    parser.add_argument(
        "--analysis-type",
        type=str,
        choices=["eda", "sentiment", "prediction", "time-series", "geographic", "classification", "cross-cultural", "all"],
        default="all",
        help="Type of analysis to perform (default: all)"
    )
    
    parser.add_argument(
        "--analysis-list",
        nargs="+",
        choices=["eda", "sentiment", "prediction", "time-series", "geographic", "classification", "cross-cultural"],
        help="List of specific analyses to run (alternative to --analysis-type)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for results (default: output)"
    )
    
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate results during processing"
    )
    
    parser.add_argument(
        "--export-format",
        choices=["json", "csv", "html", "pdf"],
        default="json",
        help="Export format for results (default: json)"
    )
    
    # Processing arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for processing (default: 500)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of worker processes (default: 4)"
    )
    
    # Dashboard arguments
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch interactive dashboard"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port for dashboard server (default: 8050)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for dashboard server (default: 127.0.0.1)"
    )
    
    # Utility arguments
    parser.add_argument(
        "--test-pipeline",
        action="store_true",
        help="Run pipeline tests"
    )
    
    parser.add_argument(
        "--validate-data",
        action="store_true",
        help="Only validate data without running analysis"
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling"
    )
    
    # Logging arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (except errors)"
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    # Check data file exists
    if not args.test_pipeline and not args.dashboard:
        data_path = Path(args.data_file)
        if not data_path.exists():
            print(f"Error: Data file not found: {args.data_file}")
            return False
    
    # Check output directory is writable
    output_path = Path(args.output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"Error: Cannot create output directory: {args.output_dir}")
        return False
    
    # Validate analysis types
    if args.analysis_list and args.analysis_type != "all":
        print("Error: Cannot specify both --analysis-type and --analysis-list")
        return False
    
    return True


def setup_configuration(args: argparse.Namespace):
    """Setup configuration based on arguments."""
    config = get_config()
    
    # Load custom configuration file if provided
    if args.config:
        try:
            config.load_from_file(args.config)
        except Exception as e:
            print(f"Warning: Could not load config file {args.config}: {e}")
    
    # Update configuration with command line arguments
    update_config(
        input_file_path=args.data_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        save_intermediate_results=args.save_intermediate
    )
    
    return config


def run_pipeline_tests():
    """Run pipeline tests."""
    print("Running pipeline tests...")
    try:
        from tests.test_pipeline import run_all_tests
        result = run_all_tests()
        return result.wasSuccessful()
    except ImportError:
        print("Error: Test modules not found. Please ensure tests are properly installed.")
        return False
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def run_data_validation(data_file: str, logger):
    """Run data validation only."""
    logger.info("Running data validation")
    
    try:
        from src.data.loader import DataLoader
        from src.data.validator import DataValidator
        
        # Load data
        loader = DataLoader()
        df = loader.load_csv(data_file)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        
        # Validate data
        validator = DataValidator()
        validation_result = validator.validate_schema(df)
        
        if validation_result.is_valid:
            logger.info("✅ Data validation passed")
            print("Data validation: PASSED")
        else:
            logger.warning("⚠️ Data validation issues found")
            print("Data validation: ISSUES FOUND")
            for error in validation_result.errors:
                print(f"  - {error}")
        
        # Additional checks
        missing_values = validator.check_missing_values(df)
        duplicates = validator.identify_duplicates(df)
        
        print(f"\nData Quality Summary:")
        print(f"  Total records: {len(df):,}")
        print(f"  Duplicate records: {len(duplicates):,}")
        print(f"  Columns with missing values: {sum(1 for v in missing_values.values() if v > 0)}")
        
        return validation_result.is_valid
        
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return False


def launch_dashboard(args: argparse.Namespace, logger):
    """Launch interactive dashboard."""
    logger.info(f"Launching dashboard on {args.host}:{args.port}")
    
    try:
        import pandas as pd
        from src.visualization.dashboard_generator import DashboardGenerator
        
        # Load data
        df = pd.read_csv(args.data_file)
        logger.info(f"Data loaded for dashboard. Shape: {df.shape}")
        
        # Create dashboard
        dashboard_gen = DashboardGenerator()
        
        # Generate Streamlit app
        streamlit_code = dashboard_gen.generate_streamlit_app(df)
        
        # Save Streamlit app to file
        app_file = Path(args.output_dir) / "dashboard_app.py"
        with open(app_file, 'w') as f:
            f.write(streamlit_code)
        
        print(f"Dashboard app saved to: {app_file}")
        print(f"To run the dashboard, execute:")
        print(f"  streamlit run {app_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Dashboard launch failed: {e}")
        return False


def run_analysis_pipeline(args: argparse.Namespace, logger) -> bool:
    """Run the main analysis pipeline."""
    logger.info("Starting analysis pipeline")
    
    try:
        # Initialize orchestrator and monitor
        orchestrator = PipelineOrchestrator()
        monitor = PipelineMonitor() if args.profile else None
        
        if monitor:
            monitor.start_monitoring()
        
        # Determine analysis types
        if args.analysis_list:
            analysis_types = args.analysis_list
        elif args.analysis_type == "all":
            analysis_types = ["eda", "sentiment", "prediction", "time-series", "geographic"]
        else:
            analysis_types = [args.analysis_type]
        
        logger.info(f"Running analysis types: {analysis_types}")
        
        # Run pipeline
        start_time = time.time()
        results = orchestrator.run_complete_pipeline(args.data_file, analysis_types)
        end_time = time.time()
        
        # Log results
        execution_time = end_time - start_time
        logger.info(f"Pipeline completed in {execution_time:.2f} seconds")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Records processed: {results['pipeline_metadata']['total_records_processed']:,}")
        print(f"Analysis types completed: {len(analysis_types)}")
        print(f"Output directory: {args.output_dir}")
        
        if monitor:
            performance_report = monitor.get_performance_report()
            print(f"Performance summary:")
            print(f"  - Stages completed: {performance_report['stages_completed']}")
            print(f"  - Average stage duration: {performance_report['average_stage_duration']:.2f}s")
            print(f"  - Total errors: {performance_report['total_errors']}")
        
        print(f"{'='*60}")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        logger.error(traceback.format_exc())
        return False


def main():
    """Main application entry point."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Setup configuration
    config = setup_configuration(args)
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else ("ERROR" if args.quiet else "INFO")
    logger = setup_logger(log_level, args.log_file)
    
    # Print banner
    if not args.quiet:
        print("📱 Multilingual Mobile App Reviews Analysis System")
        print("=" * 50)
    
    logger.info("Starting Multilingual Mobile App Reviews Analysis System")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        success = False
        
        # Run tests
        if args.test_pipeline:
            success = run_pipeline_tests()
        
        # Run data validation only
        elif args.validate_data:
            success = run_data_validation(args.data_file, logger)
        
        # Launch dashboard
        elif args.dashboard:
            success = launch_dashboard(args, logger)
        
        # Run analysis pipeline
        else:
            success = run_analysis_pipeline(args, logger)
        
        # Exit with appropriate code
        if success:
            logger.info("Application completed successfully")
            if not args.quiet:
                print("✅ Application completed successfully")
            sys.exit(0)
        else:
            logger.error("Application completed with errors")
            if not args.quiet:
                print("❌ Application completed with errors")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        if not args.quiet:
            print("\n⚠️ Application interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        if not args.quiet:
            print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()