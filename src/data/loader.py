"""
Data loading and ingestion components.
"""

import pandas as pd
import chardet
from pathlib import Path
from typing import Optional, Dict, Any
from src.data.interfaces import DataLoaderInterface
from src.data.models import ValidationResult, ReviewRecord
from src.config import get_config
from src.utils.logger import get_logger
from datetime import datetime


class DataLoader(DataLoaderInterface):
    """Data loader for CSV files with encoding detection and schema validation."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.expected_columns = [
            'review_id', 'user_id', 'app_name', 'app_category', 'review_text',
            'review_language', 'rating', 'review_date', 'verified_purchase',
            'device_type', 'num_helpful_votes', 'user_age', 'user_country',
            'user_gender', 'app_version'
        ]
    
    def detect_encoding(self, file_path: str) -> str:
        """
        Detect file encoding using chardet.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected encoding string
        """
        try:
            with open(file_path, 'rb') as file:
                # Read a sample of the file for encoding detection
                sample = file.read(10000)
                result = chardet.detect(sample)
                encoding = result['encoding']
                confidence = result['confidence']
                
                self.logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
                
                # Fallback to utf-8 if confidence is too low
                if confidence < 0.7:
                    self.logger.warning(f"Low confidence in encoding detection, using UTF-8 as fallback")
                    return 'utf-8'
                
                return encoding
                
        except Exception as e:
            self.logger.error(f"Error detecting encoding: {str(e)}")
            return 'utf-8'  # Default fallback
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load CSV file with automatic encoding detection.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            pd.errors.EmptyDataError: If file is empty
            Exception: For other loading errors
        """
        try:
            # Check if file exists
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            # Detect encoding
            encoding = self.detect_encoding(file_path)
            
            # Load CSV with detected encoding
            self.logger.info(f"Loading CSV file: {file_path}")
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                low_memory=False
            )
            
            self.logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            
            # Basic data type conversions
            df = self._convert_data_types(df)
            
            return df
            
        except FileNotFoundError:
            raise
        except pd.errors.EmptyDataError:
            self.logger.error("CSV file is empty")
            raise
        except UnicodeDecodeError as e:
            self.logger.error(f"Encoding error: {str(e)}")
            # Try with different encoding
            try:
                df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
                self.logger.info("Successfully loaded with latin-1 encoding")
                return self._convert_data_types(df)
            except Exception as fallback_error:
                self.logger.error(f"Failed with fallback encoding: {str(fallback_error)}")
                raise
        except Exception as e:
            self.logger.error(f"Error loading CSV file: {str(e)}")
            raise
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert data types for better memory usage and processing.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with converted types
        """
        try:
            # Convert date column
            if 'review_date' in df.columns:
                df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
            
            # Convert boolean column
            if 'verified_purchase' in df.columns:
                df['verified_purchase'] = df['verified_purchase'].astype('boolean')
            
            # Convert categorical columns for memory efficiency
            categorical_columns = [
                'app_name', 'app_category', 'review_language', 
                'device_type', 'user_country', 'user_gender'
            ]
            
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = df[col].astype('category')
            
            # Convert numeric columns
            if 'rating' in df.columns:
                df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            
            if 'user_age' in df.columns:
                df['user_age'] = pd.to_numeric(df['user_age'], errors='coerce')
            
            if 'num_helpful_votes' in df.columns:
                df['num_helpful_votes'] = pd.to_numeric(df['num_helpful_votes'], errors='coerce')
            
            self.logger.info("Data types converted successfully")
            return df
            
        except Exception as e:
            self.logger.warning(f"Error converting data types: {str(e)}")
            return df
    
    def validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate DataFrame schema against expected structure.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        
        # Check if DataFrame is empty
        if df.empty:
            errors.append("DataFrame is empty")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                summary={"total_rows": 0, "total_columns": 0}
            )
        
        # Check column presence
        missing_columns = set(self.expected_columns) - set(df.columns)
        extra_columns = set(df.columns) - set(self.expected_columns)
        
        if missing_columns:
            errors.append(f"Missing required columns: {list(missing_columns)}")
        
        if extra_columns:
            warnings.append(f"Extra columns found: {list(extra_columns)}")
        
        # Check data types
        type_issues = []
        
        # Check numeric columns
        numeric_columns = ['review_id', 'user_id', 'rating', 'num_helpful_votes', 'user_age']
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    # Check if it can be converted
                    try:
                        pd.to_numeric(df[col], errors='raise')
                    except:
                        type_issues.append(f"Column '{col}' should be numeric")
        
        # Check date column
        if 'review_date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['review_date']):
                try:
                    pd.to_datetime(df['review_date'], errors='raise')
                except:
                    type_issues.append("Column 'review_date' should be datetime")
        
        # Check boolean column
        if 'verified_purchase' in df.columns:
            unique_values = df['verified_purchase'].dropna().unique()
            boolean_values = {True, False, 'True', 'False', 'true', 'false', 1, 0}
            if not all(val in boolean_values for val in unique_values):
                type_issues.append("Column 'verified_purchase' should contain boolean values")
        
        if type_issues:
            warnings.extend(type_issues)
        
        # Check for completely empty columns
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            warnings.append(f"Completely empty columns: {empty_columns}")
        
        # Summary statistics
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_columns": list(missing_columns),
            "extra_columns": list(extra_columns),
            "empty_columns": empty_columns,
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        is_valid = len(errors) == 0
        
        self.logger.info(f"Schema validation completed. Valid: {is_valid}")
        if errors:
            self.logger.error(f"Validation errors: {errors}")
        if warnings:
            self.logger.warning(f"Validation warnings: {warnings}")
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            summary=summary
        )
    
    def load_and_validate(self, file_path: str) -> tuple[pd.DataFrame, ValidationResult]:
        """
        Load CSV file and validate schema in one operation.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Tuple of (DataFrame, ValidationResult)
        """
        df = self.load_csv(file_path)
        validation_result = self.validate_schema(df)
        
        return df, validation_result