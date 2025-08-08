"""
Data validation components for quality checks and data integrity.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter
from src.data.interfaces import DataValidatorInterface
from src.data.models import ValidationError
from src.utils.logger import get_logger
from difflib import SequenceMatcher


class DataValidator(DataValidatorInterface):
    """Comprehensive data validator for quality checks."""
    
    def __init__(self):
        self.logger = get_logger()
    
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Check for missing values in each column.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with column names and missing value percentages
        """
        missing_stats = {}
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            missing_stats[column] = {
                'count': int(missing_count),
                'percentage': round(missing_percentage, 2)
            }
        
        # Log summary
        high_missing = {col: stats for col, stats in missing_stats.items() 
                       if stats['percentage'] > 10}
        
        if high_missing:
            self.logger.warning(f"Columns with >10% missing values: {list(high_missing.keys())}")
        
        self.logger.info(f"Missing value analysis completed for {len(df.columns)} columns")
        
        return missing_stats
    
    def identify_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify duplicate records based on multiple criteria.
        
        Args:
            df: DataFrame to check for duplicates
            
        Returns:
            DataFrame containing duplicate records
        """
        duplicates_info = []
        
        # 1. Exact duplicates (all columns)
        exact_duplicates = df[df.duplicated(keep=False)]
        if not exact_duplicates.empty:
            self.logger.warning(f"Found {len(exact_duplicates)} exact duplicate rows")
            duplicates_info.extend([
                {'type': 'exact_duplicate', 'index': idx, 'reason': 'All columns identical'}
                for idx in exact_duplicates.index
            ])
        
        # 2. ID-based duplicates
        if 'review_id' in df.columns:
            id_duplicates = df[df.duplicated(subset=['review_id'], keep=False)]
            if not id_duplicates.empty:
                self.logger.warning(f"Found {len(id_duplicates)} duplicate review IDs")
                duplicates_info.extend([
                    {'type': 'id_duplicate', 'index': idx, 'reason': 'Duplicate review_id'}
                    for idx in id_duplicates.index
                ])
        
        # 3. Content-based duplicates (similar review text)
        if 'review_text' in df.columns and 'user_id' in df.columns:
            content_duplicates = self._find_similar_reviews(df)
            duplicates_info.extend(content_duplicates)
        
        # Create summary DataFrame
        if duplicates_info:
            duplicates_df = pd.DataFrame(duplicates_info)
            duplicates_df = duplicates_df.merge(
                df.reset_index(), 
                left_on='index', 
                right_on='index', 
                how='left'
            )
        else:
            duplicates_df = pd.DataFrame()
        
        self.logger.info(f"Duplicate analysis completed. Found {len(duplicates_info)} potential duplicates")
        
        return duplicates_df
    
    def _find_similar_reviews(self, df: pd.DataFrame, similarity_threshold: float = 0.9) -> List[Dict]:
        """
        Find reviews with similar text content.
        
        Args:
            df: DataFrame with review_text column
            similarity_threshold: Minimum similarity score to consider as duplicate
            
        Returns:
            List of similar review information
        """
        similar_reviews = []
        
        # Group by user to find similar reviews from same user
        if 'user_id' in df.columns:
            for user_id, user_reviews in df.groupby('user_id'):
                if len(user_reviews) > 1:
                    reviews_list = user_reviews['review_text'].dropna().tolist()
                    indices_list = user_reviews['review_text'].dropna().index.tolist()
                    
                    # Compare each pair of reviews
                    for i in range(len(reviews_list)):
                        for j in range(i + 1, len(reviews_list)):
                            similarity = SequenceMatcher(
                                None, 
                                str(reviews_list[i]).lower(), 
                                str(reviews_list[j]).lower()
                            ).ratio()
                            
                            if similarity >= similarity_threshold:
                                similar_reviews.extend([
                                    {
                                        'type': 'similar_content',
                                        'index': indices_list[i],
                                        'reason': f'Similar to review {indices_list[j]} (similarity: {similarity:.2f})'
                                    },
                                    {
                                        'type': 'similar_content',
                                        'index': indices_list[j],
                                        'reason': f'Similar to review {indices_list[i]} (similarity: {similarity:.2f})'
                                    }
                                ])
        
        return similar_reviews
    
    def validate_data_types(self, df: pd.DataFrame) -> List[ValidationError]:
        """
        Validate data types for each column.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            List of ValidationError objects
        """
        validation_errors = []
        
        # Define expected data types and validation rules
        validation_rules = {
            'review_id': {'type': 'int', 'min_value': 1},
            'user_id': {'type': 'int', 'min_value': 1},
            'rating': {'type': 'float', 'min_value': 1.0, 'max_value': 5.0},
            'review_date': {'type': 'datetime'},
            'verified_purchase': {'type': 'bool'},
            'num_helpful_votes': {'type': 'int', 'min_value': 0},
            'user_age': {'type': 'float', 'min_value': 13, 'max_value': 120},
            'review_text': {'type': 'str', 'min_length': 1},
            'review_language': {'type': 'str', 'valid_codes': True},
            'app_name': {'type': 'str', 'min_length': 1},
            'app_category': {'type': 'str', 'min_length': 1},
            'device_type': {'type': 'str', 'min_length': 1},
            'user_country': {'type': 'str', 'min_length': 1},
            'user_gender': {'type': 'str', 'optional': True},
            'app_version': {'type': 'str', 'min_length': 1}
        }
        
        for column, rules in validation_rules.items():
            if column not in df.columns:
                continue
                
            column_data = df[column].dropna()  # Skip NaN values for validation
            
            if column_data.empty:
                continue
            
            # Type validation
            if rules['type'] == 'int':
                invalid_rows = self._validate_integer_column(df, column, rules)
            elif rules['type'] == 'float':
                invalid_rows = self._validate_float_column(df, column, rules)
            elif rules['type'] == 'datetime':
                invalid_rows = self._validate_datetime_column(df, column)
            elif rules['type'] == 'bool':
                invalid_rows = self._validate_boolean_column(df, column)
            elif rules['type'] == 'str':
                invalid_rows = self._validate_string_column(df, column, rules)
            else:
                continue
            
            if invalid_rows:
                validation_errors.append(ValidationError(
                    column=column,
                    error_type=f"invalid_{rules['type']}",
                    message=f"Invalid {rules['type']} values in column '{column}'",
                    affected_rows=invalid_rows
                ))
        
        self.logger.info(f"Data type validation completed. Found {len(validation_errors)} validation errors")
        
        return validation_errors
    
    def _validate_integer_column(self, df: pd.DataFrame, column: str, rules: Dict) -> List[int]:
        """Validate integer column values."""
        invalid_rows = []
        
        for idx, value in df[column].items():
            if pd.isna(value):
                continue
                
            try:
                int_value = int(float(value))  # Handle float representations of integers
                
                # Check range constraints
                if 'min_value' in rules and int_value < rules['min_value']:
                    invalid_rows.append(idx)
                elif 'max_value' in rules and int_value > rules['max_value']:
                    invalid_rows.append(idx)
                    
            except (ValueError, TypeError):
                invalid_rows.append(idx)
        
        return invalid_rows
    
    def _validate_float_column(self, df: pd.DataFrame, column: str, rules: Dict) -> List[int]:
        """Validate float column values."""
        invalid_rows = []
        
        for idx, value in df[column].items():
            if pd.isna(value):
                continue
                
            try:
                float_value = float(value)
                
                # Check range constraints
                if 'min_value' in rules and float_value < rules['min_value']:
                    invalid_rows.append(idx)
                elif 'max_value' in rules and float_value > rules['max_value']:
                    invalid_rows.append(idx)
                    
            except (ValueError, TypeError):
                invalid_rows.append(idx)
        
        return invalid_rows
    
    def _validate_datetime_column(self, df: pd.DataFrame, column: str) -> List[int]:
        """Validate datetime column values."""
        invalid_rows = []
        
        for idx, value in df[column].items():
            if pd.isna(value):
                continue
                
            try:
                pd.to_datetime(value)
            except (ValueError, TypeError):
                invalid_rows.append(idx)
        
        return invalid_rows
    
    def _validate_boolean_column(self, df: pd.DataFrame, column: str) -> List[int]:
        """Validate boolean column values."""
        invalid_rows = []
        valid_boolean_values = {True, False, 'True', 'False', 'true', 'false', 1, 0, '1', '0'}
        
        for idx, value in df[column].items():
            if pd.isna(value):
                continue
                
            if value not in valid_boolean_values:
                invalid_rows.append(idx)
        
        return invalid_rows
    
    def _validate_string_column(self, df: pd.DataFrame, column: str, rules: Dict) -> List[int]:
        """Validate string column values."""
        invalid_rows = []
        
        for idx, value in df[column].items():
            if pd.isna(value):
                continue
                
            try:
                str_value = str(value).strip()
                
                # Check minimum length
                if 'min_length' in rules and len(str_value) < rules['min_length']:
                    invalid_rows.append(idx)
                
                # Special validation for language codes
                if column == 'review_language' and rules.get('valid_codes'):
                    if not self._is_valid_language_code(str_value):
                        invalid_rows.append(idx)
                        
            except (ValueError, TypeError):
                invalid_rows.append(idx)
        
        return invalid_rows
    
    def _is_valid_language_code(self, lang_code: str) -> bool:
        """Check if language code is valid (basic validation)."""
        # Basic validation - should be 2-3 characters, alphabetic
        lang_code = lang_code.lower().strip()
        return len(lang_code) in [2, 3] and lang_code.isalpha()
    
    def generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing quality metrics
        """
        self.logger.info("Generating comprehensive data quality report")
        
        # Basic statistics
        basic_stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Missing values analysis
        missing_values = self.check_missing_values(df)
        
        # Duplicate analysis
        duplicates = self.identify_duplicates(df)
        
        # Data type validation
        type_errors = self.validate_data_types(df)
        
        # Column-specific analysis
        column_analysis = {}
        for column in df.columns:
            col_data = df[column].dropna()
            
            analysis = {
                'data_type': str(df[column].dtype),
                'unique_values': col_data.nunique(),
                'missing_count': df[column].isnull().sum(),
                'missing_percentage': (df[column].isnull().sum() / len(df)) * 100
            }
            
            # Add specific analysis based on column type
            if pd.api.types.is_numeric_dtype(col_data):
                analysis.update({
                    'min': float(col_data.min()) if not col_data.empty else None,
                    'max': float(col_data.max()) if not col_data.empty else None,
                    'mean': float(col_data.mean()) if not col_data.empty else None,
                    'std': float(col_data.std()) if not col_data.empty else None
                })
            elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
                if not col_data.empty:
                    value_counts = col_data.value_counts().head(10)
                    analysis['top_values'] = value_counts.to_dict()
                    analysis['avg_length'] = col_data.astype(str).str.len().mean()
            
            column_analysis[column] = analysis
        
        # Quality score calculation
        quality_score = self._calculate_quality_score(
            missing_values, len(duplicates), len(type_errors), len(df)
        )
        
        quality_report = {
            'basic_statistics': basic_stats,
            'missing_values': missing_values,
            'duplicates': {
                'count': len(duplicates),
                'details': duplicates.to_dict('records') if not duplicates.empty else []
            },
            'validation_errors': [
                {
                    'column': error.column,
                    'error_type': error.error_type,
                    'message': error.message,
                    'affected_rows_count': len(error.affected_rows)
                }
                for error in type_errors
            ],
            'column_analysis': column_analysis,
            'quality_score': quality_score,
            'recommendations': self._generate_recommendations(missing_values, duplicates, type_errors)
        }
        
        self.logger.info(f"Data quality report generated. Quality score: {quality_score:.2f}/100")
        
        return quality_report
    
    def _calculate_quality_score(self, missing_values: Dict, duplicate_count: int, 
                                error_count: int, total_rows: int) -> float:
        """Calculate overall data quality score (0-100)."""
        score = 100.0
        
        # Penalize for missing values
        avg_missing_pct = np.mean([stats['percentage'] for stats in missing_values.values()])
        score -= min(avg_missing_pct, 30)  # Max 30 point deduction
        
        # Penalize for duplicates
        duplicate_pct = (duplicate_count / total_rows) * 100 if total_rows > 0 else 0
        score -= min(duplicate_pct * 2, 20)  # Max 20 point deduction
        
        # Penalize for validation errors
        error_pct = (error_count / total_rows) * 100 if total_rows > 0 else 0
        score -= min(error_pct * 3, 25)  # Max 25 point deduction
        
        return max(score, 0.0)
    
    def _generate_recommendations(self, missing_values: Dict, duplicates: pd.DataFrame, 
                                 errors: List[ValidationError]) -> List[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []
        
        # Missing values recommendations
        high_missing = [col for col, stats in missing_values.items() 
                       if stats['percentage'] > 20]
        if high_missing:
            recommendations.append(
                f"Consider imputation or removal for columns with high missing values: {high_missing}"
            )
        
        # Duplicates recommendations
        if not duplicates.empty:
            recommendations.append(
                f"Remove or merge {len(duplicates)} duplicate records to improve data quality"
            )
        
        # Validation errors recommendations
        if errors:
            error_columns = list(set(error.column for error in errors))
            recommendations.append(
                f"Fix data type issues in columns: {error_columns}"
            )
        
        # General recommendations
        recommendations.extend([
            "Standardize categorical values for consistency",
            "Validate language codes against ISO standards",
            "Consider outlier detection for numerical columns"
        ])
        
        return recommendations