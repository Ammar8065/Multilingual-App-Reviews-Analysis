"""
Data cleaning components for duplicate removal and missing value handling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from src.data.models import ProcessingResult
from src.config import get_config
from src.utils.logger import get_logger


class DataCleaner:
    """Data cleaner for duplicate removal and missing value handling."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        
        # Imputation strategies for different column types
        self.imputation_strategies = {
            'numerical': ['mean', 'median', 'mode', 'zero', 'forward_fill', 'backward_fill'],
            'categorical': ['mode', 'unknown', 'forward_fill', 'backward_fill'],
            'text': ['empty_string', 'unknown', 'forward_fill', 'backward_fill'],
            'datetime': ['forward_fill', 'backward_fill', 'interpolate']
        }
    
    def remove_duplicates(self, df: pd.DataFrame, 
                         strategy: str = 'comprehensive',
                         keep: str = 'first') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Remove duplicate records using various strategies.
        
        Args:
            df: Input DataFrame
            strategy: Duplicate removal strategy ('exact', 'id_based', 'content_based', 'comprehensive')
            keep: Which duplicate to keep ('first', 'last', 'most_informative')
            
        Returns:
            Tuple of (cleaned DataFrame, removal statistics)
        """
        original_count = len(df)
        df_cleaned = df.copy()
        removal_stats = {
            'original_count': original_count,
            'duplicates_removed': 0,
            'removal_details': {}
        }
        
        self.logger.info(f"Starting duplicate removal with strategy: {strategy}")
        
        if strategy in ['exact', 'comprehensive']:
            df_cleaned, exact_stats = self._remove_exact_duplicates(df_cleaned, keep)
            removal_stats['removal_details']['exact_duplicates'] = exact_stats
        
        if strategy in ['id_based', 'comprehensive']:
            df_cleaned, id_stats = self._remove_id_duplicates(df_cleaned, keep)
            removal_stats['removal_details']['id_duplicates'] = id_stats
        
        if strategy in ['content_based', 'comprehensive']:
            df_cleaned, content_stats = self._remove_content_duplicates(df_cleaned, keep)
            removal_stats['removal_details']['content_duplicates'] = content_stats
        
        # Update final statistics
        final_count = len(df_cleaned)
        removal_stats['final_count'] = final_count
        removal_stats['duplicates_removed'] = original_count - final_count
        removal_stats['removal_percentage'] = (removal_stats['duplicates_removed'] / original_count) * 100
        
        self.logger.info(f"Duplicate removal completed. Removed {removal_stats['duplicates_removed']} duplicates ({removal_stats['removal_percentage']:.2f}%)")
        
        return df_cleaned, removal_stats
    
    def _remove_exact_duplicates(self, df: pd.DataFrame, keep: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Remove exact duplicates (all columns identical)."""
        before_count = len(df)
        
        if keep == 'most_informative':
            # Keep the row with least missing values
            duplicated_mask = df.duplicated(keep=False)
            if duplicated_mask.any():
                # For each group of duplicates, keep the one with least missing values
                df_no_duplicates = []
                for group_key, group_df in df[duplicated_mask].groupby(df.columns.tolist()):
                    if len(group_df) > 1:
                        # Calculate missing values for each row
                        missing_counts = group_df.isnull().sum(axis=1)
                        best_idx = missing_counts.idxmin()
                        df_no_duplicates.append(group_df.loc[[best_idx]])
                    else:
                        df_no_duplicates.append(group_df)
                
                if df_no_duplicates:
                    df_no_duplicates = pd.concat(df_no_duplicates, ignore_index=True)
                    df_cleaned = pd.concat([df[~duplicated_mask], df_no_duplicates], ignore_index=True)
                else:
                    df_cleaned = df[~duplicated_mask].copy()
            else:
                df_cleaned = df.copy()
        else:
            df_cleaned = df.drop_duplicates(keep=keep)
        
        after_count = len(df_cleaned)
        removed_count = before_count - after_count
        
        return df_cleaned, {'removed': removed_count, 'before': before_count, 'after': after_count}
    
    def _remove_id_duplicates(self, df: pd.DataFrame, keep: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Remove duplicates based on ID columns."""
        before_count = len(df)
        
        id_columns = ['review_id']  # Primary ID column
        if 'review_id' not in df.columns:
            return df, {'removed': 0, 'before': before_count, 'after': before_count}
        
        if keep == 'most_informative':
            # Keep the row with most information (least missing values)
            duplicated_mask = df.duplicated(subset=id_columns, keep=False)
            if duplicated_mask.any():
                df_no_duplicates = []
                for review_id, group_df in df[duplicated_mask].groupby('review_id'):
                    if len(group_df) > 1:
                        # Calculate information score (inverse of missing values + helpful votes)
                        missing_counts = group_df.isnull().sum(axis=1)
                        helpful_votes = group_df.get('num_helpful_votes', 0).fillna(0)
                        info_scores = -missing_counts + helpful_votes * 0.1  # Weight helpful votes slightly
                        best_idx = info_scores.idxmax()
                        df_no_duplicates.append(group_df.loc[[best_idx]])
                    else:
                        df_no_duplicates.append(group_df)
                
                if df_no_duplicates:
                    df_no_duplicates = pd.concat(df_no_duplicates, ignore_index=True)
                    df_cleaned = pd.concat([df[~duplicated_mask], df_no_duplicates], ignore_index=True)
                else:
                    df_cleaned = df[~duplicated_mask].copy()
            else:
                df_cleaned = df.copy()
        else:
            df_cleaned = df.drop_duplicates(subset=id_columns, keep=keep)
        
        after_count = len(df_cleaned)
        removed_count = before_count - after_count
        
        return df_cleaned, {'removed': removed_count, 'before': before_count, 'after': after_count}
    
    def _remove_content_duplicates(self, df: pd.DataFrame, keep: str, 
                                  similarity_threshold: float = 0.9) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Remove duplicates based on content similarity."""
        before_count = len(df)
        
        if 'review_text' not in df.columns or 'user_id' not in df.columns:
            return df, {'removed': 0, 'before': before_count, 'after': before_count}
        
        # Find similar content within same user
        indices_to_remove = set()
        
        for user_id, user_reviews in df.groupby('user_id'):
            if len(user_reviews) <= 1:
                continue
            
            reviews_list = user_reviews['review_text'].fillna('').tolist()
            indices_list = user_reviews.index.tolist()
            
            # Compare each pair of reviews
            for i in range(len(reviews_list)):
                if indices_list[i] in indices_to_remove:
                    continue
                    
                for j in range(i + 1, len(reviews_list)):
                    if indices_list[j] in indices_to_remove:
                        continue
                    
                    # Calculate similarity
                    similarity = self._calculate_text_similarity(reviews_list[i], reviews_list[j])
                    
                    if similarity >= similarity_threshold:
                        # Decide which one to keep
                        if keep == 'first':
                            indices_to_remove.add(indices_list[j])
                        elif keep == 'last':
                            indices_to_remove.add(indices_list[i])
                        elif keep == 'most_informative':
                            # Keep the one with more helpful votes or less missing data
                            row_i = user_reviews.loc[indices_list[i]]
                            row_j = user_reviews.loc[indices_list[j]]
                            
                            score_i = row_i.get('num_helpful_votes', 0) - row_i.isnull().sum()
                            score_j = row_j.get('num_helpful_votes', 0) - row_j.isnull().sum()
                            
                            if score_i >= score_j:
                                indices_to_remove.add(indices_list[j])
                            else:
                                indices_to_remove.add(indices_list[i])
        
        # Remove identified duplicates
        df_cleaned = df.drop(index=list(indices_to_remove)).reset_index(drop=True)
        
        after_count = len(df_cleaned)
        removed_count = before_count - after_count
        
        return df_cleaned, {'removed': removed_count, 'before': before_count, 'after': after_count}
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using Jaccard similarity."""
        if not text1 or not text2:
            return 0.0
        
        # Convert to sets of words
        words1 = set(str(text1).lower().split())
        words2 = set(str(text2).lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategies: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Handle missing values using specified strategies.
        
        Args:
            df: Input DataFrame
            strategies: Dictionary mapping column names to imputation strategies
            
        Returns:
            Tuple of (cleaned DataFrame, imputation statistics)
        """
        df_cleaned = df.copy()
        imputation_stats = {
            'original_missing_count': df.isnull().sum().sum(),
            'column_stats': {},
            'strategies_applied': {}
        }
        
        # Default strategies if not provided
        if strategies is None:
            strategies = self._get_default_strategies(df)
        
        self.logger.info("Starting missing value imputation")
        
        for column, strategy in strategies.items():
            if column not in df.columns:
                continue
            
            missing_count = df[column].isnull().sum()
            if missing_count == 0:
                continue
            
            self.logger.info(f"Imputing {missing_count} missing values in column '{column}' using strategy '{strategy}'")
            
            original_missing = missing_count
            df_cleaned[column] = self._apply_imputation_strategy(df_cleaned[column], strategy)
            final_missing = df_cleaned[column].isnull().sum()
            
            imputation_stats['column_stats'][column] = {
                'original_missing': int(original_missing),
                'final_missing': int(final_missing),
                'imputed_count': int(original_missing - final_missing),
                'strategy': strategy
            }
            imputation_stats['strategies_applied'][column] = strategy
        
        # Final statistics
        imputation_stats['final_missing_count'] = df_cleaned.isnull().sum().sum()
        imputation_stats['total_imputed'] = (
            imputation_stats['original_missing_count'] - 
            imputation_stats['final_missing_count']
        )
        
        self.logger.info(f"Missing value imputation completed. Imputed {imputation_stats['total_imputed']} values")
        
        return df_cleaned, imputation_stats
    
    def _get_default_strategies(self, df: pd.DataFrame) -> Dict[str, str]:
        """Get default imputation strategies based on column types."""
        strategies = {}
        
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                # Numerical columns
                if column in ['rating', 'user_age']:
                    strategies[column] = 'median'  # More robust for ratings and age
                elif column in ['num_helpful_votes']:
                    strategies[column] = 'zero'  # Assume no votes if missing
                else:
                    strategies[column] = 'mean'
            elif df[column].dtype == 'datetime64[ns]':
                # Datetime columns
                strategies[column] = 'forward_fill'
            elif df[column].dtype == 'bool':
                # Boolean columns
                strategies[column] = 'mode'
            else:
                # Categorical/text columns
                if column in ['user_gender']:
                    strategies[column] = 'unknown'  # Explicit unknown category
                elif column in ['review_text']:
                    strategies[column] = 'empty_string'
                else:
                    strategies[column] = 'mode'
        
        return strategies
    
    def _apply_imputation_strategy(self, series: pd.Series, strategy: str) -> pd.Series:
        """Apply specific imputation strategy to a series."""
        if strategy == 'mean':
            return series.fillna(series.mean())
        elif strategy == 'median':
            return series.fillna(series.median())
        elif strategy == 'mode':
            mode_value = series.mode()
            if not mode_value.empty:
                return series.fillna(mode_value.iloc[0])
            return series
        elif strategy == 'zero':
            return series.fillna(0)
        elif strategy == 'unknown':
            return series.fillna('Unknown')
        elif strategy == 'empty_string':
            return series.fillna('')
        elif strategy == 'forward_fill':
            return series.fillna(method='ffill')
        elif strategy == 'backward_fill':
            return series.fillna(method='bfill')
        elif strategy == 'interpolate':
            return series.interpolate()
        else:
            self.logger.warning(f"Unknown imputation strategy: {strategy}")
            return series
    
    def standardize_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Standardize data formats and values.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (standardized DataFrame, standardization statistics)
        """
        df_standardized = df.copy()
        standardization_stats = {
            'standardizations_applied': [],
            'column_changes': {}
        }
        
        self.logger.info("Starting data standardization")
        
        # Standardize date formats
        if 'review_date' in df.columns:
            original_format_count = df['review_date'].apply(lambda x: str(type(x))).value_counts()
            df_standardized['review_date'] = pd.to_datetime(df_standardized['review_date'], errors='coerce')
            standardization_stats['standardizations_applied'].append('date_standardization')
            standardization_stats['column_changes']['review_date'] = {
                'original_formats': original_format_count.to_dict(),
                'standardized_format': 'datetime64[ns]'
            }
        
        # Standardize rating scale (ensure 1-5 range)
        if 'rating' in df.columns:
            original_range = {
                'min': float(df['rating'].min()),
                'max': float(df['rating'].max())
            }
            
            # Clip ratings to 1-5 range
            df_standardized['rating'] = df_standardized['rating'].clip(1.0, 5.0)
            
            standardization_stats['standardizations_applied'].append('rating_standardization')
            standardization_stats['column_changes']['rating'] = {
                'original_range': original_range,
                'standardized_range': {'min': 1.0, 'max': 5.0}
            }
        
        # Standardize boolean values
        if 'verified_purchase' in df.columns:
            original_values = df['verified_purchase'].value_counts()
            
            # Convert various boolean representations to True/False
            boolean_mapping = {
                'true': True, 'True': True, 'TRUE': True, 1: True, '1': True,
                'false': False, 'False': False, 'FALSE': False, 0: False, '0': False
            }
            
            df_standardized['verified_purchase'] = df_standardized['verified_purchase'].map(
                lambda x: boolean_mapping.get(x, x)
            )
            
            standardization_stats['standardizations_applied'].append('boolean_standardization')
            standardization_stats['column_changes']['verified_purchase'] = {
                'original_values': original_values.to_dict(),
                'standardized_values': df_standardized['verified_purchase'].value_counts().to_dict()
            }
        
        # Standardize categorical values (trim whitespace, consistent casing)
        categorical_columns = ['app_name', 'app_category', 'device_type', 'user_country', 'user_gender']
        
        for column in categorical_columns:
            if column in df.columns:
                original_unique_count = df[column].nunique()
                
                # Trim whitespace and standardize casing
                df_standardized[column] = (
                    df_standardized[column]
                    .astype(str)
                    .str.strip()
                    .str.title()  # Title case for consistency
                )
                
                # Handle special cases
                if column == 'user_gender':
                    gender_mapping = {
                        'M': 'Male', 'F': 'Female', 'Male': 'Male', 'Female': 'Female',
                        'Non-Binary': 'Non-binary', 'Nonbinary': 'Non-binary',
                        'Prefer Not To Say': 'Prefer not to say',
                        'Unknown': 'Unknown', 'Nan': 'Unknown', '': 'Unknown'
                    }
                    df_standardized[column] = df_standardized[column].map(
                        lambda x: gender_mapping.get(x, x)
                    )
                
                final_unique_count = df_standardized[column].nunique()
                
                standardization_stats['column_changes'][column] = {
                    'original_unique_values': original_unique_count,
                    'final_unique_values': final_unique_count,
                    'reduction': original_unique_count - final_unique_count
                }
        
        # Standardize numerical ranges
        if 'user_age' in df.columns:
            original_range = {
                'min': float(df['user_age'].min()) if not df['user_age'].isnull().all() else None,
                'max': float(df['user_age'].max()) if not df['user_age'].isnull().all() else None
            }
            
            # Clip age to reasonable range (13-120)
            df_standardized['user_age'] = df_standardized['user_age'].clip(13, 120)
            
            standardization_stats['column_changes']['user_age'] = {
                'original_range': original_range,
                'standardized_range': {'min': 13, 'max': 120}
            }
        
        self.logger.info(f"Data standardization completed. Applied {len(standardization_stats['standardizations_applied'])} standardizations")
        
        return df_standardized, standardization_stats
    
    def generate_cleaning_summary(self, original_df: pd.DataFrame, 
                                cleaned_df: pd.DataFrame,
                                duplicate_stats: Dict[str, Any],
                                imputation_stats: Dict[str, Any],
                                standardization_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive cleaning summary report.
        
        Args:
            original_df: Original DataFrame before cleaning
            cleaned_df: Cleaned DataFrame after processing
            duplicate_stats: Statistics from duplicate removal
            imputation_stats: Statistics from missing value imputation
            standardization_stats: Statistics from data standardization
            
        Returns:
            Dictionary containing comprehensive cleaning summary
        """
        summary = {
            'processing_timestamp': datetime.now().isoformat(),
            'data_size_changes': {
                'original_rows': len(original_df),
                'original_columns': len(original_df.columns),
                'final_rows': len(cleaned_df),
                'final_columns': len(cleaned_df.columns),
                'rows_removed': len(original_df) - len(cleaned_df),
                'removal_percentage': ((len(original_df) - len(cleaned_df)) / len(original_df)) * 100
            },
            'duplicate_removal': duplicate_stats,
            'missing_value_imputation': imputation_stats,
            'data_standardization': standardization_stats,
            'data_quality_improvement': {
                'original_missing_percentage': (original_df.isnull().sum().sum() / (len(original_df) * len(original_df.columns))) * 100,
                'final_missing_percentage': (cleaned_df.isnull().sum().sum() / (len(cleaned_df) * len(cleaned_df.columns))) * 100,
                'quality_improvement': 0  # Will be calculated below
            }
        }
        
        # Calculate quality improvement
        original_quality = 100 - summary['data_quality_improvement']['original_missing_percentage']
        final_quality = 100 - summary['data_quality_improvement']['final_missing_percentage']
        summary['data_quality_improvement']['quality_improvement'] = final_quality - original_quality
        
        # Add recommendations
        summary['recommendations'] = self._generate_cleaning_recommendations(
            duplicate_stats, imputation_stats, standardization_stats
        )
        
        return summary
    
    def _generate_cleaning_recommendations(self, duplicate_stats: Dict, 
                                         imputation_stats: Dict,
                                         standardization_stats: Dict) -> List[str]:
        """Generate recommendations based on cleaning results."""
        recommendations = []
        
        # Duplicate removal recommendations
        if duplicate_stats.get('duplicates_removed', 0) > 0:
            removal_pct = duplicate_stats.get('removal_percentage', 0)
            if removal_pct > 10:
                recommendations.append(
                    f"High duplicate rate ({removal_pct:.1f}%) detected. "
                    "Consider reviewing data collection process."
                )
        
        # Missing value recommendations
        total_imputed = imputation_stats.get('total_imputed', 0)
        if total_imputed > 0:
            recommendations.append(
                f"Imputed {total_imputed} missing values. "
                "Monitor data quality in future collections."
            )
        
        # Check for columns with high missing rates
        for column, stats in imputation_stats.get('column_stats', {}).items():
            if stats['original_missing'] > stats.get('total_rows', 1) * 0.3:  # >30% missing
                recommendations.append(
                    f"Column '{column}' had high missing rate. "
                    "Consider if this field is necessary or improve collection."
                )
        
        # Standardization recommendations
        if len(standardization_stats.get('standardizations_applied', [])) > 0:
            recommendations.append(
                "Data standardization applied. "
                "Consider implementing validation at data entry point."
            )
        
        return recommendations