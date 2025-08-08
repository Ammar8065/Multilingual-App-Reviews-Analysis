"""
Exploratory Data Analysis components for comprehensive dataset analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.data.interfaces import EDAAnalyzerInterface
from src.data.models import EDAReport
from src.config import get_config
from src.utils.logger import get_logger


class EDAAnalyzer(EDAAnalyzerInterface):
    """Comprehensive exploratory data analysis analyzer."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.visualizations = []
    
    def generate_dataset_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive dataset overview with basic statistics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing dataset overview
        """
        self.logger.info("Generating dataset overview")
        
        # Basic dataset information
        basic_info = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'data_types': df.dtypes.value_counts().to_dict(),
            'column_names': df.columns.tolist()
        }
        
        # Missing values summary
        missing_summary = {
            'total_missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'columns_with_missing': df.columns[df.isnull().any()].tolist(),
            'missing_by_column': df.isnull().sum().to_dict()
        }
        
        # Numerical columns analysis
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_summary = {}
        
        for col in numerical_columns:
            numerical_summary[col] = {
                'count': int(df[col].count()),
                'mean': float(df[col].mean()) if df[col].count() > 0 else None,
                'std': float(df[col].std()) if df[col].count() > 0 else None,
                'min': float(df[col].min()) if df[col].count() > 0 else None,
                'max': float(df[col].max()) if df[col].count() > 0 else None,
                'median': float(df[col].median()) if df[col].count() > 0 else None,
                'q25': float(df[col].quantile(0.25)) if df[col].count() > 0 else None,
                'q75': float(df[col].quantile(0.75)) if df[col].count() > 0 else None,
                'unique_values': int(df[col].nunique()),
                'zero_count': int((df[col] == 0).sum()),
                'negative_count': int((df[col] < 0).sum()) if df[col].count() > 0 else 0
            }
        
        # Categorical columns analysis
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_summary = {}
        
        for col in categorical_columns:
            value_counts = df[col].value_counts()
            categorical_summary[col] = {
                'unique_values': int(df[col].nunique()),
                'most_frequent': value_counts.index[0] if not value_counts.empty else None,
                'most_frequent_count': int(value_counts.iloc[0]) if not value_counts.empty else 0,
                'least_frequent': value_counts.index[-1] if not value_counts.empty else None,
                'least_frequent_count': int(value_counts.iloc[-1]) if not value_counts.empty else 0,
                'top_5_values': value_counts.head(5).to_dict()
            }
        
        # Date columns analysis
        date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        date_summary = {}
        
        for col in date_columns:
            date_data = df[col].dropna()
            if not date_data.empty:
                date_summary[col] = {
                    'earliest_date': date_data.min().isoformat(),
                    'latest_date': date_data.max().isoformat(),
                    'date_range_days': (date_data.max() - date_data.min()).days,
                    'unique_dates': int(date_data.nunique())
                }
        
        # Data quality indicators
        quality_indicators = {
            'completeness_score': ((len(df) * len(df.columns) - df.isnull().sum().sum()) / 
                                  (len(df) * len(df.columns))) * 100,
            'duplicate_rows': int(df.duplicated().sum()),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
            'unique_row_percentage': (len(df.drop_duplicates()) / len(df)) * 100
        }
        
        overview = {
            'basic_info': basic_info,
            'missing_values': missing_summary,
            'numerical_columns': numerical_summary,
            'categorical_columns': categorical_summary,
            'date_columns': date_summary,
            'quality_indicators': quality_indicators,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Dataset overview completed for {len(df)} rows and {len(df.columns)} columns")
        
        return overview
    
    def analyze_language_distribution(self, df: pd.DataFrame, 
                                    language_column: str = 'review_language') -> Dict[str, Any]:
        """
        Analyze distribution of languages in the dataset.
        
        Args:
            df: Input DataFrame
            language_column: Name of the language column
            
        Returns:
            Dictionary containing language analysis
        """
        self.logger.info(f"Analyzing language distribution in column '{language_column}'")
        
        if language_column not in df.columns:
            self.logger.error(f"Language column '{language_column}' not found")
            return {}
        
        # Basic language statistics
        lang_counts = df[language_column].value_counts()
        lang_percentages = df[language_column].value_counts(normalize=True) * 100
        
        basic_stats = {
            'total_reviews': len(df),
            'unique_languages': int(df[language_column].nunique()),
            'most_common_language': lang_counts.index[0] if not lang_counts.empty else None,
            'most_common_count': int(lang_counts.iloc[0]) if not lang_counts.empty else 0,
            'least_common_languages': lang_counts.tail(5).index.tolist(),
            'missing_language_info': int(df[language_column].isnull().sum())
        }
        
        # Language distribution
        distribution = {
            'counts': lang_counts.to_dict(),
            'percentages': {k: round(v, 2) for k, v in lang_percentages.to_dict().items()}
        }
        
        # Language diversity metrics
        diversity_metrics = {
            'shannon_diversity': self._calculate_shannon_diversity(lang_counts),
            'simpson_diversity': self._calculate_simpson_diversity(lang_counts),
            'language_concentration': (lang_counts.iloc[0] / len(df)) * 100 if not lang_counts.empty else 0,
            'languages_covering_80_percent': self._languages_covering_percentage(lang_counts, 80),
            'languages_covering_95_percent': self._languages_covering_percentage(lang_counts, 95)
        }
        
        # Cross-analysis with other columns
        cross_analysis = {}
        
        # Language by app category
        if 'app_category' in df.columns:
            lang_category_crosstab = pd.crosstab(df[language_column], df['app_category'])
            cross_analysis['by_app_category'] = {
                'crosstab': lang_category_crosstab.to_dict(),
                'dominant_language_per_category': lang_category_crosstab.idxmax().to_dict()
            }
        
        # Language by country
        if 'user_country' in df.columns:
            lang_country_crosstab = pd.crosstab(df[language_column], df['user_country'])
            cross_analysis['by_country'] = {
                'crosstab': lang_country_crosstab.to_dict(),
                'countries_per_language': {
                    lang: int(count) for lang, count in 
                    (lang_country_crosstab > 0).sum(axis=1).to_dict().items()
                }
            }
        
        # Language by rating patterns
        if 'rating' in df.columns:
            rating_by_lang = df.groupby(language_column)['rating'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(3)
            cross_analysis['by_rating'] = rating_by_lang.to_dict()
        
        # Text length analysis by language
        if 'review_text' in df.columns:
            df_temp = df.copy()
            df_temp['text_length'] = df_temp['review_text'].astype(str).str.len()
            text_length_by_lang = df_temp.groupby(language_column)['text_length'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(2)
            cross_analysis['by_text_length'] = text_length_by_lang.to_dict()
        
        # Create language distribution visualization
        lang_viz = self._create_language_distribution_plot(lang_counts, lang_percentages)
        self.visualizations.append(lang_viz)
        
        language_analysis = {
            'basic_statistics': basic_stats,
            'distribution': distribution,
            'diversity_metrics': diversity_metrics,
            'cross_analysis': cross_analysis,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Language analysis completed. Found {basic_stats['unique_languages']} unique languages")
        
        return language_analysis
    
    def analyze_rating_patterns(self, df: pd.DataFrame, 
                              rating_column: str = 'rating') -> Dict[str, Any]:
        """
        Analyze rating patterns and distributions.
        
        Args:
            df: Input DataFrame
            rating_column: Name of the rating column
            
        Returns:
            Dictionary containing rating analysis
        """
        self.logger.info(f"Analyzing rating patterns in column '{rating_column}'")
        
        if rating_column not in df.columns:
            self.logger.error(f"Rating column '{rating_column}' not found")
            return {}
        
        ratings = df[rating_column].dropna()
        
        # Basic rating statistics
        basic_stats = {
            'total_ratings': len(ratings),
            'missing_ratings': int(df[rating_column].isnull().sum()),
            'mean_rating': float(ratings.mean()),
            'median_rating': float(ratings.median()),
            'std_rating': float(ratings.std()),
            'min_rating': float(ratings.min()),
            'max_rating': float(ratings.max()),
            'unique_ratings': int(ratings.nunique())
        }
        
        # Rating distribution
        rating_counts = ratings.value_counts().sort_index()
        rating_percentages = ratings.value_counts(normalize=True).sort_index() * 100
        
        distribution = {
            'counts': rating_counts.to_dict(),
            'percentages': {k: round(v, 2) for k, v in rating_percentages.to_dict().items()}
        }
        
        # Rating quality metrics
        quality_metrics = {
            'rating_variance': float(ratings.var()),
            'rating_skewness': float(ratings.skew()),
            'rating_kurtosis': float(ratings.kurtosis()),
            'positive_ratings_pct': float((ratings >= 4.0).mean() * 100),
            'negative_ratings_pct': float((ratings <= 2.0).mean() * 100),
            'neutral_ratings_pct': float(((ratings > 2.0) & (ratings < 4.0)).mean() * 100)
        }
        
        # Rating patterns by other dimensions
        pattern_analysis = {}
        
        # Ratings by app category
        if 'app_category' in df.columns:
            rating_by_category = df.groupby('app_category')[rating_column].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(3)
            pattern_analysis['by_app_category'] = rating_by_category.to_dict()
        
        # Ratings by language
        if 'review_language' in df.columns:
            rating_by_language = df.groupby('review_language')[rating_column].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(3)
            pattern_analysis['by_language'] = rating_by_language.to_dict()
        
        # Ratings by country
        if 'user_country' in df.columns:
            rating_by_country = df.groupby('user_country')[rating_column].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(3)
            # Get top 10 countries by review count
            top_countries = rating_by_country.nlargest(10, 'count')
            pattern_analysis['by_country'] = top_countries.to_dict()
        
        # Ratings by device type
        if 'device_type' in df.columns:
            rating_by_device = df.groupby('device_type')[rating_column].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(3)
            pattern_analysis['by_device_type'] = rating_by_device.to_dict()
        
        # Ratings by verification status
        if 'verified_purchase' in df.columns:
            rating_by_verified = df.groupby('verified_purchase')[rating_column].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(3)
            pattern_analysis['by_verification'] = rating_by_verified.to_dict()
        
        # Rating correlation with other numerical variables
        correlations = {}
        numerical_columns = ['user_age', 'num_helpful_votes']
        
        for col in numerical_columns:
            if col in df.columns:
                correlation = df[rating_column].corr(df[col])
                if not pd.isna(correlation):
                    correlations[col] = float(correlation)
        
        # Create rating distribution visualization
        rating_viz = self._create_rating_distribution_plot(ratings, rating_counts)
        self.visualizations.append(rating_viz)
        
        rating_analysis = {
            'basic_statistics': basic_stats,
            'distribution': distribution,
            'quality_metrics': quality_metrics,
            'pattern_analysis': pattern_analysis,
            'correlations': correlations,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Rating analysis completed. Mean rating: {basic_stats['mean_rating']:.2f}")
        
        return rating_analysis
    
    def analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlations between numerical variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing correlation analysis
        """
        self.logger.info("Analyzing correlations between numerical variables")
        
        # Select numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        
        if numerical_df.empty:
            self.logger.warning("No numerical columns found for correlation analysis")
            return {}
        
        # Calculate correlation matrix
        correlation_matrix = numerical_df.corr()
        
        # Find strong correlations (absolute value > 0.5)
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:
                    strong_correlations.append({
                        'variable1': correlation_matrix.columns[i],
                        'variable2': correlation_matrix.columns[j],
                        'correlation': float(corr_value),
                        'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                    })
        
        # Sort by absolute correlation value
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # Create correlation heatmap
        corr_viz = self._create_correlation_heatmap(correlation_matrix)
        self.visualizations.append(corr_viz)
        
        correlation_analysis = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'numerical_variables': numerical_df.columns.tolist(),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Correlation analysis completed. Found {len(strong_correlations)} strong correlations")
        
        return correlation_analysis
    
    def _calculate_shannon_diversity(self, counts: pd.Series) -> float:
        """Calculate Shannon diversity index for language distribution."""
        proportions = counts / counts.sum()
        return float(-np.sum(proportions * np.log(proportions)))
    
    def _calculate_simpson_diversity(self, counts: pd.Series) -> float:
        """Calculate Simpson diversity index for language distribution."""
        proportions = counts / counts.sum()
        return float(1 - np.sum(proportions ** 2))
    
    def _languages_covering_percentage(self, counts: pd.Series, percentage: float) -> int:
        """Find number of languages covering given percentage of data."""
        cumulative_pct = (counts.cumsum() / counts.sum()) * 100
        return int((cumulative_pct <= percentage).sum())
    
    def _create_language_distribution_plot(self, counts: pd.Series, percentages: pd.Series):
        """Create language distribution visualization."""
        # Take top 15 languages for better visualization
        top_counts = counts.head(15)
        top_percentages = percentages.head(15)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Language Distribution (Counts)', 'Language Distribution (Percentages)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Bar chart for counts
        fig.add_trace(
            go.Bar(
                x=top_counts.index,
                y=top_counts.values,
                name='Review Count',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Bar chart for percentages
        fig.add_trace(
            go.Bar(
                x=top_percentages.index,
                y=top_percentages.values,
                name='Percentage',
                marker_color='lightcoral'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Language Distribution Analysis',
            showlegend=False,
            height=500
        )
        
        fig.update_xaxes(title_text="Language", row=1, col=1)
        fig.update_xaxes(title_text="Language", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Percentage (%)", row=1, col=2)
        
        return fig
    
    def _create_rating_distribution_plot(self, ratings: pd.Series, counts: pd.Series):
        """Create rating distribution visualization."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Rating Distribution', 'Rating Box Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Bar(
                x=counts.index,
                y=counts.values,
                name='Rating Count',
                marker_color='skyblue'
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=ratings,
                name='Rating Distribution',
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Rating Distribution Analysis',
            showlegend=False,
            height=500
        )
        
        fig.update_xaxes(title_text="Rating", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Rating", row=1, col=2)
        
        return fig
    
    def _create_correlation_heatmap(self, correlation_matrix: pd.DataFrame):
        """Create correlation heatmap visualization."""
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Correlation Matrix Heatmap',
            xaxis_title='Variables',
            yaxis_title='Variables',
            height=600,
            width=800
        )
        
        return fig
    
    def generate_eda_report(self, df: pd.DataFrame) -> EDAReport:
        """
        Generate comprehensive EDA report.
        
        Args:
            df: Input DataFrame
            
        Returns:
            EDAReport object containing all analysis results
        """
        self.logger.info("Generating comprehensive EDA report")
        
        # Clear previous visualizations
        self.visualizations = []
        
        # Generate all analyses
        dataset_overview = self.generate_dataset_overview(df)
        language_analysis = self.analyze_language_distribution(df)
        rating_analysis = self.analyze_rating_patterns(df)
        correlation_analysis = self.analyze_correlations(df)
        
        # Additional temporal analysis placeholder (will be implemented in next task)
        temporal_analysis = {'note': 'Temporal analysis will be implemented in Task 4.2'}
        
        # Additional geographic analysis placeholder (will be implemented in next task)
        geographic_analysis = {'note': 'Geographic analysis will be implemented in Task 4.3'}
        
        # Create comprehensive data quality report
        data_quality_report = {
            'completeness_score': dataset_overview['quality_indicators']['completeness_score'],
            'duplicate_percentage': dataset_overview['quality_indicators']['duplicate_percentage'],
            'language_diversity': language_analysis.get('diversity_metrics', {}),
            'rating_quality': rating_analysis.get('quality_metrics', {}),
            'recommendations': self._generate_eda_recommendations(
                dataset_overview, language_analysis, rating_analysis
            )
        }
        
        # Create EDA report
        eda_report = EDAReport(
            dataset_overview=dataset_overview,
            language_analysis=language_analysis,
            rating_analysis=rating_analysis,
            temporal_analysis=temporal_analysis,
            geographic_analysis=geographic_analysis,
            data_quality_report=data_quality_report,
            visualizations=self.visualizations.copy(),
            generated_at=datetime.now()
        )
        
        self.logger.info("EDA report generation completed")
        
        return eda_report
    
    def _generate_eda_recommendations(self, dataset_overview: Dict, 
                                    language_analysis: Dict, 
                                    rating_analysis: Dict) -> List[str]:
        """Generate recommendations based on EDA findings."""
        recommendations = []
        
        # Data quality recommendations
        completeness = dataset_overview['quality_indicators']['completeness_score']
        if completeness < 90:
            recommendations.append(
                f"Data completeness is {completeness:.1f}%. Consider improving data collection processes."
            )
        
        # Language diversity recommendations
        if language_analysis:
            unique_langs = language_analysis['basic_statistics']['unique_languages']
            if unique_langs > 20:
                recommendations.append(
                    f"High language diversity ({unique_langs} languages) detected. "
                    "Consider language-specific analysis strategies."
                )
        
        # Rating distribution recommendations
        if rating_analysis:
            mean_rating = rating_analysis['basic_statistics']['mean_rating']
            if mean_rating > 4.0:
                recommendations.append(
                    "High average rating detected. Consider potential rating bias or selection effects."
                )
            elif mean_rating < 2.5:
                recommendations.append(
                    "Low average rating detected. Investigate potential quality issues."
                )
        
        # Missing data recommendations
        missing_pct = dataset_overview['missing_values']['missing_percentage']
        if missing_pct > 10:
            recommendations.append(
                f"High missing data rate ({missing_pct:.1f}%). Implement robust imputation strategies."
            )
        
        return recommendations