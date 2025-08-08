"""
Temporal analysis components for time-based pattern detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.data.models import TimeSeriesResult
from src.config import get_config
from src.utils.logger import get_logger


class TemporalAnalyzer:
    """Temporal analyzer for time-based pattern detection and analysis."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.visualizations = []
    
    def analyze_review_trends(self, df: pd.DataFrame, 
                            date_column: str = 'review_date') -> TimeSeriesResult:
        """
        Analyze trends in review data over time.
        
        Args:
            df: Input DataFrame
            date_column: Name of the date column
            
        Returns:
            TimeSeriesResult containing trend analysis
        """
        self.logger.info(f"Analyzing review trends using column '{date_column}'")
        
        if date_column not in df.columns:
            self.logger.error(f"Date column '{date_column}' not found")
            return TimeSeriesResult(
                trends={}, seasonality={}, anomalies=[], forecasts=None,
                analysis_period={'start': datetime.now(), 'end': datetime.now()}
            )
        
        # Prepare time series data
        df_time = df.copy()
        df_time[date_column] = pd.to_datetime(df_time[date_column])
        df_time = df_time.dropna(subset=[date_column])
        
        if df_time.empty:
            self.logger.warning("No valid dates found for temporal analysis")
            return TimeSeriesResult(
                trends={}, seasonality={}, anomalies=[], forecasts=None,
                analysis_period={'start': datetime.now(), 'end': datetime.now()}
            )
        
        # Analysis period
        analysis_period = {
            'start': df_time[date_column].min(),
            'end': df_time[date_column].max()
        }
        
        # Volume trends
        volume_trends = self._analyze_volume_trends(df_time, date_column)
        
        # Rating trends
        rating_trends = self._analyze_rating_trends(df_time, date_column)
        
        # Language trends
        language_trends = self._analyze_language_trends(df_time, date_column)
        
        # App category trends
        category_trends = self._analyze_category_trends(df_time, date_column)
        
        # Seasonal patterns
        seasonality = self.detect_seasonal_patterns(df_time, date_column)
        
        # Anomaly detection
        anomalies = self._detect_anomalies(df_time, date_column)
        
        trends = {
            'volume_trends': volume_trends,
            'rating_trends': rating_trends,
            'language_trends': language_trends,
            'category_trends': category_trends
        }
        
        # Create temporal visualizations
        self._create_temporal_visualizations(df_time, date_column, trends, seasonality)
        
        result = TimeSeriesResult(
            trends=trends,
            seasonality=seasonality,
            anomalies=anomalies,
            forecasts=None,  # Will be implemented in forecasting task
            analysis_period=analysis_period
        )
        
        self.logger.info("Review trends analysis completed")
        
        return result
    
    def _analyze_volume_trends(self, df: pd.DataFrame, date_column: str) -> Dict[str, Any]:
        """Analyze review volume trends over time."""
        # Daily review counts
        daily_counts = df.groupby(df[date_column].dt.date).size()
        
        # Weekly review counts
        weekly_counts = df.groupby(df[date_column].dt.to_period('W')).size()
        
        # Monthly review counts
        monthly_counts = df.groupby(df[date_column].dt.to_period('M')).size()
        
        # Calculate growth rates
        daily_growth = daily_counts.pct_change().fillna(0)
        weekly_growth = weekly_counts.pct_change().fillna(0)
        monthly_growth = monthly_counts.pct_change().fillna(0)
        
        # Trend statistics
        volume_stats = {
            'total_reviews': len(df),
            'date_range_days': (df[date_column].max() - df[date_column].min()).days,
            'avg_daily_reviews': float(daily_counts.mean()),
            'max_daily_reviews': int(daily_counts.max()),
            'min_daily_reviews': int(daily_counts.min()),
            'avg_weekly_reviews': float(weekly_counts.mean()),
            'avg_monthly_reviews': float(monthly_counts.mean()),
            'daily_growth_rate_mean': float(daily_growth.mean()),
            'weekly_growth_rate_mean': float(weekly_growth.mean()),
            'monthly_growth_rate_mean': float(monthly_growth.mean())
        }
        
        # Peak periods
        peak_days = daily_counts.nlargest(5)
        peak_periods = {
            'peak_days': {
                str(date): int(count) for date, count in peak_days.items()
            },
            'peak_weeks': {
                str(period): int(count) for period, count in weekly_counts.nlargest(3).items()
            },
            'peak_months': {
                str(period): int(count) for period, count in monthly_counts.nlargest(3).items()
            }
        }
        
        return {
            'statistics': volume_stats,
            'peak_periods': peak_periods,
            'daily_counts': daily_counts.to_dict(),
            'weekly_counts': {str(k): v for k, v in weekly_counts.to_dict().items()},
            'monthly_counts': {str(k): v for k, v in monthly_counts.to_dict().items()}
        }
    
    def _analyze_rating_trends(self, df: pd.DataFrame, date_column: str) -> Dict[str, Any]:
        """Analyze rating trends over time."""
        if 'rating' not in df.columns:
            return {'note': 'Rating column not found'}
        
        # Daily rating statistics
        daily_ratings = df.groupby(df[date_column].dt.date)['rating'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(3)
        
        # Weekly rating statistics
        weekly_ratings = df.groupby(df[date_column].dt.to_period('W'))['rating'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(3)
        
        # Monthly rating statistics
        monthly_ratings = df.groupby(df[date_column].dt.to_period('M'))['rating'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(3)
        
        # Rating trend analysis
        rating_trend_stats = {
            'overall_rating_trend': float(daily_ratings['mean'].corr(pd.Series(range(len(daily_ratings))))),
            'rating_volatility': float(daily_ratings['mean'].std()),
            'highest_avg_rating_day': {
                'date': str(daily_ratings['mean'].idxmax()),
                'rating': float(daily_ratings['mean'].max())
            },
            'lowest_avg_rating_day': {
                'date': str(daily_ratings['mean'].idxmin()),
                'rating': float(daily_ratings['mean'].min())
            }
        }
        
        # Rating distribution changes over time
        rating_distribution_changes = {}
        for period, group in df.groupby(df[date_column].dt.to_period('M')):
            rating_dist = group['rating'].value_counts(normalize=True).sort_index()
            rating_distribution_changes[str(period)] = rating_dist.to_dict()
        
        return {
            'trend_statistics': rating_trend_stats,
            'daily_ratings': daily_ratings.to_dict(),
            'weekly_ratings': {str(k): v for k, v in weekly_ratings.to_dict().items()},
            'monthly_ratings': {str(k): v for k, v in monthly_ratings.to_dict().items()},
            'distribution_changes': rating_distribution_changes
        }
    
    def _analyze_language_trends(self, df: pd.DataFrame, date_column: str) -> Dict[str, Any]:
        """Analyze language usage trends over time."""
        if 'review_language' not in df.columns:
            return {'note': 'Language column not found'}
        
        # Monthly language distribution
        monthly_lang_dist = {}
        for period, group in df.groupby(df[date_column].dt.to_period('M')):
            lang_counts = group['review_language'].value_counts()
            lang_percentages = group['review_language'].value_counts(normalize=True) * 100
            monthly_lang_dist[str(period)] = {
                'counts': lang_counts.to_dict(),
                'percentages': {k: round(v, 2) for k, v in lang_percentages.to_dict().items()}
            }
        
        # Language growth trends
        language_growth = {}
        all_languages = df['review_language'].unique()
        
        for lang in all_languages:
            if pd.isna(lang):
                continue
            
            monthly_lang_counts = df[df['review_language'] == lang].groupby(
                df[date_column].dt.to_period('M')
            ).size()
            
            if len(monthly_lang_counts) > 1:
                growth_rate = monthly_lang_counts.pct_change().mean()
                language_growth[lang] = {
                    'monthly_counts': {str(k): v for k, v in monthly_lang_counts.to_dict().items()},
                    'avg_growth_rate': float(growth_rate) if not pd.isna(growth_rate) else 0.0
                }
        
        # Emerging and declining languages
        emerging_languages = []
        declining_languages = []
        
        for lang, data in language_growth.items():
            growth_rate = data['avg_growth_rate']
            if growth_rate > 0.1:  # 10% growth
                emerging_languages.append({'language': lang, 'growth_rate': growth_rate})
            elif growth_rate < -0.1:  # 10% decline
                declining_languages.append({'language': lang, 'decline_rate': abs(growth_rate)})
        
        return {
            'monthly_distribution': monthly_lang_dist,
            'language_growth': language_growth,
            'emerging_languages': sorted(emerging_languages, key=lambda x: x['growth_rate'], reverse=True),
            'declining_languages': sorted(declining_languages, key=lambda x: x['decline_rate'], reverse=True)
        }
    
    def _analyze_category_trends(self, df: pd.DataFrame, date_column: str) -> Dict[str, Any]:
        """Analyze app category trends over time."""
        if 'app_category' not in df.columns:
            return {'note': 'App category column not found'}
        
        # Monthly category distribution
        monthly_category_dist = {}
        for period, group in df.groupby(df[date_column].dt.to_period('M')):
            category_counts = group['app_category'].value_counts()
            category_percentages = group['app_category'].value_counts(normalize=True) * 100
            monthly_category_dist[str(period)] = {
                'counts': category_counts.to_dict(),
                'percentages': {k: round(v, 2) for k, v in category_percentages.to_dict().items()}
            }
        
        # Category popularity trends
        category_trends = {}
        all_categories = df['app_category'].unique()
        
        for category in all_categories:
            if pd.isna(category):
                continue
            
            monthly_counts = df[df['app_category'] == category].groupby(
                df[date_column].dt.to_period('M')
            ).size()
            
            if len(monthly_counts) > 1:
                trend_slope = np.polyfit(range(len(monthly_counts)), monthly_counts.values, 1)[0]
                category_trends[category] = {
                    'monthly_counts': {str(k): v for k, v in monthly_counts.to_dict().items()},
                    'trend_slope': float(trend_slope),
                    'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing'
                }
        
        return {
            'monthly_distribution': monthly_category_dist,
            'category_trends': category_trends
        }
    
    def detect_seasonal_patterns(self, df: pd.DataFrame, 
                               date_column: str = 'review_date') -> Dict[str, Any]:
        """
        Detect seasonal patterns in review data.
        
        Args:
            df: Input DataFrame
            date_column: Name of the date column
            
        Returns:
            Dictionary containing seasonal analysis
        """
        self.logger.info("Detecting seasonal patterns")
        
        df_seasonal = df.copy()
        df_seasonal[date_column] = pd.to_datetime(df_seasonal[date_column])
        
        # Extract time components
        df_seasonal['year'] = df_seasonal[date_column].dt.year
        df_seasonal['month'] = df_seasonal[date_column].dt.month
        df_seasonal['day_of_week'] = df_seasonal[date_column].dt.dayofweek
        df_seasonal['day_of_year'] = df_seasonal[date_column].dt.dayofyear
        df_seasonal['quarter'] = df_seasonal[date_column].dt.quarter
        
        # Monthly patterns
        monthly_patterns = df_seasonal.groupby('month').agg({
            'review_id': 'count',
            'rating': ['mean', 'std'] if 'rating' in df_seasonal.columns else 'count'
        }).round(3)
        
        # Day of week patterns
        dow_patterns = df_seasonal.groupby('day_of_week').agg({
            'review_id': 'count',
            'rating': ['mean', 'std'] if 'rating' in df_seasonal.columns else 'count'
        }).round(3)
        
        # Quarterly patterns
        quarterly_patterns = df_seasonal.groupby('quarter').agg({
            'review_id': 'count',
            'rating': ['mean', 'std'] if 'rating' in df_seasonal.columns else 'count'
        }).round(3)
        
        # Seasonal statistics
        seasonal_stats = {
            'peak_month': int(monthly_patterns[('review_id', 'count')].idxmax()),
            'low_month': int(monthly_patterns[('review_id', 'count')].idxmin()),
            'peak_day_of_week': int(dow_patterns[('review_id', 'count')].idxmax()),
            'low_day_of_week': int(dow_patterns[('review_id', 'count')].idxmin()),
            'peak_quarter': int(quarterly_patterns[('review_id', 'count')].idxmax()),
            'seasonal_variation_coefficient': float(monthly_patterns[('review_id', 'count')].std() / 
                                                  monthly_patterns[('review_id', 'count')].mean())
        }
        
        # Holiday effects (simplified - major holidays)
        holiday_effects = self._analyze_holiday_effects(df_seasonal, date_column)
        
        seasonality = {
            'monthly_patterns': monthly_patterns.to_dict(),
            'day_of_week_patterns': dow_patterns.to_dict(),
            'quarterly_patterns': quarterly_patterns.to_dict(),
            'seasonal_statistics': seasonal_stats,
            'holiday_effects': holiday_effects
        }
        
        self.logger.info("Seasonal pattern detection completed")
        
        return seasonality
    
    def _analyze_holiday_effects(self, df: pd.DataFrame, date_column: str) -> Dict[str, Any]:
        """Analyze effects of major holidays on review patterns."""
        # Define major holidays (simplified)
        holidays = {
            'New Year': [(1, 1)],
            'Christmas': [(12, 25)],
            'Independence Day': [(7, 4)],  # US
            'Thanksgiving': [(11, 24), (11, 25), (11, 26)]  # Approximate
        }
        
        holiday_effects = {}
        
        for holiday_name, dates in holidays.items():
            holiday_reviews = []
            
            for month, day in dates:
                holiday_mask = (df[date_column].dt.month == month) & (df[date_column].dt.day == day)
                holiday_data = df[holiday_mask]
                
                if not holiday_data.empty:
                    holiday_reviews.extend(holiday_data.index.tolist())
            
            if holiday_reviews:
                holiday_df = df.loc[holiday_reviews]
                normal_df = df.drop(holiday_reviews)
                
                holiday_effects[holiday_name] = {
                    'holiday_review_count': len(holiday_df),
                    'normal_avg_daily': len(normal_df) / max(1, (df[date_column].max() - df[date_column].min()).days),
                    'holiday_avg_rating': float(holiday_df['rating'].mean()) if 'rating' in df.columns and not holiday_df.empty else None,
                    'normal_avg_rating': float(normal_df['rating'].mean()) if 'rating' in df.columns and not normal_df.empty else None
                }
        
        return holiday_effects
    
    def _detect_anomalies(self, df: pd.DataFrame, date_column: str) -> List[Dict[str, Any]]:
        """Detect anomalies in review patterns."""
        # Daily review counts
        daily_counts = df.groupby(df[date_column].dt.date).size()
        
        # Calculate anomaly thresholds using IQR method
        Q1 = daily_counts.quantile(0.25)
        Q3 = daily_counts.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        anomalies = []
        
        # Volume anomalies
        for date, count in daily_counts.items():
            if count < lower_bound or count > upper_bound:
                anomaly_type = 'high_volume' if count > upper_bound else 'low_volume'
                anomalies.append({
                    'date': str(date),
                    'type': anomaly_type,
                    'value': int(count),
                    'expected_range': f"{lower_bound:.1f} - {upper_bound:.1f}",
                    'severity': 'high' if abs(count - daily_counts.median()) > 2 * daily_counts.std() else 'moderate'
                })
        
        # Rating anomalies (if rating column exists)
        if 'rating' in df.columns:
            daily_ratings = df.groupby(df[date_column].dt.date)['rating'].mean()
            rating_std = daily_ratings.std()
            rating_mean = daily_ratings.mean()
            
            for date, rating in daily_ratings.items():
                if abs(rating - rating_mean) > 2 * rating_std:
                    anomaly_type = 'high_rating' if rating > rating_mean else 'low_rating'
                    anomalies.append({
                        'date': str(date),
                        'type': anomaly_type,
                        'value': float(rating),
                        'expected_range': f"{rating_mean - 2*rating_std:.2f} - {rating_mean + 2*rating_std:.2f}",
                        'severity': 'high' if abs(rating - rating_mean) > 3 * rating_std else 'moderate'
                    })
        
        # Sort anomalies by date
        anomalies.sort(key=lambda x: x['date'])
        
        return anomalies
    
    def _create_temporal_visualizations(self, df: pd.DataFrame, date_column: str, 
                                      trends: Dict, seasonality: Dict):
        """Create temporal analysis visualizations."""
        
        # 1. Review volume over time
        daily_counts = df.groupby(df[date_column].dt.date).size()
        
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Scatter(
            x=daily_counts.index,
            y=daily_counts.values,
            mode='lines',
            name='Daily Review Count',
            line=dict(color='blue')
        ))
        
        fig_volume.update_layout(
            title='Review Volume Over Time',
            xaxis_title='Date',
            yaxis_title='Number of Reviews',
            height=400
        )
        
        self.visualizations.append(fig_volume)
        
        # 2. Rating trends over time (if available)
        if 'rating' in df.columns:
            daily_ratings = df.groupby(df[date_column].dt.date)['rating'].mean()
            
            fig_rating = go.Figure()
            fig_rating.add_trace(go.Scatter(
                x=daily_ratings.index,
                y=daily_ratings.values,
                mode='lines',
                name='Average Daily Rating',
                line=dict(color='green')
            ))
            
            fig_rating.update_layout(
                title='Average Rating Trends Over Time',
                xaxis_title='Date',
                yaxis_title='Average Rating',
                height=400
            )
            
            self.visualizations.append(fig_rating)
        
        # 3. Seasonal patterns
        if 'monthly_patterns' in seasonality:
            monthly_data = seasonality['monthly_patterns'][('review_id', 'count')]
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            fig_seasonal = go.Figure()
            fig_seasonal.add_trace(go.Bar(
                x=months,
                y=[monthly_data.get(i+1, 0) for i in range(12)],
                name='Monthly Review Count',
                marker_color='lightcoral'
            ))
            
            fig_seasonal.update_layout(
                title='Seasonal Patterns - Monthly Review Distribution',
                xaxis_title='Month',
                yaxis_title='Number of Reviews',
                height=400
            )
            
            self.visualizations.append(fig_seasonal)