"""
Time series analysis for multilingual app reviews.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    print("Warning: statsmodels not available, some features may be limited")

try:
    from prophet import Prophet
except ImportError:
    print("Warning: prophet not available, forecasting features may be limited")

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.data.interfaces import TimeSeriesAnalyzerInterface
from src.data.models import TimeSeriesResult, ForecastResult
from src.config import get_config
from src.utils.logger import get_logger


class TimeSeriesAnalyzer(TimeSeriesAnalyzerInterface):
    """
    Comprehensive time series analysis for app reviews data.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__)
        
    def analyze_review_trends(self, df: pd.DataFrame) -> TimeSeriesResult:
        """
        Analyze trends in review data over time.
        
        Args:
            df: DataFrame with review data
            
        Returns:
            TimeSeriesResult with comprehensive trend analysis
        """
        self.logger.info("Starting time series trend analysis")
        
        try:
            # Prepare time series data
            df_ts = self._prepare_time_series_data(df)
            
            # Analyze different metrics over time
            trends = {}
            
            # Review volume trends
            trends['review_volume'] = self._analyze_volume_trends(df_ts)
            
            # Rating trends
            trends['rating_trends'] = self._analyze_rating_trends(df_ts)
            
            # Language trends
            trends['language_trends'] = self._analyze_language_trends(df_ts)
            
            # App category trends
            trends['category_trends'] = self._analyze_category_trends(df_ts)
            
            # Seasonal patterns
            seasonality = self.detect_seasonal_patterns(df)
            
            # Anomaly detection
            anomalies = self._detect_anomalies(df_ts)
            
            analysis_period = {
                'start_date': df['review_date'].min(),
                'end_date': df['review_date'].max()
            }
            
            result = TimeSeriesResult(
                trends=trends,
                seasonality=seasonality,
                anomalies=anomalies,
                forecasts=None,  # Will be added in forecasting method
                analysis_period=analysis_period
            )
            
            self.logger.info("Time series trend analysis completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in time series trend analysis: {str(e)}")
            raise
    
    def detect_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect seasonal patterns in review data.
        
        Args:
            df: DataFrame with review data
            
        Returns:
            Dictionary with seasonal pattern analysis
        """
        self.logger.info("Detecting seasonal patterns")
        
        try:
            df_ts = self._prepare_time_series_data(df)
            seasonal_patterns = {}
            
            # Monthly seasonality
            seasonal_patterns['monthly'] = self._analyze_monthly_seasonality(df_ts)
            
            # Weekly seasonality
            seasonal_patterns['weekly'] = self._analyze_weekly_seasonality(df_ts)
            
            # Yearly seasonality (if data spans multiple years)
            if df_ts.index.year.nunique() > 1:
                seasonal_patterns['yearly'] = self._analyze_yearly_seasonality(df_ts)
            
            # Holiday effects
            seasonal_patterns['holiday_effects'] = self._analyze_holiday_effects(df_ts)
            
            return seasonal_patterns
            
        except Exception as e:
            self.logger.error(f"Error in seasonal pattern detection: {str(e)}")
            raise
    
    def forecast_future_trends(self, df: pd.DataFrame, periods: int) -> ForecastResult:
        """
        Forecast future trends in review data.
        
        Args:
            df: DataFrame with review data
            periods: Number of periods to forecast
            
        Returns:
            ForecastResult with predictions and confidence intervals
        """
        self.logger.info(f"Forecasting future trends for {periods} periods")
        
        try:
            df_ts = self._prepare_time_series_data(df)
            
            # Use simple forecasting if Prophet is not available
            try:
                forecast_result = self._prophet_forecast(df_ts, periods)
            except:
                forecast_result = self._simple_forecast(df_ts, periods)
            
            return forecast_result
            
        except Exception as e:
            self.logger.error(f"Error in forecasting: {str(e)}")
            raise
    
    def _prepare_time_series_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for time series analysis."""
        df_copy = df.copy()
        df_copy['review_date'] = pd.to_datetime(df_copy['review_date'])
        df_copy = df_copy.set_index('review_date').sort_index()
        return df_copy
    
    def _analyze_volume_trends(self, df_ts: pd.DataFrame) -> Dict[str, Any]:
        """Analyze review volume trends over time."""
        daily_counts = df_ts.resample('D').size()
        weekly_counts = df_ts.resample('W').size()
        monthly_counts = df_ts.resample('M').size()
        
        # Calculate trend statistics
        volume_stats = {
            'daily_avg': daily_counts.mean(),
            'daily_std': daily_counts.std(),
            'weekly_avg': weekly_counts.mean(),
            'monthly_avg': monthly_counts.mean(),
            'peak_day': daily_counts.idxmax(),
            'peak_count': daily_counts.max(),
            'trend_direction': 'increasing' if daily_counts.iloc[-30:].mean() > daily_counts.iloc[:30].mean() else 'decreasing'
        }
        
        return {
            'daily_counts': daily_counts,
            'weekly_counts': weekly_counts,
            'monthly_counts': monthly_counts,
            'statistics': volume_stats
        }
    
    def _analyze_rating_trends(self, df_ts: pd.DataFrame) -> Dict[str, Any]:
        """Analyze rating trends over time."""
        daily_ratings = df_ts.resample('D')['rating'].agg(['mean', 'std', 'count'])
        weekly_ratings = df_ts.resample('W')['rating'].agg(['mean', 'std', 'count'])
        monthly_ratings = df_ts.resample('M')['rating'].agg(['mean', 'std', 'count'])
        
        # Calculate rating trend statistics
        rating_stats = {
            'overall_trend': 'improving' if monthly_ratings['mean'].iloc[-3:].mean() > monthly_ratings['mean'].iloc[:3].mean() else 'declining',
            'volatility': daily_ratings['std'].mean(),
            'best_period': daily_ratings['mean'].idxmax(),
            'worst_period': daily_ratings['mean'].idxmin(),
            'correlation_with_volume': daily_ratings['mean'].corr(daily_ratings['count'])
        }
        
        return {
            'daily_ratings': daily_ratings,
            'weekly_ratings': weekly_ratings,
            'monthly_ratings': monthly_ratings,
            'statistics': rating_stats
        }
    
    def _analyze_language_trends(self, df_ts: pd.DataFrame) -> Dict[str, Any]:
        """Analyze language distribution trends over time."""
        language_trends = {}
        
        for lang in df_ts['review_language'].unique():
            lang_data = df_ts[df_ts['review_language'] == lang]
            monthly_counts = lang_data.resample('M').size()
            language_trends[lang] = {
                'monthly_counts': monthly_counts,
                'total_reviews': len(lang_data),
                'avg_rating': lang_data['rating'].mean(),
                'trend': 'growing' if monthly_counts.iloc[-3:].mean() > monthly_counts.iloc[:3].mean() else 'declining'
            }
        
        return language_trends
    
    def _analyze_category_trends(self, df_ts: pd.DataFrame) -> Dict[str, Any]:
        """Analyze app category trends over time."""
        category_trends = {}
        
        for category in df_ts['app_category'].unique():
            cat_data = df_ts[df_ts['app_category'] == category]
            monthly_counts = cat_data.resample('M').size()
            monthly_ratings = cat_data.resample('M')['rating'].mean()
            
            category_trends[category] = {
                'monthly_counts': monthly_counts,
                'monthly_ratings': monthly_ratings,
                'total_reviews': len(cat_data),
                'avg_rating': cat_data['rating'].mean(),
                'popularity_trend': 'rising' if monthly_counts.iloc[-3:].mean() > monthly_counts.iloc[:3].mean() else 'falling'
            }
        
        return category_trends
    
    def _analyze_monthly_seasonality(self, df_ts: pd.DataFrame) -> Dict[str, Any]:
        """Analyze monthly seasonal patterns."""
        df_ts['month'] = df_ts.index.month
        monthly_patterns = df_ts.groupby('month').agg({
            'rating': ['mean', 'count'],
            'num_helpful_votes': 'mean'
        }).round(2)
        
        # Find peak and low months
        review_counts = df_ts.groupby('month').size()
        peak_month = review_counts.idxmax()
        low_month = review_counts.idxmin()
        
        return {
            'monthly_patterns': monthly_patterns,
            'peak_month': peak_month,
            'low_month': low_month,
            'seasonality_strength': (review_counts.max() - review_counts.min()) / review_counts.mean()
        }
    
    def _analyze_weekly_seasonality(self, df_ts: pd.DataFrame) -> Dict[str, Any]:
        """Analyze weekly seasonal patterns."""
        df_ts['weekday'] = df_ts.index.dayofweek
        weekly_patterns = df_ts.groupby('weekday').agg({
            'rating': ['mean', 'count'],
            'num_helpful_votes': 'mean'
        }).round(2)
        
        # Find peak and low days
        review_counts = df_ts.groupby('weekday').size()
        peak_day = review_counts.idxmax()
        low_day = review_counts.idxmin()
        
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        return {
            'weekly_patterns': weekly_patterns,
            'peak_day': weekday_names[peak_day],
            'low_day': weekday_names[low_day],
            'weekend_vs_weekday': {
                'weekend_avg': df_ts[df_ts['weekday'].isin([5, 6])].groupby(df_ts.index.date).size().mean(),
                'weekday_avg': df_ts[df_ts['weekday'].isin([0, 1, 2, 3, 4])].groupby(df_ts.index.date).size().mean()
            }
        }
    
    def _analyze_yearly_seasonality(self, df_ts: pd.DataFrame) -> Dict[str, Any]:
        """Analyze yearly seasonal patterns."""
        yearly_patterns = df_ts.groupby(df_ts.index.year).agg({
            'rating': ['mean', 'count'],
            'num_helpful_votes': 'mean'
        }).round(2)
        
        return {
            'yearly_patterns': yearly_patterns,
            'year_over_year_growth': yearly_patterns[('rating', 'count')].pct_change().mean()
        }
    
    def _analyze_holiday_effects(self, df_ts: pd.DataFrame) -> Dict[str, Any]:
        """Analyze holiday effects on review patterns."""
        # Define major holidays (simplified)
        holidays = {
            'New Year': [(1, 1)],
            'Christmas': [(12, 25)],
            'Black Friday': [(11, 24)],  # Approximate
            'Valentine': [(2, 14)]
        }
        
        holiday_effects = {}
        
        for holiday_name, dates in holidays.items():
            holiday_data = []
            for month, day in dates:
                holiday_reviews = df_ts[(df_ts.index.month == month) & (df_ts.index.day == day)]
                if not holiday_reviews.empty:
                    holiday_data.append({
                        'date': f"{month}-{day}",
                        'review_count': len(holiday_reviews),
                        'avg_rating': holiday_reviews['rating'].mean()
                    })
            
            if holiday_data:
                holiday_effects[holiday_name] = holiday_data
        
        return holiday_effects
    
    def _detect_anomalies(self, df_ts: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect anomalies in review patterns."""
        daily_counts = df_ts.resample('D').size()
        daily_ratings = df_ts.resample('D')['rating'].mean()
        
        anomalies = []
        
        # Volume anomalies (using IQR method)
        Q1_vol = daily_counts.quantile(0.25)
        Q3_vol = daily_counts.quantile(0.75)
        IQR_vol = Q3_vol - Q1_vol
        lower_bound_vol = Q1_vol - 1.5 * IQR_vol
        upper_bound_vol = Q3_vol + 1.5 * IQR_vol
        
        volume_anomalies = daily_counts[(daily_counts < lower_bound_vol) | (daily_counts > upper_bound_vol)]
        
        for date, count in volume_anomalies.items():
            anomalies.append({
                'date': date,
                'type': 'volume_anomaly',
                'value': count,
                'severity': 'high' if count > upper_bound_vol * 1.5 else 'medium',
                'description': f"Unusual review volume: {count} reviews"
            })
        
        # Rating anomalies
        Q1_rating = daily_ratings.quantile(0.25)
        Q3_rating = daily_ratings.quantile(0.75)
        IQR_rating = Q3_rating - Q1_rating
        lower_bound_rating = Q1_rating - 1.5 * IQR_rating
        upper_bound_rating = Q3_rating + 1.5 * IQR_rating
        
        rating_anomalies = daily_ratings[(daily_ratings < lower_bound_rating) | (daily_ratings > upper_bound_rating)]
        
        for date, rating in rating_anomalies.items():
            anomalies.append({
                'date': date,
                'type': 'rating_anomaly',
                'value': rating,
                'severity': 'high' if rating < lower_bound_rating * 0.8 else 'medium',
                'description': f"Unusual average rating: {rating:.2f}"
            })
        
        return sorted(anomalies, key=lambda x: x['date'], reverse=True)
    
    def _prophet_forecast(self, df_ts: pd.DataFrame, periods: int) -> ForecastResult:
        """Use Prophet for time series forecasting."""
        # Prepare data for Prophet
        daily_counts = df_ts.resample('D').size().reset_index()
        daily_counts.columns = ['ds', 'y']
        
        # Initialize and fit Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        model.fit(daily_counts)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        # Extract forecast results
        forecast_data = forecast.tail(periods)
        predictions = forecast_data['yhat'].tolist()
        confidence_intervals = list(zip(forecast_data['yhat_lower'].tolist(), forecast_data['yhat_upper'].tolist()))
        forecast_dates = forecast_data['ds'].tolist()
        
        # Calculate accuracy metrics on historical data
        historical_forecast = forecast.head(len(daily_counts))
        mae = np.mean(np.abs(historical_forecast['yhat'] - daily_counts['y']))
        mape = np.mean(np.abs((historical_forecast['yhat'] - daily_counts['y']) / daily_counts['y'])) * 100
        
        accuracy_metrics = {
            'mae': mae,
            'mape': mape,
            'rmse': np.sqrt(np.mean((historical_forecast['yhat'] - daily_counts['y']) ** 2))
        }
        
        return ForecastResult(
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            forecast_dates=forecast_dates,
            model_type='Prophet',
            accuracy_metrics=accuracy_metrics
        )
    
    def _simple_forecast(self, df_ts: pd.DataFrame, periods: int) -> ForecastResult:
        """Simple forecasting method when Prophet is not available."""
        daily_counts = df_ts.resample('D').size()
        
        # Simple moving average forecast
        window = min(30, len(daily_counts) // 4)
        recent_avg = daily_counts.tail(window).mean()
        recent_std = daily_counts.tail(window).std()
        
        # Generate forecast dates
        last_date = daily_counts.index[-1]
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
        
        # Simple predictions (moving average with slight trend)
        trend = (daily_counts.tail(window).mean() - daily_counts.head(window).mean()) / len(daily_counts)
        predictions = [recent_avg + trend * i for i in range(1, periods + 1)]
        
        # Simple confidence intervals
        confidence_intervals = [(max(0, pred - 1.96 * recent_std), pred + 1.96 * recent_std) 
                              for pred in predictions]
        
        # Simple accuracy metrics
        accuracy_metrics = {
            'mae': recent_std,
            'mape': (recent_std / recent_avg) * 100,
            'rmse': recent_std
        }
        
        return ForecastResult(
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            forecast_dates=forecast_dates,
            model_type='Simple Moving Average',
            accuracy_metrics=accuracy_metrics
        )


class Forecaster:
    """Specialized forecasting component for future trend prediction."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__)
    
    def forecast_review_volume(self, df: pd.DataFrame, periods: int = 30) -> ForecastResult:
        """Forecast future review volume."""
        analyzer = TimeSeriesAnalyzer()
        return analyzer.forecast_future_trends(df, periods)
    
    def forecast_rating_trends(self, df: pd.DataFrame, periods: int = 30) -> Dict[str, Any]:
        """Forecast future rating trends."""
        df_ts = df.copy()
        df_ts['review_date'] = pd.to_datetime(df_ts['review_date'])
        df_ts = df_ts.set_index('review_date').sort_index()
        
        daily_ratings = df_ts.resample('D')['rating'].mean()
        
        # Simple trend forecast for ratings
        window = min(30, len(daily_ratings) // 4)
        recent_avg = daily_ratings.tail(window).mean()
        trend = (daily_ratings.tail(window).mean() - daily_ratings.head(window).mean()) / len(daily_ratings)
        
        forecast_ratings = [recent_avg + trend * i for i in range(1, periods + 1)]
        
        return {
            'forecast_ratings': forecast_ratings,
            'current_avg': recent_avg,
            'trend_direction': 'improving' if trend > 0 else 'declining',
            'confidence': 'medium'
        }