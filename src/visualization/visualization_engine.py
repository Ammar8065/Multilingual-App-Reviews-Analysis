"""
Comprehensive visualization engine for multilingual app reviews analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

from src.data.interfaces import VisualizationEngineInterface
from src.config import get_config
from src.utils.logger import get_logger


class VisualizationEngine(VisualizationEngineInterface):
    """
    Comprehensive visualization engine for creating all types of charts and plots.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.color_palette = self.config.visualization.color_palette
        self.theme = self.config.visualization.theme
        
    def create_distribution_plots(self, df: pd.DataFrame) -> List[go.Figure]:
        """
        Create distribution plots for ratings, languages, and countries.
        
        Args:
            df: DataFrame with review data
            
        Returns:
            List of Plotly figures
        """
        self.logger.info("Creating distribution plots")
        
        plots = []
        
        try:
            # Rating distribution
            plots.append(self._create_rating_distribution(df))
            
            # Language distribution
            plots.append(self._create_language_distribution(df))
            
            # Country distribution
            plots.append(self._create_country_distribution(df))
            
            # App category distribution
            plots.append(self._create_category_distribution(df))
            
            # Device type distribution
            plots.append(self._create_device_distribution(df))
            
            # Age distribution (if available)
            if 'user_age' in df.columns:
                plots.append(self._create_age_distribution(df))
            
            return plots
            
        except Exception as e:
            self.logger.error(f"Error creating distribution plots: {str(e)}")
            raise
    
    def create_time_series_plots(self, df: pd.DataFrame) -> List[go.Figure]:
        """
        Create time series visualizations for temporal analysis.
        
        Args:
            df: DataFrame with review data
            
        Returns:
            List of Plotly figures
        """
        self.logger.info("Creating time series plots")
        
        plots = []
        
        try:
            # Prepare time series data
            df_ts = df.copy()
            df_ts['review_date'] = pd.to_datetime(df_ts['review_date'])
            
            # Review volume over time
            plots.append(self._create_volume_time_series(df_ts))
            
            # Rating trends over time
            plots.append(self._create_rating_time_series(df_ts))
            
            # Language trends over time
            plots.append(self._create_language_time_series(df_ts))
            
            # Seasonal patterns
            plots.append(self._create_seasonal_patterns(df_ts))
            
            return plots
            
        except Exception as e:
            self.logger.error(f"Error creating time series plots: {str(e)}")
            raise
    
    def create_geographic_maps(self, df: pd.DataFrame) -> List[go.Figure]:
        """
        Create geographic maps for location-based analysis.
        
        Args:
            df: DataFrame with review data
            
        Returns:
            List of Plotly figures
        """
        self.logger.info("Creating geographic maps")
        
        maps = []
        
        try:
            # Country-wise review distribution
            maps.append(self._create_country_choropleth(df))
            
            # Rating by country
            maps.append(self._create_rating_choropleth(df))
            
            # Review volume by country
            maps.append(self._create_volume_choropleth(df))
            
            return maps
            
        except Exception as e:
            self.logger.error(f"Error creating geographic maps: {str(e)}")
            raise
    
    def create_correlation_heatmaps(self, df: pd.DataFrame) -> go.Figure:
        """
        Create correlation heatmaps for numerical features.
        
        Args:
            df: DataFrame with review data
            
        Returns:
            Plotly figure with correlation heatmap
        """
        self.logger.info("Creating correlation heatmap")
        
        try:
            # Select numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove ID columns
            numerical_cols = [col for col in numerical_cols if 'id' not in col.lower()]
            
            if len(numerical_cols) < 2:
                self.logger.warning("Not enough numerical columns for correlation analysis")
                return go.Figure()
            
            # Calculate correlation matrix
            corr_matrix = df[numerical_cols].corr()
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(3).values,
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Correlation Matrix of Numerical Features',
                xaxis_title='Features',
                yaxis_title='Features',
                width=800,
                height=600
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating correlation heatmap: {str(e)}")
            raise
    
    def create_sentiment_visualizations(self, df: pd.DataFrame, sentiment_column: str = 'sentiment') -> List[go.Figure]:
        """
        Create sentiment-specific visualizations.
        
        Args:
            df: DataFrame with review data
            sentiment_column: Name of the sentiment column
            
        Returns:
            List of Plotly figures
        """
        self.logger.info("Creating sentiment visualizations")
        
        plots = []
        
        try:
            if sentiment_column not in df.columns:
                self.logger.warning(f"Sentiment column '{sentiment_column}' not found")
                return plots
            
            # Sentiment distribution
            plots.append(self._create_sentiment_distribution(df, sentiment_column))
            
            # Sentiment by language
            plots.append(self._create_sentiment_by_language(df, sentiment_column))
            
            # Sentiment by country
            plots.append(self._create_sentiment_by_country(df, sentiment_column))
            
            # Sentiment vs rating
            plots.append(self._create_sentiment_vs_rating(df, sentiment_column))
            
            return plots
            
        except Exception as e:
            self.logger.error(f"Error creating sentiment visualizations: {str(e)}")
            raise
    
    def create_word_clouds(self, df: pd.DataFrame, text_column: str = 'review_text') -> Dict[str, Any]:
        """
        Create word clouds for text analysis.
        
        Args:
            df: DataFrame with review data
            text_column: Name of the text column
            
        Returns:
            Dictionary with word cloud images
        """
        self.logger.info("Creating word clouds")
        
        word_clouds = {}
        
        try:
            if text_column not in df.columns:
                self.logger.warning(f"Text column '{text_column}' not found")
                return word_clouds
            
            # Overall word cloud
            all_text = ' '.join(df[text_column].dropna().astype(str))
            if all_text.strip():
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
                word_clouds['overall'] = wordcloud
            
            # Word clouds by sentiment (if available)
            if 'sentiment' in df.columns:
                for sentiment in df['sentiment'].unique():
                    sentiment_text = ' '.join(df[df['sentiment'] == sentiment][text_column].dropna().astype(str))
                    if sentiment_text.strip():
                        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sentiment_text)
                        word_clouds[f'sentiment_{sentiment}'] = wordcloud
            
            # Word clouds by language (top 5 languages)
            top_languages = df['review_language'].value_counts().head(5).index
            for lang in top_languages:
                lang_text = ' '.join(df[df['review_language'] == lang][text_column].dropna().astype(str))
                if lang_text.strip():
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(lang_text)
                    word_clouds[f'language_{lang}'] = wordcloud
            
            return word_clouds
            
        except Exception as e:
            self.logger.error(f"Error creating word clouds: {str(e)}")
            return word_clouds
    
    def _create_rating_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create rating distribution plot."""
        fig = px.histogram(
            df, x='rating', nbins=20,
            title='Distribution of App Ratings',
            labels={'rating': 'Rating', 'count': 'Number of Reviews'},
            color_discrete_sequence=self.color_palette
        )
        
        fig.add_vline(x=df['rating'].mean(), line_dash="dash", 
                     annotation_text=f"Mean: {df['rating'].mean():.2f}")
        
        fig.update_layout(showlegend=False)
        return fig
    
    def _create_language_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create language distribution plot."""
        lang_counts = df['review_language'].value_counts().head(15)
        
        fig = px.bar(
            x=lang_counts.index, y=lang_counts.values,
            title='Top 15 Languages in Reviews',
            labels={'x': 'Language', 'y': 'Number of Reviews'},
            color=lang_counts.values,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        return fig
    
    def _create_country_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create country distribution plot."""
        country_counts = df['user_country'].value_counts().head(15)
        
        fig = px.bar(
            x=country_counts.index, y=country_counts.values,
            title='Top 15 Countries by Review Count',
            labels={'x': 'Country', 'y': 'Number of Reviews'},
            color=country_counts.values,
            color_continuous_scale='plasma'
        )
        
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        return fig
    
    def _create_category_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create app category distribution plot."""
        category_counts = df['app_category'].value_counts()
        
        fig = px.pie(
            values=category_counts.values, names=category_counts.index,
            title='Distribution of App Categories',
            color_discrete_sequence=self.color_palette
        )
        
        return fig
    
    def _create_device_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create device type distribution plot."""
        device_counts = df['device_type'].value_counts()
        
        fig = px.bar(
            x=device_counts.index, y=device_counts.values,
            title='Distribution of Device Types',
            labels={'x': 'Device Type', 'y': 'Number of Reviews'},
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(showlegend=False)
        return fig
    
    def _create_age_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create age distribution plot."""
        age_data = df['user_age'].dropna()
        
        fig = px.histogram(
            age_data, nbins=20,
            title='Distribution of User Ages',
            labels={'value': 'Age', 'count': 'Number of Users'},
            color_discrete_sequence=self.color_palette
        )
        
        fig.add_vline(x=age_data.mean(), line_dash="dash", 
                     annotation_text=f"Mean: {age_data.mean():.1f}")
        
        fig.update_layout(showlegend=False)
        return fig
    
    def _create_volume_time_series(self, df_ts: pd.DataFrame) -> go.Figure:
        """Create review volume time series plot."""
        daily_counts = df_ts.groupby(df_ts['review_date'].dt.date).size()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_counts.index, y=daily_counts.values,
            mode='lines+markers',
            name='Daily Review Count',
            line=dict(color=self.color_palette[0])
        ))
        
        # Add 7-day moving average
        ma_7 = daily_counts.rolling(window=7).mean()
        fig.add_trace(go.Scatter(
            x=ma_7.index, y=ma_7.values,
            mode='lines',
            name='7-day Moving Average',
            line=dict(color=self.color_palette[1], dash='dash')
        ))
        
        fig.update_layout(
            title='Review Volume Over Time',
            xaxis_title='Date',
            yaxis_title='Number of Reviews',
            hovermode='x unified'
        )
        
        return fig
    
    def _create_rating_time_series(self, df_ts: pd.DataFrame) -> go.Figure:
        """Create rating trends time series plot."""
        daily_ratings = df_ts.groupby(df_ts['review_date'].dt.date)['rating'].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_ratings.index, y=daily_ratings.values,
            mode='lines+markers',
            name='Daily Average Rating',
            line=dict(color=self.color_palette[2])
        ))
        
        # Add trend line
        from scipy import stats
        x_numeric = np.arange(len(daily_ratings))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, daily_ratings.values)
        trend_line = slope * x_numeric + intercept
        
        fig.add_trace(go.Scatter(
            x=daily_ratings.index, y=trend_line,
            mode='lines',
            name=f'Trend (R²={r_value**2:.3f})',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Average Rating Trends Over Time',
            xaxis_title='Date',
            yaxis_title='Average Rating',
            hovermode='x unified'
        )
        
        return fig
    
    def _create_language_time_series(self, df_ts: pd.DataFrame) -> go.Figure:
        """Create language trends time series plot."""
        top_languages = df_ts['review_language'].value_counts().head(5).index
        
        fig = go.Figure()
        
        for i, lang in enumerate(top_languages):
            lang_data = df_ts[df_ts['review_language'] == lang]
            daily_counts = lang_data.groupby(lang_data['review_date'].dt.date).size()
            
            fig.add_trace(go.Scatter(
                x=daily_counts.index, y=daily_counts.values,
                mode='lines',
                name=f'{lang}',
                line=dict(color=self.color_palette[i % len(self.color_palette)])
            ))
        
        fig.update_layout(
            title='Review Volume by Language Over Time',
            xaxis_title='Date',
            yaxis_title='Number of Reviews',
            hovermode='x unified'
        )
        
        return fig
    
    def _create_seasonal_patterns(self, df_ts: pd.DataFrame) -> go.Figure:
        """Create seasonal patterns visualization."""
        df_ts['month'] = df_ts['review_date'].dt.month
        df_ts['weekday'] = df_ts['review_date'].dt.day_name()
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Monthly Pattern', 'Weekly Pattern'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Monthly pattern
        monthly_counts = df_ts.groupby('month').size()
        fig.add_trace(
            go.Bar(x=monthly_counts.index, y=monthly_counts.values, 
                  name='Monthly Reviews', marker_color=self.color_palette[0]),
            row=1, col=1
        )
        
        # Weekly pattern
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_counts = df_ts.groupby('weekday').size().reindex(weekday_order)
        fig.add_trace(
            go.Bar(x=weekly_counts.index, y=weekly_counts.values, 
                  name='Weekly Reviews', marker_color=self.color_palette[1]),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Seasonal Patterns in Review Volume',
            showlegend=False
        )
        
        return fig
    
    def _create_country_choropleth(self, df: pd.DataFrame) -> go.Figure:
        """Create country-wise review distribution choropleth."""
        country_counts = df['user_country'].value_counts()
        
        fig = go.Figure(data=go.Choropleth(
            locations=country_counts.index,
            z=country_counts.values,
            locationmode='country names',
            colorscale='Blues',
            colorbar_title="Number of Reviews"
        ))
        
        fig.update_layout(
            title='Global Distribution of Reviews by Country',
            geo=dict(showframe=False, showcoastlines=True)
        )
        
        return fig
    
    def _create_rating_choropleth(self, df: pd.DataFrame) -> go.Figure:
        """Create rating by country choropleth."""
        country_ratings = df.groupby('user_country')['rating'].mean()
        
        fig = go.Figure(data=go.Choropleth(
            locations=country_ratings.index,
            z=country_ratings.values,
            locationmode='country names',
            colorscale='RdYlGn',
            colorbar_title="Average Rating"
        ))
        
        fig.update_layout(
            title='Average App Ratings by Country',
            geo=dict(showframe=False, showcoastlines=True)
        )
        
        return fig
    
    def _create_volume_choropleth(self, df: pd.DataFrame) -> go.Figure:
        """Create review volume choropleth."""
        country_volumes = df['user_country'].value_counts()
        
        fig = go.Figure(data=go.Choropleth(
            locations=country_volumes.index,
            z=country_volumes.values,
            locationmode='country names',
            colorscale='Viridis',
            colorbar_title="Review Volume"
        ))
        
        fig.update_layout(
            title='Review Volume by Country',
            geo=dict(showframe=False, showcoastlines=True)
        )
        
        return fig
    
    def _create_sentiment_distribution(self, df: pd.DataFrame, sentiment_column: str) -> go.Figure:
        """Create sentiment distribution plot."""
        sentiment_counts = df[sentiment_column].value_counts()
        
        colors = {'positive': 'green', 'negative': 'red', 'neutral': 'orange'}
        bar_colors = [colors.get(sentiment, 'blue') for sentiment in sentiment_counts.index]
        
        fig = px.bar(
            x=sentiment_counts.index, y=sentiment_counts.values,
            title='Distribution of Sentiment',
            labels={'x': 'Sentiment', 'y': 'Number of Reviews'},
            color=sentiment_counts.index,
            color_discrete_map=colors
        )
        
        fig.update_layout(showlegend=False)
        return fig
    
    def _create_sentiment_by_language(self, df: pd.DataFrame, sentiment_column: str) -> go.Figure:
        """Create sentiment by language plot."""
        sentiment_lang = pd.crosstab(df['review_language'], df[sentiment_column], normalize='index') * 100
        top_languages = df['review_language'].value_counts().head(10).index
        sentiment_lang_top = sentiment_lang.loc[top_languages]
        
        fig = px.bar(
            sentiment_lang_top, 
            title='Sentiment Distribution by Language (Top 10)',
            labels={'value': 'Percentage', 'index': 'Language'},
            color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'}
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        return fig
    
    def _create_sentiment_by_country(self, df: pd.DataFrame, sentiment_column: str) -> go.Figure:
        """Create sentiment by country plot."""
        sentiment_country = pd.crosstab(df['user_country'], df[sentiment_column], normalize='index') * 100
        top_countries = df['user_country'].value_counts().head(10).index
        sentiment_country_top = sentiment_country.loc[top_countries]
        
        fig = px.bar(
            sentiment_country_top,
            title='Sentiment Distribution by Country (Top 10)',
            labels={'value': 'Percentage', 'index': 'Country'},
            color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'}
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        return fig
    
    def _create_sentiment_vs_rating(self, df: pd.DataFrame, sentiment_column: str) -> go.Figure:
        """Create sentiment vs rating comparison plot."""
        fig = px.box(
            df, x=sentiment_column, y='rating',
            title='Rating Distribution by Sentiment',
            labels={'rating': 'Rating', sentiment_column: 'Sentiment'},
            color=sentiment_column,
            color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'}
        )
        
        return fig


class ChartFactory:
    """
    Factory class for creating specific chart types.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__)
    
    def create_comparison_chart(self, data: Dict[str, Any], chart_type: str = 'bar') -> go.Figure:
        """Create comparison charts for different metrics."""
        if chart_type == 'bar':
            return self._create_comparison_bar_chart(data)
        elif chart_type == 'line':
            return self._create_comparison_line_chart(data)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
    
    def create_performance_dashboard(self, model_results: Dict[str, Any]) -> go.Figure:
        """Create performance dashboard for ML models."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Accuracy', 'Feature Importance', 'Confusion Matrix', 'Performance Metrics'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "heatmap"}, {"type": "table"}]]
        )
        
        # Add model accuracy comparison
        if 'accuracy_scores' in model_results:
            models = list(model_results['accuracy_scores'].keys())
            scores = list(model_results['accuracy_scores'].values())
            
            fig.add_trace(
                go.Bar(x=models, y=scores, name='Accuracy'),
                row=1, col=1
            )
        
        # Add feature importance
        if 'feature_importance' in model_results:
            features = list(model_results['feature_importance'].keys())
            importance = list(model_results['feature_importance'].values())
            
            fig.add_trace(
                go.Bar(x=features, y=importance, name='Importance'),
                row=1, col=2
            )
        
        fig.update_layout(title='Model Performance Dashboard', showlegend=False)
        return fig
    
    def _create_comparison_bar_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create comparison bar chart."""
        fig = go.Figure()
        
        for category, values in data.items():
            fig.add_trace(go.Bar(
                name=category,
                x=list(values.keys()),
                y=list(values.values())
            ))
        
        fig.update_layout(
            title='Comparison Chart',
            xaxis_title='Categories',
            yaxis_title='Values',
            barmode='group'
        )
        
        return fig
    
    def _create_comparison_line_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create comparison line chart."""
        fig = go.Figure()
        
        for category, values in data.items():
            fig.add_trace(go.Scatter(
                name=category,
                x=list(values.keys()),
                y=list(values.values()),
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title='Trend Comparison',
            xaxis_title='Time/Categories',
            yaxis_title='Values'
        )
        
        return fig