"""
Interactive dashboard generation for multilingual app reviews analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

from src.data.models import Dashboard, EDAReport
from src.visualization.visualization_engine import VisualizationEngine
from src.analysis.sentiment_mapper import SentimentMapper
from src.config import get_config
from src.utils.logger import get_logger


class DashboardGenerator:
    """
    Interactive dashboard generator for comprehensive data analysis.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.viz_engine = VisualizationEngine()
        self.sentiment_mapper = SentimentMapper()
    
    def create_eda_dashboard(self, df: pd.DataFrame) -> Dashboard:
        """
        Create comprehensive EDA dashboard with all key metrics.
        
        Args:
            df: DataFrame with review data
            
        Returns:
            Dashboard configuration
        """
        self.logger.info("Creating EDA dashboard")
        
        try:
            components = []
            
            # Overview metrics
            components.append(self._create_overview_metrics(df))
            
            # Distribution plots
            components.append(self._create_distribution_section(df))
            
            # Time series analysis
            components.append(self._create_time_series_section(df))
            
            # Geographic analysis
            components.append(self._create_geographic_section(df))
            
            # Text analysis
            components.append(self._create_text_analysis_section(df))
            
            dashboard = Dashboard(
                title="Multilingual App Reviews - Exploratory Data Analysis",
                components=components,
                layout_config=self._get_eda_layout_config(),
                data_sources=["multilingual_mobile_app_reviews_2025.csv"],
                last_updated=pd.Timestamp.now()
            )
            
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Error creating EDA dashboard: {str(e)}")
            raise
    
    def create_ml_results_dashboard(self, results: Dict[str, Any]) -> Dashboard:
        """
        Create ML results dashboard showing model performance.
        
        Args:
            results: Dictionary with ML model results
            
        Returns:
            Dashboard configuration
        """
        self.logger.info("Creating ML results dashboard")
        
        try:
            components = []
            
            # Model performance overview
            components.append(self._create_model_performance_section(results))
            
            # Feature importance analysis
            components.append(self._create_feature_importance_section(results))
            
            # Prediction analysis
            components.append(self._create_prediction_analysis_section(results))
            
            # Model comparison
            components.append(self._create_model_comparison_section(results))
            
            dashboard = Dashboard(
                title="Machine Learning Results Dashboard",
                components=components,
                layout_config=self._get_ml_layout_config(),
                data_sources=["model_results"],
                last_updated=pd.Timestamp.now()
            )
            
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Error creating ML results dashboard: {str(e)}")
            raise
    
    def create_geographic_dashboard(self, df: pd.DataFrame, sentiment_column: str = 'sentiment') -> Dashboard:
        """
        Create geographic analysis dashboard with interactive maps.
        
        Args:
            df: DataFrame with review data
            sentiment_column: Name of the sentiment column
            
        Returns:
            Dashboard configuration
        """
        self.logger.info("Creating geographic analysis dashboard")
        
        try:
            components = []
            
            # Global overview
            components.append(self._create_global_overview_section(df))
            
            # Interactive sentiment map
            components.append(self._create_interactive_map_section(df, sentiment_column))
            
            # Regional comparisons
            components.append(self._create_regional_comparison_section(df, sentiment_column))
            
            # Country-specific insights
            components.append(self._create_country_insights_section(df))
            
            dashboard = Dashboard(
                title="Geographic Sentiment Analysis Dashboard",
                components=components,
                layout_config=self._get_geographic_layout_config(),
                data_sources=["multilingual_mobile_app_reviews_2025.csv"],
                last_updated=pd.Timestamp.now()
            )
            
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Error creating geographic dashboard: {str(e)}")
            raise
    
    def generate_streamlit_app(self, df: pd.DataFrame) -> str:
        """
        Generate Streamlit application code for interactive dashboard.
        
        Args:
            df: DataFrame with review data
            
        Returns:
            String containing Streamlit app code
        """
        self.logger.info("Generating Streamlit application")
        
        streamlit_code = '''
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.visualization.visualization_engine import VisualizationEngine
from src.analysis.sentiment_mapper import SentimentMapper

# Page configuration
st.set_page_config(
    page_title="Multilingual App Reviews Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("multilingual_mobile_app_reviews_2025.csv")

df = load_data()
viz_engine = VisualizationEngine()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", [
    "Overview", "Distributions", "Time Series", "Geographic Analysis", 
    "Sentiment Analysis", "Text Analysis"
])

# Main title
st.title("📱 Multilingual Mobile App Reviews Analysis")

if page == "Overview":
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews", len(df))
    with col2:
        st.metric("Unique Apps", df['app_name'].nunique())
    with col3:
        st.metric("Countries", df['user_country'].nunique())
    with col4:
        st.metric("Languages", df['review_language'].nunique())
    
    st.subheader("Data Sample")
    st.dataframe(df.head())
    
    st.subheader("Basic Statistics")
    st.dataframe(df.describe())

elif page == "Distributions":
    st.header("Data Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating distribution
        fig_rating = px.histogram(df, x='rating', title='Rating Distribution')
        st.plotly_chart(fig_rating, use_container_width=True)
        
        # Language distribution
        lang_counts = df['review_language'].value_counts().head(10)
        fig_lang = px.bar(x=lang_counts.index, y=lang_counts.values, 
                         title='Top 10 Languages')
        st.plotly_chart(fig_lang, use_container_width=True)
    
    with col2:
        # Country distribution
        country_counts = df['user_country'].value_counts().head(10)
        fig_country = px.bar(x=country_counts.index, y=country_counts.values,
                           title='Top 10 Countries')
        st.plotly_chart(fig_country, use_container_width=True)
        
        # Category distribution
        fig_category = px.pie(df, names='app_category', title='App Categories')
        st.plotly_chart(fig_category, use_container_width=True)

elif page == "Time Series":
    st.header("Time Series Analysis")
    
    df['review_date'] = pd.to_datetime(df['review_date'])
    
    # Daily review volume
    daily_counts = df.groupby(df['review_date'].dt.date).size()
    fig_volume = px.line(x=daily_counts.index, y=daily_counts.values,
                        title='Daily Review Volume')
    st.plotly_chart(fig_volume, use_container_width=True)
    
    # Rating trends
    daily_ratings = df.groupby(df['review_date'].dt.date)['rating'].mean()
    fig_ratings = px.line(x=daily_ratings.index, y=daily_ratings.values,
                         title='Average Rating Trends')
    st.plotly_chart(fig_ratings, use_container_width=True)

elif page == "Geographic Analysis":
    st.header("Geographic Analysis")
    
    # Country-wise metrics
    country_metrics = df.groupby('user_country').agg({
        'rating': 'mean',
        'review_id': 'count'
    }).round(2)
    country_metrics.columns = ['Avg Rating', 'Review Count']
    
    # Choropleth map
    fig_map = px.choropleth(
        locations=country_metrics.index,
        color=country_metrics['Avg Rating'],
        locationmode='country names',
        title='Average Rating by Country'
    )
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Top countries table
    st.subheader("Top Countries by Review Count")
    st.dataframe(country_metrics.sort_values('Review Count', ascending=False).head(10))

elif page == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    
    if 'sentiment' in df.columns:
        # Sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        fig_sentiment = px.pie(values=sentiment_counts.values, 
                              names=sentiment_counts.index,
                              title='Sentiment Distribution')
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # Sentiment by language
        sentiment_lang = pd.crosstab(df['review_language'], df['sentiment'], 
                                   normalize='index') * 100
        top_langs = df['review_language'].value_counts().head(8).index
        fig_sent_lang = px.bar(sentiment_lang.loc[top_langs],
                              title='Sentiment by Language (Top 8)')
        st.plotly_chart(fig_sent_lang, use_container_width=True)
    else:
        st.warning("Sentiment analysis not available. Run sentiment analysis first.")

elif page == "Text Analysis":
    st.header("Text Analysis")
    
    # Text length distribution
    df['text_length'] = df['review_text'].str.len()
    fig_length = px.histogram(df, x='text_length', title='Review Text Length Distribution')
    st.plotly_chart(fig_length, use_container_width=True)
    
    # Most common words (simplified)
    all_text = ' '.join(df['review_text'].dropna().astype(str))
    words = all_text.lower().split()
    word_freq = pd.Series(words).value_counts().head(20)
    
    fig_words = px.bar(x=word_freq.values, y=word_freq.index,
                      orientation='h', title='Top 20 Most Common Words')
    st.plotly_chart(fig_words, use_container_width=True)
'''
        
        return streamlit_code
    
    def generate_dash_app(self, df: pd.DataFrame) -> dash.Dash:
        """
        Generate Dash application for interactive dashboard.
        
        Args:
            df: DataFrame with review data
            
        Returns:
            Dash application instance
        """
        self.logger.info("Generating Dash application")
        
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("📱 Multilingual App Reviews Analysis", 
                           className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Reviews"),
                            html.H2(f"{len(df):,}", className="text-primary")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Unique Apps"),
                            html.H2(f"{df['app_name'].nunique():,}", className="text-success")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Countries"),
                            html.H2(f"{df['user_country'].nunique():,}", className="text-info")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Languages"),
                            html.H2(f"{df['review_language'].nunique():,}", className="text-warning")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='rating-distribution',
                        figure=px.histogram(df, x='rating', title='Rating Distribution')
                    )
                ], width=6),
                dbc.Col([
                    dcc.Graph(
                        id='language-distribution',
                        figure=px.bar(
                            x=df['review_language'].value_counts().head(10).index,
                            y=df['review_language'].value_counts().head(10).values,
                            title='Top 10 Languages'
                        )
                    )
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='geographic-map',
                        figure=px.choropleth(
                            locations=df.groupby('user_country')['rating'].mean().index,
                            color=df.groupby('user_country')['rating'].mean().values,
                            locationmode='country names',
                            title='Average Rating by Country'
                        )
                    )
                ], width=12)
            ])
        ], fluid=True)
        
        return app
    
    def _create_overview_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create overview metrics component."""
        return {
            'type': 'metrics',
            'title': 'Dataset Overview',
            'data': {
                'total_reviews': len(df),
                'unique_apps': df['app_name'].nunique(),
                'countries': df['user_country'].nunique(),
                'languages': df['review_language'].nunique(),
                'avg_rating': df['rating'].mean(),
                'date_range': f"{df['review_date'].min()} to {df['review_date'].max()}"
            }
        }
    
    def _create_distribution_section(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create distribution plots section."""
        plots = self.viz_engine.create_distribution_plots(df)
        return {
            'type': 'plots',
            'title': 'Data Distributions',
            'plots': plots
        }
    
    def _create_time_series_section(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create time series analysis section."""
        plots = self.viz_engine.create_time_series_plots(df)
        return {
            'type': 'plots',
            'title': 'Time Series Analysis',
            'plots': plots
        }
    
    def _create_geographic_section(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create geographic analysis section."""
        maps = self.viz_engine.create_geographic_maps(df)
        return {
            'type': 'maps',
            'title': 'Geographic Analysis',
            'maps': maps
        }
    
    def _create_text_analysis_section(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create text analysis section."""
        word_clouds = self.viz_engine.create_word_clouds(df)
        return {
            'type': 'text_analysis',
            'title': 'Text Analysis',
            'word_clouds': word_clouds
        }
    
    def _create_model_performance_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create model performance section."""
        return {
            'type': 'model_performance',
            'title': 'Model Performance Overview',
            'data': results.get('performance_metrics', {})
        }
    
    def _create_feature_importance_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create feature importance section."""
        return {
            'type': 'feature_importance',
            'title': 'Feature Importance Analysis',
            'data': results.get('feature_importance', {})
        }
    
    def _create_prediction_analysis_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create prediction analysis section."""
        return {
            'type': 'predictions',
            'title': 'Prediction Analysis',
            'data': results.get('predictions', {})
        }
    
    def _create_model_comparison_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create model comparison section."""
        return {
            'type': 'model_comparison',
            'title': 'Model Comparison',
            'data': results.get('model_comparison', {})
        }
    
    def _create_global_overview_section(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create global overview section."""
        return {
            'type': 'global_overview',
            'title': 'Global Overview',
            'data': {
                'total_countries': df['user_country'].nunique(),
                'total_reviews': len(df),
                'avg_rating_global': df['rating'].mean(),
                'top_countries': df['user_country'].value_counts().head(10).to_dict()
            }
        }
    
    def _create_interactive_map_section(self, df: pd.DataFrame, sentiment_column: str) -> Dict[str, Any]:
        """Create interactive map section."""
        if sentiment_column in df.columns:
            geo_viz = self.sentiment_mapper.create_sentiment_map(df, sentiment_column)
            return {
                'type': 'interactive_map',
                'title': 'Interactive Sentiment Map',
                'data': geo_viz
            }
        else:
            return {
                'type': 'message',
                'title': 'Interactive Sentiment Map',
                'message': 'Sentiment analysis required for this visualization'
            }
    
    def _create_regional_comparison_section(self, df: pd.DataFrame, sentiment_column: str) -> Dict[str, Any]:
        """Create regional comparison section."""
        regional_analysis = self.sentiment_mapper.analyze_regional_sentiment_patterns(df, sentiment_column)
        return {
            'type': 'regional_comparison',
            'title': 'Regional Comparisons',
            'data': regional_analysis
        }
    
    def _create_country_insights_section(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create country-specific insights section."""
        country_insights = {}
        
        for country in df['user_country'].value_counts().head(10).index:
            country_data = df[df['user_country'] == country]
            country_insights[country] = {
                'total_reviews': len(country_data),
                'avg_rating': country_data['rating'].mean(),
                'top_apps': country_data['app_name'].value_counts().head(3).to_dict(),
                'dominant_language': country_data['review_language'].mode().iloc[0] if not country_data['review_language'].mode().empty else 'unknown'
            }
        
        return {
            'type': 'country_insights',
            'title': 'Country-Specific Insights',
            'data': country_insights
        }
    
    def _get_eda_layout_config(self) -> Dict[str, Any]:
        """Get EDA dashboard layout configuration."""
        return {
            'theme': 'light',
            'sidebar': True,
            'navigation': ['Overview', 'Distributions', 'Time Series', 'Geographic', 'Text Analysis'],
            'responsive': True
        }
    
    def _get_ml_layout_config(self) -> Dict[str, Any]:
        """Get ML dashboard layout configuration."""
        return {
            'theme': 'dark',
            'sidebar': True,
            'navigation': ['Performance', 'Features', 'Predictions', 'Comparison'],
            'responsive': True
        }
    
    def _get_geographic_layout_config(self) -> Dict[str, Any]:
        """Get geographic dashboard layout configuration."""
        return {
            'theme': 'light',
            'sidebar': True,
            'navigation': ['Global', 'Regional', 'Country Details'],
            'responsive': True,
            'map_style': 'open-street-map'
        }


class ReportGenerator:
    """
    Automated report generation component.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__)
    
    def generate_comprehensive_report(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> str:
        """
        Generate comprehensive analysis report.
        
        Args:
            df: DataFrame with review data
            analysis_results: Dictionary with all analysis results
            
        Returns:
            HTML report string
        """
        self.logger.info("Generating comprehensive report")
        
        try:
            report_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Multilingual App Reviews Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; text-align: center; }}
                    .section {{ margin: 20px 0; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #e9ecef; border-radius: 5px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>📱 Multilingual Mobile App Reviews Analysis Report</h1>
                    <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>Executive Summary</h2>
                    {self._generate_executive_summary(df, analysis_results)}
                </div>
                
                <div class="section">
                    <h2>Dataset Overview</h2>
                    {self._generate_dataset_overview(df)}
                </div>
                
                <div class="section">
                    <h2>Key Findings</h2>
                    {self._generate_key_findings(df, analysis_results)}
                </div>
                
                <div class="section">
                    <h2>Recommendations</h2>
                    {self._generate_recommendations(analysis_results)}
                </div>
            </body>
            </html>
            """
            
            return report_html
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise
    
    def _generate_executive_summary(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> str:
        """Generate executive summary section."""
        avg_rating = df['rating'].mean()
        total_reviews = len(df)
        countries = df['user_country'].nunique()
        languages = df['review_language'].nunique()
        
        return f"""
        <p>This report analyzes {total_reviews:,} multilingual mobile app reviews across {countries} countries 
        and {languages} languages. The overall average rating is {avg_rating:.2f} out of 5.0.</p>
        
        <div class="metric">
            <strong>Total Reviews:</strong> {total_reviews:,}
        </div>
        <div class="metric">
            <strong>Average Rating:</strong> {avg_rating:.2f}/5.0
        </div>
        <div class="metric">
            <strong>Countries:</strong> {countries}
        </div>
        <div class="metric">
            <strong>Languages:</strong> {languages}
        </div>
        """
    
    def _generate_dataset_overview(self, df: pd.DataFrame) -> str:
        """Generate dataset overview section."""
        date_range = f"{df['review_date'].min()} to {df['review_date'].max()}"
        top_apps = df['app_name'].value_counts().head(5)
        top_countries = df['user_country'].value_counts().head(5)
        
        apps_table = "".join([f"<tr><td>{app}</td><td>{count}</td></tr>" for app, count in top_apps.items()])
        countries_table = "".join([f"<tr><td>{country}</td><td>{count}</td></tr>" for country, count in top_countries.items()])
        
        return f"""
        <p><strong>Date Range:</strong> {date_range}</p>
        
        <h3>Top 5 Apps by Review Count</h3>
        <table>
            <tr><th>App Name</th><th>Review Count</th></tr>
            {apps_table}
        </table>
        
        <h3>Top 5 Countries by Review Count</h3>
        <table>
            <tr><th>Country</th><th>Review Count</th></tr>
            {countries_table}
        </table>
        """
    
    def _generate_key_findings(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> str:
        """Generate key findings section."""
        findings = []
        
        # Rating insights
        rating_std = df['rating'].std()
        if rating_std > 1.5:
            findings.append("High rating variability indicates diverse user experiences")
        
        # Language insights
        dominant_language = df['review_language'].mode().iloc[0]
        lang_percentage = (df['review_language'].value_counts().iloc[0] / len(df)) * 100
        findings.append(f"Most reviews are in {dominant_language} ({lang_percentage:.1f}%)")
        
        # Geographic insights
        top_country = df['user_country'].value_counts().index[0]
        country_percentage = (df['user_country'].value_counts().iloc[0] / len(df)) * 100
        findings.append(f"Highest review volume from {top_country} ({country_percentage:.1f}%)")
        
        findings_html = "".join([f"<li>{finding}</li>" for finding in findings])
        return f"<ul>{findings_html}</ul>"
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> str:
        """Generate recommendations section."""
        recommendations = [
            "Focus on improving user experience in low-rated regions",
            "Expand language support for underrepresented languages",
            "Investigate cultural preferences in different markets",
            "Monitor temporal patterns for optimal release timing",
            "Leverage positive sentiment regions for marketing insights"
        ]
        
        rec_html = "".join([f"<li>{rec}</li>" for rec in recommendations])
        return f"<ul>{rec_html}</ul>"