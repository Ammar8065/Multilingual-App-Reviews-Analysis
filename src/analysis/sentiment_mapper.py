"""
Geographic sentiment mapping and visualization for multilingual app reviews.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
import json

from src.data.models import GeographicVisualization, ComparisonResult
from src.config import get_config
from src.utils.logger import get_logger


class SentimentMapper:
    """
    Geographic sentiment mapping and visualization component.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Country coordinates for mapping (simplified dataset)
        self.country_coords = {
            'United States': (39.8283, -98.5795),
            'China': (35.8617, 104.1954),
            'Germany': (51.1657, 10.4515),
            'India': (20.5937, 78.9629),
            'United Kingdom': (55.3781, -3.4360),
            'France': (46.2276, 2.2137),
            'Japan': (36.2048, 138.2529),
            'Brazil': (-14.2350, -51.9253),
            'Canada': (56.1304, -106.3468),
            'Australia': (-25.2744, 133.7751),
            'South Korea': (35.9078, 127.7669),
            'Spain': (40.4637, -3.7492),
            'Italy': (41.8719, 12.5674),
            'Russia': (61.5240, 105.3188),
            'Mexico': (23.6345, -102.5528),
            'Netherlands': (52.1326, 5.2913),
            'Sweden': (60.1282, 18.6435),
            'Norway': (60.4720, 8.4689),
            'Denmark': (56.2639, 9.5018),
            'Finland': (61.9241, 25.7482),
            'Poland': (51.9194, 19.1451),
            'Turkey': (38.9637, 35.2433),
            'Malaysia': (4.2105, 101.9758),
            'Vietnam': (14.0583, 108.2772),
            'Thailand': (15.8700, 100.9925),
            'Indonesia': (-0.7893, 113.9213),
            'Philippines': (12.8797, 121.7740),
            'Singapore': (1.3521, 103.8198),
            'Nigeria': (9.0820, 8.6753),
            'South Africa': (-30.5595, 22.9375),
            'Pakistan': (30.3753, 69.3451),
            'Bangladesh': (23.6850, 90.3563),
            'Argentina': (-38.4161, -63.6167),
            'Chile': (-35.6751, -71.5430),
            'Colombia': (4.5709, -74.2973),
            'Peru': (-9.1900, -75.0152),
            'Egypt': (26.0975, 30.0444),
            'Saudi Arabia': (23.8859, 45.0792),
            'UAE': (23.4241, 53.8478),
            'Israel': (31.0461, 34.8516),
            'Iran': (32.4279, 53.6880)
        }
    
    def create_sentiment_map(self, df: pd.DataFrame, sentiment_column: str = 'sentiment') -> GeographicVisualization:
        """
        Create interactive geographic sentiment visualization.
        
        Args:
            df: DataFrame with review data including sentiment
            sentiment_column: Name of the sentiment column
            
        Returns:
            GeographicVisualization with map and insights
        """
        self.logger.info("Creating geographic sentiment map")
        
        try:
            # Aggregate sentiment by country
            country_sentiment = self._aggregate_sentiment_by_country(df, sentiment_column)
            
            # Create interactive map using Plotly
            map_figure = self._create_plotly_choropleth(country_sentiment)
            
            # Generate insights
            insights = self._generate_geographic_insights(country_sentiment)
            
            # Calculate coverage statistics
            coverage_stats = self._calculate_coverage_stats(df, country_sentiment)
            
            return GeographicVisualization(
                map_figure=map_figure,
                data_summary=country_sentiment,
                geographic_insights=insights,
                coverage_stats=coverage_stats
            )
            
        except Exception as e:
            self.logger.error(f"Error creating sentiment map: {str(e)}")
            raise
    
    def create_folium_sentiment_map(self, df: pd.DataFrame, sentiment_column: str = 'sentiment') -> folium.Map:
        """
        Create Folium-based interactive sentiment map.
        
        Args:
            df: DataFrame with review data
            sentiment_column: Name of the sentiment column
            
        Returns:
            Folium map object
        """
        self.logger.info("Creating Folium sentiment map")
        
        try:
            # Aggregate sentiment by country
            country_sentiment = self._aggregate_sentiment_by_country(df, sentiment_column)
            
            # Create base map
            m = folium.Map(location=[20, 0], zoom_start=2, tiles='OpenStreetMap')
            
            # Add markers for each country
            for country, data in country_sentiment.items():
                if country in self.country_coords:
                    lat, lon = self.country_coords[country]
                    
                    # Determine marker color based on sentiment
                    avg_sentiment = data['avg_sentiment_score']
                    if avg_sentiment > 0.1:
                        color = 'green'
                    elif avg_sentiment < -0.1:
                        color = 'red'
                    else:
                        color = 'orange'
                    
                    # Create popup text
                    popup_text = f"""
                    <b>{country}</b><br>
                    Reviews: {data['review_count']}<br>
                    Avg Sentiment: {avg_sentiment:.3f}<br>
                    Positive: {data['positive_pct']:.1f}%<br>
                    Negative: {data['negative_pct']:.1f}%<br>
                    Neutral: {data['neutral_pct']:.1f}%
                    """
                    
                    # Add marker
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=min(max(data['review_count'] / 10, 5), 20),
                        popup=folium.Popup(popup_text, max_width=200),
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.7
                    ).add_to(m)
            
            # Add legend
            legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 150px; height: 90px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:14px; padding: 10px">
            <p><b>Sentiment Legend</b></p>
            <p><i class="fa fa-circle" style="color:green"></i> Positive</p>
            <p><i class="fa fa-circle" style="color:orange"></i> Neutral</p>
            <p><i class="fa fa-circle" style="color:red"></i> Negative</p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            return m
            
        except Exception as e:
            self.logger.error(f"Error creating Folium map: {str(e)}")
            raise
    
    def analyze_regional_sentiment_patterns(self, df: pd.DataFrame, sentiment_column: str = 'sentiment') -> Dict[str, Any]:
        """
        Analyze sentiment patterns across different regions.
        
        Args:
            df: DataFrame with review data
            sentiment_column: Name of the sentiment column
            
        Returns:
            Dictionary with regional sentiment analysis
        """
        self.logger.info("Analyzing regional sentiment patterns")
        
        try:
            # Define regions
            regions = {
                'North America': ['United States', 'Canada', 'Mexico'],
                'Europe': ['Germany', 'United Kingdom', 'France', 'Spain', 'Italy', 'Netherlands', 'Sweden', 'Norway', 'Denmark', 'Finland', 'Poland'],
                'Asia': ['China', 'India', 'Japan', 'South Korea', 'Malaysia', 'Vietnam', 'Thailand', 'Indonesia', 'Philippines', 'Singapore'],
                'South America': ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru'],
                'Middle East': ['Saudi Arabia', 'UAE', 'Israel', 'Iran', 'Egypt'],
                'Africa': ['Nigeria', 'South Africa'],
                'Oceania': ['Australia']
            }
            
            regional_analysis = {}
            
            for region, countries in regions.items():
                region_data = df[df['user_country'].isin(countries)]
                
                if not region_data.empty:
                    sentiment_dist = region_data[sentiment_column].value_counts(normalize=True) * 100
                    
                    regional_analysis[region] = {
                        'total_reviews': len(region_data),
                        'countries': countries,
                        'avg_rating': region_data['rating'].mean(),
                        'sentiment_distribution': sentiment_dist.to_dict(),
                        'top_apps': region_data['app_name'].value_counts().head(5).to_dict(),
                        'dominant_languages': region_data['review_language'].value_counts().head(3).to_dict()
                    }
            
            return regional_analysis
            
        except Exception as e:
            self.logger.error(f"Error in regional sentiment analysis: {str(e)}")
            raise
    
    def _aggregate_sentiment_by_country(self, df: pd.DataFrame, sentiment_column: str) -> Dict[str, Any]:
        """Aggregate sentiment data by country."""
        country_data = {}
        
        for country in df['user_country'].unique():
            country_reviews = df[df['user_country'] == country]
            
            if not country_reviews.empty:
                # Calculate sentiment distribution
                sentiment_counts = country_reviews[sentiment_column].value_counts()
                total_reviews = len(country_reviews)
                
                # Calculate sentiment percentages
                positive_pct = (sentiment_counts.get('positive', 0) / total_reviews) * 100
                negative_pct = (sentiment_counts.get('negative', 0) / total_reviews) * 100
                neutral_pct = (sentiment_counts.get('neutral', 0) / total_reviews) * 100
                
                # Calculate average sentiment score (positive=1, neutral=0, negative=-1)
                sentiment_scores = country_reviews[sentiment_column].map({
                    'positive': 1, 'neutral': 0, 'negative': -1
                })
                avg_sentiment_score = sentiment_scores.mean()
                
                country_data[country] = {
                    'review_count': total_reviews,
                    'avg_rating': country_reviews['rating'].mean(),
                    'positive_pct': positive_pct,
                    'negative_pct': negative_pct,
                    'neutral_pct': neutral_pct,
                    'avg_sentiment_score': avg_sentiment_score,
                    'dominant_language': country_reviews['review_language'].mode().iloc[0] if not country_reviews['review_language'].mode().empty else 'unknown',
                    'top_app': country_reviews['app_name'].mode().iloc[0] if not country_reviews['app_name'].mode().empty else 'unknown'
                }
        
        return country_data
    
    def _create_plotly_choropleth(self, country_sentiment: Dict[str, Any]) -> go.Figure:
        """Create Plotly choropleth map."""
        countries = list(country_sentiment.keys())
        sentiment_scores = [data['avg_sentiment_score'] for data in country_sentiment.values()]
        review_counts = [data['review_count'] for data in country_sentiment.values()]
        
        # Create hover text
        hover_text = []
        for country in countries:
            data = country_sentiment[country]
            text = f"""<b>{country}</b><br>
            Reviews: {data['review_count']}<br>
            Avg Rating: {data['avg_rating']:.2f}<br>
            Sentiment Score: {data['avg_sentiment_score']:.3f}<br>
            Positive: {data['positive_pct']:.1f}%<br>
            Negative: {data['negative_pct']:.1f}%<br>
            Top App: {data['top_app']}"""
            hover_text.append(text)
        
        fig = go.Figure(data=go.Choropleth(
            locations=countries,
            z=sentiment_scores,
            locationmode='country names',
            colorscale='RdYlGn',
            colorbar_title="Sentiment Score",
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_text
        ))
        
        fig.update_layout(
            title='Global Sentiment Distribution by Country',
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
            width=1200,
            height=700
        )
        
        return fig
    
    def _generate_geographic_insights(self, country_sentiment: Dict[str, Any]) -> List[str]:
        """Generate insights from geographic sentiment data."""
        insights = []
        
        # Find most positive and negative countries
        sorted_countries = sorted(country_sentiment.items(), 
                                key=lambda x: x[1]['avg_sentiment_score'], reverse=True)
        
        if sorted_countries:
            most_positive = sorted_countries[0]
            most_negative = sorted_countries[-1]
            
            insights.append(f"Most positive sentiment: {most_positive[0]} (score: {most_positive[1]['avg_sentiment_score']:.3f})")
            insights.append(f"Most negative sentiment: {most_negative[0]} (score: {most_negative[1]['avg_sentiment_score']:.3f})")
        
        # Find countries with highest review volumes
        sorted_by_volume = sorted(country_sentiment.items(), 
                                key=lambda x: x[1]['review_count'], reverse=True)
        
        if sorted_by_volume:
            top_volume = sorted_by_volume[0]
            insights.append(f"Highest review volume: {top_volume[0]} ({top_volume[1]['review_count']} reviews)")
        
        # Calculate global averages
        total_reviews = sum(data['review_count'] for data in country_sentiment.values())
        avg_global_sentiment = sum(data['avg_sentiment_score'] * data['review_count'] 
                                 for data in country_sentiment.values()) / total_reviews
        
        insights.append(f"Global average sentiment score: {avg_global_sentiment:.3f}")
        insights.append(f"Total countries analyzed: {len(country_sentiment)}")
        insights.append(f"Total reviews analyzed: {total_reviews}")
        
        return insights
    
    def _calculate_coverage_stats(self, df: pd.DataFrame, country_sentiment: Dict[str, Any]) -> Dict[str, int]:
        """Calculate coverage statistics."""
        return {
            'total_countries': len(country_sentiment),
            'total_reviews': len(df),
            'countries_with_data': len([c for c in country_sentiment if country_sentiment[c]['review_count'] > 0]),
            'avg_reviews_per_country': int(np.mean([data['review_count'] for data in country_sentiment.values()]))
        }


class RegionalComparator:
    """
    Component for comparing sentiment and behavior across regions.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__)
    
    def compare_regional_patterns(self, df: pd.DataFrame, sentiment_column: str = 'sentiment') -> ComparisonResult:
        """
        Compare patterns across different regions.
        
        Args:
            df: DataFrame with review data
            sentiment_column: Name of the sentiment column
            
        Returns:
            ComparisonResult with regional comparisons
        """
        self.logger.info("Comparing regional patterns")
        
        try:
            # Create sentiment mapper for regional analysis
            mapper = SentimentMapper()
            regional_analysis = mapper.analyze_regional_sentiment_patterns(df, sentiment_column)
            
            # Perform statistical comparisons
            statistical_tests = self._perform_statistical_tests(df, regional_analysis)
            
            # Identify significant differences
            significant_differences = self._identify_significant_differences(regional_analysis, statistical_tests)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(regional_analysis, significant_differences)
            
            return ComparisonResult(
                group_comparisons=regional_analysis,
                statistical_tests=statistical_tests,
                significant_differences=significant_differences,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error in regional comparison: {str(e)}")
            raise
    
    def _perform_statistical_tests(self, df: pd.DataFrame, regional_analysis: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Perform statistical tests between regions."""
        from scipy import stats
        
        statistical_tests = {}
        
        # Compare average ratings between regions
        region_ratings = {}
        for region, data in regional_analysis.items():
            region_countries = data['countries']
            region_data = df[df['user_country'].isin(region_countries)]
            if not region_data.empty:
                region_ratings[region] = region_data['rating'].values
        
        # Perform ANOVA test for ratings
        if len(region_ratings) > 2:
            rating_values = list(region_ratings.values())
            f_stat, p_value = stats.f_oneway(*rating_values)
            statistical_tests['rating_anova'] = {'f_statistic': f_stat, 'p_value': p_value}
        
        return statistical_tests
    
    def _identify_significant_differences(self, regional_analysis: Dict[str, Any], 
                                        statistical_tests: Dict[str, Dict[str, float]]) -> List[str]:
        """Identify significant differences between regions."""
        differences = []
        
        # Rating differences
        ratings = {region: data['avg_rating'] for region, data in regional_analysis.items()}
        max_rating_region = max(ratings, key=ratings.get)
        min_rating_region = min(ratings, key=ratings.get)
        
        if ratings[max_rating_region] - ratings[min_rating_region] > 0.5:
            differences.append(f"Significant rating difference: {max_rating_region} ({ratings[max_rating_region]:.2f}) vs {min_rating_region} ({ratings[min_rating_region]:.2f})")
        
        # Volume differences
        volumes = {region: data['total_reviews'] for region, data in regional_analysis.items()}
        max_volume_region = max(volumes, key=volumes.get)
        min_volume_region = min(volumes, key=volumes.get)
        
        if volumes[max_volume_region] > volumes[min_volume_region] * 3:
            differences.append(f"Significant volume difference: {max_volume_region} ({volumes[max_volume_region]} reviews) vs {min_volume_region} ({volumes[min_volume_region]} reviews)")
        
        return differences
    
    def _generate_recommendations(self, regional_analysis: Dict[str, Any], 
                                significant_differences: List[str]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Find regions with highest and lowest satisfaction
        ratings = {region: data['avg_rating'] for region, data in regional_analysis.items()}
        highest_rated_region = max(ratings, key=ratings.get)
        lowest_rated_region = min(ratings, key=ratings.get)
        
        recommendations.append(f"Study best practices from {highest_rated_region} (highest rated region) and apply to {lowest_rated_region}")
        
        # Volume-based recommendations
        volumes = {region: data['total_reviews'] for region, data in regional_analysis.items()}
        highest_volume_region = max(volumes, key=volumes.get)
        
        recommendations.append(f"Focus marketing efforts on regions with lower engagement, using strategies successful in {highest_volume_region}")
        
        # Language-based recommendations
        for region, data in regional_analysis.items():
            if 'dominant_languages' in data:
                top_lang = list(data['dominant_languages'].keys())[0] if data['dominant_languages'] else 'unknown'
                recommendations.append(f"Ensure {region} has adequate support for {top_lang} language")
        
        return recommendations