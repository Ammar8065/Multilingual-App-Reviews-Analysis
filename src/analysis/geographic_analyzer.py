"""
Geographic analysis components for location-based insights.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from src.data.interfaces import GeographicAnalyzerInterface
from src.data.models import GeographicVisualization, ComparisonResult
from src.config import get_config
from src.utils.logger import get_logger


class GeographicAnalyzer(GeographicAnalyzerInterface):
    """Geographic analyzer for location-based insights and patterns."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.visualizations = []
        
        # Country code mappings and coordinates (simplified set)
        self.country_coordinates = {
            'United States': {'lat': 39.8283, 'lon': -98.5795, 'code': 'US'},
            'China': {'lat': 35.8617, 'lon': 104.1954, 'code': 'CN'},
            'Germany': {'lat': 51.1657, 'lon': 10.4515, 'code': 'DE'},
            'Nigeria': {'lat': 9.0820, 'lon': 8.6753, 'code': 'NG'},
            'India': {'lat': 20.5937, 'lon': 78.9629, 'code': 'IN'},
            'South Korea': {'lat': 35.9078, 'lon': 127.7669, 'code': 'KR'},
            'Spain': {'lat': 40.4637, 'lon': -3.7492, 'code': 'ES'},
            'Australia': {'lat': -25.2744, 'lon': 133.7751, 'code': 'AU'},
            'Malaysia': {'lat': 4.2105, 'lon': 101.9758, 'code': 'MY'},
            'Brazil': {'lat': -14.2350, 'lon': -51.9253, 'code': 'BR'},
            'Vietnam': {'lat': 14.0583, 'lon': 108.2772, 'code': 'VN'},
            'Pakistan': {'lat': 30.3753, 'lon': 69.3451, 'code': 'PK'},
            'United Kingdom': {'lat': 55.3781, 'lon': -3.4360, 'code': 'GB'},
            'Japan': {'lat': 36.2048, 'lon': 138.2529, 'code': 'JP'},
            'France': {'lat': 46.6034, 'lon': 1.8883, 'code': 'FR'},
            'Canada': {'lat': 56.1304, 'lon': -106.3468, 'code': 'CA'},
            'Italy': {'lat': 41.8719, 'lon': 12.5674, 'code': 'IT'},
            'Russia': {'lat': 61.5240, 'lon': 105.3188, 'code': 'RU'},
            'Mexico': {'lat': 23.6345, 'lon': -102.5528, 'code': 'MX'},
            'Turkey': {'lat': 38.9637, 'lon': 35.2433, 'code': 'TR'}
        }
    
    def analyze_sentiment_by_country(self, df: pd.DataFrame, 
                                   country_column: str = 'user_country',
                                   rating_column: str = 'rating') -> Dict[str, Any]:
        """
        Analyze sentiment patterns by country.
        
        Args:
            df: Input DataFrame
            country_column: Name of the country column
            rating_column: Name of the rating column (used as sentiment proxy)
            
        Returns:
            Dictionary containing country-wise sentiment analysis
        """
        self.logger.info(f"Analyzing sentiment by country using columns '{country_column}' and '{rating_column}'")
        
        if country_column not in df.columns:
            self.logger.error(f"Country column '{country_column}' not found")
            return {}
        
        # Basic country statistics
        country_stats = df.groupby(country_column).agg({
            'review_id': 'count',
            rating_column: ['mean', 'std', 'min', 'max'] if rating_column in df.columns else 'count'
        }).round(3)
        
        # Rename columns for clarity
        if rating_column in df.columns:
            country_stats.columns = ['review_count', 'avg_rating', 'rating_std', 'min_rating', 'max_rating']
        else:
            country_stats.columns = ['review_count']
        
        # Sort by review count
        country_stats = country_stats.sort_values('review_count', ascending=False)
        
        # Calculate sentiment categories (if rating available)
        sentiment_by_country = {}
        if rating_column in df.columns:
            for country in df[country_column].unique():
                if pd.isna(country):
                    continue
                
                country_data = df[df[country_column] == country]
                ratings = country_data[rating_column].dropna()
                
                if not ratings.empty:
                    sentiment_by_country[country] = {
                        'positive_pct': float((ratings >= 4.0).mean() * 100),
                        'neutral_pct': float(((ratings >= 2.5) & (ratings < 4.0)).mean() * 100),
                        'negative_pct': float((ratings < 2.5).mean() * 100),
                        'avg_rating': float(ratings.mean()),
                        'total_reviews': len(ratings)
                    }
        
        # Language diversity by country
        language_diversity = {}
        if 'review_language' in df.columns:
            for country in df[country_column].unique():
                if pd.isna(country):
                    continue
                
                country_data = df[df[country_column] == country]
                lang_counts = country_data['review_language'].value_counts()
                
                language_diversity[country] = {
                    'unique_languages': int(lang_counts.nunique()),
                    'dominant_language': lang_counts.index[0] if not lang_counts.empty else None,
                    'language_distribution': lang_counts.head(5).to_dict()
                }
        
        # App category preferences by country
        category_preferences = {}
        if 'app_category' in df.columns:
            for country in df[country_column].unique():
                if pd.isna(country):
                    continue
                
                country_data = df[df[country_column] == country]
                category_counts = country_data['app_category'].value_counts()
                category_pct = country_data['app_category'].value_counts(normalize=True) * 100
                
                category_preferences[country] = {
                    'top_categories': category_counts.head(5).to_dict(),
                    'category_percentages': {k: round(v, 2) for k, v in category_pct.head(5).to_dict().items()}
                }
        
        # Device preferences by country
        device_preferences = {}
        if 'device_type' in df.columns:
            for country in df[country_column].unique():
                if pd.isna(country):
                    continue
                
                country_data = df[df[country_column] == country]
                device_counts = country_data['device_type'].value_counts()
                device_pct = country_data['device_type'].value_counts(normalize=True) * 100
                
                device_preferences[country] = {
                    'device_distribution': device_counts.to_dict(),
                    'device_percentages': {k: round(v, 2) for k, v in device_pct.to_dict().items()}
                }
        
        analysis_result = {
            'country_statistics': country_stats.to_dict(),
            'sentiment_by_country': sentiment_by_country,
            'language_diversity': language_diversity,
            'category_preferences': category_preferences,
            'device_preferences': device_preferences,
            'total_countries': int(df[country_column].nunique()),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Country sentiment analysis completed for {analysis_result['total_countries']} countries")
        
        return analysis_result
    
    def create_sentiment_map(self, df: pd.DataFrame,
                           country_column: str = 'user_country',
                           rating_column: str = 'rating') -> GeographicVisualization:
        """
        Create geographic sentiment visualization map.
        
        Args:
            df: Input DataFrame
            country_column: Name of the country column
            rating_column: Name of the rating column
            
        Returns:
            GeographicVisualization object containing map and metadata
        """
        self.logger.info("Creating geographic sentiment map")
        
        if country_column not in df.columns or rating_column not in df.columns:
            self.logger.error("Required columns not found for sentiment mapping")
            return GeographicVisualization(
                map_figure=go.Figure(),
                data_summary={},
                geographic_insights=[],
                coverage_stats={}
            )
        
        # Aggregate data by country
        country_data = df.groupby(country_column).agg({
            'review_id': 'count',
            rating_column: 'mean'
        }).round(3)
        
        country_data.columns = ['review_count', 'avg_rating']
        country_data = country_data.reset_index()
        
        # Add coordinates for countries we have data for
        map_data = []
        coverage_stats = {'mapped_countries': 0, 'unmapped_countries': 0, 'total_reviews_mapped': 0}
        
        for _, row in country_data.iterrows():
            country = row[country_column]
            if country in self.country_coordinates:
                coords = self.country_coordinates[country]
                map_data.append({
                    'country': country,
                    'lat': coords['lat'],
                    'lon': coords['lon'],
                    'avg_rating': row['avg_rating'],
                    'review_count': row['review_count'],
                    'sentiment_category': self._categorize_sentiment(row['avg_rating'])
                })
                coverage_stats['mapped_countries'] += 1
                coverage_stats['total_reviews_mapped'] += row['review_count']
            else:
                coverage_stats['unmapped_countries'] += 1
        
        # Create interactive map using Plotly
        if map_data:
            map_df = pd.DataFrame(map_data)
            
            # Create scatter mapbox
            fig = px.scatter_mapbox(
                map_df,
                lat='lat',
                lon='lon',
                size='review_count',
                color='avg_rating',
                hover_name='country',
                hover_data={'review_count': True, 'avg_rating': ':.2f'},
                color_continuous_scale='RdYlGn',
                size_max=50,
                zoom=1,
                title='Global Sentiment Map by Country'
            )
            
            fig.update_layout(
                mapbox_style="open-street-map",
                height=600,
                margin={"r": 0, "t": 50, "l": 0, "b": 0}
            )
        else:
            fig = go.Figure()
            fig.update_layout(title="No geographic data available for mapping")
        
        # Generate insights
        insights = self._generate_geographic_insights(map_data, country_data)
        
        # Data summary
        data_summary = {
            'total_countries_in_data': len(country_data),
            'countries_mapped': len(map_data),
            'mapping_coverage_pct': (len(map_data) / len(country_data)) * 100 if len(country_data) > 0 else 0,
            'avg_global_rating': float(country_data['avg_rating'].mean()),
            'highest_rated_country': country_data.loc[country_data['avg_rating'].idxmax(), country_column] if not country_data.empty else None,
            'lowest_rated_country': country_data.loc[country_data['avg_rating'].idxmin(), country_column] if not country_data.empty else None
        }
        
        self.visualizations.append(fig)
        
        return GeographicVisualization(
            map_figure=fig,
            data_summary=data_summary,
            geographic_insights=insights,
            coverage_stats=coverage_stats
        )
    
    def compare_regional_patterns(self, df: pd.DataFrame,
                                country_column: str = 'user_country') -> ComparisonResult:
        """
        Compare patterns across different regions.
        
        Args:
            df: Input DataFrame
            country_column: Name of the country column
            
        Returns:
            ComparisonResult containing regional comparisons
        """
        self.logger.info("Comparing regional patterns")
        
        # Define regions (simplified grouping)
        region_mapping = {
            'North America': ['United States', 'Canada', 'Mexico'],
            'Europe': ['Germany', 'United Kingdom', 'France', 'Italy', 'Spain', 'Russia'],
            'Asia': ['China', 'India', 'Japan', 'South Korea', 'Malaysia', 'Vietnam', 'Pakistan'],
            'Oceania': ['Australia'],
            'Africa': ['Nigeria'],
            'South America': ['Brazil']
        }
        
        # Create reverse mapping
        country_to_region = {}
        for region, countries in region_mapping.items():
            for country in countries:
                country_to_region[country] = region
        
        # Add region column
        df_regional = df.copy()
        df_regional['region'] = df_regional[country_column].map(country_to_region)
        df_regional['region'] = df_regional['region'].fillna('Other')
        
        # Regional comparisons
        group_comparisons = {}
        
        # Basic statistics by region
        if 'rating' in df.columns:
            regional_stats = df_regional.groupby('region').agg({
                'review_id': 'count',
                'rating': ['mean', 'std', 'min', 'max']
            }).round(3)
            
            regional_stats.columns = ['review_count', 'avg_rating', 'rating_std', 'min_rating', 'max_rating']
            group_comparisons['basic_statistics'] = regional_stats.to_dict()
        
        # Language patterns by region
        if 'review_language' in df.columns:
            regional_languages = {}
            for region in df_regional['region'].unique():
                region_data = df_regional[df_regional['region'] == region]
                lang_dist = region_data['review_language'].value_counts(normalize=True) * 100
                regional_languages[region] = {k: round(v, 2) for k, v in lang_dist.head(5).to_dict().items()}
            
            group_comparisons['language_patterns'] = regional_languages
        
        # App category preferences by region
        if 'app_category' in df.columns:
            regional_categories = {}
            for region in df_regional['region'].unique():
                region_data = df_regional[df_regional['region'] == region]
                cat_dist = region_data['app_category'].value_counts(normalize=True) * 100
                regional_categories[region] = {k: round(v, 2) for k, v in cat_dist.head(5).to_dict().items()}
            
            group_comparisons['category_preferences'] = regional_categories
        
        # Device preferences by region
        if 'device_type' in df.columns:
            regional_devices = {}
            for region in df_regional['region'].unique():
                region_data = df_regional[df_regional['region'] == region]
                device_dist = region_data['device_type'].value_counts(normalize=True) * 100
                regional_devices[region] = {k: round(v, 2) for k, v in device_dist.to_dict().items()}
            
            group_comparisons['device_preferences'] = regional_devices
        
        # Statistical tests (simplified)
        statistical_tests = {}
        if 'rating' in df.columns:
            # ANOVA-like comparison of ratings across regions
            regional_ratings = []
            region_names = []
            
            for region in df_regional['region'].unique():
                region_ratings = df_regional[df_regional['region'] == region]['rating'].dropna()
                if len(region_ratings) > 0:
                    regional_ratings.extend(region_ratings.tolist())
                    region_names.extend([region] * len(region_ratings))
            
            if len(set(region_names)) > 1:
                # Calculate F-statistic (simplified)
                region_means = df_regional.groupby('region')['rating'].mean()
                overall_mean = df_regional['rating'].mean()
                
                between_group_var = sum(
                    df_regional[df_regional['region'] == region]['rating'].count() * 
                    (region_mean - overall_mean) ** 2
                    for region, region_mean in region_means.items()
                ) / (len(region_means) - 1)
                
                within_group_var = sum(
                    ((df_regional[df_regional['region'] == region]['rating'] - region_mean) ** 2).sum()
                    for region, region_mean in region_means.items()
                ) / (len(df_regional) - len(region_means))
                
                f_statistic = between_group_var / within_group_var if within_group_var > 0 else 0
                
                statistical_tests['rating_comparison'] = {
                    'f_statistic': float(f_statistic),
                    'significance': 'significant' if f_statistic > 2.0 else 'not_significant'
                }
        
        # Identify significant differences
        significant_differences = []
        
        if 'basic_statistics' in group_comparisons:
            ratings_by_region = {k: v['avg_rating'] for k, v in group_comparisons['basic_statistics'].items()}
            max_rating_region = max(ratings_by_region, key=ratings_by_region.get)
            min_rating_region = min(ratings_by_region, key=ratings_by_region.get)
            
            rating_diff = ratings_by_region[max_rating_region] - ratings_by_region[min_rating_region]
            
            if rating_diff > 0.5:  # Significant difference threshold
                significant_differences.append(
                    f"Significant rating difference between {max_rating_region} "
                    f"({ratings_by_region[max_rating_region]:.2f}) and {min_rating_region} "
                    f"({ratings_by_region[min_rating_region]:.2f})"
                )
        
        # Generate recommendations
        recommendations = self._generate_regional_recommendations(group_comparisons, significant_differences)
        
        # Create regional comparison visualization
        self._create_regional_comparison_plots(df_regional, group_comparisons)
        
        return ComparisonResult(
            group_comparisons=group_comparisons,
            statistical_tests=statistical_tests,
            significant_differences=significant_differences,
            recommendations=recommendations
        )
    
    def _categorize_sentiment(self, rating: float) -> str:
        """Categorize sentiment based on rating."""
        if rating >= 4.0:
            return 'Positive'
        elif rating >= 2.5:
            return 'Neutral'
        else:
            return 'Negative'
    
    def _generate_geographic_insights(self, map_data: List[Dict], 
                                    country_data: pd.DataFrame) -> List[str]:
        """Generate insights from geographic analysis."""
        insights = []
        
        if not map_data:
            return ["No geographic data available for analysis"]
        
        # Highest and lowest rated countries
        map_df = pd.DataFrame(map_data)
        highest_rated = map_df.loc[map_df['avg_rating'].idxmax()]
        lowest_rated = map_df.loc[map_df['avg_rating'].idxmin()]
        
        insights.append(
            f"Highest rated country: {highest_rated['country']} "
            f"(avg rating: {highest_rated['avg_rating']:.2f})"
        )
        
        insights.append(
            f"Lowest rated country: {lowest_rated['country']} "
            f"(avg rating: {lowest_rated['avg_rating']:.2f})"
        )
        
        # Countries with most reviews
        most_reviews = map_df.loc[map_df['review_count'].idxmax()]
        insights.append(
            f"Most active country: {most_reviews['country']} "
            f"({most_reviews['review_count']} reviews)"
        )
        
        # Rating distribution insights
        positive_countries = len(map_df[map_df['avg_rating'] >= 4.0])
        negative_countries = len(map_df[map_df['avg_rating'] < 2.5])
        
        insights.append(
            f"Countries with positive sentiment: {positive_countries} "
            f"({positive_countries/len(map_df)*100:.1f}%)"
        )
        
        if negative_countries > 0:
            insights.append(
                f"Countries with negative sentiment: {negative_countries} "
                f"({negative_countries/len(map_df)*100:.1f}%)"
            )
        
        return insights
    
    def _generate_regional_recommendations(self, comparisons: Dict, 
                                         differences: List[str]) -> List[str]:
        """Generate recommendations based on regional analysis."""
        recommendations = []
        
        if differences:
            recommendations.append(
                "Significant regional differences detected. Consider region-specific strategies."
            )
        
        if 'language_patterns' in comparisons:
            recommendations.append(
                "Language preferences vary by region. Implement localized content strategies."
            )
        
        if 'category_preferences' in comparisons:
            recommendations.append(
                "App category preferences differ across regions. Tailor app recommendations accordingly."
            )
        
        if 'device_preferences' in comparisons:
            recommendations.append(
                "Device usage patterns vary by region. Optimize for regional device preferences."
            )
        
        recommendations.append(
            "Monitor regional trends regularly to identify emerging patterns and opportunities."
        )
        
        return recommendations
    
    def _create_regional_comparison_plots(self, df: pd.DataFrame, comparisons: Dict):
        """Create visualizations for regional comparisons."""
        
        # Regional rating comparison
        if 'basic_statistics' in comparisons:
            regions = list(comparisons['basic_statistics'].keys())
            avg_ratings = [comparisons['basic_statistics'][region]['avg_rating'] for region in regions]
            review_counts = [comparisons['basic_statistics'][region]['review_count'] for region in regions]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Average Rating by Region', 'Review Count by Region')
            )
            
            fig.add_trace(
                go.Bar(x=regions, y=avg_ratings, name='Avg Rating', marker_color='lightblue'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=regions, y=review_counts, name='Review Count', marker_color='lightcoral'),
                row=1, col=2
            )
            
            fig.update_layout(
                title='Regional Comparison Analysis',
                showlegend=False,
                height=500
            )
            
            fig.update_xaxes(title_text="Region", row=1, col=1)
            fig.update_xaxes(title_text="Region", row=1, col=2)
            fig.update_yaxes(title_text="Average Rating", row=1, col=1)
            fig.update_yaxes(title_text="Number of Reviews", row=1, col=2)
            
            self.visualizations.append(fig)