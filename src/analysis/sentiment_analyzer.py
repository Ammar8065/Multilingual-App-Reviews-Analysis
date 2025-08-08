"""
Multilingual sentiment analysis components.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# NLP libraries
from textblob import TextBlob
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from src.data.interfaces import SentimentAnalyzerInterface
from src.data.models import SentimentResult
from src.config import get_config
from src.utils.logger import get_logger


class MultilingualSentimentAnalyzer(SentimentAnalyzerInterface):
    """Multilingual sentiment analyzer supporting multiple analysis methods."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.visualizations = []
        
        # Initialize sentiment analyzers
        self.vader_analyzer = None
        self.transformers_pipeline = None
        self.transformers_tokenizer = None
        
        self._initialize_analyzers()
        
        # Language-specific sentiment mappings
        self.language_sentiment_models = {
            'en': 'textblob',  # Default to TextBlob for English
            'es': 'textblob',  # Spanish
            'fr': 'textblob',  # French
            'de': 'textblob',  # German
            'it': 'textblob',  # Italian
            'pt': 'textblob',  # Portuguese
            'nl': 'textblob',  # Dutch
            # For other languages, we'll use multilingual transformers if available
        }
        
        # Sentiment thresholds
        self.sentiment_thresholds = {
            'positive': 0.1,
            'negative': -0.1
        }
    
    def _initialize_analyzers(self):
        """Initialize available sentiment analysis tools."""
        try:
            # Initialize VADER
            if SentimentIntensityAnalyzer:
                self.vader_analyzer = SentimentIntensityAnalyzer()
                self.logger.info("VADER sentiment analyzer initialized")
            
            # Initialize Transformers pipeline
            if TRANSFORMERS_AVAILABLE:
                try:
                    # Use a multilingual sentiment model
                    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
                    self.transformers_pipeline = pipeline(
                        "sentiment-analysis",
                        model=model_name,
                        tokenizer=model_name,
                        return_all_scores=True
                    )
                    self.logger.info("Transformers multilingual sentiment pipeline initialized")
                except Exception as e:
                    self.logger.warning(f"Could not initialize transformers pipeline: {str(e)}")
                    # Fallback to a simpler model
                    try:
                        self.transformers_pipeline = pipeline(
                            "sentiment-analysis",
                            model="distilbert-base-uncased-finetuned-sst-2-english",
                            return_all_scores=True
                        )
                        self.logger.info("Fallback transformers pipeline initialized")
                    except Exception as e2:
                        self.logger.warning(f"Could not initialize fallback pipeline: {str(e2)}")
            
        except Exception as e:
            self.logger.error(f"Error initializing sentiment analyzers: {str(e)}")
    
    def analyze_sentiment(self, text: str, language: str = 'en') -> SentimentResult:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Input text to analyze
            language: Language code of the text
            
        Returns:
            SentimentResult containing sentiment analysis
        """
        if not text or pd.isna(text) or len(str(text).strip()) == 0:
            return SentimentResult(
                sentiment='neutral',
                confidence=0.0,
                language=language,
                processing_method='empty_text'
            )
        
        text = str(text).strip()
        
        # Choose analysis method based on language and availability
        if language in self.language_sentiment_models:
            method = self.language_sentiment_models[language]
        else:
            # Use transformers for non-supported languages if available
            method = 'transformers' if self.transformers_pipeline else 'textblob'
        
        # Perform sentiment analysis
        try:
            if method == 'transformers' and self.transformers_pipeline:
                return self._analyze_with_transformers(text, language)
            elif method == 'vader' and self.vader_analyzer:
                return self._analyze_with_vader(text, language)
            else:
                return self._analyze_with_textblob(text, language)
                
        except Exception as e:
            self.logger.warning(f"Error in sentiment analysis: {str(e)}")
            # Fallback to simple rule-based analysis
            return self._analyze_with_simple_rules(text, language)
    
    def _analyze_with_textblob(self, text: str, language: str) -> SentimentResult:
        """Analyze sentiment using TextBlob."""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Convert polarity to sentiment category
            if polarity > self.sentiment_thresholds['positive']:
                sentiment = 'positive'
            elif polarity < self.sentiment_thresholds['negative']:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            # Convert polarity to confidence (0-1 scale)
            confidence = min(abs(polarity), 1.0)
            
            return SentimentResult(
                sentiment=sentiment,
                confidence=confidence,
                language=language,
                processing_method='textblob'
            )
            
        except Exception as e:
            self.logger.warning(f"TextBlob analysis failed: {str(e)}")
            return self._analyze_with_simple_rules(text, language)
    
    def _analyze_with_vader(self, text: str, language: str) -> SentimentResult:
        """Analyze sentiment using VADER."""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            compound_score = scores['compound']
            
            # Convert compound score to sentiment category
            if compound_score >= 0.05:
                sentiment = 'positive'
            elif compound_score <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            # Use compound score as confidence
            confidence = min(abs(compound_score), 1.0)
            
            return SentimentResult(
                sentiment=sentiment,
                confidence=confidence,
                language=language,
                processing_method='vader'
            )
            
        except Exception as e:
            self.logger.warning(f"VADER analysis failed: {str(e)}")
            return self._analyze_with_simple_rules(text, language)
    
    def _analyze_with_transformers(self, text: str, language: str) -> SentimentResult:
        """Analyze sentiment using Transformers."""
        try:
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            results = self.transformers_pipeline(text)
            
            # Extract sentiment and confidence
            if isinstance(results[0], list):
                # Multiple scores returned
                scores = {result['label'].lower(): result['score'] for result in results[0]}
                
                # Map labels to our sentiment categories
                label_mapping = {
                    'positive': 'positive',
                    'negative': 'negative',
                    'neutral': 'neutral',
                    'label_0': 'negative',  # Some models use numeric labels
                    'label_1': 'neutral',
                    'label_2': 'positive'
                }
                
                best_sentiment = 'neutral'
                best_confidence = 0.0
                
                for label, score in scores.items():
                    mapped_label = label_mapping.get(label, 'neutral')
                    if score > best_confidence:
                        best_sentiment = mapped_label
                        best_confidence = score
            else:
                # Single result
                result = results[0]
                label = result['label'].lower()
                confidence = result['score']
                
                # Map label to sentiment
                if 'pos' in label or label == 'positive':
                    sentiment = 'positive'
                elif 'neg' in label or label == 'negative':
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                best_sentiment = sentiment
                best_confidence = confidence
            
            return SentimentResult(
                sentiment=best_sentiment,
                confidence=best_confidence,
                language=language,
                processing_method='transformers'
            )
            
        except Exception as e:
            self.logger.warning(f"Transformers analysis failed: {str(e)}")
            return self._analyze_with_simple_rules(text, language)
    
    def _analyze_with_simple_rules(self, text: str, language: str) -> SentimentResult:
        """Simple rule-based sentiment analysis as fallback."""
        text_lower = text.lower()
        
        # Simple positive/negative word lists (English-focused)
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'like', 'best', 'perfect', 'awesome', 'brilliant',
            'outstanding', 'superb', 'magnificent', 'incredible'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate',
            'dislike', 'poor', 'disappointing', 'useless', 'broken',
            'annoying', 'frustrating', 'pathetic', 'disgusting'
        ]
        
        # Count positive and negative words
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Determine sentiment
        if positive_count > negative_count:
            sentiment = 'positive'
            confidence = min(positive_count / 10.0, 1.0)  # Normalize to 0-1
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = min(negative_count / 10.0, 1.0)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return SentimentResult(
            sentiment=sentiment,
            confidence=confidence,
            language=language,
            processing_method='simple_rules'
        )
    
    def batch_sentiment_analysis(self, df: pd.DataFrame,
                               text_column: str = 'review_text',
                               language_column: str = 'review_language') -> pd.DataFrame:
        """
        Perform batch sentiment analysis on DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column
            language_column: Name of the language column
            
        Returns:
            DataFrame with sentiment analysis results
        """
        self.logger.info(f"Starting batch sentiment analysis on {len(df)} texts")
        
        df_result = df.copy()
        
        # Initialize result columns
        df_result['sentiment'] = ''
        df_result['sentiment_confidence'] = 0.0
        df_result['sentiment_method'] = ''
        
        # Process in batches for better performance
        batch_size = self.config.processing.batch_size
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for i in range(0, len(df), batch_size):
            batch_end = min(i + batch_size, len(df))
            batch_df = df.iloc[i:batch_end]
            
            self.logger.info(f"Processing batch {i//batch_size + 1}/{total_batches}")
            
            # Analyze each text in the batch
            for idx, row in batch_df.iterrows():
                text = row.get(text_column, '')
                language = row.get(language_column, 'en')
                
                result = self.analyze_sentiment(text, language)
                
                df_result.loc[idx, 'sentiment'] = result.sentiment
                df_result.loc[idx, 'sentiment_confidence'] = result.confidence
                df_result.loc[idx, 'sentiment_method'] = result.processing_method
        
        # Add sentiment statistics
        sentiment_stats = self._calculate_sentiment_statistics(df_result)
        
        self.logger.info(f"Batch sentiment analysis completed. Statistics: {sentiment_stats}")
        
        return df_result
    
    def compare_sentiment_by_language(self, df: pd.DataFrame,
                                    language_column: str = 'review_language') -> Dict[str, Any]:
        """
        Compare sentiment patterns across languages.
        
        Args:
            df: DataFrame with sentiment analysis results
            language_column: Name of the language column
            
        Returns:
            Dictionary containing cross-language sentiment comparison
        """
        self.logger.info("Comparing sentiment patterns across languages")
        
        if 'sentiment' not in df.columns:
            self.logger.error("Sentiment column not found. Run sentiment analysis first.")
            return {}
        
        # Sentiment distribution by language
        sentiment_by_language = {}
        
        for language in df[language_column].unique():
            if pd.isna(language):
                continue
            
            lang_data = df[df[language_column] == language]
            sentiment_counts = lang_data['sentiment'].value_counts()
            sentiment_percentages = lang_data['sentiment'].value_counts(normalize=True) * 100
            
            sentiment_by_language[language] = {
                'total_reviews': len(lang_data),
                'sentiment_counts': sentiment_counts.to_dict(),
                'sentiment_percentages': {k: round(v, 2) for k, v in sentiment_percentages.to_dict().items()},
                'avg_confidence': float(lang_data['sentiment_confidence'].mean()),
                'dominant_sentiment': sentiment_counts.index[0] if not sentiment_counts.empty else 'neutral'
            }
        
        # Cross-language sentiment statistics
        overall_stats = {
            'total_languages': len(sentiment_by_language),
            'most_positive_languages': [],
            'most_negative_languages': [],
            'highest_confidence_languages': []
        }
        
        # Find most positive/negative languages
        positive_percentages = []
        negative_percentages = []
        confidence_scores = []
        
        for lang, stats in sentiment_by_language.items():
            positive_pct = stats['sentiment_percentages'].get('positive', 0)
            negative_pct = stats['sentiment_percentages'].get('negative', 0)
            confidence = stats['avg_confidence']
            
            positive_percentages.append((lang, positive_pct))
            negative_percentages.append((lang, negative_pct))
            confidence_scores.append((lang, confidence))
        
        # Sort and get top languages
        positive_percentages.sort(key=lambda x: x[1], reverse=True)
        negative_percentages.sort(key=lambda x: x[1], reverse=True)
        confidence_scores.sort(key=lambda x: x[1], reverse=True)
        
        overall_stats['most_positive_languages'] = positive_percentages[:5]
        overall_stats['most_negative_languages'] = negative_percentages[:5]
        overall_stats['highest_confidence_languages'] = confidence_scores[:5]
        
        # Create visualization
        self._create_sentiment_comparison_plots(sentiment_by_language)
        
        comparison_result = {
            'sentiment_by_language': sentiment_by_language,
            'overall_statistics': overall_stats,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Sentiment comparison completed for {overall_stats['total_languages']} languages")
        
        return comparison_result
    
    def _calculate_sentiment_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall sentiment statistics."""
        if 'sentiment' not in df.columns:
            return {}
        
        sentiment_counts = df['sentiment'].value_counts()
        sentiment_percentages = df['sentiment'].value_counts(normalize=True) * 100
        
        stats = {
            'total_analyzed': len(df),
            'sentiment_distribution': sentiment_counts.to_dict(),
            'sentiment_percentages': {k: round(v, 2) for k, v in sentiment_percentages.to_dict().items()},
            'avg_confidence': float(df['sentiment_confidence'].mean()) if 'sentiment_confidence' in df.columns else 0.0,
            'method_distribution': df['sentiment_method'].value_counts().to_dict() if 'sentiment_method' in df.columns else {}
        }
        
        return stats
    
    def _create_sentiment_comparison_plots(self, sentiment_by_language: Dict[str, Any]):
        """Create visualizations for sentiment comparison across languages."""
        
        # Prepare data for visualization
        languages = list(sentiment_by_language.keys())[:15]  # Top 15 languages
        positive_pcts = [sentiment_by_language[lang]['sentiment_percentages'].get('positive', 0) for lang in languages]
        negative_pcts = [sentiment_by_language[lang]['sentiment_percentages'].get('negative', 0) for lang in languages]
        neutral_pcts = [sentiment_by_language[lang]['sentiment_percentages'].get('neutral', 0) for lang in languages]
        
        # Create stacked bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Positive',
            x=languages,
            y=positive_pcts,
            marker_color='lightgreen'
        ))
        
        fig.add_trace(go.Bar(
            name='Neutral',
            x=languages,
            y=neutral_pcts,
            marker_color='lightgray'
        ))
        
        fig.add_trace(go.Bar(
            name='Negative',
            x=languages,
            y=negative_pcts,
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title='Sentiment Distribution by Language',
            xaxis_title='Language',
            yaxis_title='Percentage (%)',
            barmode='stack',
            height=500
        )
        
        self.visualizations.append(fig)
        
        # Create confidence comparison
        confidences = [sentiment_by_language[lang]['avg_confidence'] for lang in languages]
        
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Bar(
            x=languages,
            y=confidences,
            name='Average Confidence',
            marker_color='skyblue'
        ))
        
        fig_conf.update_layout(
            title='Sentiment Analysis Confidence by Language',
            xaxis_title='Language',
            yaxis_title='Average Confidence Score',
            height=400
        )
        
        self.visualizations.append(fig_conf)
    
    def analyze_sentiment_trends(self, df: pd.DataFrame,
                               date_column: str = 'review_date') -> Dict[str, Any]:
        """
        Analyze sentiment trends over time.
        
        Args:
            df: DataFrame with sentiment analysis results
            date_column: Name of the date column
            
        Returns:
            Dictionary containing sentiment trend analysis
        """
        self.logger.info("Analyzing sentiment trends over time")
        
        if 'sentiment' not in df.columns or date_column not in df.columns:
            self.logger.error("Required columns not found for sentiment trend analysis")
            return {}
        
        # Prepare time series data
        df_time = df.copy()
        df_time[date_column] = pd.to_datetime(df_time[date_column])
        df_time = df_time.dropna(subset=[date_column])
        
        # Daily sentiment trends
        daily_sentiment = df_time.groupby([df_time[date_column].dt.date, 'sentiment']).size().unstack(fill_value=0)
        daily_sentiment_pct = daily_sentiment.div(daily_sentiment.sum(axis=1), axis=0) * 100
        
        # Monthly sentiment trends
        monthly_sentiment = df_time.groupby([df_time[date_column].dt.to_period('M'), 'sentiment']).size().unstack(fill_value=0)
        monthly_sentiment_pct = monthly_sentiment.div(monthly_sentiment.sum(axis=1), axis=0) * 100
        
        # Calculate trend statistics
        trend_stats = {}
        for sentiment_type in ['positive', 'negative', 'neutral']:
            if sentiment_type in daily_sentiment_pct.columns:
                values = daily_sentiment_pct[sentiment_type].values
                trend_slope = np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0
                trend_stats[sentiment_type] = {
                    'trend_slope': float(trend_slope),
                    'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing',
                    'avg_percentage': float(daily_sentiment_pct[sentiment_type].mean()),
                    'volatility': float(daily_sentiment_pct[sentiment_type].std())
                }
        
        # Create trend visualization
        self._create_sentiment_trend_plots(daily_sentiment_pct, monthly_sentiment_pct)
        
        trend_analysis = {
            'daily_trends': daily_sentiment_pct.to_dict(),
            'monthly_trends': {str(k): v for k, v in monthly_sentiment_pct.to_dict().items()},
            'trend_statistics': trend_stats,
            'analysis_period': {
                'start': df_time[date_column].min().isoformat(),
                'end': df_time[date_column].max().isoformat()
            }
        }
        
        self.logger.info("Sentiment trend analysis completed")
        
        return trend_analysis
    
    def _create_sentiment_trend_plots(self, daily_trends: pd.DataFrame, monthly_trends: pd.DataFrame):
        """Create sentiment trend visualizations."""
        
        # Daily sentiment trends
        fig_daily = go.Figure()
        
        colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
        
        for sentiment in daily_trends.columns:
            if sentiment in colors:
                fig_daily.add_trace(go.Scatter(
                    x=daily_trends.index,
                    y=daily_trends[sentiment],
                    mode='lines',
                    name=sentiment.title(),
                    line=dict(color=colors[sentiment])
                ))
        
        fig_daily.update_layout(
            title='Daily Sentiment Trends',
            xaxis_title='Date',
            yaxis_title='Percentage (%)',
            height=400
        )
        
        self.visualizations.append(fig_daily)
        
        # Monthly sentiment trends
        fig_monthly = go.Figure()
        
        for sentiment in monthly_trends.columns:
            if sentiment in colors:
                fig_monthly.add_trace(go.Scatter(
                    x=[str(idx) for idx in monthly_trends.index],
                    y=monthly_trends[sentiment],
                    mode='lines+markers',
                    name=sentiment.title(),
                    line=dict(color=colors[sentiment])
                ))
        
        fig_monthly.update_layout(
            title='Monthly Sentiment Trends',
            xaxis_title='Month',
            yaxis_title='Percentage (%)',
            height=400
        )
        
        self.visualizations.append(fig_monthly)