"""
Text classification components for multilingual topic categorization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import Counter
import re

# Machine learning libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# NLP libraries
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.interfaces import TextClassifierInterface
from src.data.models import ClassificationModel, ModelPerformance
from src.config import get_config
from src.utils.logger import get_logger


class TextClassifier(TextClassifierInterface):
    """Multilingual text classifier for topic categorization."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.visualizations = []
        
        # Predefined topic categories based on common app review themes
        self.predefined_topics = {
            'functionality': [
                'feature', 'function', 'work', 'working', 'works', 'functionality',
                'tool', 'option', 'setting', 'capability', 'ability'
            ],
            'usability': [
                'easy', 'difficult', 'hard', 'simple', 'complex', 'user-friendly',
                'intuitive', 'confusing', 'interface', 'navigation', 'menu'
            ],
            'performance': [
                'fast', 'slow', 'speed', 'performance', 'lag', 'crash', 'freeze',
                'responsive', 'smooth', 'quick', 'loading', 'battery'
            ],
            'design': [
                'design', 'look', 'appearance', 'beautiful', 'ugly', 'color',
                'layout', 'theme', 'style', 'visual', 'graphics'
            ],
            'bugs': [
                'bug', 'error', 'problem', 'issue', 'glitch', 'broken',
                'fix', 'crash', 'freeze', 'malfunction', 'fault'
            ],
            'updates': [
                'update', 'version', 'new', 'latest', 'upgrade', 'improvement',
                'change', 'added', 'removed', 'modified'
            ],
            'support': [
                'support', 'help', 'customer', 'service', 'response', 'contact',
                'assistance', 'team', 'staff', 'reply'
            ],
            'pricing': [
                'price', 'cost', 'expensive', 'cheap', 'free', 'paid',
                'subscription', 'premium', 'money', 'worth', 'value'
            ]
        }
        
        # Initialize models
        self.trained_models = {}
        self.label_encoders = {}
        
        # Multilingual topic keywords (simplified)
        self.multilingual_keywords = {
            'es': {  # Spanish
                'functionality': ['función', 'funciona', 'herramienta', 'opción'],
                'usability': ['fácil', 'difícil', 'simple', 'interfaz'],
                'performance': ['rápido', 'lento', 'velocidad', 'rendimiento'],
                'design': ['diseño', 'apariencia', 'bonito', 'color'],
                'bugs': ['error', 'problema', 'fallo', 'roto'],
                'updates': ['actualización', 'versión', 'nuevo', 'mejora'],
                'support': ['soporte', 'ayuda', 'servicio', 'equipo'],
                'pricing': ['precio', 'costo', 'caro', 'gratis']
            },
            'fr': {  # French
                'functionality': ['fonction', 'fonctionne', 'outil', 'option'],
                'usability': ['facile', 'difficile', 'simple', 'interface'],
                'performance': ['rapide', 'lent', 'vitesse', 'performance'],
                'design': ['design', 'apparence', 'beau', 'couleur'],
                'bugs': ['erreur', 'problème', 'bug', 'cassé'],
                'updates': ['mise à jour', 'version', 'nouveau', 'amélioration'],
                'support': ['support', 'aide', 'service', 'équipe'],
                'pricing': ['prix', 'coût', 'cher', 'gratuit']
            },
            'de': {  # German
                'functionality': ['funktion', 'funktioniert', 'werkzeug', 'option'],
                'usability': ['einfach', 'schwierig', 'einfach', 'benutzeroberfläche'],
                'performance': ['schnell', 'langsam', 'geschwindigkeit', 'leistung'],
                'design': ['design', 'aussehen', 'schön', 'farbe'],
                'bugs': ['fehler', 'problem', 'bug', 'kaputt'],
                'updates': ['update', 'version', 'neu', 'verbesserung'],
                'support': ['support', 'hilfe', 'service', 'team'],
                'pricing': ['preis', 'kosten', 'teuer', 'kostenlos']
            }
        }
    
    def classify_review_topics(self, text: str, language: str = 'en') -> List[str]:
        """
        Classify review into topics using rule-based approach.
        
        Args:
            text: Input text to classify
            language: Language code of the text
            
        Returns:
            List of identified topics
        """
        if not text or pd.isna(text):
            return []
        
        text_lower = str(text).lower()
        identified_topics = []
        
        # Get keywords for the language
        if language in self.multilingual_keywords:
            keywords_dict = self.multilingual_keywords[language]
        else:
            keywords_dict = self.predefined_topics  # Default to English
        
        # Check for each topic
        for topic, keywords in keywords_dict.items():
            topic_score = 0
            for keyword in keywords:
                # Count occurrences of keyword
                topic_score += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
            
            # If topic score is above threshold, include it
            if topic_score > 0:
                identified_topics.append(topic)
        
        # If no topics found, assign 'general'
        if not identified_topics:
            identified_topics = ['general']
        
        return identified_topics
    
    def train_multilingual_classifier(self, df: pd.DataFrame,
                                    text_column: str = 'review_text',
                                    language_column: str = 'review_language') -> ClassificationModel:
        """
        Train multilingual classification model.
        
        Args:
            df: Training DataFrame
            text_column: Name of the text column
            language_column: Name of the language column
            
        Returns:
            Trained ClassificationModel
        """
        self.logger.info(f"Training multilingual classifier on {len(df)} samples")
        
        # Generate topic labels using rule-based classification
        df_train = df.copy()
        df_train['topics'] = df_train.apply(
            lambda row: self.classify_review_topics(
                row.get(text_column, ''), 
                row.get(language_column, 'en')
            ), axis=1
        )
        
        # Convert multi-label to single label (take first topic)
        df_train['primary_topic'] = df_train['topics'].apply(
            lambda topics: topics[0] if topics else 'general'
        )
        
        # Prepare training data
        texts = df_train[text_column].fillna('').tolist()
        labels = df_train['primary_topic'].tolist()
        
        # Filter out empty texts
        valid_indices = [i for i, text in enumerate(texts) if len(str(text).strip()) > 0]
        texts = [texts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        if len(texts) == 0:
            self.logger.error("No valid texts found for training")
            return None
        
        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, encoded_labels, 
            test_size=self.config.ml.test_size,
            random_state=self.config.ml.random_state,
            stratify=encoded_labels
        )
        
        # Train multiple models and select the best one
        models_to_try = {
            'naive_bayes': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
                ('classifier', MultinomialNB())
            ]),
            'svm': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
                ('classifier', SVC(kernel='linear', probability=True))
            ]),
            'random_forest': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
        }
        
        best_model = None
        best_score = 0
        best_model_name = ''
        
        for model_name, model in models_to_try.items():
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                avg_score = cv_scores.mean()
                
                self.logger.info(f"{model_name} CV accuracy: {avg_score:.3f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    best_model_name = model_name
                    
            except Exception as e:
                self.logger.warning(f"Error training {model_name}: {str(e)}")
        
        if best_model is None:
            self.logger.error("No model could be trained successfully")
            return None
        
        # Train the best model on full training data
        best_model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)
        
        # Calculate performance metrics
        accuracy = (y_pred == y_test).mean()
        
        # Classification report
        class_report = classification_report(
            y_test, y_pred, 
            target_names=label_encoder.classes_,
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Feature importance (for tree-based models)
        feature_importance = None
        if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
            feature_names = best_model.named_steps['tfidf'].get_feature_names_out()
            importances = best_model.named_steps['classifier'].feature_importances_
            feature_importance = dict(zip(feature_names, importances))
            # Get top 20 features
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True)[:20])
        
        # Create performance object
        performance = ModelPerformance(
            model_type=best_model_name,
            accuracy_metrics={
                'accuracy': float(accuracy),
                'macro_avg_f1': float(class_report['macro avg']['f1-score']),
                'weighted_avg_f1': float(class_report['weighted avg']['f1-score'])
            },
            confusion_matrix=conf_matrix,
            feature_importance=feature_importance,
            cross_validation_scores=cv_scores.tolist(),
            training_time=None,
            prediction_time=None
        )
        
        # Store model and encoder
        self.trained_models[best_model_name] = best_model
        self.label_encoders[best_model_name] = label_encoder
        
        # Create classification model object
        classification_model = ClassificationModel(
            model=best_model,
            model_type=best_model_name,
            training_languages=df[language_column].unique().tolist(),
            feature_names=best_model.named_steps['tfidf'].get_feature_names_out().tolist(),
            performance=performance,
            created_at=datetime.now()
        )
        
        # Create performance visualization
        self._create_classification_performance_plots(
            conf_matrix, label_encoder.classes_, class_report
        )
        
        self.logger.info(f"Model training completed. Best model: {best_model_name} (accuracy: {accuracy:.3f})")
        
        return classification_model
    
    def predict_topics_batch(self, texts: List[str], 
                           languages: List[str],
                           model_name: str = 'naive_bayes') -> List[List[str]]:
        """
        Predict topics for a batch of texts.
        
        Args:
            texts: List of texts to classify
            languages: List of corresponding language codes
            model_name: Name of the trained model to use
            
        Returns:
            List of topic predictions for each text
        """
        if model_name not in self.trained_models:
            self.logger.warning(f"Model {model_name} not found. Using rule-based classification.")
            return [self.classify_review_topics(text, lang) 
                   for text, lang in zip(texts, languages)]
        
        model = self.trained_models[model_name]
        label_encoder = self.label_encoders[model_name]
        
        try:
            # Predict using trained model
            predictions = model.predict(texts)
            predicted_labels = label_encoder.inverse_transform(predictions)
            
            # Convert single labels to lists for consistency
            return [[label] for label in predicted_labels]
            
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {str(e)}")
            # Fallback to rule-based classification
            return [self.classify_review_topics(text, lang) 
                   for text, lang in zip(texts, languages)]
    
    def analyze_topic_distribution(self, df: pd.DataFrame,
                                 topics_column: str = 'topics') -> Dict[str, Any]:
        """
        Analyze distribution of topics in the dataset.
        
        Args:
            df: DataFrame with topic classifications
            topics_column: Name of the topics column
            
        Returns:
            Dictionary containing topic distribution analysis
        """
        self.logger.info("Analyzing topic distribution")
        
        if topics_column not in df.columns:
            self.logger.error(f"Topics column '{topics_column}' not found")
            return {}
        
        # Flatten topic lists and count occurrences
        all_topics = []
        for topics in df[topics_column]:
            if isinstance(topics, list):
                all_topics.extend(topics)
            elif isinstance(topics, str):
                all_topics.append(topics)
        
        topic_counts = Counter(all_topics)
        total_topics = len(all_topics)
        
        # Calculate percentages
        topic_percentages = {
            topic: (count / total_topics) * 100 
            for topic, count in topic_counts.items()
        }
        
        # Topic co-occurrence analysis
        topic_cooccurrence = {}
        for topics in df[topics_column]:
            if isinstance(topics, list) and len(topics) > 1:
                for i, topic1 in enumerate(topics):
                    for topic2 in topics[i+1:]:
                        pair = tuple(sorted([topic1, topic2]))
                        topic_cooccurrence[pair] = topic_cooccurrence.get(pair, 0) + 1
        
        # Sort co-occurrences
        sorted_cooccurrence = sorted(
            topic_cooccurrence.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]  # Top 10 co-occurrences
        
        # Topic trends by other dimensions
        topic_analysis = {}
        
        # Topics by language
        if 'review_language' in df.columns:
            topic_by_language = {}
            for language in df['review_language'].unique():
                if pd.isna(language):
                    continue
                
                lang_data = df[df['review_language'] == language]
                lang_topics = []
                for topics in lang_data[topics_column]:
                    if isinstance(topics, list):
                        lang_topics.extend(topics)
                    elif isinstance(topics, str):
                        lang_topics.append(topics)
                
                lang_topic_counts = Counter(lang_topics)
                topic_by_language[language] = dict(lang_topic_counts.most_common(5))
            
            topic_analysis['by_language'] = topic_by_language
        
        # Topics by rating
        if 'rating' in df.columns:
            topic_by_rating = {}
            rating_ranges = [(1, 2), (2, 3), (3, 4), (4, 5)]
            
            for min_rating, max_rating in rating_ranges:
                range_name = f"{min_rating}-{max_rating}"
                range_data = df[
                    (df['rating'] >= min_rating) & (df['rating'] < max_rating)
                ]
                
                range_topics = []
                for topics in range_data[topics_column]:
                    if isinstance(topics, list):
                        range_topics.extend(topics)
                    elif isinstance(topics, str):
                        range_topics.append(topics)
                
                range_topic_counts = Counter(range_topics)
                topic_by_rating[range_name] = dict(range_topic_counts.most_common(5))
            
            topic_analysis['by_rating'] = topic_by_rating
        
        # Create topic distribution visualization
        self._create_topic_distribution_plots(topic_counts, topic_percentages)
        
        distribution_result = {
            'topic_counts': dict(topic_counts.most_common()),
            'topic_percentages': {k: round(v, 2) for k, v in topic_percentages.items()},
            'total_topic_instances': total_topics,
            'unique_topics': len(topic_counts),
            'most_common_topic': topic_counts.most_common(1)[0] if topic_counts else None,
            'topic_cooccurrence': [
                {'topics': list(pair), 'count': count} 
                for pair, count in sorted_cooccurrence
            ],
            'topic_analysis': topic_analysis,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Topic distribution analysis completed. Found {len(topic_counts)} unique topics")
        
        return distribution_result
    
    def _create_classification_performance_plots(self, confusion_matrix: np.ndarray,
                                               class_names: List[str],
                                               classification_report: Dict):
        """Create performance visualization plots."""
        
        # Confusion matrix heatmap
        fig_cm = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=confusion_matrix,
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig_cm.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=500
        )
        
        self.visualizations.append(fig_cm)
        
        # Performance metrics by class
        classes = [cls for cls in class_names if cls in classification_report]
        precision_scores = [classification_report[cls]['precision'] for cls in classes]
        recall_scores = [classification_report[cls]['recall'] for cls in classes]
        f1_scores = [classification_report[cls]['f1-score'] for cls in classes]
        
        fig_metrics = go.Figure()
        
        fig_metrics.add_trace(go.Bar(
            name='Precision',
            x=classes,
            y=precision_scores,
            marker_color='lightblue'
        ))
        
        fig_metrics.add_trace(go.Bar(
            name='Recall',
            x=classes,
            y=recall_scores,
            marker_color='lightgreen'
        ))
        
        fig_metrics.add_trace(go.Bar(
            name='F1-Score',
            x=classes,
            y=f1_scores,
            marker_color='lightcoral'
        ))
        
        fig_metrics.update_layout(
            title='Classification Performance by Topic',
            xaxis_title='Topic',
            yaxis_title='Score',
            barmode='group',
            height=500
        )
        
        self.visualizations.append(fig_metrics)
    
    def _create_topic_distribution_plots(self, topic_counts: Counter, 
                                       topic_percentages: Dict[str, float]):
        """Create topic distribution visualization plots."""
        
        # Topic frequency bar chart
        topics = list(topic_counts.keys())[:15]  # Top 15 topics
        counts = [topic_counts[topic] for topic in topics]
        percentages = [topic_percentages[topic] for topic in topics]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Topic Frequency (Counts)', 'Topic Distribution (%)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Bar(x=topics, y=counts, name='Count', marker_color='skyblue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=topics, y=percentages, name='Percentage', marker_color='lightcoral'),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Topic Distribution Analysis',
            showlegend=False,
            height=500
        )
        
        fig.update_xaxes(title_text="Topic", row=1, col=1)
        fig.update_xaxes(title_text="Topic", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Percentage (%)", row=1, col=2)
        
        self.visualizations.append(fig)