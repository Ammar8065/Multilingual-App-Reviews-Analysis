"""
Model evaluation components for comprehensive model assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score, cross_validate
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.data.interfaces import ModelEvaluatorInterface
from src.data.models import ModelPerformance
from src.config import get_config
from src.utils.logger import get_logger


class ModelEvaluator(ModelEvaluatorInterface):
    """Comprehensive model evaluator for regression and classification tasks."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.visualizations = []
    
    def evaluate_regression(self, y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """
        Evaluate regression model performance.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary containing regression metrics
        """
        self.logger.info("Evaluating regression model performance")
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Handle edge cases
        if len(y_true) == 0 or len(y_pred) == 0:
            self.logger.warning("Empty prediction arrays provided")
            return {}
        
        if len(y_true) != len(y_pred):
            self.logger.error(f"Mismatched array lengths: {len(y_true)} vs {len(y_pred)}")
            return {}
        
        # Calculate regression metrics
        metrics = {}
        
        try:
            # Mean Squared Error
            metrics['mse'] = float(mean_squared_error(y_true, y_pred))
            
            # Root Mean Squared Error
            metrics['rmse'] = float(np.sqrt(metrics['mse']))
            
            # Mean Absolute Error
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
            
            # R-squared Score
            metrics['r2_score'] = float(r2_score(y_true, y_pred))
            
            # Mean Absolute Percentage Error
            # Handle division by zero
            non_zero_mask = y_true != 0
            if np.any(non_zero_mask):
                metrics['mape'] = float(mean_absolute_percentage_error(
                    y_true[non_zero_mask], y_pred[non_zero_mask]
                ))
            else:
                metrics['mape'] = float('inf')
            
            # Additional custom metrics
            residuals = y_true - y_pred
            
            # Mean Residual (should be close to 0 for unbiased models)
            metrics['mean_residual'] = float(np.mean(residuals))
            
            # Standard deviation of residuals
            metrics['residual_std'] = float(np.std(residuals))
            
            # Maximum absolute error
            metrics['max_error'] = float(np.max(np.abs(residuals)))
            
            # Explained variance score
            metrics['explained_variance'] = float(1 - np.var(residuals) / np.var(y_true))
            
            # Median Absolute Error
            metrics['median_ae'] = float(np.median(np.abs(residuals)))
            
            # Mean Squared Log Error (for positive values only)
            if np.all(y_true > 0) and np.all(y_pred > 0):
                metrics['msle'] = float(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
            else:
                metrics['msle'] = None
            
            # Prediction accuracy within tolerance bands
            tolerance_bands = [0.1, 0.25, 0.5, 1.0]  # For rating scale 1-5
            for tolerance in tolerance_bands:
                within_tolerance = np.abs(residuals) <= tolerance
                metrics[f'accuracy_within_{tolerance}'] = float(np.mean(within_tolerance))
            
            self.logger.info(f"Regression evaluation completed. R² = {metrics['r2_score']:.4f}, RMSE = {metrics['rmse']:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error calculating regression metrics: {str(e)}")
            return {}
        
        return metrics
    
    def evaluate_classification(self, y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        """
        Evaluate classification model performance.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            
        Returns:
            Dictionary containing classification metrics
        """
        self.logger.info("Evaluating classification model performance")
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Handle edge cases
        if len(y_true) == 0 or len(y_pred) == 0:
            self.logger.warning("Empty prediction arrays provided")
            return {}
        
        if len(y_true) != len(y_pred):
            self.logger.error(f"Mismatched array lengths: {len(y_true)} vs {len(y_pred)}")
            return {}
        
        metrics = {}
        
        try:
            # Basic classification metrics
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            
            # Get unique classes
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            
            # Multi-class metrics with different averaging strategies
            if len(unique_classes) > 2:
                # Macro averages (unweighted mean of per-class metrics)
                metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
                metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
                metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
                
                # Weighted averages (weighted by support)
                metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
                metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
                metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
                
                # Micro averages (global average)
                metrics['precision_micro'] = float(precision_score(y_true, y_pred, average='micro', zero_division=0))
                metrics['recall_micro'] = float(recall_score(y_true, y_pred, average='micro', zero_division=0))
                metrics['f1_micro'] = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
            else:
                # Binary classification
                metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
                metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
                metrics['f1_score'] = float(f1_score(y_true, y_pred, zero_division=0))
            
            # Per-class metrics
            class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            # Extract per-class metrics
            per_class_metrics = {}
            for class_name, class_metrics in class_report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']:
                    per_class_metrics[f'class_{class_name}'] = {
                        'precision': float(class_metrics['precision']),
                        'recall': float(class_metrics['recall']),
                        'f1_score': float(class_metrics['f1-score']),
                        'support': int(class_metrics['support'])
                    }
            
            metrics['per_class_metrics'] = per_class_metrics
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
            metrics['confusion_matrix'] = cm.tolist()
            metrics['confusion_matrix_labels'] = unique_classes.tolist()
            
            # Additional metrics
            metrics['num_classes'] = len(unique_classes)
            metrics['total_samples'] = len(y_true)
            
            # Class distribution in predictions vs true labels
            true_dist = pd.Series(y_true).value_counts(normalize=True).to_dict()
            pred_dist = pd.Series(y_pred).value_counts(normalize=True).to_dict()
            
            metrics['true_class_distribution'] = {str(k): float(v) for k, v in true_dist.items()}
            metrics['pred_class_distribution'] = {str(k): float(v) for k, v in pred_dist.items()}
            
            # Balanced accuracy (for imbalanced datasets)
            from sklearn.metrics import balanced_accuracy_score
            metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
            
            self.logger.info(f"Classification evaluation completed. Accuracy = {metrics['accuracy']:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error calculating classification metrics: {str(e)}")
            return {}
        
        return metrics
    
    def cross_validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform cross-validation on model.
        
        Args:
            model: Trained model object
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary containing cross-validation results
        """
        self.logger.info(f"Performing {self.config.ml.cv_folds}-fold cross-validation")
        
        if len(X) == 0 or len(y) == 0:
            self.logger.warning("Empty data provided for cross-validation")
            return {}
        
        if len(X) != len(y):
            self.logger.error(f"Mismatched data lengths: {len(X)} vs {len(y)}")
            return {}
        
        cv_results = {}
        
        try:
            # Determine if it's a regression or classification task
            is_regression = self._is_regression_task(y)
            
            if is_regression:
                # Regression cross-validation
                scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
                
                cv_scores = cross_validate(
                    model, X, y,
                    cv=self.config.ml.cv_folds,
                    scoring=scoring_metrics,
                    n_jobs=self.config.ml.n_jobs,
                    return_train_score=True
                )
                
                # Process regression CV results
                cv_results['regression_metrics'] = {
                    'mse': {
                        'test_scores': (-cv_scores['test_neg_mean_squared_error']).tolist(),
                        'train_scores': (-cv_scores['train_neg_mean_squared_error']).tolist(),
                        'test_mean': float(np.mean(-cv_scores['test_neg_mean_squared_error'])),
                        'test_std': float(np.std(-cv_scores['test_neg_mean_squared_error'])),
                        'train_mean': float(np.mean(-cv_scores['train_neg_mean_squared_error'])),
                        'train_std': float(np.std(-cv_scores['train_neg_mean_squared_error']))
                    },
                    'mae': {
                        'test_scores': (-cv_scores['test_neg_mean_absolute_error']).tolist(),
                        'train_scores': (-cv_scores['train_neg_mean_absolute_error']).tolist(),
                        'test_mean': float(np.mean(-cv_scores['test_neg_mean_absolute_error'])),
                        'test_std': float(np.std(-cv_scores['test_neg_mean_absolute_error'])),
                        'train_mean': float(np.mean(-cv_scores['train_neg_mean_absolute_error'])),
                        'train_std': float(np.std(-cv_scores['train_neg_mean_absolute_error']))
                    },
                    'r2': {
                        'test_scores': cv_scores['test_r2'].tolist(),
                        'train_scores': cv_scores['train_r2'].tolist(),
                        'test_mean': float(np.mean(cv_scores['test_r2'])),
                        'test_std': float(np.std(cv_scores['test_r2'])),
                        'train_mean': float(np.mean(cv_scores['train_r2'])),
                        'train_std': float(np.std(cv_scores['train_r2']))
                    }
                }
                
                # Calculate RMSE from MSE
                rmse_test = np.sqrt(-cv_scores['test_neg_mean_squared_error'])
                rmse_train = np.sqrt(-cv_scores['train_neg_mean_squared_error'])
                
                cv_results['regression_metrics']['rmse'] = {
                    'test_scores': rmse_test.tolist(),
                    'train_scores': rmse_train.tolist(),
                    'test_mean': float(np.mean(rmse_test)),
                    'test_std': float(np.std(rmse_test)),
                    'train_mean': float(np.mean(rmse_train)),
                    'train_std': float(np.std(rmse_train))
                }
                
            else:
                # Classification cross-validation
                scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
                
                cv_scores = cross_validate(
                    model, X, y,
                    cv=self.config.ml.cv_folds,
                    scoring=scoring_metrics,
                    n_jobs=self.config.ml.n_jobs,
                    return_train_score=True
                )
                
                # Process classification CV results
                cv_results['classification_metrics'] = {}
                
                for metric in scoring_metrics:
                    test_key = f'test_{metric}'
                    train_key = f'train_{metric}'
                    
                    cv_results['classification_metrics'][metric] = {
                        'test_scores': cv_scores[test_key].tolist(),
                        'train_scores': cv_scores[train_key].tolist(),
                        'test_mean': float(np.mean(cv_scores[test_key])),
                        'test_std': float(np.std(cv_scores[test_key])),
                        'train_mean': float(np.mean(cv_scores[train_key])),
                        'train_std': float(np.std(cv_scores[train_key]))
                    }
            
            # General CV information
            cv_results['cv_info'] = {
                'n_folds': self.config.ml.cv_folds,
                'n_samples': len(X),
                'n_features': len(X.columns) if hasattr(X, 'columns') else X.shape[1],
                'task_type': 'regression' if is_regression else 'classification'
            }
            
            # Overfitting analysis
            cv_results['overfitting_analysis'] = self._analyze_overfitting(cv_results)
            
            self.logger.info("Cross-validation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cross-validation: {str(e)}")
            return {}
        
        return cv_results
    
    def _is_regression_task(self, y: pd.Series) -> bool:
        """Determine if the task is regression or classification."""
        # Check if target is continuous (float) or has many unique values
        if pd.api.types.is_numeric_dtype(y):
            unique_values = y.nunique()
            total_values = len(y)
            
            # If more than 10 unique values or more than 50% unique values, likely regression
            if unique_values > 10 or (unique_values / total_values) > 0.5:
                return True
        
        return False
    
    def _analyze_overfitting(self, cv_results: Dict) -> Dict[str, Any]:
        """Analyze overfitting based on train/test score differences."""
        overfitting_analysis = {}
        
        if 'regression_metrics' in cv_results:
            metrics = cv_results['regression_metrics']
            
            # Calculate train-test gaps
            for metric_name, metric_data in metrics.items():
                if metric_name in ['r2']:  # Higher is better
                    gap = metric_data['train_mean'] - metric_data['test_mean']
                else:  # Lower is better (MSE, MAE, RMSE)
                    gap = metric_data['test_mean'] - metric_data['train_mean']
                
                overfitting_analysis[f'{metric_name}_gap'] = float(gap)
            
            # Overall overfitting assessment
            r2_gap = overfitting_analysis.get('r2_gap', 0)
            if r2_gap > 0.1:
                overfitting_analysis['overfitting_severity'] = 'high'
            elif r2_gap > 0.05:
                overfitting_analysis['overfitting_severity'] = 'moderate'
            else:
                overfitting_analysis['overfitting_severity'] = 'low'
                
        elif 'classification_metrics' in cv_results:
            metrics = cv_results['classification_metrics']
            
            # Calculate train-test gaps for classification
            for metric_name, metric_data in metrics.items():
                gap = metric_data['train_mean'] - metric_data['test_mean']
                overfitting_analysis[f'{metric_name}_gap'] = float(gap)
            
            # Overall overfitting assessment
            accuracy_gap = overfitting_analysis.get('accuracy_gap', 0)
            if accuracy_gap > 0.1:
                overfitting_analysis['overfitting_severity'] = 'high'
            elif accuracy_gap > 0.05:
                overfitting_analysis['overfitting_severity'] = 'moderate'
            else:
                overfitting_analysis['overfitting_severity'] = 'low'
        
        return overfitting_analysis
    
    def create_model_performance_report(self, model_name: str, 
                                      performance_metrics: Dict[str, Any],
                                      cv_results: Optional[Dict[str, Any]] = None,
                                      feature_importance: Optional[Dict[str, float]] = None) -> ModelPerformance:
        """
        Create comprehensive model performance report.
        
        Args:
            model_name: Name of the model
            performance_metrics: Performance metrics dictionary
            cv_results: Cross-validation results (optional)
            feature_importance: Feature importance scores (optional)
            
        Returns:
            ModelPerformance object containing all performance information
        """
        self.logger.info(f"Creating performance report for model: {model_name}")
        
        # Determine model type
        model_type = 'regression' if 'mse' in performance_metrics else 'classification'
        
        # Extract cross-validation scores if available
        cv_scores = []
        if cv_results:
            if model_type == 'regression' and 'regression_metrics' in cv_results:
                cv_scores = cv_results['regression_metrics']['r2']['test_scores']
            elif model_type == 'classification' and 'classification_metrics' in cv_results:
                cv_scores = cv_results['classification_metrics']['accuracy']['test_scores']
        
        # Extract confusion matrix if available
        confusion_matrix = None
        if 'confusion_matrix' in performance_metrics:
            confusion_matrix = np.array(performance_metrics['confusion_matrix'])
        
        # Create performance object
        performance = ModelPerformance(
            model_type=model_type,
            accuracy_metrics=performance_metrics,
            confusion_matrix=confusion_matrix,
            feature_importance=feature_importance,
            cross_validation_scores=cv_scores,
            training_time=None,  # To be filled by calling code
            prediction_time=None  # To be filled by calling code
        )
        
        # Create performance visualizations
        self._create_performance_visualizations(model_name, performance_metrics, cv_results)
        
        self.logger.info(f"Performance report created for {model_name}")
        
        return performance
    
    def _create_performance_visualizations(self, model_name: str, 
                                         metrics: Dict[str, Any],
                                         cv_results: Optional[Dict[str, Any]] = None):
        """Create visualizations for model performance."""
        
        # 1. Metrics summary plot
        if 'mse' in metrics:  # Regression
            metric_names = ['RMSE', 'MAE', 'R²', 'MAPE']
            metric_values = [
                metrics.get('rmse', 0),
                metrics.get('mae', 0),
                metrics.get('r2_score', 0),
                metrics.get('mape', 0) if metrics.get('mape') != float('inf') else 0
            ]
            
            fig = go.Figure(data=go.Bar(
                x=metric_names,
                y=metric_values,
                marker_color=['red', 'orange', 'green', 'blue']
            ))
            
            fig.update_layout(
                title=f'{model_name} - Regression Performance Metrics',
                xaxis_title='Metrics',
                yaxis_title='Values',
                height=400
            )
            
        else:  # Classification
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            metric_values = [
                metrics.get('accuracy', 0),
                metrics.get('precision', metrics.get('precision_macro', 0)),
                metrics.get('recall', metrics.get('recall_macro', 0)),
                metrics.get('f1_score', metrics.get('f1_macro', 0))
            ]
            
            fig = go.Figure(data=go.Bar(
                x=metric_names,
                y=metric_values,
                marker_color=['blue', 'green', 'orange', 'red']
            ))
            
            fig.update_layout(
                title=f'{model_name} - Classification Performance Metrics',
                xaxis_title='Metrics',
                yaxis_title='Values',
                height=400
            )
        
        self.visualizations.append(fig)
        
        # 2. Cross-validation results plot
        if cv_results:
            self._create_cv_visualization(model_name, cv_results)
        
        # 3. Confusion matrix plot (for classification)
        if 'confusion_matrix' in metrics:
            self._create_confusion_matrix_plot(model_name, metrics)
    
    def _create_cv_visualization(self, model_name: str, cv_results: Dict[str, Any]):
        """Create cross-validation results visualization."""
        
        if 'regression_metrics' in cv_results:
            metrics = cv_results['regression_metrics']
            
            # Create box plots for each metric
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('R² Scores', 'RMSE Scores', 'MAE Scores', 'Train vs Test R²')
            )
            
            # R² box plot
            fig.add_trace(
                go.Box(y=metrics['r2']['test_scores'], name='R² Test', marker_color='blue'),
                row=1, col=1
            )
            
            # RMSE box plot
            fig.add_trace(
                go.Box(y=metrics['rmse']['test_scores'], name='RMSE Test', marker_color='red'),
                row=1, col=2
            )
            
            # MAE box plot
            fig.add_trace(
                go.Box(y=metrics['mae']['test_scores'], name='MAE Test', marker_color='green'),
                row=2, col=1
            )
            
            # Train vs Test comparison
            fig.add_trace(
                go.Bar(x=['Train', 'Test'], 
                      y=[metrics['r2']['train_mean'], metrics['r2']['test_mean']],
                      name='R² Comparison', marker_color=['lightblue', 'darkblue']),
                row=2, col=2
            )
            
        elif 'classification_metrics' in cv_results:
            metrics = cv_results['classification_metrics']
            
            # Create box plots for classification metrics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Accuracy Scores', 'Precision Scores', 'Recall Scores', 'F1 Scores')
            )
            
            fig.add_trace(
                go.Box(y=metrics['accuracy']['test_scores'], name='Accuracy', marker_color='blue'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Box(y=metrics['precision_macro']['test_scores'], name='Precision', marker_color='green'),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Box(y=metrics['recall_macro']['test_scores'], name='Recall', marker_color='orange'),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Box(y=metrics['f1_macro']['test_scores'], name='F1', marker_color='red'),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f'{model_name} - Cross-Validation Results',
            showlegend=False,
            height=600
        )
        
        self.visualizations.append(fig)
    
    def _create_confusion_matrix_plot(self, model_name: str, metrics: Dict[str, Any]):
        """Create confusion matrix heatmap."""
        
        cm = np.array(metrics['confusion_matrix'])
        labels = metrics.get('confusion_matrix_labels', [])
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f'{model_name} - Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=500,
            width=500
        )
        
        self.visualizations.append(fig)
    
    def compare_models(self, model_performances: Dict[str, ModelPerformance]) -> Dict[str, Any]:
        """
        Compare multiple model performances.
        
        Args:
            model_performances: Dictionary mapping model names to ModelPerformance objects
            
        Returns:
            Dictionary containing model comparison results
        """
        self.logger.info(f"Comparing {len(model_performances)} models")
        
        if not model_performances:
            return {}
        
        comparison_results = {
            'model_rankings': {},
            'best_models': {},
            'performance_summary': {},
            'recommendations': []
        }
        
        # Determine task type
        first_model = list(model_performances.values())[0]
        is_regression = first_model.model_type == 'regression'
        
        if is_regression:
            # Regression model comparison
            comparison_results = self._compare_regression_models(model_performances)
        else:
            # Classification model comparison
            comparison_results = self._compare_classification_models(model_performances)
        
        # Create comparison visualization
        self._create_model_comparison_plot(model_performances, is_regression)
        
        return comparison_results
    
    def _compare_regression_models(self, models: Dict[str, ModelPerformance]) -> Dict[str, Any]:
        """Compare regression models."""
        
        model_scores = {}
        
        for name, performance in models.items():
            metrics = performance.accuracy_metrics
            model_scores[name] = {
                'r2_score': metrics.get('r2_score', 0),
                'rmse': metrics.get('rmse', float('inf')),
                'mae': metrics.get('mae', float('inf')),
                'mape': metrics.get('mape', float('inf'))
            }
        
        # Rank models by R² (higher is better)
        r2_ranking = sorted(model_scores.items(), key=lambda x: x[1]['r2_score'], reverse=True)
        
        # Rank models by RMSE (lower is better)
        rmse_ranking = sorted(model_scores.items(), key=lambda x: x[1]['rmse'])
        
        return {
            'model_rankings': {
                'by_r2': [{'model': name, 'score': scores['r2_score']} for name, scores in r2_ranking],
                'by_rmse': [{'model': name, 'score': scores['rmse']} for name, scores in rmse_ranking]
            },
            'best_models': {
                'highest_r2': r2_ranking[0][0],
                'lowest_rmse': rmse_ranking[0][0]
            },
            'performance_summary': model_scores
        }
    
    def _compare_classification_models(self, models: Dict[str, ModelPerformance]) -> Dict[str, Any]:
        """Compare classification models."""
        
        model_scores = {}
        
        for name, performance in models.items():
            metrics = performance.accuracy_metrics
            model_scores[name] = {
                'accuracy': metrics.get('accuracy', 0),
                'f1_score': metrics.get('f1_score', metrics.get('f1_macro', 0)),
                'precision': metrics.get('precision', metrics.get('precision_macro', 0)),
                'recall': metrics.get('recall', metrics.get('recall_macro', 0))
            }
        
        # Rank models by accuracy
        accuracy_ranking = sorted(model_scores.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        # Rank models by F1 score
        f1_ranking = sorted(model_scores.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        return {
            'model_rankings': {
                'by_accuracy': [{'model': name, 'score': scores['accuracy']} for name, scores in accuracy_ranking],
                'by_f1': [{'model': name, 'score': scores['f1_score']} for name, scores in f1_ranking]
            },
            'best_models': {
                'highest_accuracy': accuracy_ranking[0][0],
                'highest_f1': f1_ranking[0][0]
            },
            'performance_summary': model_scores
        }
    
    def _create_model_comparison_plot(self, models: Dict[str, ModelPerformance], is_regression: bool):
        """Create model comparison visualization."""
        
        model_names = list(models.keys())
        
        if is_regression:
            r2_scores = [models[name].accuracy_metrics.get('r2_score', 0) for name in model_names]
            rmse_scores = [models[name].accuracy_metrics.get('rmse', 0) for name in model_names]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('R² Score Comparison', 'RMSE Comparison')
            )
            
            fig.add_trace(
                go.Bar(x=model_names, y=r2_scores, name='R²', marker_color='blue'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=model_names, y=rmse_scores, name='RMSE', marker_color='red'),
                row=1, col=2
            )
            
        else:
            accuracy_scores = [models[name].accuracy_metrics.get('accuracy', 0) for name in model_names]
            f1_scores = [models[name].accuracy_metrics.get('f1_score', 
                        models[name].accuracy_metrics.get('f1_macro', 0)) for name in model_names]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Accuracy Comparison', 'F1 Score Comparison')
            )
            
            fig.add_trace(
                go.Bar(x=model_names, y=accuracy_scores, name='Accuracy', marker_color='green'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=model_names, y=f1_scores, name='F1 Score', marker_color='orange'),
                row=1, col=2
            )
        
        fig.update_layout(
            title='Model Performance Comparison',
            showlegend=False,
            height=500
        )
        
        self.visualizations.append(fig)