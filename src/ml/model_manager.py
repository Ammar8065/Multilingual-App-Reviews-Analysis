"""
Model management components for versioning, storage, and deployment.
"""

import os
import pickle
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
import joblib
from src.data.models import ModelPerformance, ClassificationModel, RegressionModel
from src.config import get_config
from src.utils.logger import get_logger


class ModelManager:
    """Model manager for versioning, storage, and deployment operations."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        
        # Model storage paths
        self.models_dir = self.config.models_dir
        self.metadata_dir = self.models_dir / "metadata"
        self.versions_dir = self.models_dir / "versions"
        
        # Create directories if they don't exist
        self._create_directories()
        
        # Model registry file
        self.registry_file = self.metadata_dir / "model_registry.json"
        self.registry = self._load_registry()
    
    def _create_directories(self):
        """Create necessary directories for model storage."""
        directories = [self.models_dir, self.metadata_dir, self.versions_dir]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Model storage directories initialized")
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading model registry: {str(e)}")
                return {}
        else:
            return {}
    
    def _save_registry(self):
        """Save model registry to file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2, default=str)
            self.logger.debug("Model registry saved")
        except Exception as e:
            self.logger.error(f"Error saving model registry: {str(e)}")
    
    def save_model(self, model: Any, model_name: str, model_type: str,
                   performance: ModelPerformance, metadata: Optional[Dict[str, Any]] = None,
                   version: Optional[str] = None) -> str:
        """
        Save model with versioning and metadata.
        
        Args:
            model: Trained model object
            model_name: Name of the model
            model_type: Type of model ('regression' or 'classification')
            performance: Model performance metrics
            metadata: Additional metadata
            version: Version string (auto-generated if not provided)
            
        Returns:
            Version string of the saved model
        """
        self.logger.info(f"Saving model: {model_name}")
        
        # Generate version if not provided
        if version is None:
            version = self._generate_version(model_name)
        
        # Create model directory
        model_dir = self.versions_dir / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model file
        model_file = model_dir / "model.pkl"
        try:
            joblib.dump(model, model_file)
            self.logger.info(f"Model saved to: {model_file}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
        
        # Save performance metrics
        performance_file = model_dir / "performance.json"
        try:
            performance_dict = {
                'model_type': performance.model_type,
                'accuracy_metrics': performance.accuracy_metrics,
                'confusion_matrix': performance.confusion_matrix.tolist() if performance.confusion_matrix is not None else None,
                'feature_importance': performance.feature_importance,
                'cross_validation_scores': performance.cross_validation_scores,
                'training_time': performance.training_time,
                'prediction_time': performance.prediction_time
            }
            
            with open(performance_file, 'w') as f:
                json.dump(performance_dict, f, indent=2, default=str)
            
            self.logger.info(f"Performance metrics saved to: {performance_file}")
        except Exception as e:
            self.logger.error(f"Error saving performance metrics: {str(e)}")
            raise
        
        # Save metadata
        model_metadata = {
            'model_name': model_name,
            'model_type': model_type,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'model_file': str(model_file),
            'performance_file': str(performance_file),
            'model_size_bytes': model_file.stat().st_size,
            'model_hash': self._calculate_file_hash(model_file),
            'custom_metadata': metadata or {}
        }
        
        metadata_file = model_dir / "metadata.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(model_metadata, f, indent=2, default=str)
            
            self.logger.info(f"Metadata saved to: {metadata_file}")
        except Exception as e:
            self.logger.error(f"Error saving metadata: {str(e)}")
            raise
        
        # Update registry
        if model_name not in self.registry:
            self.registry[model_name] = {
                'versions': [],
                'latest_version': None,
                'model_type': model_type,
                'created_at': datetime.now().isoformat()
            }
        
        # Add version to registry
        version_info = {
            'version': version,
            'created_at': model_metadata['created_at'],
            'performance_summary': self._extract_performance_summary(performance),
            'model_path': str(model_dir),
            'is_active': True
        }
        
        self.registry[model_name]['versions'].append(version_info)
        self.registry[model_name]['latest_version'] = version
        self.registry[model_name]['updated_at'] = datetime.now().isoformat()
        
        self._save_registry()
        
        self.logger.info(f"Model {model_name} version {version} saved successfully")
        
        return version
    
    def load_model(self, model_name: str, version: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Load model and its metadata.
        
        Args:
            model_name: Name of the model
            version: Version to load (latest if not specified)
            
        Returns:
            Tuple of (model_object, metadata_dict)
        """
        self.logger.info(f"Loading model: {model_name}, version: {version or 'latest'}")
        
        if model_name not in self.registry:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        # Get version to load
        if version is None:
            version = self.registry[model_name]['latest_version']
            if version is None:
                raise ValueError(f"No versions available for model '{model_name}'")
        
        # Check if version exists
        version_exists = any(v['version'] == version for v in self.registry[model_name]['versions'])
        if not version_exists:
            raise ValueError(f"Version '{version}' not found for model '{model_name}'")
        
        # Load model
        model_dir = self.versions_dir / model_name / version
        model_file = model_dir / "model.pkl"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        try:
            model = joblib.load(model_file)
            self.logger.info(f"Model loaded from: {model_file}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Load metadata
        metadata_file = model_dir / "metadata.json"
        metadata = {}
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading metadata: {str(e)}")
        
        # Load performance metrics
        performance_file = model_dir / "performance.json"
        if performance_file.exists():
            try:
                with open(performance_file, 'r') as f:
                    performance_data = json.load(f)
                    metadata['performance'] = performance_data
            except Exception as e:
                self.logger.warning(f"Error loading performance metrics: {str(e)}")
        
        return model, metadata
    
    def list_models(self) -> Dict[str, Any]:
        """
        List all available models and their versions.
        
        Returns:
            Dictionary containing model information
        """
        self.logger.info("Listing all available models")
        
        model_list = {}
        
        for model_name, model_info in self.registry.items():
            model_list[model_name] = {
                'model_type': model_info['model_type'],
                'total_versions': len(model_info['versions']),
                'latest_version': model_info['latest_version'],
                'created_at': model_info['created_at'],
                'updated_at': model_info.get('updated_at'),
                'versions': [
                    {
                        'version': v['version'],
                        'created_at': v['created_at'],
                        'performance_summary': v['performance_summary'],
                        'is_active': v['is_active']
                    }
                    for v in model_info['versions']
                ]
            }
        
        return model_list
    
    def delete_model_version(self, model_name: str, version: str) -> bool:
        """
        Delete a specific model version.
        
        Args:
            model_name: Name of the model
            version: Version to delete
            
        Returns:
            True if deletion was successful
        """
        self.logger.info(f"Deleting model version: {model_name} v{version}")
        
        if model_name not in self.registry:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Find version in registry
        version_index = None
        for i, v in enumerate(self.registry[model_name]['versions']):
            if v['version'] == version:
                version_index = i
                break
        
        if version_index is None:
            raise ValueError(f"Version '{version}' not found for model '{model_name}'")
        
        # Delete files
        model_dir = self.versions_dir / model_name / version
        
        try:
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
                self.logger.info(f"Deleted model directory: {model_dir}")
        except Exception as e:
            self.logger.error(f"Error deleting model files: {str(e)}")
            return False
        
        # Remove from registry
        del self.registry[model_name]['versions'][version_index]
        
        # Update latest version if necessary
        if self.registry[model_name]['latest_version'] == version:
            if self.registry[model_name]['versions']:
                # Set latest to most recent remaining version
                latest_version = max(
                    self.registry[model_name]['versions'],
                    key=lambda x: x['created_at']
                )['version']
                self.registry[model_name]['latest_version'] = latest_version
            else:
                self.registry[model_name]['latest_version'] = None
        
        # Remove model entry if no versions left
        if not self.registry[model_name]['versions']:
            del self.registry[model_name]
        
        self._save_registry()
        
        self.logger.info(f"Model version {model_name} v{version} deleted successfully")
        
        return True
    
    def compare_model_versions(self, model_name: str, 
                             versions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare different versions of a model.
        
        Args:
            model_name: Name of the model
            versions: List of versions to compare (all versions if not specified)
            
        Returns:
            Dictionary containing comparison results
        """
        self.logger.info(f"Comparing versions of model: {model_name}")
        
        if model_name not in self.registry:
            raise ValueError(f"Model '{model_name}' not found")
        
        available_versions = [v['version'] for v in self.registry[model_name]['versions']]
        
        if versions is None:
            versions = available_versions
        else:
            # Validate requested versions
            invalid_versions = set(versions) - set(available_versions)
            if invalid_versions:
                raise ValueError(f"Invalid versions: {invalid_versions}")
        
        comparison_results = {
            'model_name': model_name,
            'compared_versions': versions,
            'version_details': {},
            'performance_comparison': {},
            'recommendations': []
        }
        
        # Load performance data for each version
        performance_data = {}
        
        for version in versions:
            try:
                _, metadata = self.load_model(model_name, version)
                if 'performance' in metadata:
                    performance_data[version] = metadata['performance']
                    
                    comparison_results['version_details'][version] = {
                        'created_at': metadata.get('created_at'),
                        'model_size_bytes': metadata.get('model_size_bytes'),
                        'training_time': metadata['performance'].get('training_time'),
                        'prediction_time': metadata['performance'].get('prediction_time')
                    }
            except Exception as e:
                self.logger.warning(f"Error loading version {version}: {str(e)}")
                continue
        
        # Compare performance metrics
        if performance_data:
            model_type = list(performance_data.values())[0]['model_type']
            
            if model_type == 'regression':
                comparison_results['performance_comparison'] = self._compare_regression_versions(performance_data)
            else:
                comparison_results['performance_comparison'] = self._compare_classification_versions(performance_data)
            
            # Generate recommendations
            comparison_results['recommendations'] = self._generate_version_recommendations(
                performance_data, comparison_results['performance_comparison']
            )
        
        return comparison_results
    
    def get_model_performance_history(self, model_name: str) -> Dict[str, Any]:
        """
        Get performance history for a model across all versions.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary containing performance history
        """
        self.logger.info(f"Getting performance history for model: {model_name}")
        
        if model_name not in self.registry:
            raise ValueError(f"Model '{model_name}' not found")
        
        history = {
            'model_name': model_name,
            'versions': [],
            'performance_trends': {},
            'best_performing_version': None
        }
        
        performance_data = []
        
        # Collect performance data from all versions
        for version_info in self.registry[model_name]['versions']:
            version = version_info['version']
            
            try:
                _, metadata = self.load_model(model_name, version)
                if 'performance' in metadata:
                    perf_data = metadata['performance']
                    perf_data['version'] = version
                    perf_data['created_at'] = version_info['created_at']
                    performance_data.append(perf_data)
            except Exception as e:
                self.logger.warning(f"Error loading version {version}: {str(e)}")
                continue
        
        # Sort by creation date
        performance_data.sort(key=lambda x: x['created_at'])
        
        if not performance_data:
            return history
        
        # Extract performance trends
        model_type = performance_data[0]['model_type']
        
        if model_type == 'regression':
            metrics = ['r2_score', 'rmse', 'mae']
            best_metric = 'r2_score'
            best_is_higher = True
        else:
            metrics = ['accuracy', 'f1_score', 'precision', 'recall']
            best_metric = 'accuracy'
            best_is_higher = True
        
        # Build trends
        for metric in metrics:
            if metric in performance_data[0]['accuracy_metrics']:
                history['performance_trends'][metric] = [
                    {
                        'version': data['version'],
                        'created_at': data['created_at'],
                        'value': data['accuracy_metrics'].get(metric, 0)
                    }
                    for data in performance_data
                ]
        
        # Find best performing version
        if best_metric in performance_data[0]['accuracy_metrics']:
            best_version_data = max(
                performance_data,
                key=lambda x: x['accuracy_metrics'].get(best_metric, 0) if best_is_higher
                else -x['accuracy_metrics'].get(best_metric, float('inf'))
            )
            history['best_performing_version'] = {
                'version': best_version_data['version'],
                'metric': best_metric,
                'value': best_version_data['accuracy_metrics'][best_metric],
                'created_at': best_version_data['created_at']
            }
        
        history['versions'] = [
            {
                'version': data['version'],
                'created_at': data['created_at'],
                'key_metrics': {
                    metric: data['accuracy_metrics'].get(metric, 0)
                    for metric in metrics[:3]  # Top 3 metrics
                    if metric in data['accuracy_metrics']
                }
            }
            for data in performance_data
        ]
        
        return history
    
    def deploy_model(self, model_name: str, version: Optional[str] = None,
                    deployment_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Deploy model for production use.
        
        Args:
            model_name: Name of the model to deploy
            version: Version to deploy (latest if not specified)
            deployment_config: Deployment configuration
            
        Returns:
            Dictionary containing deployment information
        """
        self.logger.info(f"Deploying model: {model_name}, version: {version or 'latest'}")
        
        # Load model to validate it works
        try:
            model, metadata = self.load_model(model_name, version)
            actual_version = metadata.get('version', version)
        except Exception as e:
            self.logger.error(f"Error loading model for deployment: {str(e)}")
            raise
        
        # Create deployment record
        deployment_info = {
            'model_name': model_name,
            'version': actual_version,
            'deployed_at': datetime.now().isoformat(),
            'deployment_config': deployment_config or {},
            'status': 'deployed',
            'model_metadata': metadata
        }
        
        # Save deployment info
        deployment_file = self.metadata_dir / f"deployment_{model_name}_{actual_version}.json"
        
        try:
            with open(deployment_file, 'w') as f:
                json.dump(deployment_info, f, indent=2, default=str)
            
            self.logger.info(f"Deployment info saved to: {deployment_file}")
        except Exception as e:
            self.logger.error(f"Error saving deployment info: {str(e)}")
            raise
        
        # Update registry with deployment status
        for version_info in self.registry[model_name]['versions']:
            if version_info['version'] == actual_version:
                version_info['deployed'] = True
                version_info['deployed_at'] = deployment_info['deployed_at']
                break
        
        self._save_registry()
        
        self.logger.info(f"Model {model_name} v{actual_version} deployed successfully")
        
        return deployment_info
    
    def _generate_version(self, model_name: str) -> str:
        """Generate version string for model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check if model exists and get next version number
        if model_name in self.registry:
            version_count = len(self.registry[model_name]['versions']) + 1
            return f"v{version_count}_{timestamp}"
        else:
            return f"v1_{timestamp}"
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.warning(f"Error calculating file hash: {str(e)}")
            return ""
    
    def _extract_performance_summary(self, performance: ModelPerformance) -> Dict[str, float]:
        """Extract key performance metrics for summary."""
        metrics = performance.accuracy_metrics
        
        if performance.model_type == 'regression':
            return {
                'r2_score': metrics.get('r2_score', 0),
                'rmse': metrics.get('rmse', 0),
                'mae': metrics.get('mae', 0)
            }
        else:
            return {
                'accuracy': metrics.get('accuracy', 0),
                'f1_score': metrics.get('f1_score', metrics.get('f1_macro', 0)),
                'precision': metrics.get('precision', metrics.get('precision_macro', 0))
            }
    
    def _compare_regression_versions(self, performance_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare regression model versions."""
        comparison = {
            'metric_comparison': {},
            'best_version_by_metric': {},
            'performance_trends': {}
        }
        
        metrics = ['r2_score', 'rmse', 'mae', 'mape']
        
        for metric in metrics:
            metric_values = {}
            
            for version, perf_data in performance_data.items():
                if metric in perf_data['accuracy_metrics']:
                    metric_values[version] = perf_data['accuracy_metrics'][metric]
            
            if metric_values:
                comparison['metric_comparison'][metric] = metric_values
                
                # Find best version for this metric
                if metric == 'r2_score':  # Higher is better
                    best_version = max(metric_values, key=metric_values.get)
                else:  # Lower is better for error metrics
                    best_version = min(metric_values, key=metric_values.get)
                
                comparison['best_version_by_metric'][metric] = {
                    'version': best_version,
                    'value': metric_values[best_version]
                }
        
        return comparison
    
    def _compare_classification_versions(self, performance_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare classification model versions."""
        comparison = {
            'metric_comparison': {},
            'best_version_by_metric': {},
            'performance_trends': {}
        }
        
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        
        for metric in metrics:
            metric_values = {}
            
            for version, perf_data in performance_data.items():
                # Handle different metric naming conventions
                metric_key = metric
                if metric_key not in perf_data['accuracy_metrics']:
                    if metric == 'f1_score':
                        metric_key = 'f1_macro'
                    elif metric == 'precision':
                        metric_key = 'precision_macro'
                    elif metric == 'recall':
                        metric_key = 'recall_macro'
                
                if metric_key in perf_data['accuracy_metrics']:
                    metric_values[version] = perf_data['accuracy_metrics'][metric_key]
            
            if metric_values:
                comparison['metric_comparison'][metric] = metric_values
                
                # Find best version for this metric (higher is better for all classification metrics)
                best_version = max(metric_values, key=metric_values.get)
                
                comparison['best_version_by_metric'][metric] = {
                    'version': best_version,
                    'value': metric_values[best_version]
                }
        
        return comparison
    
    def _generate_version_recommendations(self, performance_data: Dict[str, Dict],
                                        comparison: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on version comparison."""
        recommendations = []
        
        if not performance_data:
            return recommendations
        
        # Check if there's a clear best version
        best_versions = list(comparison.get('best_version_by_metric', {}).values())
        
        if best_versions:
            version_counts = {}
            for best_info in best_versions:
                version = best_info['version']
                version_counts[version] = version_counts.get(version, 0) + 1
            
            most_frequent_best = max(version_counts, key=version_counts.get)
            
            if version_counts[most_frequent_best] > len(best_versions) / 2:
                recommendations.append(
                    f"Version {most_frequent_best} performs best across multiple metrics. "
                    "Consider using this version for deployment."
                )
            else:
                recommendations.append(
                    "No single version dominates all metrics. "
                    "Choose version based on your priority metric."
                )
        
        # Check for performance degradation
        versions_by_date = sorted(
            performance_data.items(),
            key=lambda x: x[1].get('created_at', '')
        )
        
        if len(versions_by_date) >= 2:
            latest_version = versions_by_date[-1]
            previous_version = versions_by_date[-2]
            
            # Compare key metric (accuracy for classification, r2 for regression)
            model_type = latest_version[1]['model_type']
            key_metric = 'accuracy' if model_type == 'classification' else 'r2_score'
            
            latest_score = latest_version[1]['accuracy_metrics'].get(key_metric, 0)
            previous_score = previous_version[1]['accuracy_metrics'].get(key_metric, 0)
            
            if latest_score < previous_score * 0.95:  # 5% degradation threshold
                recommendations.append(
                    f"Latest version shows performance degradation in {key_metric}. "
                    "Consider investigating or reverting to previous version."
                )
        
        # Storage recommendations
        if len(performance_data) > 5:
            recommendations.append(
                "Multiple model versions detected. "
                "Consider cleaning up old versions to save storage space."
            )
        
        return recommendations