
import mlflow
import yaml
from typing import Dict, Any
import logging

class ExperimentManager:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        self.experiment_name = self.config['mlflow']['experiment_name']
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
            
    def start_run(self, run_name: str = None):
        """Start a new MLflow run."""
        return mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name
        )
        
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        mlflow.log_params(params)
        
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to MLflow."""
        mlflow.log_metrics(metrics, step=step)
        
    def log_artifact(self, local_path: str):
        """Log an artifact to MLflow."""
        mlflow.log_artifact(local_path)
        
    def log_model(self, model, artifact_path: str):
        """Log a model to MLflow."""
        mlflow.sklearn.log_model(model, artifact_path)