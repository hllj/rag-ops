import os
import time
import mlflow
import yaml
from typing import Dict, Any
import logging

class ExperimentManager:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        mlflow.set_tracking_uri(tracking_uri)
        self.experiment_name = self.config['mlflow']['experiment_name']
        
        # Wait for MLflow to be ready
        self._wait_for_mlflow(tracking_uri)
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                self.experiment_id = experiment.experiment_id
        except Exception as e:
            logging.error(f"Failed to create/get experiment: {str(e)}")
            raise
        
    def _wait_for_mlflow(self, tracking_uri, max_retries=5, delay=5):
        """Wait for MLflow server to be ready"""
        for i in range(max_retries):
            try:
                client = mlflow.tracking.MlflowClient()
                client.search_experiments()
                return True
            except Exception as e:
                if i < max_retries - 1:
                    logging.warning(f"Waiting for MLflow server... (attempt {i+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    logging.error(f"MLflow server not available after {max_retries} attempts")
                    raise
            
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