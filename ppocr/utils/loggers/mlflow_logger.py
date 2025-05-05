import os
from .base_logger import BaseLogger  # Import đúng module chứa BaseLogger của bạn
from ppocr.utils.logging import get_logger
import mlflow

class MlflowLogger(BaseLogger):
    def __init__(self, 
        experiment_name=None, 
        run_name=None, 
        save_dir=None, 
        config=None,
        tracking_server_uri=None,
        **kwargs):
        try:

            self.mlflow = mlflow
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install mlflow using `pip install mlflow`"
            )

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.save_dir = save_dir
        self.config = config
        self.tracking_server_uri = tracking_server_uri
        self.kwargs = kwargs
        self._run = None

        if self.tracking_server_uri:
            self.mlflow.set_tracking_uri(self.tracking_server_uri)

        if self.experiment_name:
            self.mlflow.set_experiment(self.experiment_name)
        
        _ = self.run  # ensure run is initialized

        if self.config:
            self.mlflow.log_params(self.config)

    @property
    def run(self):
        if self._run is None:
            active_run = self.mlflow.active_run()
            if active_run is not None:
                self._run = active_run
            else:
                self._run = self.mlflow.start_run(run_name=self.run_name)
        return self._run

    def log_metrics(self, metrics, prefix=None, step=None):
        if not prefix:
            prefix = ""
        updated_metrics = {prefix.lower() + "/" + k: v for k, v in metrics.items()}

        if step is not None:
            for k, v in updated_metrics.items():
                self.mlflow.log_metric(k, v, step=step)
        else:
            self.mlflow.log_metrics(updated_metrics)

    def log_model(self, is_best, prefix, metadata=None):
        model_path = os.path.join(self.save_dir, prefix + '.pdparams')
        states_path = os.path.join(self.save_dir, prefix + '.states')
        pdopt_path = os.path.join(self.save_dir, prefix + '.pdopt')

        
        self.mlflow.log_artifact(model_path, artifact_path="models")
        self.mlflow.log_artifact(states_path, artifact_path="models")
        self.mlflow.log_artifact(pdopt_path, artifact_path="models")


        if metadata:
            metadata_path = os.path.join(self.save_dir, prefix + '_metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            self.mlflow.log_artifact(metadata_path, artifact_path="models")

    def close(self):
        if self.mlflow.active_run() is not None:
            self.mlflow.end_run()
