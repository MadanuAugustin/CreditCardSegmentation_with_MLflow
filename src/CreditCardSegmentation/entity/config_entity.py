


from dataclasses import dataclass
from pathlib import Path


#################################### DATA_INGESTION_CONFIG ################################


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir : Path
    local_data_file : Path



#################################### DATA-VALIDATION-CONFIG ################################


@dataclass(frozen= True)
class DataValidationConfig:
    root_dir : Path
    local_data_file : Path
    STATUS_FILE : Path
    all_schema : Path


#################################### DATA-TRANSFORMATION-CONFIG ################################


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir : Path
    local_data_file : Path
    data_path : Path


#################################### MODEL-TRAINER-CONFIG ################################


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir : Path
    data_path : Path
    model_name : str
    n_clusters : int
    init : str



#################################### MODEL-EVALUATION-CONFIG ################################


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir : Path
    model_path : Path
    metric_file_name : Path
    all_params : dict
    mlflow_uri : str
    data_path : Path