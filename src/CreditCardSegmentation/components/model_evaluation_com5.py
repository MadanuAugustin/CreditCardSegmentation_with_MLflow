

import pandas as pd
import joblib
import dagshub
import mlflow
import os
import mlflow.sklearn
from pathlib import Path
from src.CreditCardSegmentation.entity.config_entity import ModelEvaluationConfig
from sklearn.metrics import silhouette_score 
from src.CreditCardSegmentation.utils.common import save_json
from src.CreditCardSegmentation.logger_file.logger_obj import logger


class ModelEvaluation:
    def __init__(self, config : ModelEvaluationConfig):
        self.config = config


    def log_into_mlflow(self):

        logger.info(f'-----------Entered log_into_mlflow function----------------')

        model = joblib.load(self.config.model_path)

        transformed_data = pd.read_csv(self.config.data_path)

        # transformed_data.dropna(inplace=True)

        logger.info(f'-----------successfully loaded model joblib--------------------------')

        os.environ["MLFLOW_TRACKING_URI"]='https://dagshub.com/augustin7766/CreditCardSegmentation_with_MLflow.mlflow'
        os.environ["MLFLOW_TRACKING_USERNAME"]="augustin7766"
        os.environ["MLFLOW_TRACKING_PASSWORD"]="8a01ee4bec043666cf3ced22edc7d308526b4b42"


        mlflow.set_experiment('ninth_09_exp')

        with mlflow.start_run():

            logger.info(f'------------------mlflow function started--------------------------------')

            inertia_score = model.inertia_

            silhouetteScore = silhouette_score(transformed_data, model.labels_)

            scores = {'inertia' : inertia_score, 'silhouetteScore' : silhouetteScore}

            logger.info(f'--------The model inertia score is :{inertia_score} and silhouetteScore is : {silhouetteScore}')

            save_json(path = Path(self.config.metric_file_name), data = scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric('inertia', model.inertia_)

            mlflow.log_metric('silhouetteScore', silhouetteScore)

            mlflow.sklearn.log_model(model, 'model', registered_model_name = 'Kmeansclustering')

            logger.info(f'------------------------mlflow function completed-----------------------')

