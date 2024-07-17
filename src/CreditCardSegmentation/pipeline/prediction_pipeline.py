




import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.CreditCardSegmentation.logger_file.logger_obj import logger
from src.CreditCardSegmentation.Exception.custom_exception import CustomException


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts//model_trainer//model.joblib'))
        self.preprocessorObj = joblib.load(Path('artifacts//data_transformation//preprocessor_obj.joblib'))


    # the below method takes the data from the user to predict

    def predictDatapoint(self, data):
        
        try:

            data_df = data.rename(columns = {0 : "BALANCE", 1 : 'PURCHASES', 2 : 'CREDIT_LIMIT'})

            # data_df = data
            
            print(data_df)

            transformed_numeric_cols = self.preprocessorObj.transform(data_df)

            logger.info(f'---------Below is the transformed user input----------------')

            print(transformed_numeric_cols)

            prediction = self.model.predict(transformed_numeric_cols)

            cluster_mapping = {0: 'cluster_A', 1: 'cluster_B', 2: 'cluster_C', 3: 'cluster_D', 4: 'cluster_E', 5: 'cluster_F', 6: 'cluster_G',
                               7: 'cluster_H', 8: 'cluster_I', 9: 'cluster_J', 10: 'cluster_K', 11: 'cluster_L'}
            
            mapped_predictions = [cluster_mapping[pred] for pred in prediction]

            logger.info(f'-----------Below output is predicted by the model---------------')

            print(mapped_predictions)

            return mapped_predictions
        
        
        except Exception as e:
            raise CustomException(e, sys)