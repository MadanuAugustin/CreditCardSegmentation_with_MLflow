
import os
import sys
import numpy as np
import pandas as pd
import joblib
from src.CreditCardSegmentation.logger_file.logger_obj import logger
from src.CreditCardSegmentation.Exception.custom_exception import CustomException
from sklearn.model_selection import train_test_split
from src.CreditCardSegmentation.entity.config_entity import DataTransformationConfig
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer





class DataTransformation:
    def __init__(self, config : DataTransformationConfig):
        
        self.config = config




    def data_split(self):
        try:

            logger.info(f'-----------Entered data_split method---------------')
        
            data = pd.read_csv(self.config.local_data_file)

            data = data[["BALANCE", 'PURCHASES', 'CREDIT_LIMIT']]

            train, test = train_test_split(data, test_size=0.2, random_state=42)

            train.to_csv(os.path.join(self.config.train_data_path, 'train.csv'), index = False, header = True)
        
            test.to_csv(os.path.join(self.config.test_data_path, 'test.csv'), index = False, header = True)

            logger.info(f'----------------saved train test data in csv format------------------')

            logger.info(f'------------The shape of the train data is {train.shape}')

            logger.info(f'--------------The shape of the test data is {test.shape}')

            logger.info(f'------------completed data splitting----------------------')

        except Exception as e:
            raise CustomException(e, sys)
        

    

    def preprocessor_fun(self):
        try:

            logger.info(f'---------------Entered preprocessor function------------------')


            numeric_columns = ["BALANCE", 'PURCHASES', 'CREDIT_LIMIT']
            
            logger.info(f'----------creating transformer pipelines---------------')

            numeric_pipeline = Pipeline(
                steps=[
                    ('standardscaler', MinMaxScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('numericpipeline', numeric_pipeline, numeric_columns)
                ]
            )

            logger.info(f'---------------completed creating transformer pipelines---------------')

            logger.info(f'--------------completed preprocessor function------------------')

            return preprocessor
            

        except Exception as e:
            raise CustomException(e, sys)
        

    
    def initiate_data_transformation(self):

        try:

            logger.info(f'------------started initiate_data_transformation method------------')

            train_df = pd.read_csv('artifacts//data_transformation//train.csv')
            
            test_df = pd.read_csv('artifacts//data_transformation//test.csv')

            logger.info(f'----------obtaining the preprocessor obj-----------')

            preprocessor_obj = self.preprocessor_fun()

            transformed_train_df = pd.DataFrame(preprocessor_obj.fit_transform(train_df))

            joblib.dump(preprocessor_obj, os.path.join(self.config.root_dir, 'preprocessor_obj.joblib'))

            # transformed_test_df = pd.DataFrame(preprocessor_obj.transform(test_df))

            transformed_train_df.rename(columns={0 : 'BALANCE', 1 : 'PURCHASES', 2 : 'CREDIT_LIMIT'}, inplace=True)

            # transformed_test_df.rename(columns={0 : 'BALANCE', 1 : 'PURCHASES', 2 : 'CREDIT_LIMIT'}, inplace=True)

            transformed_train_df.to_csv(os.path.join(self.config.root_dir, 'transformed_train_df.csv'), index = False, header = True)

            # transformed_test_df.to_csv(os.path.join(self.config.root_dir, 'transformed_test_df.csv'), index = False, header = True)

            logger.info(f'-------------transformed data using preprocessor obj and saved in csv format----------')

            logger.info(f'--------------completed initiate_data_transformation method--------------')

        except Exception as e:
            raise CustomException(e, sys)