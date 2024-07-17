
import pandas as pd
import joblib
import os
import numpy as np
from src.CreditCardSegmentation.entity.config_entity import ModelTrainerConfig
from sklearn.cluster import KMeans
from src.CreditCardSegmentation.logger_file.logger_obj import logger


class ModelTrainer:
    def __init__(self, config : ModelTrainerConfig):
        self.config = config


    def initiate_model_training(self):

        data_df = pd.read_csv(self.config.data_path)

        # data_df.dropna(inplace=True)

        kmeans = KMeans(n_clusters=self.config.n_clusters, init=self.config.init, random_state=42)

        kmeans.fit(data_df)

        logger.info(f'---------Model training completed------------')

        labels = kmeans.labels_

        labeled_data_df = pd.DataFrame(np.c_[data_df, labels])

        labeled_data_df.to_csv(os.path.join(self.config.root_dir, 'labeled_data_df.csv'), index = False, header = True)

        joblib.dump(kmeans, os.path.join(self.config.root_dir, self.config.model_name))

        logger.info(f'-----------saved model as pickle file---------------')


