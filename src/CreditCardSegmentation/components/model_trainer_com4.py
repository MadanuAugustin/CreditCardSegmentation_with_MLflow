
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

        # train_data = pd.read_csv(self.config.train_data_path)
        train_df = pd.read_csv(self.config.train_data_path)

        train_df.dropna(inplace=True)

        kmeans = KMeans(n_clusters=2, random_state=42)

        kmeans.fit(train_df)

        logger.info(f'---------Model training completed------------')

        labels = kmeans.labels_

        labeled_train_df = pd.DataFrame(np.c_[train_df, labels])

        labeled_train_df.to_csv(os.path.join(self.config.root_dir, 'labeled_train_df.csv'), index = False, header = True)

        joblib.dump(kmeans, os.path.join(self.config.root_dir, self.config.model_name))

        logger.info(f'-----------saved model as pickle file---------------')


