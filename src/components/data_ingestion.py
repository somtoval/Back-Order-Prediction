from src.logger import logging
from src.exception import CustomException

import pandas as pd

import os
import sys
from dataclasses import dataclass

from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    raw_data_path:str = os.path.join('artifacts', 'raw.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered initiate_data_ingestion")
        try:
            df = pd.read_csv('C:/Users/user/My ML Projects/Back Order Prediction/notebooks/data/backorder.csv', nrows=2000)
            if 'sku' in df.columns:
                df.drop(columns=['sku'], axis=1)
            df.drop_duplicates(inplace=True)
            df.dropna(subset=["went_on_backorder"], inplace=True)

            encoder = LabelEncoder()
            cat_cols = df.select_dtypes(exclude=['int', 'float']).columns
            for col in cat_cols:
                df[col] = encoder.fit_transform(df[col])
            
            df = df.fillna(df.median().iloc[0])

            X = df.drop(columns=['went_on_backorder'], axis=1)
            y = df['went_on_backorder']

            smt = SMOTETomek(sampling_strategy='auto')
            X_new, y_new = smt.fit_resample(X,y)

            df = pd.DataFrame(X_new, columns=df.columns)
            df['went_on_backorder'] = y_new

            logging.info('The dataset has been Loaded')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            df.to_csv(self.ingestion_config.train_data_path, index=False)
            df.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info('Exited initiate_data_ingestion method of DataIngestion Class')

            return(
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
                    )
        except Exception as e:
            raise CustomException(e, sys)



