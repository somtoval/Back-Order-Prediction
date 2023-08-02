import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

def replace_na_with_nan(X):
    return np.where(X == 'na', np.nan, X)

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            nan_replacement_step = ('nan_replacement', FunctionTransformer(replace_na_with_nan))
            imputer_step = ('imputer', SimpleImputer(strategy="mean"))
            scaler_step = ('scaler', RobustScaler())

            preprocessor = Pipeline(
                steps=[
                    nan_replacement_step,
                    imputer_step,
                    scaler_step
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info('Started initiate_data_transformation')

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            preprocessor = self.get_data_transformer_object()
            target_column_name = 'went_on_backorder'

            input_feature_train_df = train_df.drop(columns=['went_on_backorder'])
            target_feature_train_df = train_df['went_on_backorder']

            input_feature_test_df = test_df.drop(columns=['went_on_backorder'])
            target_feature_test_df = test_df['went_on_backorder']

            print('>>>>>>>>>>>>>>>>>>>>>>>Train: ', input_feature_train_df.columns)
            print('>>>>>>>>>>>>>>>>>>>>>>>Test: ', input_feature_test_df.columns)

            transformed_input_train_feature = preprocessor.fit_transform(input_feature_train_df)
            print('Length of train', transformed_input_train_feature.shape)
            transformed_input_test_feature = preprocessor.transform(input_feature_test_df)
            print('Length of test', transformed_input_test_feature.shape)

            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            save_object(self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessor)

            logging.info('Finished Data Transformation')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
                )
        
        except Exception as e:
            raise CustomException(e, sys)