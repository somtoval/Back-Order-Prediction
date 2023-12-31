import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, load_object, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class CustomModel:
    def __init__(self, preprocessing_object, trained_model_object) -> None:
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X):
        transformed_feature = self.preprocessing_object.transform(X)
        
        return self.trained_model_object.predict(transformed_feature)
    
    def __repr__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"
    
    def __str__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"
    
class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info(f"Splitting training and testing input and target feature")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "XGBClassifier": XGBClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }

            logging.info(f'Passing models and data to evaluate_models utility function')

            model_report:dict = evaluate_models(X=X_train, y=y_train, models=models)

            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            if best_model_score < 0.6:
                raise Exception("No best model found")

            logging.info(f"Best model found in training and testing data")

            preprocessing_obj = load_object(file_path=preprocessor_path)

            custom_model = CustomModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model,
            )

            logging.info(f"Saving model at path: {self.model_trainer_config.trained_model_file_path}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=custom_model
            )

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            print('>>> Test Accuracy >>>->:', accuracy)

            train_accuracy = accuracy_score(y_train, best_model.predict(X_train))
            print('>>> Train Accuracy >>>->:', train_accuracy)

            return accuracy
        
        except Exception as e:
            raise CustomException(e, sys)