import sys
from dataclasses import dataclass
import os
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model, save_object


@dataclass
class ModelTraningConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTraningConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info(
                'splitting dependent and independent columns from train and test array')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]

            )

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet()
            }

            model_report: dict = evaluate_model(
                X_train, y_train, X_test, y_test, models)

            print(model_report)
            print(
                '\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(
                f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print(
                '\n====================================================================================\n')
            logging.info(
                f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info('error occured at model training ')
            raise CustomException(e, sys)
