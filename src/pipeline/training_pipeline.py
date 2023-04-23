import os
import sys


from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransormation
from src.components.model_trainer import ModelTrainer

# run data ingestion
if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransormation()
    train_array, test_array, preprocessor_file_path = data_transformation.initiate_data_transformation(
        train_data_path, test_data_path)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_array, test_array)
