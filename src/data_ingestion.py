import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from src.logger import get_log
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml
import sys

logger = get_log(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config['data_ingestion']
        self.train_test_ratio = self.config['train_ratio']
        self.file_downlod_path = self.config['file_path']

        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info(f"Data Ingestion initialized with file")
    
    def download_csv_from_storage(self):
        try:
            df = pd.read_csv(self.file_downlod_path)
            df.to_csv(RAW_FILEPATH, index=False)
            logger.info(f"Successfully downloaded raw file into {RAW_FILEPATH}")
        except Exception as e:
            logger.error("Error downloading CSV file")
            raise CustomException("Failed to download CSV file", sys)  # Pass sys module
    
    def split_data(self):
        try:
            logger.info("Starting to split data")
            data = pd.read_csv(RAW_FILEPATH)
            train_data, test_data = train_test_split(
                data, test_size=1 - self.train_test_ratio, random_state=42
            )
            train_data.to_csv(TRAIN_FILEPATH, index=False)
            test_data.to_csv(TEST_FILEPATH, index=False)
            logger.info(f"Train data saved to {TRAIN_FILEPATH} and test data saved to {TEST_FILEPATH}")
        except Exception as e:
            logger.error("Error in splitting data")
            raise CustomException("Failed to split data", sys)  # Pass sys module
    
    def run(self):
        try:
            logger.info("Starting Data Ingestion process")
            self.download_csv_from_storage()
            self.split_data()
            logger.info("Data Ingestion process completed successfully")
        except CustomException as ce:
            logger.error(f"Custom Exception: {str(ce)}")
       
        finally:
            logger.info("Data Ingestion pipeline finished execution")

if __name__ == "__main__":
    data_ingestion_obj = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion_obj.run()