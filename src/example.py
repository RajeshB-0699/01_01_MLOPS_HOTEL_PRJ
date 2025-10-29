from utils.common_functions import read_yaml
import yaml
from config.paths_config import *
import pandas as pd
from src.logger import get_log
from src.custom_exception import CustomException
import sys
from sklearn.model_selection import train_test_split

logger = get_log(__name__)
# filepath = yaml.safe_load(open('config/config.yaml'))['data_ingestion']['file_path']
# ratio = yaml.safe_load(open('config/config.yaml'))['data_ingestion']['train_ratio']
class NewExample:
    
    filepath = read_yaml('config/config.yaml')['data_ingestion']['file_path']
    ratio = read_yaml('config/config.yaml')['data_ingestion']['train_ratio']

    os.makedirs(RAW_DIR, exist_ok=True)

    def read_file_from_input():
        try:
            logger.info("Reading file input")
            df = pd.read_csv(NewExample.filepath)
            df.to_csv(RAW_FILEPATH, index=False)
            logger.info("File got stored")
        except Exception as e:
            logger.error("Error in reading..")
            raise CustomException("Failed to load and download data ",sys)
    
    def split_data():
        try:
            logger.info("Initiate splitting data")
            df = pd.read_csv(RAW_FILEPATH)
            train_data, test_data = train_test_split(df, test_size = 1-NewExample.ratio, random_state=42)
            train_data.to_csv(TRAIN_FILEPATH, index=False)
            test_data.to_csv(TEST_FILEPATH, index=False)
            logger.info("COmpleted....")
        except Exception as e:
            logger.error("Error in splittig data")
            raise CustomException("Failed to split data ", sys)
    def run():
        try:
            logger.info("Trying....")
            NewExample.read_file_from_input()
            NewExample.split_data()
            logger.info("endss/..")
        except CustomException as ce:
            logger.error(f"Error in {str(ce)}")
        finally:
            logger.info("comle/...")

if __name__ == "__main__":
    NewExample.run()
    


 


   

