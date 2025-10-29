import os
import pandas as pd
import yaml

from src.custom_exception import CustomException
from src.logger import get_log

logger = get_log(__name__)

def read_yaml(filepath):
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Error in finding file. Please provide correct path or file may not exists")
        with open(filepath, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info("Successfully loaded yaml file in config variable")
            return config
    except Exception as e:
        logger.error("Error in reading yaml file")
        raise CustomException("Failed to read yaml file",e)

def load_data(path):
    try:
        logger.info("Loading file csv data from path")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error in loading csv file from {path}")
        raise CustomException("Failed to load csv from path",e)
    

