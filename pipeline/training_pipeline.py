from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTraining
from config.model_params import *
from config.paths_config import *
from utils.common_functions import *

if __name__ == "__main__":
    data_ingestion_obj = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion_obj.run()

    processor = DataProcessor(TRAIN_FILEPATH, TEST_FILEPATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()

    model_training = ModelTraining(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,MODEL_OUTPUT_PATH,CONFIG_PATH)
    model_training.run()