import os

#######   DATA INGESTION #######################

RAW_DIR = "artifacts/raw"
RAW_FILEPATH = os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILEPATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILEPATH = os.path.join(RAW_DIR, "test.csv")

CONFIG_PATH = "config/config.yaml"


############# DATA PROCESSING ######################

PROCESSED_DIR = "artifacts/processed"

PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_train.csv")
PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_test.csv")


############# MODEL BUILDING ##################

MODEL_OUTPUT_PATH = "artifacts/models/lgbm_model.pkl"

