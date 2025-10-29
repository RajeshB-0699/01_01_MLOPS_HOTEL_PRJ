import os
import pandas as pd
import numpy as np
from utils.common_functions import read_yaml, load_data
from config.paths_config import *
from src.custom_exception import CustomException
from src.logger import get_log
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE



logger = get_log("Starting data preprocessing")

class DataProcessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config= read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def preprocess_data(self, df):
        try:
            logger.info("Starts Data preprocess_data")
            logger.info("Dropping columns")
            df.drop(columns = 'Booking_ID', inplace=True)
            df.drop_duplicates(inplace=True)

            cat_cols = self.config['data_processing']['cat_cols']
            num_cols = self.config['data_processing']['num_cols']

            logger.info("Applying label encoder")
            le = LabelEncoder()
            mappings = {}
            for col in cat_cols:
                df[col] = le.fit_transform(df[col])
                mappings[col] = {label: code for label,code in zip(le.classes_, le.fit_transform(le.classes_))}
            logger.info(f"Mapped labels {mappings}")

            logger.info("Skewness threshold")
            skewness_threshold = self.config['data_processing']['skewness_threshold']
            skewness = df[num_cols].apply(lambda x : x.skew())
            for column in skewness[skewness > skewness_threshold].index:
                df[column] = np.log1p(df[column])
            return df
        except Exception as e:
            logger.error(f"Error in data preprocess steps {e}")
            raise CustomException("Error in data preprocess steps - 1",e)
    
    def balance_data(self, df):
        try:
            logger.info(" Handling data imbalance")
            smote = SMOTE(random_state=42)
            X = df.drop(columns = 'booking_status')
            y = df['booking_status']
            X_res , y_res = smote.fit_resample(X,y)
            balanced_df = pd.DataFrame(X_res, columns=X.columns)
            balanced_df['booking_status']= y_res
            logger.info("Data balanced ...")
            return balanced_df
        except Exception as e:
            logger.error(f"Error in balancing data...")
            raise CustomException(f"Error in balancing data...",e)
    
    def select_features(self, df):
        try:
            logger.info("Selectinf features")
            X = df.drop(columns = 'booking_status')
            y = df['booking_status']
            model =  RandomForestClassifier(random_state=42)
            model.fit(X,y)
            feature_imp = model.feature_importances_
            feature_imp_df = pd.DataFrame({
                'features': X.columns,
                'importance': feature_imp 
            })

            top_feature_df = feature_imp_df.sort_values(by = 'importance', ascending=False)
            num_features_to_select = self.config['data_processing']['no_of_features']
            top_10_features = top_feature_df['features'].head(num_features_to_select).values
            top_10_df = df[top_10_features.tolist() + ['booking_status']]
            logger.info("Feature selection completed")
            return top_10_df
        except Exception as e:
            logger.error(f"Error in selecting features {e}")
            raise CustomException("Error in selecting features", e)
        
    def save_data(self, df, file_path):
        try:
            logger.info("Saving csv files which were processed")
            df.to_csv(file_path, index=False)
            logger.info(f"Data files csv files saved successfully {file_path}")
        except Exception as e:
            logger.error(f"Error in saving data {e}")
            raise CustomException("Error in saving file ", e)
    
    def process(self):
        try:
            logger.info("Loading data from all..")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            train_df = self.select_features(train_df)
            test_df = self.select_features(test_df)

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data Processing process run.. over..")
        except Exception as e:
            logger.error("Error in process method of data_preprocessing")
            raise CustomException(f"Error in Data Preprocess process...",e)
        
if __name__ == "__main__":
    processor = DataProcessor(TRAIN_FILEPATH, TEST_FILEPATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()