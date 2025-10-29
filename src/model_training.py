import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import joblib
import sys 

from src.logger import get_log
from src.custom_exception import CustomException
from config.model_params import *
from config.paths_config import *

from utils.common_functions import read_yaml, load_data

from scipy.stats import randint, uniform

import mlflow
import mlflow.sklearn

logger = get_log(__name__)

class ModelTraining:
    def __init__(self, train_path, test_path, model_output_path, config_file):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS
        self.config = read_yaml(config_file)

    def load_and_split_data(self):
        try:
            logger.info(f"Staring Model building and loading train data from {self.train_path}")
            train_df = load_data(self.train_path)
            #train_df = pd.read_csv("./artifacts/processed/processed_train.csv")
            logger.info(f"Loading test data from {self.test_path}")
            test_df = load_data(self.test_path)
            #test_df = pd.read_csv("./artifacts/processed/processed_test.csv")
            X_train = train_df.drop(columns=self.config['data_processing']['target_col'])
            y_train = train_df[self.config['data_processing']['target_col']]
            X_test = test_df.drop(columns=self.config['data_processing']['target_col'])
            y_test = test_df[self.config['data_processing']['target_col']]
            
            logger.info("Loaded data and splitted data")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Error in loading ad splitting data ,{e}")
            raise CustomException("Error in loading and splitting data ", sys)
        
    
    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Initializing model LGBM")
            lgbm_model = lgb.LGBMClassifier(random_state = self.random_search_params['random_state'])
            logger.info("Starting hyperparameter tuning of model")
            random_search =  RandomizedSearchCV(estimator = lgbm_model,
                                                  param_distributions= self.params_dist,
                                                  n_iter= self.random_search_params["n_iter"],
                                                  cv = self.random_search_params["cv"],
                                                  n_jobs = self.random_search_params["n_jobs"],
                                                  verbose = self.random_search_params['verbose'],
                                                  random_state= self.random_search_params['random_state'],
                                                  scoring = self.random_search_params['metrics'])
            logger.info("Starting hyperparameter tuning")
            random_search.fit(X_train, y_train)
            logger.info("Hyperparameter tuning completed")
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_
            logger.info(f"Best parameters are {best_params}")
            return best_lgbm_model
        except Exception as e:
            logger.error("Error in training lgbm model")
            raise CustomException("Error in training LGBM model", sys)
    
    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating model")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            logger.info(f"Accuracy : {accuracy}, Precision : {precision}, Recall : {recall},  f1  {f1}")
            return {"accuracy": accuracy, "precision": precision, "Recall": recall, "F1" : f1}
        except Exception as e:
            logger.error(f"Error in evaluating model {e}")
            raise CustomException("Error in evaluating model", sys)
    
    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            logger.info("Dumping LGBM model")
            joblib.dump(model, self.model_output_path)
            logger.info(f"Saving model to {self.model_output_path}")
        except Exception as e:
            logger.error(f"Error in saving model {e}")
            raise CustomException("Error in saving model", sys)

    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Running model building pipeline")
                logger.info("MLflow starting...")
                logger.info("ML flow logging training and test data")
                logger.info("Logging training artifacts")
                mlflow.log_artifact(self.train_path, artifact_path='dataset')
                logger.info("Logging test dataset artifacts")
                mlflow.log_artifact(self.test_path, artifact_path='dataset')
                X_train, y_train, X_test , y_test = self.load_and_split_data()
                best_model_lgbm = self.train_lgbm(X_train, y_train)
                metric = self.evaluate_model(best_model_lgbm,X_test, y_test)
                self.save_model(best_model_lgbm)    
                logger.info(metric)
                logger.info("Versioning models logging into mlflow")
                mlflow.sklearn.log_model(sk_model=best_model_lgbm, name="model")

                mlflow.log_params(best_model_lgbm.get_params())
                mlflow.log_metrics(metric)
                logger.info("Logged metrics into mlflow")
        except CustomException as ce:
            logger.error(f"Error in model building pipeline, {ce}")

if __name__ == "__main__":
    model_training = ModelTraining(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,MODEL_OUTPUT_PATH,CONFIG_PATH)
    model_training.run()