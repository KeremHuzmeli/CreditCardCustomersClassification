import os
import sys
from dataclasses import dataclass

from xgboost import XGBClassifier
from sklearn.metrics import recall_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            params = {
                    "colsample_bytree": 0.6109547429295861,
                    "learning_rate": 0.019161672175990796,
                    "max_depth": 8,
                    "min_child_weight": 2,
                    "n_estimators": 153,
                    "subsample": 0.9788478159629733
                }
            best_model = XGBClassifier(**params)

            best_model.fit(X_train, y_train)
            best_model_score = recall_score(y_train, best_model.predict(X_train))

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            recall = recall_score(y_test, predicted)
            return recall
            
        except Exception as e:
            raise CustomException(e,sys)