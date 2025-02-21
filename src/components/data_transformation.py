import os
import sys
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from imblearn.over_sampling import SMOTE


from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "proprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["Customer_Age", "Dependent_count", "Months_on_book",
                                 "Total_Relationship_Count", "Months_Inactive_12_mon",
                                 "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal",
                                 "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt",
                                 "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio"]
            categorical_columns_onehot = ["Gender", "Marital_Status"]
            categorical_columns_ordinal = ["Education_Level", "Income_Category", "Card_Category"]

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline_onehot = Pipeline(
                steps=[
                    ("onehotEncoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline_ordinal = Pipeline(
                steps=[
                    ("ordinalEncoder", OrdinalEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline_onehot", cat_pipeline_onehot, categorical_columns_onehot),
                    ("cat_pipeline_ordinal", cat_pipeline_ordinal, categorical_columns_ordinal)
                ]
            )

            return preprocessor
             
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")

            train_df = train_df.drop(["CLIENTNUM", 
                                       "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1", 
                                       "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"], axis=1)
            test_df = test_df.drop(["CLIENTNUM", 
                                     "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1", 
                                     "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"], axis=1)
            logging.info("Dropping unnecessary columns")
            
            train_df["Attrition_Flag"] = np.where(train_df["Attrition_Flag"] == "Attrited Customer", 1, 0)
            test_df["Attrition_Flag"] = np.where(test_df["Attrition_Flag"] == "Attrited Customer", 1, 0)
            logging.info("Altering the target columns")
            
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = "Attrition_Flag"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Generating artificial data")
            sm = SMOTE()
            input_feature_train_arr, target_feature_train_df = sm.fit_resample(input_feature_train_arr, target_feature_train_df)       

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("Saved preprocessing object")
            
            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj 
            )
            
            return (
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
