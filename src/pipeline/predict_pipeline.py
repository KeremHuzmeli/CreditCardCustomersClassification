import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:

            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\proprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self, 
        Customer_Age: int, 
        Gender, 
        Dependent_count: int, 
        Education_Level, 
        Marital_Status: str, 
        Income_Category, 
        Card_Category, 
        Months_on_book: int,
        Total_Relationship_Count: int, 
        Months_Inactive_12_mon: int,
        Contacts_Count_12_mon: int, 
        Credit_Limit: float, 
        Total_Revolving_Bal: int,
        Avg_Open_To_Buy: float, 
        Total_Amt_Chng_Q4_Q1: float, 
        Total_Trans_Amt: int,
        Total_Trans_Ct: int, 
        Total_Ct_Chng_Q4_Q1: float, 
        Avg_Utilization_Ratio: float
    ):
        self.Customer_Age = Customer_Age
        self.Gender = Gender
        self.Dependent_count = Dependent_count
        self.Education_Level = Education_Level
        self.Marital_Status = Marital_Status
        self.Income_Category = Income_Category
        self.Card_Category = Card_Category
        self.Months_on_book = Months_on_book
        self.Total_Relationship_Count = Total_Relationship_Count
        self.Months_Inactive_12_mon = Months_Inactive_12_mon
        self.Contacts_Count_12_mon = Contacts_Count_12_mon
        self.Credit_Limit = Credit_Limit
        self.Total_Revolving_Bal = Total_Revolving_Bal
        self.Avg_Open_To_Buy = Avg_Open_To_Buy
        self.Total_Amt_Chng_Q4_Q1 = Total_Amt_Chng_Q4_Q1
        self.Total_Trans_Amt = Total_Trans_Amt
        self.Total_Trans_Ct = Total_Trans_Ct
        self.Total_Ct_Chng_Q4_Q1 = Total_Ct_Chng_Q4_Q1
        self.Avg_Utilization_Ratio = Avg_Utilization_Ratio

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Customer_Age": [self.Customer_Age],
                "Gender": [self.Gender],
                "Dependent_count": [self.Dependent_count],
                "Education_Level": [self.Education_Level],
                "Marital_Status": [self.Marital_Status],
                "Income_Category": [self.Income_Category],
                "Card_Category": [self.Card_Category],
                "Months_on_book": [self.Months_on_book],
                "Total_Relationship_Count": [self.Total_Relationship_Count],
                "Months_Inactive_12_mon": [self.Months_Inactive_12_mon],
                "Contacts_Count_12_mon": [self.Contacts_Count_12_mon],
                "Credit_Limit": [self.Credit_Limit],
                "Total_Revolving_Bal": [self.Total_Revolving_Bal],
                "Avg_Open_To_Buy": [self.Avg_Open_To_Buy],
                "Total_Amt_Chng_Q4_Q1": [self.Total_Amt_Chng_Q4_Q1],
                "Total_Trans_Amt": [self.Total_Trans_Amt],
                "Total_Trans_Ct": [self.Total_Trans_Ct],
                "Total_Ct_Chng_Q4_Q1": [self.Total_Ct_Chng_Q4_Q1],
                "Avg_Utilization_Ratio": [self.Avg_Utilization_Ratio]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)








