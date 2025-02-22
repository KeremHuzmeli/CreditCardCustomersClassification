from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.utils import determine_customer
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            Customer_Age=request.form.get("customer_age"),
            Gender=request.form.get("gender"),
            Dependent_count=request.form.get("dependent_count"),
            Education_Level=request.form.get("education_level"),
            Marital_Status=request.form.get("marital_status"),
            Income_Category=request.form.get("income_category"),
            Card_Category=request.form.get("card_category"),
            Months_on_book=request.form.get("months_on_book"),
            Total_Relationship_Count=request.form.get("total_relationship_count"),
            Months_Inactive_12_mon=request.form.get("months_inactive_12_mon"),
            Contacts_Count_12_mon=request.form.get("contacts_count_12_mon"),
            Credit_Limit=request.form.get("credit_limit"),
            Total_Revolving_Bal=request.form.get("total_revolving_bal"),
            Avg_Open_To_Buy=request.form.get("avg_open_to_buy"),
            Total_Amt_Chng_Q4_Q1=request.form.get("total_amt_chng_q4_q1"),
            Total_Trans_Amt=request.form.get("total_trans_amt"),
            Total_Trans_Ct=request.form.get("total_trans_ct"),
            Total_Ct_Chng_Q4_Q1=request.form.get("total_ct_chng_q4_q1"),
            Avg_Utilization_Ratio=request.form.get("avg_utilization_ratio")
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        result = determine_customer(results[0])
        return render_template("home.html", results=result)


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)
    

