# Credit Card Customer Churn Prediction

## Project Overview
This project aims to predict customer churn in a credit card business using machine learning. By analyzing customer demographics, transaction history, and account details, we can determine the likelihood of a customer leaving the service.

## Features
The dataset includes the following key features:
- **Customer Age**
- **Gender** (M/F)
- **Dependent Count**
- **Education Level** (High School, Graduate, Uneducated, Unknown, College, Post-Graduate, Doctorate)
- **Marital Status**
- **Income Category** ($60K - $80K, Less than $40K, $80K - $120K, $40K - $60K, $120K+, Unknown)
- **Card Category** (Blue, Gold, Silver, Platinum)
- **Months on Book** (Number of months the customer has been with the bank)
- **Total Relationship Count**
- **Credit Limit**
- **Total Revolving Balance**
- **Total Transaction Amount**
- **Total Transaction Count**
- **And more...**

## How to Run the Project
### 1. Set up the Environment
Create a new conda environment and install the required dependencies:
```bash
conda create --name credit_churn_env python=3.8 -y
conda activate credit_churn_env
pip install -r requirements.txt
```

### 2. Train the Model
To train the model, execute the following command:
```bash
python data_ingestion.py
```

### 3. Run the Application
To start the Flask web application, run:
```bash
python app.py
```
This will launch the application, and you can access it via `http://127.0.0.1:5000/` in your browser.

## Project Structure
```
project_directory/
│── artifacts/
│   ├── data.csv
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── test.csv
│   ├── train.csv
│
│── notebook/
│   ├── 1.EDA.ipynb
│   ├── 2.MODEL_TRAINER.ipynb
│   ├── data/
│   │   ├── credit_card_churn.csv
│
│── template/
│   ├── home.html
│   ├── index.html
│
│── src/
│   ├── __init__.py
│   ├── exception.py
│   ├── logger.py
│   ├── utils.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── predict_pipeline.py
│   │   ├── train_pipeline.py
│
│── requirements.txt
│── README.md
│── setup.py
│── app.py
```

## API Endpoints
- `/`: Home page (Form to input customer details)
- `/predict`: Endpoint to predict customer churn

## Future Improvements
- Add more feature engineering techniques
- Implement additional machine learning models
- Deploy the model as a REST API
- Improve UI/UX of the web application

## Author
- **Kerem Hüzmeli**

