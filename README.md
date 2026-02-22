# Customer-Churn-Prediction

## 📌 Project Overview
This project predicts whether a customer will churn (Loss of customer) based on demographic, account, and service usage information. It uses a Random Forest Classifier and includes data cleaning, visualization, feature engineering, and model evaluation.

## 🧠 Skills Demonstrated
- Data Cleaning & Preprocessing  
- Categorical Encoding
- Machine Learning Model Training
- Random Forest Classifier
- Confusion Matrix & Classification Report
- Feature Importance Visualization

## 🚀 Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn

## 📂 Folder Structure
customer-churn-prediction/
│── data/
│── notebooks/
│── src/
│── README.md
│── requirements.txt

- Data folder containes dataset used.
- src folder containes "model.py" file which has complete python code.
- Notebooks folder has 2 notebooks. "churn_prediction_complete_code.ipynb" containes complete analysis. "churn_prediction.ipynb" containes code which uses model.py file as a package and does the same thing as the other notebook.
- "requirements.txt" file has package names that are used in the code.

## 📊 Model Performance
- Accuracy: ~80–85%
- Most important features: Tenure, MonthlyCharges, Contract type

## 📁 Dataset
Kaggle: Telco Customer Churn Dataset

## How to run this project
### python file
```bash
pip install -r requirements.txt
python src/model.py
```
### notebooks are in notebooks/ folder