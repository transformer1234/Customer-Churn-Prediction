# model.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_data(path):
    df = pd.read_csv(path, index_col = 'customerID')
    return df


def preprocess(df):
    df.replace(" ", None, inplace=True)
    df.dropna(inplace=True)
    df["TotalCharges"] = df["TotalCharges"].astype(float)
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col])
    return df


def train_model(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred)
    }

    return model, results


if __name__ == '__main__':
    df = load_data("./data/Telco-Customer-Churn.csv")
    df = preprocess(df)
    model, results = train_model(df)

    print(results)

