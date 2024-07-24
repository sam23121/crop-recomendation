

import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.metrics import root_mean_squared_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import mlflow
import os

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

mlflow.autolog()
mlflow_dir = "/home/sam/Documents/projects/practice/mlops/crop-recomendation"
mlflow.set_tracking_uri(os.path.join(mlflow_dir, "mlruns"))

mlflow.set_experiment("crop-recommendation")


def read(filename: str) -> pd.DataFrame:
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(filename)

    label_encoder = LabelEncoder()

    # Fit and transform the data
    df['label'] = label_encoder.fit_transform(df['label'])

    with open('../models/label_encoder.bin', 'wb') as f_out:
        pickle.dump(label_encoder, f_out)
    
    return df

def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:


    X = df.drop(['label', 'P'], axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    # Step 4: Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    
    return X_train, X_test, y_train, y_test


def train(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    


    with open('../models/log_reg.bin', 'wb') as f_out:
        pickle.dump(lr, f_out)


    return lr

    



def evaluate(X_test: np.ndarray, y_test: np.ndarray, lr_model=None) -> Tuple[int, int]:

    if lr_model == None:
        with open('../models/log_reg.bin', 'rb') as file:
            lr_model = pickle.load(file)

    y_pred = lr_model.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)
    confu_mat = confusion_matrix(y_test, y_pred)

    print(f'Accuracy: {acc_score}')
    print(f'Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}')

    return acc_score



def predict(
        feature1: float, feature2: float, 
        feature3: float, feature4: float, 
        feature5: float, feature6: float
        ) -> None:

    with open('../models/label_encoder.bin', 'rb') as f_in:
        label_encoder = pickle.load(f_in)

    with open('../models/log_reg.bin', 'rb') as f_in:
        lr = pickle.load(f_in)

    # Prepare new data for prediction
    new_data = np.array([[feature1, feature2, feature3, feature4, feature5, feature6]])

    # Make a prediction
    prediction = lr.predict(new_data)
    print("Prediction (encoded):", prediction)

    # Reverse the label encoding
    original_label = label_encoder.inverse_transform([round(prediction[0])])
    print("Prediction (original label):", original_label[0])


def objective(params: dict):
    with mlflow.start_run():
        mlflow.set_tag("model", "LogisticRegression")
        mlflow.log_params(params)

        # Instantiate the Logistic Regression model
        lr = LogisticRegression(
            C=params['C'],
            # penalty=params['penalty'],
            solver=params['solver'],
            max_iter=params['max_iter'],
            random_state=params['random_state'],
        )

        # Fit the model
        lr.fit(X_train, y_train)

        # Evaluate the model on the test set
        y_pred = lr.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        mlflow.log_metric("test_acc_score", acc_score)

    return -acc_score

    # return {'accuracy_score': -acc_score, 'status': STATUS_OK}


if __name__ == '__main__':
    df = read("../data/Crop_recommendation.csv")
    X_train, X_test, y_train, y_test = preprocess(df)

    # Define the search space
    search_space = {
        'C': hp.loguniform('C', -5, 5),
        # 'penalty': hp.choice('penalty', [None, 'l2']),
        'solver': hp.choice('solver', ['liblinear', 'lbfgs']),
        'max_iter': scope.int(hp.quniform('max_iter', 100, 1000, 100)),
        'random_state': 42
    }

    # Optimize the hyperparameters
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials()
    )

    # lr = train(X_train, y_train, X_test, y_test)
    # acc_score= evaluate(X_test, y_test, lr)
    # predict(10,	81,	10.879744, 2.002744, 3.502985, 2.935536)
    # df.sort_values()

