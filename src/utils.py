from statistics import mode

import numpy as np 
import pandas as pd
import pickle
import os
import sys

from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV


from src.exceptions import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        tuned_models = {}
        
        for model_name, model in models.items():
            param_grid = params.get(model_name, {})
            gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            y_test_pred = best_model.predict(X_test)


            acc = accuracy_score(y_test, y_test_pred)
            prec = precision_score(y_test, y_test_pred, average='weighted')
            rec = recall_score(y_test, y_test_pred, average='weighted')
            f1 = f1_score(y_test, y_test_pred, average='weighted')

            report[model_name] = {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1
            }
            tuned_models[model_name] = best_model

            best_model_name = max(report, key=lambda x: report[x]['F1 Score'])
            best_score = report[best_model_name]['F1 Score']

            print(f"🏆 Best Model: {best_model_name} with F1 Score: {best_score:.4f}")


            test_model_accuracy = accuracy_score(y_test, y_test_pred)
            report[model_name] = test_model_accuracy
            tuned_models[model_name] = best_model

            return report, tuned_models



    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
