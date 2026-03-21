import os
from pyexpat import model
import sys
from dataclasses import dataclass

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier


from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score
import test
from xgboost.dask import predict

from src.logger import logger as LOG
from src.exceptions import CustomException
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            LOG.info("Splitting training and testing input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                'Logistic Regression': LogisticRegression(random_state=42),
                'Random Forest': RandomForestClassifier(random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'K-Nearest Neighbors': KNeighborsClassifier()
            }

            params = {
                'Logistic Regression': {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear']
                },
                'Random Forest': {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20]
                },
                'Decision Tree': {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                },
                'Gradient Boosting': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1]
                },
                'K-Nearest Neighbors': {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance']
                }
            }

            model_report, tuned_models = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                 models=models, params=params
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = tuned_models[best_model_name]

            LOG.info(f"Best model found on both training and testing dataset: {best_model_name} with score: {best_model_score}")
            LOG.info(f"Best model parameters: {best_model.get_params()}")


            if best_model_score < 0.6:
                raise CustomException("No best model found with score greater than 0.6")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model 
            )

            prediction = best_model.predict(X_test)

            
            test_model_score = {
                "test_model_accuracy": accuracy_score(y_test, prediction),
                "test_model_precision": precision_score(y_test, prediction, average='weighted'),
                "test_model_recall": recall_score(y_test, prediction, average='weighted'),
                "test_model_f1": f1_score(y_test, prediction, average='weighted')
            }

            
            LOG.info(f"Test Model Score: {test_model_score}")
            LOG.info(f"Model training completed successfully.")

            return test_model_score, best_model_name, best_model_score


            




            


        except Exception as e:
            LOG.error(f"Error occurred at model training stage: {e}")
            raise CustomException(e, sys)


