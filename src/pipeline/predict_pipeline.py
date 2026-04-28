import sys
import pandas as pd


from src.logger import logger as LOG
from src.utils import load_object
from src.exceptions import CustomException


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            model = load_object(file_path=model_path)

            preprocessor_path = "artifacts/preprocessor.pkl"
            preprocessor = load_object(file_path=preprocessor_path)

            scaled_data = preprocessor.transform(features)

            pred = model.predict(scaled_data)

            return pred
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,
                 gender: str,
                 age: int,
                 hypertension: int,
                 heart_disease: int,
                 smoking_history: str,
                 bmi: float,
                 HbA1c_level: float,
                blood_glucose_level: float):
        
        self.gender = gender
        self.age = age
        self.hypertension = hypertension
        self.heart_disease = heart_disease
        self.smoking_history = smoking_history
        self.bmi = bmi
        self.HbA1c_level = HbA1c_level
        self.blood_glucose_level = blood_glucose_level


    def get_data_as_dataframe(self):

        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "age": [self.age],
                "hypertension": [self.hypertension],
                "heart_disease": [self.heart_disease],
                "smoking_history": [self.smoking_history],
                "bmi": [self.bmi],
                "HbA1c_level": [self.HbA1c_level],
                "blood_glucose_level": [self.blood_glucose_level]
            }

            LOG.info(custom_data_input_dict)

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
        
        
        


# if __name__ == "__main__":
#     data = CustomData(
#         gender="Male",
#         age=45,
#         hypertension=1,
#         heart_disease=0,
#         smoking_history="Never",
#         bmi=28.5,
#         HbA1c_level=6.5,
#         blood_glucose_level=180
#     )
#     df = data.get_data_as_dataframe()
    

#     prediction_pipeline = PredictPipeline()
#     pred = prediction_pipeline.predict(df)
#     print(f"Predicted class: {pred}")



