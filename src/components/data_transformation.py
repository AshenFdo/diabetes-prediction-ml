import os
import sys
from src.logger import logger as LOG
from src.exceptions import CustomException
from dataclasses import dataclass
from src.utils import save_object

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer # for handling missing values

from sklearn.model_selection import train_test_split

@dataclass
class DataTransformationConfig:
    '''Data Transformation Configurations
    
    This class defines the configuration for data transformation, including the file path for saving the preprocessor object.
    '''
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    '''Data Transformation Class
    
    This class is responsible for performing data transformation tasks, including creating a preprocessor object and applying it to the training and testing datasets.
    '''
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''Get Data Transformer Object
        
        This method creates and returns a preprocessor object that includes pipelines for numerical and categorical features.
        
        Returns:
            preprocessor (ColumnTransformer): A preprocessor object that can be used to transform data.
        '''
        try:
            numeric_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
            categorical_features = ['gender', 'smoking_history']

            # Numerical pipeline
            numeric_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            LOG.info("Numerical and categorical pipelines created successfully.")

            preprocesser =ColumnTransformer(transformers=[
                ('num_pipeline', numeric_pipeline, numeric_features),
                ('cat_pipeline', categorical_pipeline, categorical_features)
            ])

            return preprocesser

        except Exception as e:
            LOG.error("Error in get_data_transformer_object method of DataTransformation class")
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        '''Initiate Data Transformation
        
        This method reads the training and testing datasets, applies the preprocessor object to transform the data, and saves the preprocessor object for future use.
        
        Args:
            train_path (str): The file path for the training dataset.
            test_path (str): The file path for the testing dataset.
        '''

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            LOG.info("Read train and test data completed")

            LOG.info("Obtaining preprocessor object")
            preprocessor_obj = self.get_data_transformer_object()

            targer_column_name = 'diabetes'

            input_feature_train_df = train_df.drop(columns=[targer_column_name])
            target_feature_train_df = train_df[targer_column_name]

            input_feature_test_df = test_df.drop(columns=[targer_column_name])
            target_feature_test_df = test_df[targer_column_name]

            LOG.info("Applying preprocessor object on training and testing datasets")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            LOG.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
 


        except Exception as e:
            LOG.error("Error in initiate_data_transformation method of DataTransformation class")
            raise CustomException(e, sys)

