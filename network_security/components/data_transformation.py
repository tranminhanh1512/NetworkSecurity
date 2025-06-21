import os, sys
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from network_security.constants.training_pipeline import TARGET_COLUMN
from network_security.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from network_security.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from network_security.entity.config_entity import DataTransformationConfig
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

from network_security.utils.main_utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys) 
    
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Reads data from a CSV file and returns it as a DataFrame.
        
        :param file_path: Path to the CSV file.
        :return: DataFrame containing the data.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(cls) -> Pipeline:
        """"
        Creates a data transformation pipeline with KNN imputer.
        
        :return: A Pipeline object for data transformation.
        """
        logging.info("Entered the get_data_transformer_object method of DataTransformation class")
        try:
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f"Initialize KNNImputer with parameters: {DATA_TRANSFORMATION_IMPUTER_PARAMS}")
            processor: Pipeline = Pipeline([("imputer", imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered the initiate_data_transformation method of DataTransformation class")
        try:
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Training dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1, 0)

            # Testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1, 0)

            # Initialize the transformation pipeline
            preprocessor = self.get_data_transformer_object()

            # Fit the preprocessor on the data
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            # Concatenate the transformed features with the target feature
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            # Save the transformed data
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_file_path,
                array=train_arr,
            )
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_file_path,
                array=test_arr,
            )

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=preprocessor_object,
            )

            # Prepare the artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )

            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)