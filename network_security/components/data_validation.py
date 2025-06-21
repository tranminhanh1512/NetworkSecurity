from network_security.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from network_security.entity.config_entity import DataValidationConfig
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.constants.training_pipeline import SCHEMA_FILE_PATH
from network_security.utils.main_utils.utils import read_yaml_file, write_yaml_file

from scipy.stats import ks_2samp
import pandas as pd
import os, sys

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._scheme_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys) 
    
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Reads a CSV file and returns a DataFrame.
        
        :param file_path: Path to the CSV file.
        :return: DataFrame containing the data.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys) 
    
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validates the number of columns in the DataFrame against the expected schema.
        
        :param dataframe: DataFrame to validate.
        :return: True if the number of columns matches the schema, False otherwise.
        """
        try:
            number_of_columns = len(self._scheme_config)
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Actual number of columns: {len(dataframe.columns)}")

            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drif(self, base_df, current_df, threshold = 0.05) -> bool:
        """
        Detects dataset drift using the Kolmogorov-Smirnov test.
        
        :param base_df: Base DataFrame (e.g., training data).
        :param current_df: Current DataFrame (e.g., testing data).
        :param threshold: Significance level for the KS test.
        :return: True if drift is detected, False otherwise.
        """
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({column: {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found
                }})
            
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            # Create directory if it does not exist
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            # Save the drift report as a yaml file
            write_yaml_file(file_path = drift_report_file_path, content = report)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            # Initialize error message
            error_message = "" 

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            
            # Read the training and testing data
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            # Validate the number of columns in both datasets
            status = self.validate_number_of_columns(train_dataframe)
            if not status:
                error_message = f"{error_message} Train dataframe does not contain the required number of columns."
            status = self.validate_number_of_columns(test_dataframe)
            if not status:
                error_message = f"{error_message} Test dataframe does not contain the required number of columns."
            
            # Validate the numeric columns existence
            numeric_columns = self._scheme_config["numerical_columns"]
            for column in numeric_columns:
                if column not in train_dataframe.columns:
                    error_message = f"{error_message} Column '{column}' is missing in the train dataframe."
                if column not in test_dataframe.columns:
                    error_message = f"{error_message} Column '{column}' is missing in the test dataframe."

            # Check data drift
            status = self.detect_dataset_drif(base_df=train_dataframe, current_df=test_dataframe)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Save the validated train dataframes
            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            data_validation_artifact = DataValidationArtifact(
                validation_status = status,
                valid_train_file_path = self.data_ingestion_artifact.train_file_path,
                valid_test_file_path = self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path = None,
                invalid_test_file_path = None,
                drift_report_file_path = self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        