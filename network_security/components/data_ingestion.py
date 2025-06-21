from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.entity.config_entity import DataIngestionConfig
from network_security.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import numpy as np
import pandas as pd
import pymongo
from typing import List
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            # Get the data ingestion configuration
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    def export_collection_as_dataframe(self):
        """
        Reads a MongoDB collection and converts it into a clean Pandas DataFrame.

        Workflow:
        - Connects to the MongoDB instance using the configured database and collection.
        - Reads all documents from the collection and converts them into a DataFrame.
        - Drops the MongoDB `_id` column if present.
        - Replaces any string "na" with actual NaN values for proper handling of missing data.

        Returns:
            pd.DataFrame: The cleaned DataFrame containing data from the MongoDB collection.

        Raises:
            NetworkSecurityException: If any error occurs during the MongoDB read or transformation process.
        """
        try:
            # Get the database and collection names from the configuration
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            # Create MongoDB client and access the specified collection
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]

            # Load data from MongoDB into a DataFrame
            df = pd.DataFrame(list(collection.find()))

            # Drop the '_id' column if it exists
            if "_id" in list(df.columns):
                df = df.drop(columns=["_id"], axis=1)

            # Replace string "na" with np.nan for proper handling of missing values
            df.replace({"na": np.nan}, inplace=True)
            
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e 
    
    def export_data_to_feature_store(self, dataframe: pd.DataFrame):
        """
        Exports a Pandas DataFrame to a CSV file in the configured feature store directory.

        Workflow:
        - Retrieves the path to the feature store file.
        - Ensures the directory exists (creates it if necessary).
        - Saves the DataFrame to a CSV file with headers and no index column.

        Args:
            dataframe (pd.DataFrame): The DataFrame to be saved.

        Returns:
            pd.DataFrame: The same DataFrame, returned for chaining or reuse.

        Raises:
            NetworkSecurityException: If any error occurs while writing to the file system.
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            # Create the feature store directory 
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Export the DataFrame to CSV
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        """
        Splits the DataFrame into training and testing sets based on the configured split ratio.

        Workflow:
        - Uses sklearn's train_test_split to divide the DataFrame into training and testing sets.
        - The split is performed based on the specified ratio in the configuration.

        Args:
            dataframe (pd.DataFrame): The DataFrame to be split.

        Raises:
            NetworkSecurityException: If any error occurs during the data splitting process.
        """
        try:
            # Perform train-test split
            train_set, test_set = train_test_split(
                dataframe, 
                test_size = self.data_ingestion_config.train_test_split_ratio, 
            )

            # Log the information about the split
            logging.info("Perfomed train test split on the dataframe")
            logging.info("Exited the split_data_as_train_test method of DataIngestion class")

            # Make directory for training and testing files
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)

            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting the train and test data to the respective file paths")
            
            # Export the train and test sets to CSV files
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info("Exported the train and test data to the respective file paths")
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_ingestion(self):
        """ 
        Initiates the data ingestion process by exporting data from MongoDB, saving it to the feature store,
        and splitting it into training and testing datasets.

        Workflow:
        - Reads data from the MongoDB collection and converts it into a DataFrame.
        - Exports the DataFrame to a CSV file in the feature store directory.
        - Splits the DataFrame into training and testing sets based on the configured split ratio.

        Returns:
            DataIngestionArtifact: An artifact containing paths to the training and testing files.

        Raises:
            NetworkSecurityException: If any error occurs during the data ingestion process.
        """
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_to_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            dataingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            return dataingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e