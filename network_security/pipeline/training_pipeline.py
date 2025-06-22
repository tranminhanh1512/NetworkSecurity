import os, sys

from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

from network_security.components.data_ingestion import DataIngestion
from network_security.components.data_validation import DataValidation
from network_security.components.data_transformation import DataTransformation
from network_security.components.model_trainer import ModelTrainer

from network_security.entity.config_entity import(
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainerConfig
)

from network_security.entity.artifact_entity import(
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from network_security.constants.training_pipeline import TRAINING_BUCKET_NAME, SAVED_MODEL_DIR
from network_security.cloud.s3_syncer import S3Sync

import sys

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()
    
    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info(f"Start Data Ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
            logging.info("Initiate the data validation process")
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Data validation completed successfully")
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            logging.info("Initiate the data transformation process")
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact, data_transformation_config=data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Data transformation completed successfully")
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            self.model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            logging.info("successfully load artifact")
            self.s3_sync.sync_folder_to_s3(folder = self.training_pipeline_config.artifact_dir,aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def sync_saved_model_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            logging.info("successfully load model")
            self.s3_sync.sync_folder_to_s3(folder = self.training_pipeline_config.model_dir,aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)