from network_security.components.data_ingestion import DataIngestion
from network_security.components.data_validation import DataValidation
from network_security.components.data_transformation import DataTransformation
from network_security.components.model_trainer import ModelTrainer

from network_security.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from network_security.entity.config_entity import TrainingPipelineConfig
from network_security.entity.config_entity import ModelTrainerConfig

from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

import sys 

if __name__ == "__main__":
    try:
        trainingpipeline_config = TrainingPipelineConfig()
        
        dataingestion_config = DataIngestionConfig(trainingpipeline_config)
        dataingestion = DataIngestion(dataingestion_config)
        logging.info("Initiate the data ingestion process")
        dataingestion_artifact = dataingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed successfully")
        print(dataingestion_artifact)
        
        data_validation_config = DataValidationConfig(trainingpipeline_config)
        data_validation = DataValidation(dataingestion_artifact, data_validation_config)
        logging.info("Initiate the data validation process")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed successfully")
        print(data_validation_artifact)
        logging.info("Initiate the data transformation process")
        
        data_transformation_config = DataTransformationConfig(trainingpipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data transformation completed successfully")
        print(data_transformation_artifact)

        logging.info("Initiate the model trainer process")
        model_trainer_config = ModelTrainerConfig(trainingpipeline_config)
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model training artifact successfully created")

    except Exception as e:
        raise NetworkSecurityException(e, sys) 