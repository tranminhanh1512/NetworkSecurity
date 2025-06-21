import os, sys

from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

from network_security.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME

class NetworkModel:
    """
    A class to represent a machine learning model for network security.
    It encapsulates the preprocessor and model, providing a method to make predictions.

    Attributes:
        preprocessor: An object that preprocesses the input data.
        model: A trained machine learning model.
        
    Methods:
        __init__(preprocessor, model): Initializes the NetworkModel with a preprocessor and model.
        predict(X): Transforms the input data using the preprocessor and makes predictions using the model.
    """
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def predict(self, X):
        try:
            X_transformed = self.preprocessor.transform(X)
            y_hat = self.model.predict(X_transformed)
            return y_hat
        except Exception as e:
            raise NetworkSecurityException(e, sys)
