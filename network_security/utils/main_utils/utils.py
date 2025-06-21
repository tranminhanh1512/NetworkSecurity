import yaml
import os, sys
import numpy as np
import dill
import pickle

from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its content as a dictionary.
    
    :param file_path: Path to the YAML file.
    :return: Dictionary containing the YAML file content.
    """
    try:
        with open(file_path, 'rb') as file:
            content = yaml.safe_load(file)
        return content
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes content to a YAML file.
    
    :param file_path: Path to the YAML file.
    :param content: Content to write to the file.
    :param replace: If True, replaces the existing file; otherwise, appends.
    """
    try:
        if replace:
            if os.path.exists(file_path):
               os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
           yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) 

def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Saves a NumPy array to a file.
    
    :param file_path: Path to the file where the array will be saved.
    :param array: NumPy array to save.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            np.save(file, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) 

def save_object(file_path: str, obj: object) -> None:
    """
    Saves an object to a file using picke.
    
    :param file_path: Path to the file where the object will be saved.
    :param obj: Object to save.
    """
    try:
        logging.info(f"Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
           pickle.dump(obj, file)
        logging.info(f"Exited the save_object method of MainUtils class")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def load_object(file_path: str) -> object:
    """
    Loads an object from a file using pickle.
    
    :param file_path: Path to the file from which the object will be loaded.
    :return: Loaded object.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, 'rb') as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) 

def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Loads a NumPy array from a file.
    
    :param file_path: Path to the file from which the array will be loaded.
    :return: Loaded NumPy array.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv = 3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        
        return report
    except Exception as e:
        raise NetworkSecurityException(e, sys)