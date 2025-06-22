import os, sys
from urllib.parse import urlparse

from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

from network_security.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from network_security.entity.config_entity import ModelTrainerConfig

from network_security.utils.main_utils.utils import save_object, load_object
from network_security.utils.main_utils.utils import load_numpy_array_data, evaluate_models

from network_security.utils.ml_utils.model.estimator import NetworkModel
from network_security.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

import mlflow
import dagshub
dagshub.init(repo_owner='tranminhanh1512', repo_name='NetworkSecurity', mlflow=True)

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        """
        Initialize the ModelTrainer with configuration and data transformation artifacts.

        :param model_trainer_config: Configuration for model training.
        :param data_transformation_artifact: Artifacts
        :raises NetworkSecurityException: If there is an error during initialization.
        """
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classification_metric):
        with mlflow.start_run():
            f1_score = classification_metric.f1_score
            precision_score = classification_metric.precision_score
            recall_score = classification_metric.recall_score

            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision", precision_score)
            mlflow.log_metric("recall", recall_score)
            mlflow.sklearn.log_model(best_model, "model")

    def train_model(self, X_train, y_train, X_test, y_test):
        #mlflow.set_registry_uri("https://dagshub.com/tranminhanh1512/NetworkSecurity.mlflow")
        #tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        models = {
            "Random Forest": RandomForestClassifier(verbose = 1),
            "Decision Tree": DecisionTreeClassifier(),
             "Gradient Boosting": GradientBoostingClassifier(verbose = 1),
            "Logistic Regression": LogisticRegression(verbose = 1),
            "AdaBoost": AdaBoostClassifier()
        }
       
        params = {
           "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                'splitter':['best','random'],
                'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                'criterion':['gini', 'entropy', 'log_loss'],
                'max_features':['sqrt','log2',None],
                'n_estimators': [128,256]
            },
            "Gradient Boosting":{
                'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05],
                'subsample':[0.7,0.8,0.9],
                'criterion':['squared_error', 'friedman_mse'],
                'max_features':['auto','sqrt','log2'],
                'n_estimators': [64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [64,128,256]
            }
        }

        model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                             models=models, params=params)
        
        # Select the best model score from the report
        best_model_score = max(sorted(model_report.values()))

        # Select the best model name based on the best score
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]

        # Predict using the best model
        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

        # Track the experiment with MLFlow
        self.track_mlflow(best_model, classification_train_metric)

        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        self.track_mlflow(best_model, classification_test_metric)
        
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        Network_Model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=Network_Model)

        save_object("final_models/model.pkl", best_model)

        # Model Trainer Artifact
        model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path, 
                                                    train_metric_artifact=classification_train_metric,
                                                    test_metric_artifact=classification_test_metric
                                                    )

        logging.info(f"Model trainer artifact: {ModelTrainerArtifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiates the model training process.

        :return: ModelTrainerArtifact containing the trained model and metrics.
        :raises NetworkSecurityException: If there is an error during model training.
        """
        try:
            # Load the transformed training and testing data
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_array = load_numpy_array_data(file_path=train_file_path)
            test_array = load_numpy_array_data(file_path=test_file_path)

            # Split the data into features and target variable
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return model_trainer_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)