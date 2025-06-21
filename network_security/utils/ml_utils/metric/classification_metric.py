from network_security.entity.artifact_entity import ClassificationMetricArtifact
from network_security.exception.exception import NetworkSecurityException

from sklearn.metrics import precision_score, recall_score, f1_score

def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    """
    Calculate classification metrics and return them in a ClassificationMetricArtifact.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: ClassificationMetricArtifact containing precision, recall, and F1 score.
    """
    try:
        model_f1_score = f1_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)

        classification_metric_artifact = ClassificationMetricArtifact(
            f1_score = model_f1_score,
            precision_score = model_precision_score,
            recall_score = model_recall_score
        )

        return classification_metric_artifact
    except Exception as e:
        raise NetworkSecurityException(e) 