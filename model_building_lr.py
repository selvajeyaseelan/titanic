# In model_building_lr.py
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder
import logging

def build_model(model_name="logistic_regression"):
    """
    Builds and returns a specified machine learning model for tuning.
    """
    logging.info(f"Building base model for tuning: {model_name}")
    try:
        if model_name == "logistic_regression":
            model = LogisticRegression(labelCol="Survived", featuresCol="features")
        elif model_name == "decision_tree":
            model = DecisionTreeClassifier(labelCol="Survived", featuresCol="features")
        elif model_name == "random_forest":
            model = RandomForestClassifier(labelCol="Survived", featuresCol="features")
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        logging.info("Base model built successfully.")
        return model
    except Exception as e:
        logging.error(f"Error building model: {e}")
        raise

def build_param_grid(model_name="logistic_regression"):
    """
    Builds and returns a parameter grid for hyperparameter tuning.
    """
    logging.info(f"Building parameter grid for {model_name}.")
    
    if model_name == "logistic_regression":
        param_grid = ParamGridBuilder() \
            .addGrid(LogisticRegression.regParam, [0.01, 0.1, 0.5]) \
            .addGrid(LogisticRegression.elasticNetParam, [0.0, 0.5, 1.0]) \
            .build()
    elif model_name == "decision_tree":
        param_grid = ParamGridBuilder() \
            .addGrid(DecisionTreeClassifier.maxDepth, [2, 5, 10]) \
            .addGrid(DecisionTreeClassifier.minInstancesPerNode, [1, 5, 10]) \
            .build()
    elif model_name == "random_forest":
        param_grid = ParamGridBuilder() \
            .addGrid(RandomForestClassifier.numTrees, [10, 20, 50]) \
            .addGrid(RandomForestClassifier.maxDepth, [5, 10]) \
            .build()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
        
    logging.info("Parameter grid built successfully.")
    return param_grid