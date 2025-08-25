import logging
import os
from pyspark.sql import SparkSession
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler, Imputer, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when, split, lit
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Define the local path to save the fitted preprocessing pipeline
PREPROCESSING_PIPELINE_PATH = "./preprocessing_pipeline_model"

def create_spark_session():
    """
    Initializes and returns a SparkSession with necessary configurations.
    """
    spark = SparkSession.builder \
        .appName("TitanicMLOpsPipeline") \
        .config("spark.driver.extraJavaOptions", "--add-opens=java.base/javax.security.auth.Subject=ALL-UNNAMED") \
        .getOrCreate()
    return spark

def get_full_pipeline():
    """
    Builds and returns a complete Spark ML Pipeline including preprocessing and feature engineering.
    """
    # Handle Missing Values using Imputer
    imputer = Imputer(
        inputCols=['Age', 'Fare'],
        outputCols=['Age_imputed', 'Fare_imputed']
    ).setStrategy("mean")

    # Categorical Feature Encoding
    gender_indexer = StringIndexer(inputCol="Sex", outputCol="SexIndex", handleInvalid="skip")
    gender_encoder = OneHotEncoder(inputCol="SexIndex", outputCol="SexVec")
    embarked_indexer = StringIndexer(inputCol="Embarked", outputCol="EmbarkedIndex", handleInvalid="skip")
    embarked_encoder = OneHotEncoder(inputCol="EmbarkedIndex", outputCol="EmbarkedVec")
    title_indexer = StringIndexer(inputCol="Title", outputCol="TitleIndex", handleInvalid="skip")
    title_encoder = OneHotEncoder(inputCol="TitleIndex", outputCol="TitleVec")

    # Feature Assembly: Combines all feature columns into a single vector
    assembler = VectorAssembler(
        inputCols=[
            "Pclass", "Age_imputed", "SibSp", "Parch", "Fare_imputed",
            "SexVec", "EmbarkedVec", "TitleVec"
        ],
        outputCol="features"
    )

    # Create a complete Spark ML Pipeline
    pipeline = Pipeline(stages=[
        imputer,
        gender_indexer,
        gender_encoder,
        embarked_indexer,
        embarked_encoder,
        title_indexer,
        title_encoder,
        assembler
    ])
    
    return pipeline

def build_model(model_name="logistic_regression"):
    """
    Builds and returns a specified machine learning model for tuning.
    """
    logging.info(f"Building base model for tuning: {model_name}")
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

def main():
    """
    Main function to run the complete MLOps pipeline.
    """
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting the MLOps pipeline run.")

    try:
        spark = create_spark_session()
        logging.info("Spark session created successfully.")
    except Exception as e:
        logging.error(f"Failed to create Spark session: {e}")
        return
    
    # --- DVC Integration ---
    try:
        logging.info("Pulling data from DVC.")
        os.system("dvc pull")
        logging.info("Data pull from DVC completed.")
    except Exception as e:
        logging.error(f"Failed to pull data from DVC: {e}")
        spark.stop()
        return

    # Step 1: Data Ingestion and Initial Cleaning
    try:
        logging.info("Starting data ingestion.")
        # Ingest both train and test data
        train_df = spark.read.csv("data/train.csv", header=True, inferSchema=True)
        test_df = spark.read.csv("data/test.csv", header=True, inferSchema=True)
        
        train_df = train_df.drop('Cabin', 'Ticket').fillna({'Embarked': 'S'})
        test_df = test_df.drop('Cabin', 'Ticket').fillna({'Embarked': 'S'}) # Apply the same cleaning to the test set
        
        # Add the 'Title' feature to both datasets before preprocessing
        train_df = train_df.withColumn("Title", when(split(col("Name"), r"\s+")[1].isin("Mr.", "Mrs.", "Miss.", "Master."), split(col("Name"), r"\s+")[1]).otherwise(lit("Other")))
        test_df = test_df.withColumn("Title", when(split(col("Name"), r"\s+")[1].isin("Mr.", "Mrs.", "Miss.", "Master."), split(col("Name"), r"\s+")[1]).otherwise(lit("Other")))
        
        logging.info("Data ingested and initially cleaned.")
    except Exception as e:
        logging.error(f"Data ingestion failed: {e}")
        spark.stop()
        return

    # Step 2: Data Preprocessing and Feature Engineering
    try:
        logging.info("Starting data preprocessing and feature engineering.")
        
        # Build and fit the full pipeline ONCE on the training data
        pipeline = get_full_pipeline()
        pipeline_model = pipeline.fit(train_df)
        
        # Save the fitted preprocessing pipeline model for deployment
        pipeline_model.write().overwrite().save(PREPROCESSING_PIPELINE_PATH)
        logging.info(f"Fitted preprocessing pipeline model saved to {PREPROCESSING_PIPELINE_PATH}")

        # Transform both the training and test data using the fitted pipeline
        processed_train_df = pipeline_model.transform(train_df)
        processed_test_df = pipeline_model.transform(test_df)
        
        logging.info("Data preprocessing and feature engineering completed.")
    except Exception as e:
        logging.error(f"Preprocessing/Feature Engineering failed: {e}")
        spark.stop()
        return

    # Step 3: Train and Evaluate Multiple Models
    logging.info("Starting multi-model training and evaluation.")

    # Split the processed training data for cross-validation and a final validation
    (training_data, validation_data) = processed_train_df.randomSplit([0.8, 0.2], seed=42)

    # Define the models to train
    models_to_train = ["logistic_regression", "decision_tree", "random_forest"]
    
    best_overall_auc = -1
    best_overall_model = None
    best_overall_model_name = ""

    for model_name in models_to_train:
        with mlflow.start_run(run_name=f"Training_{model_name}"):
            try:
                # Model Building and Parameter Grid
                base_model = build_model(model_name)
                param_grid = build_param_grid(model_name)
                
                mlflow.log_param("model_name", model_name)

                # Cross-validator setup
                evaluator = BinaryClassificationEvaluator(labelCol="Survived", metricName="areaUnderROC")
                cross_validator = CrossValidator(
                    estimator=base_model,
                    estimatorParamMaps=param_grid,
                    evaluator=evaluator,
                    numFolds=5
                )

                # Fit the cross-validator
                cv_model = cross_validator.fit(training_data)
                best_model = cv_model.bestModel
                
                logging.info(f"Hyperparameter tuning for {model_name} complete.")

                # Model Evaluation on the validation set
                predictions = best_model.transform(validation_data)
                auc = evaluator.evaluate(predictions)
                
                mlflow.log_metric("validation_auc", auc)
                logging.info(f"Final Model {model_name} AUC on validation set: {auc}")
                
                # Check if this is the best model so far
                if auc > best_overall_auc:
                    best_overall_auc = auc
                    best_overall_model = best_model
                    best_overall_model_name = model_name
                    
            except Exception as e:
                logging.error(f"Model training/tuning for {model_name} failed: {e}")
                mlflow.end_run("FAILED")
    
    # After the loop, log the single best model to MLflow
    if best_overall_model:
        with mlflow.start_run(run_name="Final_Best_Model_Eval"):
            logging.info(f"The best overall model is: {best_overall_model_name} with AUC: {best_overall_auc}")
            mlflow.log_param("best_overall_model_name", best_overall_model_name)
            mlflow.log_metric("best_overall_auc", best_overall_auc)
            
            # Use the best model to make predictions on the processed test data
            test_predictions = best_overall_model.transform(processed_test_df)
            
            # The test set does not have a 'Survived' column, so we cannot evaluate it.
            # Instead, we will log the predictions to a file.
            test_predictions_to_save = test_predictions.select("PassengerId", "prediction")
            test_predictions_to_save.write.csv("./final_predictions.csv", header=True, mode="overwrite")
            logging.info("Final predictions for the test set saved to ./final_predictions.csv")

            # Log the best model to MLflow
            mlflow.spark.log_model(
                spark_model=best_overall_model,
                artifact_path="best-model",
                registered_model_name="TitanicBestModel"
            )
            logging.info("Best overall model logged to MLflow.")

            # --- MLflow Model Registry Stage Transition ---
            client = MlflowClient()
            latest_version = client.get_latest_versions("TitanicBestModel", stages=["None"])[0]
            client.transition_model_version_stage(
                name="TitanicBestModel",
                version=latest_version.version,
                stage="Staging"
            )
            logging.info(f"Model version {latest_version.version} transitioned to Staging.")

    # Step 4: Stop Spark session
    spark.stop()
    logging.info("Pipeline run finished.")

    # Step 4: Stop Spark session
    spark.stop()
    logging.info("Pipeline run finished.")

if __name__ == "__main__":
    main()