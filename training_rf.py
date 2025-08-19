
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark

def create_spark_session():
    """
    Initializes and returns a SparkSession with necessary configurations.
    """
    
    spark = SparkSession.builder \
        .appName("TitanicModelTraining") \
        .config("spark.driver.extraJavaOptions", "--add-opens=java.base/javax.security.auth.Subject=ALL-UNNAMED") \
        .getOrCreate()
    return spark

def train_and_evaluate_random_forest(spark, train_path):
    """
    Loads data, trains a Random Forest model, and evaluates it.
    Splits the training data for evaluation.
    Logs parameters, metrics, and the model to MLflow.
    """
    # Load preprocessed training data
    train_df = spark.read.parquet(train_path)

    # Split the data into a training set and a validation set
    (training_data, validation_data) = train_df.randomSplit([0.8, 0.2], seed=42)

    # Define model parameters
    num_trees = 20
    max_depth = 5
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("num_trees", num_trees)
        mlflow.log_param("max_depth", max_depth)

        # Initialize and train the Random Forest model
        rfc = RandomForestClassifier(
            numTrees=num_trees,
            maxDepth=max_depth,
            labelCol="Survived",
            featuresCol="features"
        )
        rfc_model = rfc.fit(training_data)

        # Make predictions on the validation data
        predictions = rfc_model.transform(validation_data)

        # Evaluate the model
        evaluator = BinaryClassificationEvaluator(labelCol="Survived", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        auc = evaluator.evaluate(predictions)

        # Log metrics
        mlflow.log_metric("AUC", auc)

        # Log the trained model to MLflow
        mlflow.spark.log_model(
            spark_model=rfc_model,
            artifact_path="random-forest-model",
            registered_model_name="TitanicRandomForestModel"
        )
    
    print(f"Random Forest training complete. AUC: {auc}")
    return rfc_model, auc

if __name__ == "__main__":
    spark = create_spark_session()
    
    # Define the path to your preprocessed training data
    train_data_path = "data/processed/train.parquet"
    
    # Train and evaluate the model
    train_and_evaluate_random_forest(spark, train_data_path)
    
    # Stop the Spark session
    spark.stop()