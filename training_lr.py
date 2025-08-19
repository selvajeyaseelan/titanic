
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
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

def train_and_evaluate_model(spark, train_path):
    """
    Loads data, trains a Logistic Regression model, and evaluates it.
    Splits the training data for evaluation.
    Logs parameters, metrics, and the model to MLflow.
    """
    # Load preprocessed training data
    train_df = spark.read.parquet(train_path)

    # Split the data into a training set and a validation set
    (training_data, validation_data) = train_df.randomSplit([0.8, 0.2], seed=42)
    
    # Define model parameters
    max_iter = 100
    reg_param = 0.01
    elastic_net_param = 0.8
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("reg_param", reg_param)
        mlflow.log_param("elastic_net_param", elastic_net_param)

        # Initialize and train the Logistic Regression model
        lr = LogisticRegression(
            maxIter=max_iter,
            regParam=reg_param,
            elasticNetParam=elastic_net_param,
            labelCol="Survived",
            featuresCol="features"
        )
        lr_model = lr.fit(training_data)  # Fit on training_data
        
        # Make predictions on the validation data
        predictions = lr_model.transform(validation_data)
        
        # Evaluate the model
        evaluator = BinaryClassificationEvaluator(labelCol="Survived", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        auc = evaluator.evaluate(predictions)
        
        # Log metrics
        mlflow.log_metric("AUC", auc)
        
        # Log the trained model to MLflow
        mlflow.spark.log_model(
            spark_model=lr_model,
            artifact_path="logistic-regression-model",
            registered_model_name="TitanicLogisticRegressionModel"
        )
        
    print(f"Model training complete. AUC: {auc}")
    return lr_model, auc

if __name__ == "__main__":
    spark = create_spark_session()
    
    # Define the path to your preprocessed training data
    train_data_path = "data/processed/train.parquet"
    
    # Train and evaluate the model
    train_and_evaluate_model(spark, train_data_path)
    
    # Stop the Spark session
    spark.stop()