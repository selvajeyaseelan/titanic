import logging
from flask import Flask, request, jsonify
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import when, split, lit, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# Define the local paths to load the models
REGISTERED_MODEL_NAME = "TitanicBestModel"
PREPROCESSING_PIPELINE_PATH = "./preprocessing_pipeline_model"

# Initialize Spark Session (required for PySpark model prediction)
try:
    spark = SparkSession.builder \
        .appName("TitanicModelAPI") \
        .config("spark.driver.extraJavaOptions", "--add-opens=java.base/javax.security.auth.Subject=ALL-UNNAMED") \
        .getOrCreate()
    logging.info("Spark session created for API.")
except Exception as e:
    logging.error(f"Failed to create Spark session: {e}")
    exit(1)

# Load the trained Spark ML model from MLflow Registry
try:
    # Use the latest version of the registered model
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/latest"
    best_model = mlflow.spark.load_model(model_uri)
    logging.info(f"Loaded best model from MLflow Registry at URI: {model_uri}")
except Exception as e:
    logging.error(f"Failed to load model from MLflow Registry: {e}")
    spark.stop()
    exit(1)

# Load the fitted preprocessing pipeline model
try:
    preprocessing_pipeline_model = PipelineModel.load(PREPROCESSING_PIPELINE_PATH)
    logging.info("Loaded fitted preprocessing pipeline model.")
except Exception as e:
    logging.error(f"Failed to load preprocessing pipeline model: {e}")
    spark.stop()
    exit(1)

app = Flask(__name__)

# Define the schema for the incoming data
# This is crucial for creating a Spark DataFrame correctly
data_schema = StructType([
    StructField("PassengerId", IntegerType(), True),
    StructField("Pclass", IntegerType(), True),
    StructField("Name", StringType(), True),
    StructField("Sex", StringType(), True),
    StructField("Age", DoubleType(), True),
    StructField("SibSp", IntegerType(), True),
    StructField("Parch", IntegerType(), True),
    StructField("Fare", DoubleType(), True),
    StructField("Embarked", StringType(), True)
])

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests for model prediction.
    Expects a JSON payload with Titanic passenger data.
    """
    try:
        data = request.get_json(force=True)
        logging.info(f"Received prediction request with data: {data}")

        # Create a Spark DataFrame from the JSON data
        # We need to add the 'Title' column before transforming with the pipeline
        data_df = spark.createDataFrame([data], schema=data_schema)
        
        # Manually add the Title column as a pre-processing step
        data_df = data_df.withColumn("Title", when(split(col("Name"), r"\s+")[1].isin("Mr.", "Mrs.", "Miss.", "Master."), split(col("Name"), r"\s+")[1]).otherwise(lit("Other")))

        # Apply the loaded preprocessing pipeline to the new data
        processed_data = preprocessing_pipeline_model.transform(data_df)

        # Use the best model to make a prediction
        prediction_df = best_model.transform(processed_data)
        
        # Extract the prediction from the DataFrame
        # The 'prediction' column contains the final predicted class (0 or 1)
        # The 'probability' column contains the probability distribution
        prediction = prediction_df.select("prediction").collect()[0]["prediction"]
        probability = prediction_df.select("probability").collect()[0]["probability"][1] # Probability of survival

        response = {
            "prediction": int(prediction),
            "survival_probability": probability,
            "message": "Survived" if prediction == 1.0 else "Did not survive"
        }
        
        logging.info(f"Prediction result: {response}")
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

