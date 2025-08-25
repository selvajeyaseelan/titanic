import logging
import os
from pyspark.sql import SparkSession
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from data_ingestion import ingest_data # Import the ingest_data function
from preprocessing import build_preprocessing_pipeline # Import the preprocessing pipeline builder
from model_building_lr import build_model, build_param_grid # Import model and param grid builders
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, when, split, lit

def create_spark_session():
    # Your existing Spark session creation function
    ...

def main():
    logging.basicConfig(level=logging.INFO)
    spark = create_spark_session()
    
    # 1. Pull data from DVC
    logging.info("Pulling data from DVC.")
    os.system("dvc pull")

    # 2. Ingest Data
    train_df, test_df = ingest_data(spark)

    # 3. Add 'Title' feature before pipeline
    train_df = train_df.withColumn("Title", when(split(col("Name"), r"\s+")[1].isin("Mr.", "Mrs.", "Miss.", "Master."), split(col("Name"), r"\s+")[1]).otherwise(lit("Other")))
    test_df = test_df.withColumn("Title", when(split(col("Name"), r"\s+")[1].isin("Mr.", "Mrs.", "Miss.", "Master."), split(col("Name"), r"\s+")[1]).otherwise(lit("Other")))

    # 4. Build and fit preprocessing pipeline on training data
    preprocessing_pipeline = build_preprocessing_pipeline()
    fitted_pipeline = preprocessing_pipeline.fit(train_df)
    
    # 5. Transform both train and test data
    processed_train_df = fitted_pipeline.transform(train_df)
    processed_test_df = fitted_pipeline.transform(test_df)

    # 6. Model Training and Evaluation
    # ... your existing logic for model training and selection
    # Use processed_train_df for cross-validation
    (training_data, validation_data) = processed_train_df.randomSplit([0.8, 0.2], seed=42)
    # ... your existing code for model training loop
    
    # 7. Final Model Evaluation on the Test Set
    if best_overall_model:
        with mlflow.start_run(run_name="Final_Model_Test_Evaluation"):
            # Use the best model to make predictions on the processed test data
            test_predictions = best_overall_model.transform(processed_test_df)
            test_evaluator = BinaryClassificationEvaluator(labelCol="Survived", metricName="areaUnderROC")
            test_auc = test_evaluator.evaluate(test_predictions)
            
            mlflow.log_metric("final_test_auc", test_auc)
            logging.info(f"Final Model AUC on test set: {test_auc}")
    
    spark.stop()

if __name__ == "__main__":
    main()