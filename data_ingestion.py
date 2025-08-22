from pyspark.sql import SparkSession
import logging

def create_spark_session():
    # Initializes a SparkSession with necessary configurations.
    spark = SparkSession.builder \
        .appName("DataIngestion") \
        .config("spark.driver.extraJavaOptions", "--add-opens=java.base/javax.security.auth.Subject=ALL-UNNAMED") \
        .getOrCreate()
    return spark

def ingest_data(spark):
    """
    Ingests raw data and performs initial cleaning.
    """
    logging.info("Starting data ingestion.")
    try:
        train_df = spark.read.csv("data/train.csv", header=True, inferSchema=True)
        test_df = spark.read.csv("data/test.csv", header=True, inferSchema=True)
        logging.info("Data ingestion successful.")
        
        # Initial cleaning: dropping columns with many missing values
        train_df = train_df.drop('Cabin', 'Ticket')
        test_df = test_df.drop('Cabin', 'Ticket')
        
        # Filling a few missing values in 'Embarked'
        train_df = train_df.fillna({'Embarked': 'S'})
        
        return train_df, test_df
    except Exception as e:
        logging.error(f"Error during data ingestion: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    spark = create_spark_session()
    train_data, test_data = ingest_data(spark)
    # You would typically save these cleaned dataframes here for the next step,
    # or pass them directly to the preprocessing script.
    spark.stop()