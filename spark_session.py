
from pyspark.sql import SparkSession
import logging

def create_spark_session():
    """
    Initializes and returns a SparkSession with necessary configurations.
    """
    logging.info("Initializing Spark session.")
    try:
        
        spark = SparkSession.builder \
            .appName("TitanicMLOpsPipeline") \
            .config("spark.driver.extraJavaOptions", "--add-opens=java.base/javax.security.auth.Subject=ALL-UNNAMED") \
            .getOrCreate()
        logging.info("Spark session created successfully.")
        return spark
    except Exception as e:
        logging.error(f"Error creating Spark session: {e}")
        raise