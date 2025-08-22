from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, when, split, lit
import logging

def add_features(df):
    """
    Performs feature engineering by adding a 'Title' column.
    """
    logging.info("Starting feature engineering.")
    try:
        df = df.withColumn("Title", when(split(col("Name"), r"\s+")[1].isin("Mr.", "Mrs.", "Miss.", "Master."), split(col("Name"), r"\s+")[1]).otherwise(lit("Other")))
        logging.info("Feature engineering successful.")
        return df
    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
        raise

def assemble_features(df):
    """
    Assembles all feature columns into a single vector.
    """
    logging.info("Assembling features.")
    assembler = VectorAssembler(
        inputCols=[
            "Pclass", "Age_imputed", "SibSp", "Parch", "Fare_imputed",
            "SexVec", "EmbarkedVec", "TitleVec"
        ],
        outputCol="features"
    )
    return assembler.transform(df)