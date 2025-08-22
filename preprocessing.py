from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import col, lit, when, split
import logging

def build_preprocessing_pipeline():
    """
    Builds and returns a Spark ML Pipeline for data preprocessing and feature engineering.
    """
    logging.info("Building preprocessing pipeline.")
    
    # Feature Engineering: Extract Title from Name
    title_indexer = StringIndexer(inputCol="Title", outputCol="TitleIndex", handleInvalid="skip")
    title_encoder = OneHotEncoder(inputCol="TitleIndex", outputCol="TitleVec")

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

    # Feature Assembly: Combines all feature columns into a single vector
    assembler = VectorAssembler(
        inputCols=[
            "Pclass", "Age_imputed", "SibSp", "Parch", "Fare_imputed",
            "SexVec", "EmbarkedVec", "TitleVec"
        ],
        outputCol="features"
    )
    
    # Build the complete pipeline
    preprocessing_pipeline = Pipeline(stages=[
        imputer,
        gender_indexer,
        gender_encoder,
        embarked_indexer,
        embarked_encoder,
        title_indexer,
        title_encoder,
        assembler
    ])
    
    logging.info("Preprocessing pipeline built successfully.")
    return preprocessing_pipeline