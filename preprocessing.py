
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when, split, lit

def create_spark_session():
    """
    Initializes and returns a SparkSession with necessary configurations.
    """
    
    spark = SparkSession.builder \
        .appName("TitanicPreprocessing") \
        .config("spark.driver.extraJavaOptions", "--add-opens=java.base/javax.security.auth=ALL-UNNAMED") \
        .getOrCreate()
    return spark

def build_preprocessing_pipeline():
    """
    Builds and returns a Spark ML Pipeline for data preprocessing.
    """
    # 1. Feature Engineering: Extract Title from Name
    # We will use a regular expression to find titles like 'Mr.', 'Mrs.', etc.
    # We'll treat common titles separately and group the rest into 'Other'.
    title_udf = lambda name: when(split(col("Name"), r"\s+")[1] == "Mr.", "Mr") \
        .when(split(col("Name"), r"\s+")[1] == "Mrs.", "Mrs") \
        .when(split(col("Name"), r"\s+")[1] == "Miss.", "Miss") \
        .when(split(col("Name"), r"\s+")[1] == "Master.", "Master") \
        .otherwise("Other")
    
    # 2. Handle Missing Values using Imputer
    # We'll impute missing 'Age' values with the mean.
    imputer = Imputer(
        inputCols=['Age', 'Fare'],
        outputCols=['Age_imputed', 'Fare_imputed']
    ).setStrategy("mean")

    # 3. Categorical Feature Encoding
    # StringIndexer converts categorical strings into numerical indices.
    # OneHotEncoder converts the indices into a one-hot encoded vector.
    gender_indexer = StringIndexer(inputCol="Sex", outputCol="SexIndex")
    gender_encoder = OneHotEncoder(inputCol="SexIndex", outputCol="SexVec")

    embarked_indexer = StringIndexer(inputCol="Embarked", outputCol="EmbarkedIndex")
    embarked_encoder = OneHotEncoder(inputCol="EmbarkedIndex", outputCol="EmbarkedVec")

    title_indexer = StringIndexer(inputCol="Title", outputCol="TitleIndex")
    title_encoder = OneHotEncoder(inputCol="TitleIndex", outputCol="TitleVec")

    # 4. Feature Assembly
    # VectorAssembler combines all feature columns into a single vector column.
    assembler = VectorAssembler(
        inputCols=[
            "Pclass", "Age_imputed", "SibSp", "Parch", "Fare_imputed",
            "SexVec", "EmbarkedVec", "TitleVec"
        ],
        outputCol="features"
    )

    # 5. Create a Spark ML Pipeline
    # The pipeline ensures that all preprocessing steps are applied in the correct order.
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

def main():
    """
    Main function to execute the preprocessing pipeline.
    """
    spark = create_spark_session()
    
    # Load raw data from the data folder
    train_df = spark.read.csv("data/train.csv", header=True, inferSchema=True)
    test_df = spark.read.csv("data/test.csv", header=True, inferSchema=True)
    
    # Drop the Cabin and Ticket columns as they have too many missing values or are not useful
    train_df = train_df.drop('Cabin', 'Ticket')
    test_df = test_df.drop('Cabin', 'Ticket')
    
    # Fill in a few missing values in 'Embarked' with a default value
    train_df = train_df.fillna({'Embarked': 'S'})

    # Add a 'Title' column to the DataFrames
    train_df = train_df.withColumn("Title", when(split(col("Name"), r"\s+")[1].isin("Mr.", "Mrs.", "Miss.", "Master."), split(col("Name"), r"\s+")[1]).otherwise(lit("Other")))
    test_df = test_df.withColumn("Title", when(split(col("Name"), r"\s+")[1].isin("Mr.", "Mrs.", "Miss.", "Master."), split(col("Name"), r"\s+")[1]).otherwise(lit("Other")))

    # Build the pipeline
    preprocessing_pipeline = build_preprocessing_pipeline()

    # Fit the pipeline on the training data. This calculates necessary statistics (like mean for imputation).
    pipeline_model = preprocessing_pipeline.fit(train_df)
    
    # Transform both the training and test datasets using the fitted pipeline.
    processed_train_df = pipeline_model.transform(train_df)
    processed_test_df = pipeline_model.transform(test_df)
    
    # Display the processed data schema and a few rows to verify the output.
    print("Processed Training Data Schema:")
    processed_train_df.printSchema()
    
    print("Processed Training Data:")
    processed_train_df.show(5)

    # Save the processed data in Parquet format, which is optimized for Spark.
    # This step is crucial for the next phase of the pipeline: model training.
    # The processed data can be easily loaded for training without re-running the preprocessing steps.
    processed_train_df.write.parquet("data/processed/train.parquet", mode="overwrite")
    processed_test_df.write.parquet("data/processed/test.parquet", mode="overwrite")
    
    print("Preprocessing complete. Processed data saved to 'data/processed/' directory.")

if __name__ == "__main__":
    main()