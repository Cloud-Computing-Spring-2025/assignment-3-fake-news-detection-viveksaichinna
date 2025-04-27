from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark Session
spark = SparkSession.builder.appName("FakeNewsDetection").getOrCreate()

# ---------------------------
# Task 1: Load & Basic Exploration
# ---------------------------
# Load CSV
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

# Create Temp View
df.createOrReplaceTempView("news_data")

# Basic Queries
df.show(5)
print("Total number of articles:", df.count())
df.select("label").distinct().show()

# Save sample output
df.limit(5).toPandas().to_csv("task1_output.csv", index=False)

# ---------------------------
# Task 2: Text Preprocessing
# ---------------------------
# Convert text to lowercase
df = df.withColumn("text", lower(col("text")))

# Tokenization
tokenizer = Tokenizer(inputCol="text", outputCol="words")
df_tokenized = tokenizer.transform(df)

# Remove Stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df_cleaned = remover.transform(df_tokenized)

# Select necessary columns
df_task2 = df_cleaned.select("id", "title", "filtered_words", "label")

# Save output
df_task2.toPandas().to_csv("task2_output.csv", index=False)

# ---------------------------
# Task 3: Feature Extraction
# ---------------------------
# HashingTF
hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=10000)
df_hashed = hashingTF.transform(df_task2)

# IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(df_hashed)
df_featurized = idfModel.transform(df_hashed)

# Label Indexing
indexer = StringIndexer(inputCol="label", outputCol="label_index")
df_final = indexer.fit(df_featurized).transform(df_featurized)

# Select important columns
df_task3 = df_final.select("id", "filtered_words", "features", "label_index")

# Save output
df_task3.toPandas().to_csv("task3_output.csv", index=False)

# ---------------------------
# ---------------------------
# Task 4: Model Training (Fixed)
# ---------------------------
# Merge title back for prediction output
df_with_title = df_cleaned.select("id", "title")

# Split data
train_data, test_data = df_task3.randomSplit([0.8, 0.2], seed=42)

# Train Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label_index")
model = lr.fit(train_data)

# Predictions
predictions = model.transform(test_data)

# Join predictions with original title
predictions_with_title = predictions.join(df_with_title, on="id", how="left")

# Select and Save predictions
predictions_with_title.select("id", "title", "label_index", "prediction").toPandas().to_csv("task4_output.csv", index=False)
# ---------------------------
# Task 5: Evaluate the Model
# ---------------------------
# ---------------------------
# Task 5: Evaluate the Model
# ---------------------------
# Evaluators
evaluator_accuracy = MulticlassClassificationEvaluator(
    labelCol="label_index", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label_index", predictionCol="prediction", metricName="f1")

accuracy = evaluator_accuracy.evaluate(predictions)
f1 = evaluator_f1.evaluate(predictions)

# Save evaluation results
import pandas as pd

eval_results = pd.DataFrame({
    'Metric': ['Accuracy', 'F1 Score'],
    'Value': [accuracy, f1]
})

eval_results.to_csv("task5_output.csv", index=False)


# Also print
print("Evaluation Results:")
print(eval_results)

# ---------------------------
# Finish
# ---------------------------
spark.stop()