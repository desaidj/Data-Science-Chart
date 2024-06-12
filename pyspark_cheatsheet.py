# PySpark Cheat Sheet

# Initialization
from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder \
    .appName("example") \
    .getOrCreate()

# DataFrame Basics

# Creating DataFrames

# From an existing RDD
# df = rdd.toDF()

# From a list
data = [("Alice", 1), ("Bob", 2)]
df = spark.createDataFrame(data, ["name", "age"])

# From a CSV file
# df = spark.read.csv("file_path.csv", header=True, inferSchema=True)

# From a JSON file
# df = spark.read.json("file_path.json")

# Showing Data
df.show()
df.show(5)  # Show top 5 rows
df.printSchema()  # Print the schema
df.describe().show()  # Summary statistics

# DataFrame Operations

# Selecting Columns
df.select("name", "age").show()
df.select(df.name, df.age).show()

# Filtering Data
df.filter(df.age > 21).show()
df.filter("age > 21").show()
df.where(df.age > 21).show()

# Adding Columns
from pyspark.sql.functions import lit

df = df.withColumn("new_column", lit(10))
df = df.withColumn("age_double", df.age * 2)

# Renaming Columns
df = df.withColumnRenamed("age", "years")

# Dropping Columns
df = df.drop("new_column")

# Aggregations
df.groupBy("name").count().show()
df.groupBy("name").agg({"years": "mean"}).show()

# Joins
# Assuming df1 and df2 are already defined DataFrames
# df1.join(df2, df1.name == df2.name, "inner").show()
# df1.join(df2, "name", "outer").show()

# SQL Queries
# Register the DataFrame as a SQL temporary view
df.createOrReplaceTempView("people")

# SQL query
sqlDF = spark.sql("SELECT * FROM people")
sqlDF.show()

# Working with RDDs

# Creating RDDs
rdd = spark.sparkContext.parallelize([1, 2, 3, 4])
# rdd = spark.sparkContext.textFile("file_path.txt")

# Basic RDD Operations

# Map
rdd.map(lambda x: x * 2).collect()

# Filter
rdd.filter(lambda x: x % 2 == 0).collect()

# Reduce
rdd.reduce(lambda a, b: a + b)

# FlatMap
rdd.flatMap(lambda x: (x, x*2)).collect()

# Machine Learning with MLlib

# Basic Example: Linear Regression
from pyspark.ml.regression import LinearRegression

# Load training data
training = spark.read.format("libsvm").load("data/mllib/sample_linear_regression_data.txt")

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(training)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

# Saving and Loading Models

# Save model
lrModel.save("path_to_save_model")

# Load model
from pyspark.ml.regression import LinearRegressionModel
lrModel = LinearRegressionModel.load("path_to_save_model")

# Configurations and Performance Tuning

# Set configuration
spark.conf.set("spark.sql.shuffle.partitions", "50")
spark.conf.set("spark.executor.memory", "2g")

# View configuration
print(spark.conf.get("spark.sql.shuffle.partitions"))
