from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("CheckProcessed").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet("data/processed/taxi_parquet")

print("Processed rows:", df.count())
print("Processed cols:", len(df.columns))
df.select("tpep_pickup_datetime", "trip_distance", "total_amount").show(5, truncate=False)

spark.stop()
