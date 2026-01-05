from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, DoubleType, StringType, TimestampType
)
from pyspark.sql.functions import col

RAW_GLOB = "data/raw/*.csv"
OUT_PATH = "data/processed/taxi_parquet"

# Schema for Jan 2015-style TLC Yellow Taxi files (19 columns)
TAXI_SCHEMA = StructType([
    StructField("VendorID", IntegerType(), True),
    StructField("tpep_pickup_datetime", TimestampType(), True),
    StructField("tpep_dropoff_datetime", TimestampType(), True),
    StructField("passenger_count", IntegerType(), True),
    StructField("trip_distance", DoubleType(), True),
    StructField("pickup_longitude", DoubleType(), True),
    StructField("pickup_latitude", DoubleType(), True),
    StructField("RateCodeID", IntegerType(), True),
    StructField("store_and_fwd_flag", StringType(), True),
    StructField("dropoff_longitude", DoubleType(), True),
    StructField("dropoff_latitude", DoubleType(), True),
    StructField("payment_type", IntegerType(), True),
    StructField("fare_amount", DoubleType(), True),
    StructField("extra", DoubleType(), True),
    StructField("mta_tax", DoubleType(), True),
    StructField("tip_amount", DoubleType(), True),
    StructField("tolls_amount", DoubleType(), True),
    StructField("improvement_surcharge", DoubleType(), True),
    StructField("total_amount", DoubleType(), True),
])

def build_spark():
    return (
        SparkSession.builder
        .appName("TaxiIngestion2015")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    df = (
        spark.read
        .option("header", True)
        .schema(TAXI_SCHEMA)               # explicit schema (fast + stable)
        .option("mode", "DROPMALFORMED")   # drop bad rows safely
        .csv(RAW_GLOB)
    )

    print("\n=== PHASE 1: INGESTION CHECKS ===")
    print("Rows:", df.count())
    print("Columns:", len(df.columns))
    df.printSchema()

    # Basic sanity checks
    print("\n=== SANITY CHECKS ===")
    if "trip_distance" in df.columns:
        print("Negative trip_distance:", df.filter(col("trip_distance") < 0).count())
    if "total_amount" in df.columns:
        print("Negative total_amount:", df.filter(col("total_amount") < 0).count())
    if "passenger_count" in df.columns:
        print("Negative passenger_count:", df.filter(col("passenger_count") < 0).count())

    # Write to parquet (industry)
    df.write.mode("overwrite").parquet(OUT_PATH)
    print(f"\nSaved parquet to: {OUT_PATH}")

    spark.stop()

if __name__ == "__main__":
    main()
