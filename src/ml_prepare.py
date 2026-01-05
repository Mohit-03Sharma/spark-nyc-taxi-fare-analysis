from pyspark.sql import SparkSession
from pyspark.sql import functions as F

IN_PATH = "data/processed/taxi_parquet"
OUT_PATH = "data/processed/ml_fare_dataset"

TOP_N_GRIDS = 300  # keeps location feature compact & RF-friendly

def build_spark():
    return (
        SparkSession.builder
        .appName("MLPrepareFareDataset")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(IN_PATH)

    base = (
        df
        .filter(F.col("trip_distance").isNotNull())
        .filter(F.col("fare_amount").isNotNull())
        .filter(F.col("tpep_pickup_datetime").isNotNull())
        .filter(F.col("tpep_dropoff_datetime").isNotNull())
        .filter(F.col("pickup_latitude").isNotNull())
        .filter(F.col("pickup_longitude").isNotNull())
        .filter(F.col("trip_distance") > 0)
        .filter(F.col("fare_amount") > 0)
        # NYC bounds to remove (0,0) and garbage coords
        .filter(F.col("pickup_latitude").between(40.5, 41.0))
        .filter(F.col("pickup_longitude").between(-74.5, -73.5))
    )

    # Fare per mile (for outlier guard)
    base = base.withColumn("fare_per_mile", F.col("fare_amount") / F.col("trip_distance"))
    base = base.filter((F.col("fare_per_mile") >= 0) & (F.col("fare_per_mile") <= 50))

    # Time features
    base = (
        base
        .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
        .withColumn("pickup_dow", F.date_format("tpep_pickup_datetime", "E"))
        .withColumn("is_weekend", F.col("pickup_dow").isin("Sat", "Sun"))
    )

    # Duration feature (minutes)
    # Cast to long seconds, then / 60
    base = base.withColumn(
        "trip_duration_mins",
        (F.col("tpep_dropoff_datetime").cast("long") - F.col("tpep_pickup_datetime").cast("long")) / 60.0
    ).filter((F.col("trip_duration_mins") > 0) & (F.col("trip_duration_mins") <= 180))

    # Location bins + grid id
    base = (
        base
        .withColumn("pickup_lat_bin", F.round("pickup_latitude", 2))
        .withColumn("pickup_lon_bin", F.round("pickup_longitude", 2))
        .withColumn("pickup_grid", F.concat_ws("_", F.col("pickup_lat_bin").cast("string"), F.col("pickup_lon_bin").cast("string")))
    )

    # Fare components
    base = (
        base
        .withColumn(
            "fixed_fees",
            F.coalesce(F.col("extra"), F.lit(0.0))
            + F.coalesce(F.col("mta_tax"), F.lit(0.0))
            + F.coalesce(F.col("improvement_surcharge"), F.lit(0.0))
        )
        .withColumn("tolls", F.coalesce(F.col("tolls_amount"), F.lit(0.0)))
    )

    # Distance regime (optional, useful for analysis later)
    base = base.withColumn(
        "distance_regime",
        F.when(F.col("trip_distance") <= 1, F.lit("short"))
         .when((F.col("trip_distance") > 1) & (F.col("trip_distance") <= 5), F.lit("medium"))
         .otherwise(F.lit("long"))
    )

    # ----------------------------
    # Compress location categories: keep top N grids
    # ----------------------------
    top_grids = (
        base.groupBy("pickup_grid")
        .count()
        .orderBy(F.desc("count"))
        .limit(TOP_N_GRIDS)
        .select("pickup_grid")
        .collect()
    )
    top_grid_set = set([r["pickup_grid"] for r in top_grids])

    # Map rare grids to "other"
    # (Spark-side: use a broadcast literal list)
    base = base.withColumn(
        "pickup_grid_top",
        F.when(F.col("pickup_grid").isin(list(top_grid_set)), F.col("pickup_grid")).otherwise(F.lit("other"))
    )

    # Final modeling dataset
    model_df = (
        base.select(
            F.col("fare_amount").alias("label_fare_amount"),
            "trip_distance",
            "pickup_hour",
            "is_weekend",
            "RateCodeID",
            "pickup_grid_top",
            "fixed_fees",
            "tolls",
            "trip_duration_mins",
            "fare_per_mile",
            "distance_regime",
        )
        .dropna()
    )

    print("\n=== ML PREP (UPDATED): FINAL DATASET ===")
    print("Rows:", model_df.count())
    print("Columns:", len(model_df.columns))
    model_df.printSchema()
    model_df.show(5, truncate=False)

    (
        model_df
        .repartition(8)
        .write.mode("overwrite")
        .parquet(OUT_PATH)
    )

    print(f"\nSaved ML dataset to: {OUT_PATH}")
    spark.stop()

if __name__ == "__main__":
    main()