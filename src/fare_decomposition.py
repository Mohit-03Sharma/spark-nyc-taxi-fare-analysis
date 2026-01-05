from pyspark.sql import SparkSession
from pyspark.sql import functions as F

IN_PATH = "data/processed/taxi_parquet"
OUT_DIR = "outputs"

def build_spark():
    return (
        SparkSession.builder
        .appName("FareDecomposition")
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
        .filter(F.col("trip_distance") > 0)
        .filter(F.col("fare_amount") > 0)
        .filter(F.col("tpep_pickup_datetime").isNotNull())
        .filter(F.col("pickup_latitude").isNotNull())
        .filter(F.col("pickup_longitude").isNotNull())
        # NYC bounds to remove (0,0) and garbage
        .filter(F.col("pickup_latitude").between(40.5, 41.0))
        .filter(F.col("pickup_longitude").between(-74.5, -73.5))
    )

    # Distance regimes
    base = base.withColumn(
        "distance_regime",
        F.when(F.col("trip_distance") <= 1, "short")
         .when((F.col("trip_distance") > 1) & (F.col("trip_distance") <= 5), "medium")
         .otherwise("long")
    )

    # Fare per mile (outlier guard)
    base = base.withColumn("fare_per_mile", F.col("fare_amount") / F.col("trip_distance"))
    base = base.filter((F.col("fare_per_mile") >= 0) & (F.col("fare_per_mile") <= 50))

    # Time + geo
    base = (
        base
        .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
        .withColumn("pickup_dow", F.date_format("tpep_pickup_datetime", "E"))
        .withColumn("is_weekend", F.col("pickup_dow").isin("Sat", "Sun"))
        .withColumn("pickup_lat_bin", F.round("pickup_latitude", 2))
        .withColumn("pickup_lon_bin", F.round("pickup_longitude", 2))
    )

    # Decompose components (only using available columns)
    base = (
        base
        .withColumn("fixed_fees", F.coalesce(F.col("extra"), F.lit(0.0))
                              + F.coalesce(F.col("mta_tax"), F.lit(0.0))
                              + F.coalesce(F.col("improvement_surcharge"), F.lit(0.0)))
        .withColumn("tolls", F.coalesce(F.col("tolls_amount"), F.lit(0.0)))
        .withColumn("fixed_fee_share", F.col("fixed_fees") / F.col("fare_amount"))
        .withColumn("toll_share", F.col("tolls") / F.col("fare_amount"))
    )

    # Aggregate decomposition by time
    by_time = (
        base.groupBy("distance_regime", "is_weekend", "pickup_hour")
        .agg(
            F.count("*").alias("trips"),
            F.round(F.expr("percentile_approx(fixed_fees, 0.5)"), 2).alias("median_fixed_fees"),
            F.round(F.expr("percentile_approx(tolls, 0.5)"), 2).alias("median_tolls"),
            F.round(F.expr("percentile_approx(fixed_fee_share, 0.5)"), 3).alias("median_fixed_fee_share"),
            F.round(F.expr("percentile_approx(toll_share, 0.9)"), 3).alias("p90_toll_share"),
        )
        .orderBy("distance_regime", "is_weekend", "pickup_hour")
    )

    # Top pickup grids by median tolls (proxy for toll-heavy areas)
    toll_grids = (
        base.groupBy("distance_regime", "pickup_lat_bin", "pickup_lon_bin")
        .agg(
            F.count("*").alias("trips"),
            F.round(F.expr("percentile_approx(tolls, 0.5)"), 2).alias("median_tolls"),
            F.round(F.expr("percentile_approx(tolls, 0.9)"), 2).alias("p90_tolls"),
        )
        .filter(F.col("trips") >= 5000)
        .orderBy(F.desc("p90_tolls"))
        .limit(100)
    )

    by_time.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUT_DIR}/decomposition_by_time")
    toll_grids.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUT_DIR}/toll_heavy_pickup_grids")

    print("\n=== DECOMPOSITION BY TIME (sample) ===")
    by_time.show(15, truncate=False)

    print("\n=== TOLL-HEAVY PICKUP GRIDS (sample) ===")
    toll_grids.show(15, truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()