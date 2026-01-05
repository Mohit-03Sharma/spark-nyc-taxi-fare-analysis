from pyspark.sql import SparkSession
from pyspark.sql import functions as F

IN_PATH = "data/processed/taxi_parquet"
OUT_DIR = "outputs"

def build_spark():
    return (
        SparkSession.builder
        .appName("FareDeviationContext")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(IN_PATH)

    # Basic hygiene
    base = (
        df
        .filter(F.col("trip_distance").isNotNull())
        .filter(F.col("fare_amount").isNotNull())
        .filter(F.col("tpep_pickup_datetime").isNotNull())
        .filter(F.col("trip_distance") > 0)
        .filter(F.col("fare_amount") > 0)
    )

    # Distance regimes (same as Step 1)
    base = base.withColumn(
        "distance_regime",
        F.when(F.col("trip_distance") <= 1, "short")
         .when((F.col("trip_distance") > 1) & (F.col("trip_distance") <= 5), "medium")
         .otherwise("long")
    )

    # Fare per mile + simple outlier guard
    base = base.withColumn("fare_per_mile", F.col("fare_amount") / F.col("trip_distance"))
    base = base.filter((F.col("fare_per_mile") >= 0) & (F.col("fare_per_mile") <= 50))

    # Time + geo features
    base = (
        base
        .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
        .withColumn("pickup_dow", F.date_format("tpep_pickup_datetime", "E"))
        .withColumn("is_weekend", F.col("pickup_dow").isin("Sat", "Sun"))
        .withColumn("pickup_lat_bin", F.round("pickup_latitude", 2))
        .withColumn("pickup_lon_bin", F.round("pickup_longitude", 2))
    )

    # Baseline medians per regime
    baseline_by_regime = (
        base.groupBy("distance_regime")
        .agg(F.expr("percentile_approx(fare_per_mile, 0.5)").alias("median_fare_per_mile"))
    )

    with_baseline = (
        base.join(baseline_by_regime, on="distance_regime", how="left")
        .withColumn("expected_fare_baseline", F.col("trip_distance") * F.col("median_fare_per_mile"))
        .withColumn("fare_deviation", F.col("fare_amount") - F.col("expected_fare_baseline"))
        .withColumn("abs_deviation", F.abs(F.col("fare_deviation")))
    )

    # ----------------------------
    # 1) Deviation by hour & weekend
    # ----------------------------
    by_time = (
        with_baseline.groupBy("distance_regime", "is_weekend", "pickup_hour")
        .agg(
            F.count("*").alias("trips"),
            F.round(F.expr("percentile_approx(abs_deviation, 0.5)"), 2).alias("median_abs_dev"),
            F.round(F.expr("percentile_approx(abs_deviation, 0.9)"), 2).alias("p90_abs_dev"),
            F.round(F.avg("fare_deviation"), 2).alias("avg_dev"),
        )
        .orderBy("distance_regime", "is_weekend", "pickup_hour")
    )

    # ----------------------------
    # 2) Top locations by unpredictability (use p90 abs deviation)
    # Only consider grids with enough trips to be meaningful
    # ----------------------------
    by_grid = (
        with_baseline.groupBy("distance_regime", "pickup_lat_bin", "pickup_lon_bin")
        .agg(
            F.count("*").alias("trips"),
            F.round(F.expr("percentile_approx(abs_deviation, 0.9)"), 2).alias("p90_abs_dev"),
            F.round(F.expr("percentile_approx(abs_deviation, 0.5)"), 2).alias("median_abs_dev"),
        )
        .filter(F.col("trips") >= 5000)
        .orderBy(F.desc("p90_abs_dev"))
        .limit(100)
    )

    # Save
    by_time.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUT_DIR}/deviation_by_time")
    by_grid.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUT_DIR}/deviation_top_pickup_grids")

    print("\n=== DEVIATION BY TIME (sample) ===")
    by_time.show(15, truncate=False)

    print("\n=== TOP GRIDS BY UNPREDICTABILITY (sample) ===")
    by_grid.show(15, truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()