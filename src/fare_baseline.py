from pyspark.sql import SparkSession
from pyspark.sql import functions as F

IN_PATH = "data/processed/taxi_parquet"
OUT_DIR = "outputs"

def build_spark():
    return (
        SparkSession.builder
        .appName("FareBaselineDistanceOnly")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(IN_PATH)

    # ----------------------------
    # Basic hygiene for baseline
    # ----------------------------
    base = (
        df
        .filter(F.col("trip_distance").isNotNull())
        .filter(F.col("fare_amount").isNotNull())
        .filter(F.col("trip_distance") > 0)
        .filter(F.col("fare_amount") >= 0)
    )

    # ----------------------------
    # Distance regimes (Choice B)
    # ----------------------------
    base = base.withColumn(
        "distance_regime",
        F.when(F.col("trip_distance") <= 1, "short")
         .when((F.col("trip_distance") > 1) & (F.col("trip_distance") <= 5), "medium")
         .otherwise("long")
    )

    # ----------------------------
    # Fare per mile
    # ----------------------------
    base = base.withColumn(
        "fare_per_mile",
        F.col("fare_amount") / F.col("trip_distance")
    )

    # ----------------------------
    # Baseline constant per regime
    # (median fare_per_mile)
    # ----------------------------
    baseline_by_regime = (
        base.groupBy("distance_regime")
        .agg(
            F.expr("percentile_approx(fare_per_mile, 0.5)").alias("median_fare_per_mile"),
            F.count("*").alias("trips")
        )
    )

    # ----------------------------
    # Expected fare under distance-only baseline
    # ----------------------------
    with_baseline = (
        base.join(baseline_by_regime, on="distance_regime", how="left")
        .withColumn(
            "expected_fare_baseline",
            F.col("trip_distance") * F.col("median_fare_per_mile")
        )
        .withColumn(
            "fare_deviation",
            F.col("fare_amount") - F.col("expected_fare_baseline")
        )
    )

    # ----------------------------
    # Save outputs
    # ----------------------------
    (
        baseline_by_regime
        .orderBy("distance_regime")
        .coalesce(1)
        .write.mode("overwrite").option("header", True)
        .csv(f"{OUT_DIR}/fare_baseline_by_distance_regime")
    )

    deviation_summary = (
        with_baseline.groupBy("distance_regime")
        .agg(
            F.count("*").alias("trips"),
            F.round(F.avg("fare_deviation"), 2).alias("avg_deviation"),
            F.round(F.expr("percentile_approx(fare_deviation, 0.5)"), 2).alias("median_deviation"),
            F.round(F.expr("percentile_approx(fare_deviation, 0.9)"), 2).alias("p90_deviation"),
            F.round(F.expr("percentile_approx(fare_deviation, 0.1)"), 2).alias("p10_deviation"),
        )
        .orderBy("distance_regime")
    )

    (
        deviation_summary
        .coalesce(1)
        .write.mode("overwrite").option("header", True)
        .csv(f"{OUT_DIR}/fare_deviation_summary")
    )

    # Small console sample (for README)
    print("\n=== BASELINE (MEDIAN FARE PER MILE) ===")
    baseline_by_regime.show(truncate=False)

    print("\n=== DEVIATION SUMMARY ===")
    deviation_summary.show(truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()