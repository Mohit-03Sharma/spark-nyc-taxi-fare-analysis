from pyspark.sql import SparkSession
from pyspark.sql import functions as F

IN_PATH = "data/processed/taxi_parquet"
OUT_DIR = "outputs"

def build_spark():
    return (
        SparkSession.builder
        .appName("TaxiAnalytics2015")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(IN_PATH)

    # -----------------------------
    # 1) Basic cleaning (analyst-safe)
    # -----------------------------
    df_clean = (
        df
        .filter(F.col("tpep_pickup_datetime").isNotNull())
        .filter(F.col("tpep_dropoff_datetime").isNotNull())
        .filter(F.col("trip_distance").isNotNull())
        .filter(F.col("total_amount").isNotNull())
        .filter(F.col("trip_distance") >= 0)
        .filter(F.col("total_amount") >= 0)  # drop negative totals (refund/edge rows)
    )

    # Trip duration in minutes (remove extreme outliers)
    df_clean = df_clean.withColumn(
        "trip_minutes",
        (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")) / 60.0
    ).filter((F.col("trip_minutes") > 0) & (F.col("trip_minutes") < 180))

    # Add time features
    df_feat = (
        df_clean
        .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
        .withColumn("pickup_date", F.to_date("tpep_pickup_datetime"))
        .withColumn("pickup_dow", F.date_format("tpep_pickup_datetime", "E"))  # Mon, Tue...
        .withColumn("is_weekend", F.col("pickup_dow").isin("Sat", "Sun"))
    )

    # -----------------------------
    # 2) Hourly demand + revenue + distance
    # -----------------------------
    hourly = (
        df_feat.groupBy("pickup_hour")
        .agg(
            F.count("*").alias("total_trips"),
            F.round(F.avg("trip_distance"), 2).alias("avg_trip_distance_miles"),
            F.round(F.avg("trip_minutes"), 2).alias("avg_trip_duration_mins"),
            F.round(F.sum("total_amount"), 2).alias("total_revenue_usd"),
        )
        .orderBy("pickup_hour")
    )

    # -----------------------------
    # 3) Weekday vs weekend pattern
    # -----------------------------
    weekend = (
        df_feat.groupBy("is_weekend", "pickup_hour")
        .agg(
            F.count("*").alias("trips"),
            F.round(F.avg("total_amount"), 2).alias("avg_total_amount_usd")
        )
        .orderBy("is_weekend", "pickup_hour")
    )

    # -----------------------------
    # 4) Payment type split
    # payment_type (per TLC): 1=Credit card, 2=Cash, 3=No charge, 4=Dispute, 5=Unknown, 6=Voided
    # -----------------------------
    payment_map = (
        F.when(F.col("payment_type") == 1, F.lit("Credit Card"))
         .when(F.col("payment_type") == 2, F.lit("Cash"))
         .when(F.col("payment_type") == 3, F.lit("No Charge"))
         .when(F.col("payment_type") == 4, F.lit("Dispute"))
         .when(F.col("payment_type") == 5, F.lit("Unknown"))
         .when(F.col("payment_type") == 6, F.lit("Voided"))
         .otherwise(F.lit("Other"))
    )

    payment = (
        df_feat.withColumn("payment_label", payment_map)
        .groupBy("payment_label")
        .agg(
            F.count("*").alias("trips"),
            F.round(F.sum("total_amount"), 2).alias("revenue_usd"),
            F.round(F.avg("tip_amount"), 2).alias("avg_tip_usd")
        )
        .orderBy(F.desc("trips"))
    )

    # -----------------------------
    # 5) Top pickup hotspots (geo-binning)
    # Use coarse rounding to 2 decimals (~1km-ish resolution)
    # -----------------------------
    pickups = df_feat.filter(
        (F.col("pickup_latitude").between(40.5, 41.0)) &
        (F.col("pickup_longitude").between(-74.5, -73.5))
    )

    pickup_grid = (
        pickups
        .withColumn("pickup_lat_bin", F.round("pickup_latitude", 2))
        .withColumn("pickup_lon_bin", F.round("pickup_longitude", 2))
        .groupBy("pickup_lat_bin", "pickup_lon_bin")
        .agg(
            F.count("*").alias("trips"),
            F.round(F.avg("total_amount"), 2).alias("avg_total_amount_usd")
        )
        .orderBy(F.desc("trips"))
        .limit(50)
    )

    # -----------------------------
    # Write outputs
    # -----------------------------
    hourly.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUT_DIR}/hourly_demand_revenue")
    weekend.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUT_DIR}/weekend_vs_weekday_hourly")
    payment.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUT_DIR}/payment_split")
    pickup_grid.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUT_DIR}/top_pickup_grids")

    # Print samples (so you can paste into README)
    print("\n=== HOURLY SAMPLE ===")
    hourly.show(10, truncate=False)

    print("\n=== PAYMENT SPLIT SAMPLE ===")
    payment.show(10, truncate=False)

    print("\n=== TOP PICKUP GRIDS SAMPLE ===")
    pickup_grid.show(10, truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()
