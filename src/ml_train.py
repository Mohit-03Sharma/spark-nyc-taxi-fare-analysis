import os
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

IN_PATH = "data/processed/ml_fare_dataset"
OUT_DIR = "outputs"

SAMPLE_FRAC = 0.01
SEED = 42

def build_spark():
    return (
        SparkSession.builder
        .appName("MLTrainFareModels_RF")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(IN_PATH)

    # sample for sklearn training
    pdf = df.sample(withReplacement=False, fraction=SAMPLE_FRAC, seed=SEED).toPandas().dropna()

    y = pdf["label_fare_amount"].astype(float)

    # One split for fair comparison
    idx = np.arange(len(pdf))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED)

    # -----------------------------
    # Model 0: distance-only baseline
    # -----------------------------
    X0 = pdf[["trip_distance"]].astype(float)

    model0 = LinearRegression()
    model0.fit(X0.iloc[train_idx], y.iloc[train_idx])
    pred0 = model0.predict(X0.iloc[test_idx])

    mae0 = float(mean_absolute_error(y.iloc[test_idx], pred0))
    rmse0 = rmse(y.iloc[test_idx], pred0)

    # -----------------------------
    # Model 1: context-aware Random Forest
    # -----------------------------
    num_features = ["trip_distance", "pickup_hour", "fixed_fees", "tolls", "trip_duration_mins"]
    cat_features = ["is_weekend", "RateCodeID", "pickup_grid_top"]

    X1 = pdf[num_features + cat_features].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=SEED,
    )

    model1 = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("rf", rf),
        ]
    )

    model1.fit(X1.iloc[train_idx], y.iloc[train_idx])
    pred1 = model1.predict(X1.iloc[test_idx])

    mae1 = float(mean_absolute_error(y.iloc[test_idx], pred1))
    rmse1 = rmse(y.iloc[test_idx], pred1)

    metrics = pd.DataFrame([
        {"model": "Model0_DistanceOnly_LinearRegression", "sample_frac": SAMPLE_FRAC, "MAE": mae0, "RMSE": rmse0},
        {"model": "Model1_ContextAware_RandomForest+Duration", "sample_frac": SAMPLE_FRAC, "MAE": mae1, "RMSE": rmse1},
    ])

    metrics_path = os.path.join(OUT_DIR, "ml_metrics.csv")
    metrics.to_csv(metrics_path, index=False)

    residuals = pd.DataFrame({
        "y_true": y.iloc[test_idx].values,
        "y_pred_model0": pred0,
        "y_pred_model1": pred1,
        "trip_distance": pdf["trip_distance"].iloc[test_idx].values,
        "pickup_hour": pdf["pickup_hour"].iloc[test_idx].values,
        "is_weekend": pdf["is_weekend"].iloc[test_idx].values,
    })
    residuals["abs_err_model0"] = np.abs(residuals["y_true"] - residuals["y_pred_model0"])
    residuals["abs_err_model1"] = np.abs(residuals["y_true"] - residuals["y_pred_model1"])

    residuals_path = os.path.join(OUT_DIR, "residuals_sample.csv")
    residuals.head(5000).to_csv(residuals_path, index=False)

    print("\n=== ML RESULTS (RF + duration, fair split) ===")
    print(metrics)

    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {residuals_path}")

    spark.stop()

if __name__ == "__main__":
    main()