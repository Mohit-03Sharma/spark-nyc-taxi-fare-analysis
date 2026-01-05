import pandas as pd
import os

RESIDUALS_PATH = "outputs/residuals_sample.csv"
OUT_PATH = "outputs/residuals_summary.csv"

def main():
    df = pd.read_csv(RESIDUALS_PATH)

    print("\n=== Residual sanity check ===")
    print(df.head())

    # -------------------------
    # Overall comparison
    # -------------------------
    overall = pd.DataFrame({
        "model": ["Distance-only", "Context-aware RF"],
        "avg_abs_error": [
            df["abs_err_model0"].mean(),
            df["abs_err_model1"].mean()
        ]
    })

    # -------------------------
    # By pickup hour
    # -------------------------
    by_hour = (
        df.groupby("pickup_hour")[["abs_err_model0", "abs_err_model1"]]
        .mean()
        .reset_index()
        .sort_values("pickup_hour")
    )

    # -------------------------
    # By weekend
    # -------------------------
    by_weekend = (
        df.groupby("is_weekend")[["abs_err_model0", "abs_err_model1"]]
        .mean()
        .reset_index()
    )

    # Save combined report
    overall.to_csv("outputs/residuals_overall.csv", index=False)
    by_hour.to_csv("outputs/residuals_by_hour.csv", index=False)
    by_weekend.to_csv("outputs/residuals_by_weekend.csv", index=False)
    
    print("\n=== Overall avg absolute error ===")
    print(overall)

    print("\n=== Avg absolute error by hour (first 10 rows) ===")
    print(by_hour.head(10))

    print("\n=== Avg absolute error by weekend ===")
    print(by_weekend)

    print(f"\nSaved residual summary to: {OUT_PATH.replace('.csv', '.xlsx')}")

if __name__ == "__main__":
    main()