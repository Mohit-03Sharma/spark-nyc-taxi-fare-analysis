import os
import glob
import shutil

OUT = "outputs"

MAPPINGS = {
    "hourly_demand_revenue": "hourly_demand_revenue.csv",
    "weekend_vs_weekday_hourly": "weekend_vs_weekday_hourly.csv",
    "payment_split": "payment_split.csv",
    "top_pickup_grids": "top_pickup_grids.csv",
}

def find_part_csv(folder):
    candidates = glob.glob(os.path.join(OUT, folder, "part-*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No part-*.csv found in {OUT}/{folder}")
    return candidates[0]

def main():
    for folder, target in MAPPINGS.items():
        src = find_part_csv(folder)
        dst = os.path.join(OUT, target)
        shutil.copyfile(src, dst)
        print(f"Saved: {dst}")

if __name__ == "__main__":
    main()