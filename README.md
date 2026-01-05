# Understanding NYC Taxi Fare Structure Using PySpark and Machine Learning

## Overview
Taxi fares are commonly assumed to scale linearly with trip distance.  
This project tests that assumption using **12.4 million NYC Yellow Taxi trips (January 2015)** and systematically analyzes **when, why, and by how much** distance alone fails to explain fares.

Built a data-analytics pipeline in **PySpark** to process **12,748,986** NYC Yellow Taxi trips (Jan 2015) and generate scalable, analyst-style insights on demand, revenue, payments, and pickup hotspots.

## Dataset
- **Source:** NYC Taxi & Limousine Commission (TLC)
- **Type:** Yellow Taxi trip records
- **Period:** January 2015
- **Size:** ~2 GB raw CSV, 12.4M valid trips after cleaning
- **Key fields:** pickup/dropoff time, trip distance, fare amount, surcharges, tolls, pickup coordinates


## Tech Stack
- **PySpark** for ingestion, cleaning, aggregation, and feature engineering
- **Parquet** for columnar storage and efficient reuse
- **scikit-learn** for modeling and evaluation
- **Local Spark (`local[*]`)** on Windows

---

## Step 1 — Distance-Based Baseline
A distance-only baseline was constructed to represent the common mental model:

> **fare ≈ distance × constant**

Trips were segmented into regimes:
- **Short:** ≤ 1 mile  
- **Medium:** 1–5 miles  
- **Long:** > 5 miles  

Median fare-per-mile was computed per regime.

### Finding
There is **no single universal cost-per-mile**:
- Short trips exhibit inflated cost-per-mile due to fixed charges
- Long trips dilute fixed components

**Conclusion:** Distance is important, but insufficient on its own.

---

## Step 2 — Deviation Analysis
Deviation from the distance-only baseline was analyzed across:
- Pickup hour
- Weekday vs weekend
- Pickup location (coarse spatial grids)

Robust metrics (median, percentiles) were emphasized over means.

### Findings
- Fare predictability deteriorates during **peak hours**
- Certain pickup zones consistently show **high variability**
- Variability is structured, not random

---

## Step 3 — Fare Decomposition
Fare was decomposed into:
- Distance-driven component
- **Fixed fees** (extras, MTA tax, improvement surcharge)
- **Tolls**

### Findings
- Fixed fees dominate short trips
- Tolls are episodic but introduce large deviations
- These components explain much of the observed unpredictability

---

## Step 4 — Machine Learning: Quantifying Explanatory Power

Two models were trained on a **representative 1% sample** for local experimentation.

### Models
- **Model 0:** Distance-only Linear Regression  
- **Model 1:** Context-aware Random Forest  
  - Features: distance, trip duration, fixed fees, tolls, pickup hour, weekend flag, rate code, pickup grid

> *Trip duration acts as a proxy for traffic and time-based meter effects.*

### Results

| Model | MAE | RMSE |
|-----|----|----|
| Distance-only Linear Regression | **1.64** | **2.79** |
| Context-aware Random Forest | **0.31** | **1.14** |

### Interpretation
Adding trip duration, fare components, and contextual features reduces:
- **Typical error (MAE)** by ~80%
- **Large errors (RMSE)** by ~60%

> Note: Duration is known only post-trip; models are used for **explanatory analysis**, not real-time fare prediction.

---

## Residual Analysis
Residuals were analyzed by:
- Pickup hour
- Weekend vs weekday
- Distance regime

### Observations
- Distance-only errors spike during congestion-heavy hours
- Context-aware model consistently reduces error across all segments
- Remaining error aligns with peak traffic variability

---

## Key Takeaways
- Taxi fares are **not linear in distance**
- Fare variability arises systematically from:
  - Time (traffic)
  - Location
  - Fixed fees
  - Tolls
- Machine learning complements analysis by **quantifying**, not replacing, domain reasoning

---

## Why Spark Was Necessary
Although the data is available as CSV, Pandas is impractical due to:
- Memory constraints
- Repeated full-table scans
- Inefficient I/O

**Parquet + Spark** enables:
- Columnar access
- Predicate pushdown
- Reusable, scalable pipelines

---

## Limitations
- Analysis is based on historical data (2015)
- Duration-based modeling is explanatory, not predictive at pickup time
- Models trained on sampled data for local execution

---

## Future Extensions
- Apply pipeline to newer TLC datasets
- Compare regulated taxi pricing with dynamic pricing (e.g., ride-hailing)
- Add visual dashboards for spatial residual patterns

---

## Skills Demonstrated
- Large-scale data processing (PySpark)
- Analytical hypothesis testing
- Robust statistical reasoning
- Feature engineering
- Interpretable machine learning
- Clear communication of assumptions and limitations