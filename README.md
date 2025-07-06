
# Predictive Maintenance with CMAPSS Dataset

This project applies machine learning techniques (Random Forest) to predict the Remaining Useful Life (RUL) of engines using the CMAPSS dataset from NASA.

---

## Dataset: CMAPSS - FD001, FD002

The CMAPSS dataset includes sensor readings from turbofan engines over time, under various operating conditions. Each engine runs until failure. The goal is to predict how many cycles are left before failure (RUL).

### Files:
FD001: 1 operating condition, 1 fault mode, 100 engines<br>
FD002: 6 operating conditions, 1 fault mode, 260 engines<br>
FD003: 1 operating condition, 2 fault modes, 100 engines<br>
FD004: 6 operating conditions, 2 fault modes, 248 engines<br>

- `train_FD001.txt`: Full sensor data for engines until failure.
- `test_FD001.txt`: Partial data for engines currently in operation.
- `RUL_FD001.txt`: True remaining cycles for test engines.

Each row includes:
- `unit`: Engine ID
- `cycle`: Time cycle
- `op_setting_1` to `op_setting_3`: Operating conditions
- `sensor_1` to `sensor_21`: Sensor readings
---

## Code Overview

### 1. `read_dataset.py`

```python
train_df, column_names = load_train_data("CMAPSSData")
test_df, rul_true = load_test_data("CMAPSSData", column_names)
```
- Loads training and test datasets
- Computes RUL for each training instance

### 2. `solver.py`

```python
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)
```
- Trains a Random Forest Regressor
- Scales sensor data using MinMaxScaler
- Evaluates model using MAE and RMSE

---

## How to Run

Ensure the folder `CMAPSSData/` contains:
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

Then run:

```bash
python solver.py
```

---

## Example Output (FD001)

```
Validation MAE: 29.57
Validation RMSE: 41.41
Test MAE: 24.21
Test RMSE: 33.15
```

## Example Output (FD002)

```
Validation MAE: 31.78
Validation RMSE: 43.33
Test MAE: 23.08
Test RMSE: 31.32
```

## Example Output (FD003)

```
Validation MAE: 37.94
Validation RMSE: 55.55
Test MAE: 32.02
Test RMSE: 44.61
```

## Example Output (FD004)

```
Validation MAE: 40.13
Validation RMSE: 55.59
Test MAE: 31.94
Test RMSE: 43.78
```

---

## Conclusion

This project demonstrates the use of supervised learning to forecast machine failure and support predictive maintenance. The accuracy achieved shows promise for real-world deployment in industrial settings.
