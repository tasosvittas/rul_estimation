
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from read_dataset import load_train_data, load_test_data

#fortwsi dedomenwn
train_df, column_names = load_train_data("CMAPSSData")
test_df, rul_true = load_test_data("CMAPSSData", column_names)

#proetoimasia dedomenwn
features = [f"sensor_{i}" for i in range(1, 22)]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(train_df[features])
y = train_df["RUL"]
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#ekpaideusi montelou me to training set 
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

#validation set
y_pred = rf.predict(X_val)
print(f"Validation MAE: {mean_absolute_error(y_val, y_pred):.2f}")
print(f"Validation RMSE: {np.sqrt(mean_squared_error(y_val, y_pred)):.2f}")

#test set
last_cycles = test_df.groupby("unit").last().reset_index()
X_test = scaler.transform(last_cycles[features])
y_test_pred = rf.predict(X_test)
print(f"Test MAE: {mean_absolute_error(rul_true, y_test_pred):.2f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(rul_true, y_test_pred)):.2f}")
