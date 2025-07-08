import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, accuracy_score
from prophet import Prophet
import joblib
import os

# --- Setup ---
DATA_DIR = 'data'
MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print("--- IntelliChill Model Training Suite ---")

# --- Load Data ---
print("Loading data...")
products_df = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))
journeys_df = pd.read_csv(os.path.join(DATA_DIR, 'shipment_journeys.csv'), parse_dates=['Timestamp'])
demand_df = pd.read_csv(os.path.join(DATA_DIR, 'historical_demand.csv'), parse_dates=['Date'])
reefer_df = pd.read_csv(os.path.join(DATA_DIR, 'reefer_telemetry.csv'), parse_dates=['Timestamp'])

# --- Feature Engineering for Shipment-level models ---
print("Performing feature engineering for RSL and Spoilage models...")
journeys_df = pd.merge(journeys_df, products_df, on='ProductID')
shipment_summary = journeys_df.groupby('ShipmentID').agg(
    ProductID=('ProductID', 'first'), BaseShelfLife_days=('BaseShelfLife_days', 'first'),
    Temp_Min_C=('Temp_Min_C', 'first'), Temp_Max_C=('Temp_Max_C', 'first'),
    Avg_Temp=('Temperature_C', 'mean'), Std_Temp=('Temperature_C', 'std'),
    Avg_Humidity=('Humidity_pct', 'mean'),
    Journey_Duration_hours=('Timestamp', lambda x: (x.max() - x.min()).total_seconds() / 3600),
    Spoiled_GroundTruth=('Spoiled', 'max')
).reset_index()
shipment_summary['Temp_Spikes'] = journeys_df.groupby('ShipmentID').apply(
    lambda g: ((g['Temperature_C'] > g['Temp_Max_C']) | (g['Temperature_C'] < g['Temp_Min_C'])).sum()
).values

# --- 1. Train RSL (Remaining Shelf Life) Regression Model ---
print("\n--- Training RSL Model ---")
temp_deviation = np.abs(shipment_summary['Avg_Temp'] - (shipment_summary['Temp_Max_C'] + shipment_summary['Temp_Min_C'])/2)
penalty_hours = (shipment_summary['Temp_Spikes'] * 12) + (temp_deviation * 4)
shipment_summary['Actual_RSL_hours'] = (shipment_summary['BaseShelfLife_days'] * 24) - penalty_hours
shipment_summary['Actual_RSL_hours'] = shipment_summary['Actual_RSL_hours'].clip(lower=0)
features_rsl = ['Avg_Temp', 'Std_Temp', 'Avg_Humidity', 'Journey_Duration_hours', 'Temp_Spikes', 'BaseShelfLife_days']
target_rsl = 'Actual_RSL_hours'
X_train, X_test, y_train, y_test = train_test_split(shipment_summary[features_rsl], shipment_summary[target_rsl], test_size=0.2, random_state=42)
rsl_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
rsl_model.fit(X_train, y_train)
print(f"RSL Model R^2 Score: {r2_score(y_test, rsl_model.predict(X_test)):.2f}")
joblib.dump(rsl_model, os.path.join(MODEL_DIR, 'rsl_model.pkl'))
print(f"RSL model saved to {MODEL_DIR}/rsl_model.pkl")

# --- 2. Train Spoilage Risk Classification Model ---
print("\n--- Training Spoilage Risk Model ---")
features_spoilage = ['Avg_Temp', 'Std_Temp', 'Temp_Spikes', 'Journey_Duration_hours']
target_spoilage = 'Spoiled_GroundTruth'
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(shipment_summary[features_spoilage], shipment_summary[target_spoilage], test_size=0.2, random_state=42, stratify=shipment_summary[target_spoilage])
spoilage_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
spoilage_model.fit(X_train_s, y_train_s)
print(f"Spoilage Model Accuracy: {accuracy_score(y_test_s, spoilage_model.predict(X_test_s)):.2f}")
joblib.dump(spoilage_model, os.path.join(MODEL_DIR, 'spoilage_model.pkl'))
print(f"Spoilage model saved to {MODEL_DIR}/spoilage_model.pkl")

# --- 3. Train Demand Forecasting Models ---
print("\n--- Training Demand Forecasting Models ---")
demand_models = {}
unique_combinations = demand_df[['WarehouseID', 'ProductID']].drop_duplicates().sample(5, random_state=42)
for _, row in unique_combinations.iterrows():
    wh_id, prod_id = row['WarehouseID'], row['ProductID']
    print(f"Training demand model for {wh_id} - {prod_id}...")
    df_subset = demand_df[(demand_df['WarehouseID'] == wh_id) & (demand_df['ProductID'] == prod_id)].copy()
    df_subset.rename(columns={'Date': 'ds', 'SalesVolume': 'y'}, inplace=True)
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    prophet_model.fit(df_subset[['ds', 'y']])
    demand_models[f'{wh_id}_{prod_id}'] = prophet_model
joblib.dump(demand_models, os.path.join(MODEL_DIR, 'demand_models.pkl'))
print(f"\nSaved {len(demand_models)} demand models to {MODEL_DIR}/demand_models.pkl")

# --- 4. Train Predictive Maintenance Model ---
print("\n--- Training Predictive Maintenance Model ---")
reefer_summary = reefer_df.groupby('ShipmentID').agg(
    Avg_PowerDraw=('PowerDraw_kW', 'mean'), Std_PowerDraw=('PowerDraw_kW', 'std'),
    Avg_Vibration=('Vibration_mm_s', 'mean'), Std_Vibration=('Vibration_mm_s', 'std'),
    Failure_Likely=('Failure_Likely', 'max')
).reset_index()
features_maint = ['Avg_PowerDraw', 'Std_PowerDraw', 'Avg_Vibration', 'Std_Vibration']
target_maint = 'Failure_Likely'
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(reefer_summary[features_maint], reefer_summary[target_maint], test_size=0.2, random_state=42, stratify=reefer_summary[target_maint])
maintenance_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
maintenance_model.fit(X_train_m, y_train_m)
print(f"Maintenance Model Accuracy: {maintenance_model.score(X_test_m, y_test_m):.2f}")
joblib.dump(maintenance_model, os.path.join(MODEL_DIR, 'maintenance_model.pkl'))
print(f"Maintenance model saved to {MODEL_DIR}/maintenance_model.pkl")

# --- 5. Train Warehouse Inventory Clustering Model ---
print("\n--- Training Warehouse Clustering Model ---")
inventory_summary = journeys_df.groupby('ShipmentID').last().reset_index()
inventory_summary['Predicted_RSL_days'] = np.random.uniform(1, 20, len(inventory_summary))
inventory_summary['Demand_Forecast'] = np.random.uniform(10, 100, len(inventory_summary))
features_cluster = ['Predicted_RSL_days', 'Demand_Forecast']
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
inventory_summary['Cluster'] = kmeans.fit_predict(inventory_summary[features_cluster])
joblib.dump(kmeans, os.path.join(MODEL_DIR, 'warehouse_cluster_model.pkl'))
print(f"Warehouse clustering model saved to {MODEL_DIR}/warehouse_cluster_model.pkl")

print("\n✅ All models trained and saved successfully!")