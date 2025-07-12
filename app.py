import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from haversine import haversine
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="IntelliChill v2.0",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Path Configuration ---
MODEL_DIR = 'models'
DATA_DIR = 'data'


# --- Caching: Load Models & Data ---
@st.cache_resource
def load_all():
    """Loads all models and data from disk, caching the results."""
    models = {
        "rsl": joblib.load(os.path.join(MODEL_DIR, 'rsl_model.pkl')),
        "spoilage": joblib.load(os.path.join(MODEL_DIR, 'spoilage_model.pkl')),
        "demand": joblib.load(os.path.join(MODEL_DIR, 'demand_models.pkl')),
        "maintenance": joblib.load(os.path.join(MODEL_DIR, 'maintenance_model.pkl')),
        "clustering": joblib.load(os.path.join(MODEL_DIR, 'warehouse_cluster_model.pkl'))
    }
    data = {
        "products": pd.read_csv(os.path.join(DATA_DIR, 'products.csv')),
        "journeys": pd.read_csv(os.path.join(DATA_DIR, 'shipment_journeys.csv'), parse_dates=['Timestamp']),
        "demand": pd.read_csv(os.path.join(DATA_DIR, 'historical_demand.csv')),
        "risks": pd.read_csv(os.path.join(DATA_DIR, 'geospatial_risks.csv')),
        "reefer": pd.read_csv(os.path.join(DATA_DIR, 'reefer_telemetry.csv'), parse_dates=['Timestamp'])
    }
    data["journeys"] = pd.merge(data["journeys"], data["products"], on='ProductID')
    return models, data


try:
    models, data = load_all()
except FileNotFoundError:
    st.error("Error: Model or data files not found. Please run `data_generator.py` and `model_trainer.py` first.")
    st.stop()


# --- Helper Functions ---
def predict_live_shipment(shipment_df):
    """Predicts RSL and Spoilage for a single, ongoing shipment."""
    features_rsl = ['Avg_Temp', 'Std_Temp', 'Avg_Humidity', 'Journey_Duration_hours', 'Temp_Spikes',
                    'BaseShelfLife_days']
    features_spoilage = ['Avg_Temp', 'Std_Temp', 'Temp_Spikes', 'Journey_Duration_hours']
    summary = {
        'Avg_Temp': shipment_df['Temperature_C'].mean(), 'Std_Temp': shipment_df['Temperature_C'].std(),
        'Avg_Humidity': shipment_df['Humidity_pct'].mean(),
        'Journey_Duration_hours': (shipment_df['Timestamp'].max() - shipment_df[
            'Timestamp'].min()).total_seconds() / 3600,
        'Temp_Spikes': ((shipment_df['Temperature_C'] > shipment_df['Temp_Max_C'].iloc[0]) | (
                    shipment_df['Temperature_C'] < shipment_df['Temp_Min_C'].iloc[0])).sum(),
        'BaseShelfLife_days': shipment_df['BaseShelfLife_days'].iloc[0]
    }
    summary_df = pd.DataFrame([summary])
    summary_df.fillna(0, inplace=True)
    predicted_rsl_hours = models['rsl'].predict(summary_df[features_rsl])[0]
    risk_proba = models['spoilage'].predict_proba(summary_df[features_spoilage])[:, 1][0]
    return predicted_rsl_hours, risk_proba


# --- Sidebar Navigation ---
st.sidebar.title("🧊 IntelliChill v2.0")
PAGES = ["Live Operations", "Product Lifecycle", "Advanced Warehouse Mgt.", "Predictive Maintenance",
         "Geo-Spatial Risk Analysis"]
page = st.sidebar.radio("Select Dashboard", PAGES)


# --- Page Rendering Functions ---

def render_live_operations():
    st.title("🔴 Live Operations Dashboard")
    st.markdown("Monitor in-transit shipments in real-time.")

    # Simulate "live" data by taking a subset of shipments
    live_shipment_ids = data['journeys']['ShipmentID'].unique()[:15]
    live_journeys = data['journeys'][data['journeys']['ShipmentID'].isin(live_shipment_ids)].copy()

    map_data = []
    alerts = []

    # Check if there are any live journeys to process
    if not live_journeys.empty:
        for shipment_id in live_shipment_ids:
            shipment_df = live_journeys[live_journeys['ShipmentID'] == shipment_id]

            # Ensure the shipment has data points before proceeding
            if not shipment_df.empty:
                latest_point = shipment_df.iloc[-1]
                rsl, risk = predict_live_shipment(shipment_df)

                map_data.append({
                    'lat': latest_point['Latitude'],
                    'lon': latest_point['Longitude'],
                    'risk': risk,
                    'tooltip': f"ID: {shipment_id}\nProduct: {latest_point['ProductName']}\nRisk: {risk:.2%}"
                })

                if risk > 0.5 or (rsl / 24) < 2:
                    alerts.append({
                        "Shipment ID": shipment_id,
                        "Product": latest_point['ProductName'],
                        "Risk Score": f"{risk:.2%}",
                        "Predicted RSL (days)": f"{rsl / 24:.1f}",
                        "Destination": latest_point['DestinationWarehouseID']
                    })

    # --- KPI Metrics Section (Robustly handled) ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Live Shipments", len(live_shipment_ids))
    col2.metric("High-Risk Alerts", len(alerts))

    if map_data:
        avg_risk = np.mean([d['risk'] for d in map_data])
        col3.metric("Avg. Spoilage Risk", f"{avg_risk:.2%}")
    else:
        col3.metric("Avg. Spoilage Risk", "N/A")

    # --- Map Display Section (FIXED) ---
    st.subheader("Live Shipment Map")
    if map_data:
        # **THIS IS THE CORRECTED LINE**
        st.map(pd.DataFrame(map_data),
               latitude='lat',
               longitude='lon',
               color='#FF0000',  # Set a fixed red color using a hex string
               size='risk',  # The size of the dot is proportional to the 'risk' column
               zoom=3)
    else:
        st.info("No live shipment data to display on the map at this moment.")

    # --- Alerts Display Section (Robustly handled) ---
    st.subheader("🚨 Critical Alerts")
    if alerts:
        st.dataframe(pd.DataFrame(alerts), use_container_width=True)
    else:
        st.success("No critical alerts at this time. All monitored shipments are within safe parameters.")
def render_product_lifecycle():
    st.title("ιχ Product Lifecycle Management")
    st.markdown("Trace the entire health journey of a single shipment.")
    shipment_id = st.selectbox("Select a Shipment ID", data['journeys']['ShipmentID'].unique())
    if shipment_id:
        shipment_df = data['journeys'][data['journeys']['ShipmentID'] == shipment_id].copy()
        rsl_over_time = []
        for i in range(10, len(shipment_df), 10):
            subset = shipment_df.iloc[:i + 1]
            rsl, _ = predict_live_shipment(subset)
            rsl_over_time.append({'Timestamp': subset['Timestamp'].iloc[-1], 'Predicted_RSL_hours': rsl})

        rsl_df = pd.DataFrame(rsl_over_time)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rsl_df['Timestamp'], y=rsl_df['Predicted_RSL_hours'], name='Predicted RSL (hours)',
                                 line=dict(color='blue', width=3)))
        fig.add_trace(
            go.Scatter(x=shipment_df['Timestamp'], y=shipment_df['Temperature_C'], name='Temperature (°C)', yaxis='y2',
                       line=dict(color='red', dash='dot')))
        fig.update_layout(title_text=f"Health & Temperature Journey for {shipment_id}",
                          yaxis=dict(title='Predicted Remaining Shelf-Life (hours)'),
                          yaxis2=dict(title='Temperature (°C)', overlaying='y', side='right', showgrid=False))
        st.plotly_chart(fig, use_container_width=True)


def render_warehouse_management():
    st.title("📦 Advanced Warehouse Management")
    st.markdown("Use clustering to create intelligent action zones for your inventory.")
    inventory_df = data['journeys'].groupby('ShipmentID').last().reset_index()
    inventory_df['Predicted_RSL_days'] = inventory_df['ShipmentID'].apply(
        lambda id: predict_live_shipment(data['journeys'][data['journeys']['ShipmentID'] == id])[0] / 24)
    inventory_df['Demand_Forecast'] = np.random.uniform(50, 200, len(inventory_df))

    inventory_df['ActionZone'] = models['clustering'].predict(inventory_df[['Predicted_RSL_days', 'Demand_Forecast']])
    zone_map = {0: "⚠️ Relocate / Promote", 1: "✅ Healthy Stock", 2: "🚀 Prioritize Dispatch"}
    inventory_df['ActionZone'] = inventory_df['ActionZone'].map(zone_map)

    fig = px.scatter(inventory_df, x='Predicted_RSL_days', y='Demand_Forecast', color='ActionZone',
                     hover_data=['ShipmentID', 'ProductName', 'DestinationWarehouseID'],
                     title="Warehouse Inventory Action Zones",
                     color_discrete_map={"⚠️ Relocate / Promote": "orange", "✅ Healthy Stock": "green",
                                         "🚀 Prioritize Dispatch": "blue"})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(inventory_df[
                     ['ShipmentID', 'ProductName', 'DestinationWarehouseID', 'Predicted_RSL_days', 'Demand_Forecast',
                      'ActionZone']], use_container_width=True)


def render_predictive_maintenance():
    st.title("🔧 Predictive Equipment Maintenance")
    st.markdown("Monitor reefer health and predict failures before they impact your cargo.")
    reefer_summary = data['reefer'].groupby('ShipmentID').agg(
        Avg_PowerDraw=('PowerDraw_kW', 'mean'), Std_PowerDraw=('PowerDraw_kW', 'std'),
        Avg_Vibration=('Vibration_mm_s', 'mean'), Std_Vibration=('Vibration_mm_s', 'std'),
    ).reset_index()
    features = ['Avg_PowerDraw', 'Std_PowerDraw', 'Avg_Vibration', 'Std_Vibration']
    reefer_summary['Failure_Risk'] = models['maintenance'].predict_proba(reefer_summary[features])[:, 1]

    st.subheader("Reefer Unit Health Overview")
    st.dataframe(reefer_summary.sort_values('Failure_Risk', ascending=False), use_container_width=True)

    st.subheader("Units Requiring Immediate Attention")
    at_risk_units = reefer_summary[reefer_summary['Failure_Risk'] > 0.5]
    if not at_risk_units.empty:
        for _, row in at_risk_units.iterrows():
            st.warning(
                f"**Shipment {row['ShipmentID']}**: High failure risk detected ({row['Failure_Risk']:.1%}). Advise immediate inspection.")
    else:
        st.success("All monitored refrigeration units are operating within normal parameters.")


def render_geo_risk_analysis():
    st.title("🗺️ Geo-Spatial Risk Analysis")
    st.markdown("Analyze shipment routes against external threats like weather and natural disasters.")
    risk_df = data['risks']
    shipment_id = st.selectbox("Select a Shipment to Analyze", data['journeys']['ShipmentID'].unique())
    shipment_route = data['journeys'][data['journeys']['ShipmentID'] == shipment_id]

    st.subheader("Route and Active Risk Zones")
    map_layer = pd.concat([
        shipment_route[['Latitude', 'Longitude']].rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}),
        risk_df[['Latitude', 'Longitude']].rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
    ])
    st.map(map_layer)

    st.subheader("Route Risk Assessment")
    total_risk_score, risky_segments = 0, []
    for _, point in shipment_route.iterrows():
        for _, risk in risk_df.iterrows():
            distance = haversine((point['Latitude'], point['Longitude']), (risk['Latitude'], risk['Longitude']))
            if distance <= risk['Radius_km']:
                total_risk_score += risk['Severity']
                risky_segments.append(risk['RiskType'])

    st.metric("Total Route Risk Score", f"{total_risk_score:.2f} (Higher is worse)")
    if risky_segments:
        st.error(
            f"**Alert:** Route for `{shipment_id}` intersects with high-risk zones: **{', '.join(sorted(list(set(risky_segments))))}**. Consider re-routing or delaying shipment.")
    else:
        st.success(f"Route for `{shipment_id}` appears clear of major known geo-spatial risks.")
    st.dataframe(risk_df, use_container_width=True)


# --- Main App Router ---
if page == "Live Operations":
    render_live_operations()
elif page == "Product Lifecycle":
    render_product_lifecycle()
elif page == "Advanced Warehouse Mgt.":
    render_warehouse_management()
elif page == "Predictive Maintenance":
    render_predictive_maintenance()
elif page == "Geo-Spatial Risk Analysis":
    render_geo_risk_analysis()