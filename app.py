import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from haversine import haversine
import numpy as np
# --- Page Config ---
st.set_page_config(page_title="IntelliChill v2.0", layout="wide")


# --- Load Models & Data (cached) ---
@st.cache_resource
def load_all():
    models = {
        "rsl": joblib.load('models/rsl_model.pkl'),
        "spoilage": joblib.load('models/spoilage_model.pkl'),
        "demand": joblib.load('models/demand_models.pkl'),
        "maintenance": joblib.load('models/maintenance_model.pkl'),
        "clustering": joblib.load('models/warehouse_cluster_model.pkl')
    }
    data = {
        "products": pd.read_csv('data/products.csv'),
        "journeys": pd.read_csv('data/shipment_journeys.csv', parse_dates=['Timestamp']),
        "demand": pd.read_csv('data/historical_demand.csv'),
        "risks": pd.read_csv('data/geospatial_risks.csv'),
        "reefer": pd.read_csv('data/reefer_telemetry.csv', parse_dates=['Timestamp'])
    }
    data["journeys"] = pd.merge(data["journeys"], data["products"], on='ProductID')
    return models, data


models, data = load_all()

# --- Sidebar Navigation ---
st.sidebar.title("🧊 IntelliChill v2.0")
PAGES = [
    "Live Operations",
    "Product Lifecycle",
    "Advanced Warehouse Mgt.",
    "Predictive Maintenance",
    "Geo-Spatial Risk Analysis"
]
page = st.sidebar.radio("Select Dashboard", PAGES)


# --- Page Rendering Functions ---

def render_live_operations():
    # This page remains largely the same as the "Operations Dashboard" from the previous response
    st.title("🔴 Live Operations Dashboard")
    # ... (Paste code from previous response's 'Operations Dashboard' page)
    pass


def render_product_lifecycle():
    st.title("ιχ Product Lifecycle Management")
    st.markdown("Trace the entire health journey of a single shipment.")

    shipment_id = st.selectbox("Select a Shipment ID", data['journeys']['ShipmentID'].unique())

    if shipment_id:
        shipment_df = data['journeys'][data['journeys']['ShipmentID'] == shipment_id].copy()

        # Calculate RSL evolution over time
        rsl_over_time = []
        for i in range(1, len(shipment_df)):
            subset = shipment_df.iloc[:i + 1]
            # (Simplified RSL calculation for demo)
            spikes = ((subset['Temperature_C'] > subset['Temp_Max_C'].iloc[0]) | (
                        subset['Temperature_C'] < subset['Temp_Min_C'].iloc[0])).sum()
            base_life_hours = subset['BaseShelfLife_days'].iloc[0] * 24
            rsl = base_life_hours - (spikes * 12)  # 12-hour penalty per spike
            rsl_over_time.append({'Timestamp': subset['Timestamp'].iloc[-1], 'Predicted_RSL_hours': rsl})

        rsl_df = pd.DataFrame(rsl_over_time)

        fig = px.line(rsl_df, x='Timestamp', y='Predicted_RSL_hours', title=f"RSL Evolution for {shipment_id}",
                      markers=True)
        fig.add_scatter(x=shipment_df['Timestamp'], y=shipment_df['Temperature_C'], name='Temperature', yaxis='y2')
        fig.update_layout(yaxis2=dict(title='Temperature (°C)', overlaying='y', side='right'))
        st.plotly_chart(fig, use_container_width=True)


def render_warehouse_management():
    st.title("📦 Advanced Warehouse Management")
    st.markdown("Use clustering to create intelligent action zones for your inventory.")

    # Simulate current inventory with RSL and Demand
    inventory_df = data['journeys'].groupby('ShipmentID').last().reset_index()
    inventory_df['Predicted_RSL_days'] = np.random.uniform(1, 20, len(inventory_df))
    inventory_df['Demand_Forecast'] = np.random.uniform(10, 100, len(inventory_df))

    # Predict clusters
    cluster_model = models['clustering']
    inventory_df['ActionZone'] = cluster_model.predict(inventory_df[['Predicted_RSL_days', 'Demand_Forecast']])

    # Interpret clusters (this requires knowing what the clusters mean)
    zone_map = {0: "⚠️ Relocate / Promote", 1: "✅ Healthy Stock", 2: "🚀 Prioritize Dispatch"}
    inventory_df['ActionZone'] = inventory_df['ActionZone'].map(zone_map)

    fig = px.scatter(inventory_df, x='Predicted_RSL_days', y='Demand_Forecast', color='ActionZone',
                     hover_data=['ShipmentID', 'ProductName'], title="Warehouse Inventory Action Zones")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(inventory_df[
                     ['ShipmentID', 'ProductName', 'DestinationWarehouseID', 'Predicted_RSL_days', 'Demand_Forecast',
                      'ActionZone']], use_container_width=True)


def render_predictive_maintenance():
    st.title("🔧 Predictive Equipment Maintenance")
    st.markdown("Monitor reefer health and predict failures before they impact your cargo.")

    reefer_summary = data['reefer'].groupby('ShipmentID').agg(
        Avg_PowerDraw=('PowerDraw_kW', 'mean'),
        Std_PowerDraw=('PowerDraw_kW', 'std'),
        Avg_Vibration=('Vibration_mm_s', 'mean'),
        Std_Vibration=('Vibration_mm_s', 'std'),
    ).reset_index()

    maint_model = models['maintenance']
    features = ['Avg_PowerDraw', 'Std_PowerDraw', 'Avg_Vibration', 'Std_Vibration']
    reefer_summary['Failure_Risk'] = maint_model.predict_proba(reefer_summary[features])[:, 1]

    st.dataframe(reefer_summary.sort_values('Failure_Risk', ascending=False), use_container_width=True)

    st.subheader("At-Risk Units")
    at_risk_units = reefer_summary[reefer_summary['Failure_Risk'] > 0.5]
    if not at_risk_units.empty:
        st.warning(f"Found {len(at_risk_units)} units with high failure risk. Advise immediate inspection.")
    else:
        st.success("All monitored refrigeration units are operating within normal parameters.")


def render_geo_risk_analysis():
    st.title("🗺️ Geo-Spatial Risk Analysis")
    st.markdown("Analyze shipment routes against external threats like weather and natural disasters.")

    risk_df = data['risks']
    shipment_id = st.selectbox("Select a Shipment to Analyze", data['journeys']['ShipmentID'].unique())

    shipment_route = data['journeys'][data['journeys']['ShipmentID'] == shipment_id]

    # Plot map with route and risk zones
    st.map(shipment_route, latitude='Latitude', longitude='Longitude')

    st.subheader("Active Risk Zones")
    st.dataframe(risk_df)

    # Calculate route risk
    total_risk_score = 0
    risky_segments = []
    for _, point in shipment_route.iterrows():
        for _, risk in risk_df.iterrows():
            distance = haversine((point['Latitude'], point['Longitude']), (risk['Latitude'], risk['Longitude']))
            if distance <= risk['Radius_km']:
                total_risk_score += risk['Severity']
                risky_segments.append(risk['RiskType'])

    st.metric("Route Risk Score", f"{total_risk_score:.2f}")
    if risky_segments:
        st.error(
            f"Route for {shipment_id} intersects with high-risk zones: {', '.join(set(risky_segments))}. Consider re-routing.")
    else:
        st.success(f"Route for {shipment_id} appears clear of major known risks.")


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