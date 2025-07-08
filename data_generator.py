import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# --- Configuration ---
NUM_SHIPMENTS = 100
NUM_PRODUCTS = 5
NUM_WAREHOUSES = 3
START_DATE = datetime(2023, 1, 1)
JOURNEY_DAYS = 7
DATA_POINTS_PER_HOUR = 4


def generate_product_data(path='data/products.csv'):
    """Generates a CSV with product information."""
    products = {
        'ProductID': [f'P{100 + i}' for i in range(NUM_PRODUCTS)],
        'ProductName': ['Organic Milk', 'Fresh Strawberries', 'Insulin Vials', 'Gourmet Cheese', 'Frozen Pizza'],
        'BaseShelfLife_days': [14, 7, 365, 30, 180],
        'Temp_Min_C': [1, 1, 2, 2, -20],
        'Temp_Max_C': [4, 5, 8, 7, -15]
    }
    df = pd.DataFrame(products)
    df.to_csv(path, index=False)
    print(f"Generated {path}")
    return df


def generate_shipment_journeys(products_df, path='data/shipment_journeys.csv'):
    """Generates the core 'IoT' sensor data for multiple shipments."""
    shipment_data = []
    locations = {"WH1": (40.7128, -74.0060), "WH2": (34.0522, -118.2437), "WH3": (41.8781, -87.6298)}
    warehouse_ids = list(locations.keys())

    for i in range(NUM_SHIPMENTS):
        shipment_id = f'S{1000 + i}'
        product = products_df.sample(1).iloc[0]
        start_wh = random.choice(warehouse_ids)
        end_wh = random.choice([wh for wh in warehouse_ids if wh != start_wh])
        lat_start, lon_start = locations[start_wh]
        lat_end, lon_end = locations[end_wh]

        total_data_points = JOURNEY_DAYS * 24 * DATA_POINTS_PER_HOUR
        spoiled_shipment = random.random() < 0.2
        spoilage_events = 0

        for j in range(total_data_points):
            timestamp = START_DATE + timedelta(days=i // 10) + timedelta(minutes=j * (60 // DATA_POINTS_PER_HOUR))
            base_temp = (product['Temp_Min_C'] + product['Temp_Max_C']) / 2
            temp = base_temp + np.random.randn() * 0.5

            if spoiled_shipment and random.random() < 0.05:
                excursion = random.uniform(3, 8)
                temp += excursion if random.random() < 0.5 else -excursion
                spoilage_events += 1

            humidity = random.uniform(40, 60) + np.random.randn() * 5
            progress = j / total_data_points
            lat = lat_start + (lat_end - lat_start) * progress + np.random.randn() * 0.01
            lon = lon_start + (lon_end - lon_start) * progress + np.random.randn() * 0.01

            shipment_data.append({
                'Timestamp': timestamp, 'ShipmentID': shipment_id, 'ProductID': product['ProductID'],
                'Temperature_C': round(temp, 2), 'Humidity_pct': round(humidity, 2),
                'Latitude': round(lat, 4), 'Longitude': round(lon, 4),
                'DestinationWarehouseID': end_wh, 'Spoiled': spoilage_events > 3
            })

    df = pd.DataFrame(shipment_data)
    df.to_csv(path, index=False)
    print(f"Generated {path}")


def generate_historical_demand(products_df, path='data/historical_demand.csv'):
    """Generates simulated sales data for demand forecasting."""
    demand_data = []
    warehouses = [f'WH{i + 1}' for i in range(NUM_WAREHOUSES)]
    cities = ['New York', 'Los Angeles', 'Chicago']
    current_date = START_DATE - timedelta(days=365 * 2)
    end_date = START_DATE + timedelta(days=JOURNEY_DAYS + 10)

    while current_date < end_date:
        for i, wh_id in enumerate(warehouses):
            for _, product in products_df.iterrows():
                day_of_year = current_date.timetuple().tm_yday
                seasonality = (np.sin(2 * np.pi * (day_of_year - 80) / 365) + 1) * 50
                weekly_spike = 30 if current_date.weekday() >= 4 else 0
                base_volume = random.randint(50, 150)
                sales = base_volume + seasonality + weekly_spike + random.randint(-20, 20)
                demand_data.append({
                    'Date': current_date.date(), 'WarehouseID': wh_id, 'WarehouseCity': cities[i],
                    'ProductID': product['ProductID'], 'SalesVolume': max(0, int(sales))
                })
        current_date += timedelta(days=1)

    df = pd.DataFrame(demand_data)
    df.to_csv(path, index=False)
    print(f"Generated {path}")


def generate_reefer_telemetry(path='data/reefer_telemetry.csv'):
    """Generates telemetry data for predictive maintenance of refrigeration units."""
    telemetry_data = []
    shipment_ids = [f'S{1000 + i}' for i in range(NUM_SHIPMENTS)]

    for shipment_id in shipment_ids:
        is_failing_unit = random.random() < 0.1
        total_data_points = JOURNEY_DAYS * 24 * DATA_POINTS_PER_HOUR
        for j in range(total_data_points):
            timestamp = START_DATE + timedelta(days=int(shipment_id[1:]) % 10) + timedelta(
                minutes=j * (60 // DATA_POINTS_PER_HOUR))
            power_draw = 4.5 + np.random.randn() * 0.2
            vibration = 0.5 + np.random.randn() * 0.1
            compressor_cycles = 10 + np.random.randn()
            failure_imminent = False
            if is_failing_unit and j > total_data_points * 0.6:
                progress_to_failure = (j - total_data_points * 0.6) / (total_data_points * 0.4)
                power_draw += progress_to_failure * 1.5
                vibration += progress_to_failure * 0.8
                failure_imminent = progress_to_failure > 0.8
            telemetry_data.append({
                'Timestamp': timestamp, 'ShipmentID': shipment_id, 'PowerDraw_kW': round(power_draw, 2),
                'Vibration_mm_s': round(vibration, 2), 'CompressorCycles_per_hr': round(compressor_cycles, 2),
                'Failure_Likely': failure_imminent
            })
    df = pd.DataFrame(telemetry_data)
    df.to_csv(path, index=False)
    print(f"Generated {path}")


def generate_geospatial_risks(path='data/geospatial_risks.csv'):
    """Generates a static file of external risks."""
    risks = [
        {'RiskID': 'R1', 'RiskType': 'Hurricane Warning', 'Latitude': 28.5383, 'Longitude': -81.3792, 'Radius_km': 300,
         'Severity': 0.9},
        {'RiskID': 'R2', 'RiskType': 'Wildfire', 'Latitude': 34.0522, 'Longitude': -118.2437, 'Radius_km': 100,
         'Severity': 0.7},
        {'RiskID': 'R3', 'RiskType': 'Extreme Heatwave', 'Latitude': 33.4484, 'Longitude': -112.0740, 'Radius_km': 400,
         'Severity': 0.6},
        {'RiskID': 'R4', 'RiskType': 'Flood Zone', 'Latitude': 29.7604, 'Longitude': -95.3698, 'Radius_km': 150,
         'Severity': 0.8},
    ]
    df = pd.DataFrame(risks)
    df.to_csv(path, index=False)
    print(f"Generated {path}")


if __name__ == '__main__':
    DATA_DIR = 'data'
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    products = generate_product_data(path=os.path.join(DATA_DIR, 'products.csv'))
    generate_shipment_journeys(products, path=os.path.join(DATA_DIR, 'shipment_journeys.csv'))
    generate_historical_demand(products, path=os.path.join(DATA_DIR, 'historical_demand.csv'))
    generate_reefer_telemetry(path=os.path.join(DATA_DIR, 'reefer_telemetry.csv'))
    generate_geospatial_risks(path=os.path.join(DATA_DIR, 'geospatial_risks.csv'))

    print("\n✅ All data files generated successfully in 'data/' directory!")
