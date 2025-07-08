# 🧊 IntelliChill: AI-Powered Cold Chain Command Center

**Tagline:** *From Predictive Maintenance to Geo-Spatial Risk Analysis, Command Your Supply Chain with Intelligence.*

IntelliChill is an end-to-end cold chain monitoring and management platform. It moves beyond simple tracking to provide a predictive, multi-faceted command center for large-scale logistics operations like Walmart or Amazon. Our goal is to minimize waste, mitigate risks, and maximize the value of perishable goods through data science.

---

### ✨ Key Features (v2.0)

This platform is built around several interconnected, ML-powered dashboards:

1.  **Live Operations Dashboard:**
    *   **What it does:** Real-time map of all shipments, color-coded by a live spoilage risk score.
    *   **ML Model:** Spoilage Risk Classifier.

2.  **Product Lifecycle Management (PLM):**
    *   **What it does:** A full, end-to-end "health-over-time" view for any specific shipment. See how its Remaining Shelf Life (RSL) evolved throughout its journey and what events impacted it.
    *   **ML Model:** Dynamic RSL Regression Model.

3.  **Advanced Warehouse Management:**
    *   **What it does:** Moves beyond simple inventory lists. It uses clustering to group inventory into "Action Zones" (e.g., "Relocate Urgently," "Promote Locally," "Standard Dispatch").
    *   **ML Model:** K-Means Clustering.

4.  **Predictive Equipment Maintenance:**
    *   **What it does:** Monitors the health of the refrigeration units (reefers) themselves. Predicts potential failures *before* they happen, allowing for preventive maintenance.
    *   **ML Model:** Anomaly Detection / Failure Prediction Classifier.

5.  **Geo-Spatial Risk Analysis:**
    *   **What it does:** Analyzes planned shipment routes against a live map of external risks like extreme weather (hurricanes, heatwaves) and natural disasters. It calculates a "Route Risk Score" and can suggest safer alternatives.
    *   **Engine:** Risk Scoring Algorithm.

---

### 🛠️ Technology Stack

- **Dashboard:** Streamlit
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Gradient Boosting, K-Means), Prophet
- **Visualization:** Plotly, Matplotlib
- **Deployment (Conceptual):** Docker, AWS/GCP

---

### 🚀 How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/IntelliChill.git
    cd IntelliChill
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Generate the simulated data:**
    ```bash
    python src/data_generator.py
    ```
    This will create all necessary CSV files inside the `data/` directory.

4.  **Train the machine learning models:**
    ```bash
    python src/model_trainer.py
    ```
    This will train all models and save the `.pkl` files into the `models/` directory.

5.  **Run the Streamlit application:**
    ```bash
    streamlit run src/app.py
    ```
    Your browser will open with the IntelliChill dashboard.

---

### 📁 Project Structure
