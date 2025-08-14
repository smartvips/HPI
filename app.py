
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="House Pricing Prediction", page_icon="🏠")
st.title("🏠 House Pricing Prediction")

st.write("Upload a CSV with features and a **Price** target column.")

# Upload CSV
file = st.file_uploader("Upload CSV", type=["csv"])

# Your continuous columns
continuous_cols = [
    'lot area',
    'Area of the house(excluding basement)',
    'Area of the basement',
    'living_area_renov',
    'lot_area_renov',
    'Distance from the airport',
    'Property Age',
    'Renovation Age'
]

if not file:
    df = pd.read_csv("Data.csv") 
    st.info("No file uploaded. Using sample data.")
    
else:
    df = pd.read_csv(file)

if "Price" not in df.columns:
    st.error("Your dataset must have a target column named 'Price'.")
    st.stop()

st.subheader("Data Preview")
st.dataframe(df.head())

# Features/Target
feature_cols = [c for c in df.columns if c != "Price"]
X = df[feature_cols].copy()
y = df["Price"].copy()

# Keep only numeric cols
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Scale only the continuous columns that actually exist in the data
cols_to_scale = [c for c in continuous_cols if c in numeric_cols]
if not cols_to_scale:
    cols_to_scale = numeric_cols  # fallback

preprocess = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), cols_to_scale),
        ("pass", "passthrough", [c for c in numeric_cols if c not in cols_to_scale])
    ],
    remainder="drop",
)

pipe = Pipeline([
    ("prep", preprocess),
    ("lr", LinearRegression())
])

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X[numeric_cols], y, test_size=0.2, random_state=45
)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.subheader("Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"{mae:,.2f}")
c2.metric("MSE", f"{mse:,.2f}")
c3.metric("RMSE", f"{rmse:,.2f}")
c4.metric("R²", f"{r2:.3f}")

# Scatter plot
st.subheader("Actual vs Predicted Prices")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.7)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Actual vs Predicted")
st.pyplot(fig)

# ---- Predict with fewer inputs ----
st.subheader("Predict (Fewer Inputs)")

# Choose a small subset of "basic" inputs to ask from the user
basic_inputs = [
    'Area of the house(excluding basement)',
    'lot area',
    'Property Age',
    'Distance from the airport'
]
# Keep only those that truly exist in the dataset's numeric columns
basic_inputs = [c for c in basic_inputs if c in numeric_cols]

# Compute dataset medians for all numeric features to auto-fill missing ones
medians = X[numeric_cols].median(numeric_only=True).to_dict()

with st.form("predict_basic"):
    st.caption("Enter a few basic values; the rest of the model inputs will use the dataset medians.")
    user_vals = {}
    for col in basic_inputs:
        default = float(medians.get(col, 0.0))
        user_vals[col] = st.number_input(col, value=default)
    submitted = st.form_submit_button("Predict Price")
    if submitted:
        # Start with medians for all numeric features
        row = {c: float(medians.get(c, 0.0)) for c in numeric_cols}
        # Override with user-provided values for the basic subset
        for k, v in user_vals.items():
            row[k] = v
        new_df = pd.DataFrame([row], columns=numeric_cols)
        pred = pipe.predict(new_df)[0]
        st.success(f"Predicted Price: {pred:,.2f}")
