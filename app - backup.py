import streamlit as st
import pickle
import pandas as pd

# Load trained model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load model columns
with open("model_columns.pkl", "rb") as file:
    model_columns = pickle.load(file)

# Load sample data
with open("house_sample.pkl", "rb") as file:
    sample = pickle.load(file)

# Convert sample data into DataFrame
input_data = pd.DataFrame([sample])
input_data = input_data.reindex(columns=model_columns)

# Streamlit UI
st.title("🏠 House Price Prediction")
st.write("Predict house prices using Machine Learning")

st.header("Enter House Details")

area = st.number_input(
    "Living Area (Square Feet)",
    min_value=100,
    max_value=10000,
    value=1500
)

lot_area = st.number_input(
    "Lot Area (Square Feet)",
    min_value=500,
    max_value=100000,
    value=8400
)

quality = st.slider(
    "Overall House Quality",
    min_value=1,
    max_value=10,
    value=5
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

bathrooms = st.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

year_built = st.number_input(
    "Year Built",
    min_value=1800,
    max_value=2026,
    value=2000
)

if st.button("Predict House Price"):

    # Update user inputs
    input_data["GrLivArea"] = area
    input_data["LotArea"] = lot_area
    input_data["OverallQual"] = quality
    input_data["BedroomAbvGr"] = bedrooms
    input_data["FullBath"] = bathrooms
    input_data["YearBuilt"] = year_built

    # Predict price
    prediction = model.predict(input_data)

    st.success("Prediction Successful!")

    st.subheader("Estimated House Price")

    st.write(f"${prediction[0]:,.2f}")