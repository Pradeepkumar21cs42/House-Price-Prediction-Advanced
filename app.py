import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e0e0e0;
}

h1 {
    color: #1e3a8a;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD MODEL FILES
# ==========================================

@st.cache_resource
def load_model_files():

    model = joblib.load("best_house_price_model.pkl")

    features = joblib.load("house_price_features.pkl")

    defaults = joblib.load("house_price_defaults.pkl")

    neighborhoods = joblib.load("neighborhoods.pkl")

    return model, features, defaults, neighborhoods


try:

    model, features, defaults, neighborhoods = load_model_files()

except Exception as e:

    st.error(f"Error loading model files: {e}")

    st.stop()


# ==========================================
# 4. HEADER
# ==========================================

st.title("🏠 House Price Prediction")

st.markdown("""
### Advanced Machine Learning Application

Predict house prices using Machine Learning based on
property characteristics, quality, size and location.
""")

st.divider()


# ==========================================
# 5. SIDEBAR INPUTS
# ==========================================

st.sidebar.title("🏡 Property Details")

st.sidebar.markdown(
    "Enter the house details below to predict the estimated price."
)


house_area = st.sidebar.number_input(
    "House Area (sq ft)",
    min_value=100,
    max_value=10000,
    value=1500,
    step=100
)


lot_area = st.sidebar.number_input(
    "Lot Area (sq ft)",
    min_value=100,
    max_value=100000,
    value=8000,
    step=500
)


overall_quality = st.sidebar.slider(
    "Overall Quality",
    min_value=1,
    max_value=10,
    value=7
)


bedrooms = st.sidebar.number_input(
    "Bedrooms",
    min_value=0,
    max_value=15,
    value=3
)


bathrooms = st.sidebar.number_input(
    "Full Bathrooms",
    min_value=0,
    max_value=10,
    value=2
)


year_built = st.sidebar.number_input(
    "Year Built",
    min_value=1800,
    max_value=2026,
    value=2005
)


garage_capacity = st.sidebar.selectbox(
    "Garage Capacity",
    options=[0, 1, 2, 3, 4, 5],
    index=2
)


neighborhood = st.sidebar.selectbox(
    "Neighborhood",
    options=neighborhoods
)


st.sidebar.divider()

predict_button = st.sidebar.button(
    "🔮 Predict House Price",
    use_container_width=True
)


# ==========================================
# 6. MAIN DASHBOARD
# ==========================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "🏠 House Area",
        f"{house_area:,} sq ft"
    )


with col2:

    st.metric(
        "🛏️ Bedrooms",
        bedrooms
    )


with col3:

    st.metric(
        "⭐ Overall Quality",
        f"{overall_quality}/10"
    )


st.divider()


# ==========================================
# 7. PREDICTION
# ==========================================

if predict_button:

    try:

        input_data = defaults.copy()


        # User inputs

        input_data["GrLivArea"] = house_area

        input_data["LotArea"] = lot_area

        input_data["OverallQual"] = overall_quality

        input_data["BedroomAbvGr"] = bedrooms

        input_data["FullBath"] = bathrooms

        input_data["YearBuilt"] = year_built

        input_data["GarageCars"] = garage_capacity

        input_data["Neighborhood"] = neighborhood


        # Feature engineering

        input_data["HouseAge"] = (
            input_data["YrSold"] - year_built
        )


        input_data["YearsSinceRemodel"] = (
            input_data["YrSold"]
            - input_data["YearRemodAdd"]
        )


        input_data["TotalSF"] = (
            input_data["TotalBsmtSF"]
            + house_area
        )


        input_data["TotalBathrooms"] = (
            bathrooms
            + 0.5 * input_data["HalfBath"]
            + input_data["BsmtFullBath"]
            + 0.5 * input_data["BsmtHalfBath"]
        )


        input_data["TotalPorchSF"] = (
            input_data["OpenPorchSF"]
            + input_data["EnclosedPorch"]
            + input_data["3SsnPorch"]
            + input_data["ScreenPorch"]
            + input_data["WoodDeckSF"]
        )


        # DataFrame creation

        input_df = pd.DataFrame([input_data])

        input_df = input_df.reindex(
            columns=features
        )


        # Prediction

        prediction = model.predict(input_df)

        predicted_price = prediction[0]


        # ==========================================
        # 8. PREDICTION RESULT
        # ==========================================

        st.success("Prediction Completed Successfully!")

        st.header("💰 Estimated House Price")


        st.metric(
            label="Predicted Sale Price",
            value=f"${predicted_price:,.2f}"
        )


        st.markdown(
            f"""
            <div style="
                background-color:#e8f4ff;
                padding:25px;
                border-radius:15px;
                text-align:center;
                border:1px solid #b6d9ff;
            ">
                <h2 style="color:#1e3a8a;">
                    Estimated Value
                </h2>

                <h1 style="color:#2563eb;">
                    ${predicted_price:,.2f}
                </h1>

                <p>
                    Machine Learning Prediction
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


        # ==========================================
        # 9. PRICE VISUALIZATION
        # ==========================================

        st.divider()

        st.subheader("📊 Price Visualization")


        price_data = pd.DataFrame({
            "Category": ["Predicted House Price"],
            "Price": [predicted_price]
        })


        st.bar_chart(
            price_data,
            x="Category",
            y="Price"
        )


        # ==========================================
        # 10. PROPERTY SUMMARY
        # ==========================================

        st.divider()

        st.header("📋 Property Summary")


        summary_col1, summary_col2 = st.columns(2)


        with summary_col1:

            st.markdown("### 🏠 Property Information")

            st.write(f"**House Area:** {house_area:,} sq ft")

            st.write(f"**Lot Area:** {lot_area:,} sq ft")

            st.write(f"**Bedrooms:** {bedrooms}")

            st.write(f"**Bathrooms:** {bathrooms}")


        with summary_col2:

            st.markdown("### 🏡 Property Features")

            st.write(f"**Overall Quality:** {overall_quality}/10")

            st.write(f"**Year Built:** {year_built}")

            st.write(f"**Garage Capacity:** {garage_capacity} Cars")

            st.write(f"**Neighborhood:** {neighborhood}")


        # ==========================================
        # 11. DISCLAIMER
        # ==========================================

        st.divider()

        st.info("""
        **Disclaimer:**

        This price is an estimated value generated by a
        Machine Learning model. Actual house prices may vary
        depending on market conditions, property features,
        location and other factors.
        """)


    except Exception as e:

        st.error(f"Prediction Error: {e}")


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.markdown("""
<center>

**House Price Prediction – Advanced ML Project**

Developed using Python, Pandas, Scikit-learn and Streamlit

</center>
""", unsafe_allow_html=True)