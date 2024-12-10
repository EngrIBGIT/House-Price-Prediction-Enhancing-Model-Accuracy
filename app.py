import streamlit as st
import numpy as np
import pickle

# Load the pre-trained model
model = pickle.load(open('House_price_mod.pkl', 'rb'))

# App Logo
st.image("https://via.placeholder.com/400x100.png?text=House+Price+Prediction", use_container_width=True)

# Page Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>House Price Prediction App</h1>", unsafe_allow_html=True)
st.write("This interactive app predicts house prices based on various features.")


# Custom CSS for Styling
st.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-color: #f4f4f4;
        }
        .stButton>button {
            color: white;
            background-color: #4CAF50;
            border-radius: 5px;
        }
        .stMarkdown h1 {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Input Features")
st.sidebar.markdown("## Enter the values below:")

# Sidebar Inputs
crime_rate = st.sidebar.slider("Crime Rate", 0.0, 10.0, step=0.1, value=1.0)
resid_area = st.sidebar.selectbox("Residential Area (1 = Yes, 0 = No)", [0, 1])
air_qual = st.sidebar.slider("Air Quality Index", 0.0, 10.0, step=0.1, value=5.0)
room_num = st.sidebar.slider("Number of Rooms", 1, 10, step=1, value=3)
age = st.sidebar.slider("Age of Property (Years)", 0, 100, step=1, value=30)
teachers = st.sidebar.slider("Number of Teachers", 1, 50, step=1, value=20)
poor_prop = st.sidebar.slider("Percentage of Poor Population", 0.0, 1.0, step=0.01, value=0.1)
n_hos_beds = st.sidebar.slider("Number of Hospital Beds", 0, 100, step=1, value=10)
n_hot_rooms = st.sidebar.slider("Number of Hot Rooms", 0.0, 50.0, step=0.1, value=5.0)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 500.0, step=1.0, value=200.0)
parks = st.sidebar.slider("Number of Parks", 0, 10, step=1, value=3)
avg_dist = st.sidebar.slider("Average Distance to Facilities", 0.0, 10.0, step=0.1, value=5.0)
airport_YES = st.sidebar.selectbox("Proximity to Airport (1 = Yes, 0 = No)", [0, 1])
waterbody_Lake_and_River = st.sidebar.selectbox("Proximity to Lake and River (1 = Yes, 0 = No)", [0, 1])
waterbody_River = st.sidebar.selectbox("Proximity to River (1 = Yes, 0 = No)", [0, 1])

# Sidebar News Section
st.sidebar.markdown("## Latest Real Estate News")
st.sidebar.write("- Housing market hits record highs.")
st.sidebar.write("- Expert tips for first-time home buyers.")
st.sidebar.write("- New trends in real estate investing.")

# Collect Input Data
input_data = np.array([
    crime_rate, resid_area, air_qual, room_num, age, teachers, poor_prop,
    n_hos_beds, n_hot_rooms, rainfall, parks, avg_dist, airport_YES,
    waterbody_Lake_and_River, waterbody_River
]).reshape(1, -1)

# Prediction and Results Section
st.markdown("### Prediction Results")
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    predicted_price = f"The predicted house price is: ${prediction[0]:,.2f}"
    st.success(predicted_price)

    # Option to Download Prediction
    result_file = f"Predicted Price: ${prediction[0]:,.2f}"
    st.download_button(
        label="Download Prediction",
        data=result_file,
        file_name="predicted_price.txt",
        mime="text/plain"
    )

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Powered by AI | Designed for Home Buyers</p>", unsafe_allow_html=True)
