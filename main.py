import numpy as np
import pandas as pd
import re
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Load models, encoders, and scalers once
with open(r'model.pkl', 'rb') as file:
    load_model = pickle.load(file)
with open(r'scaler.pkl', 'rb') as f:
    scaler_load = pickle.load(f)
with open(r't.pkl', 'rb') as f:
    t_load = pickle.load(f)
with open(r's.pkl', 'rb') as f:
    s_load = pickle.load(f)

with open(r'c_model.pkl', 'rb') as f:
    c_load = pickle.load(f)
with open(r'c_encode.pkl', 'rb') as f:
    c_encode = pickle.load(f)
with open(r'c_scaler.pkl', 'rb') as f:
    c_scaler = pickle.load(f)

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Explore data", "About"],
        icons=["house", "search", "info-circle"],
        default_index=0,
    )

if selected == "Home":
    st.title("Industrial Copper Modeling")
    st.image("D:/Error/DS_Industrial Copper Modeling/image.jpg",width=300)

elif selected == "Explore data":
    tab_1, tab_2 = st.tabs(["PREDICT SELLING PRICE", "PRICE STATUS"])

    # First tab for selling price prediction
    with tab_1:
        # Options for dropdown menus
        status_options = ["Won", "Draft", "to be approved", "lost", "Not for AM", "Wonderful", "Revised", "Offered", "Offerable"]
        item_type_options = ["W", "WI", "S", "Others", "PL", "IPL", "SLAWR"]
        country_options = [28., 25., 30., 32., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
        application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79.]
        product = ['611993', '611728', '628112', '628117', '628377', '640400', '640405', '640665', '611993', '929423819', '1282007633', '1332077137']

        with st.form('my_form'):
            col_1, col_3 = st.columns([5, 5])  # Adjust columns for better layout

            with col_1:
                status = st.selectbox("Status", status_options, key=1)
                item_type = st.selectbox("Item type", item_type_options, key=2)
                country = st.selectbox("Country", sorted(country_options), key=3)
                application = st.selectbox("Application", sorted(application_options), key=4)
                product_ref = st.selectbox("Product Reference", product, key=5)

            with col_3:
                quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                thickness = st.text_input("Enter Thickness (min:0.18 & max:400)")
                width = st.text_input("Enter Width (min:1 & max:2999)")
                customer = st.text_input("Enter Customer ID (min:12458 & max:30408185)")
                submit_button = st.form_submit_button(label="Predict Selling price")

            # Input validation
            flag = 0
            invalid_field = ""
            pattern = r'^(?:\d+|\d*\.\d+)$'
            for field, value in {"Quantity Tons": quantity_tons, "Thickness": thickness, "Width": width, "Customer ID": customer}.items():
                if not re.match(pattern, value):
                    flag = 1
                    invalid_field = field
                    break

        if submit_button and flag == 1:
            st.write(f"Please enter a valid number for {invalid_field}")

        if submit_button and flag == 0:
            new_sample = np.array([[np.log(float(quantity_tons)), application, np.log(float(thickness)), float(width), country, float(customer), int(product_ref), item_type, status]])
            new_sample_ohe = t_load.transform(new_sample[:, [7]]).reshape(-1, 1)  # Ensure it is 2D

            s_load.fit(np.array(status_options).reshape(-1, 1))
            new_sample_be = s_load.transform(new_sample[:, [8]]).reshape(1, -1)  # Ensure it is 2D
            new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6]], new_sample_ohe, new_sample_be), axis=1)
            new_sample1 = scaler_load.transform(new_sample)
            new_pred = load_model.predict(new_sample1)[0]
            st.write('## :green[Predicted selling price:] ', np.exp(new_pred))

    # Second tab for price status prediction
    with tab_2:
        with st.form('my_forms'):
            col_1, col_3 = st.columns([5, 5])

            with col_1:
                c_quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                c_thickness = st.text_input("Enter Thickness (min:0.18 & max:400)")
                c_width = st.text_input("Enter Width (min:1 & max:2999)")
                c_customer = st.text_input("Enter Customer ID (min:12458 & max:30408185)")
                c_selling = st.text_input("Enter Selling Price (min:1 & max:100001015)")

            with col_3:
                c_item_type = st.selectbox("Item type", item_type_options, key=21)
                c_country = st.selectbox("Country", sorted(country_options), key=31)
                c_application = st.selectbox("Application", sorted(application_options), key=41)
                c_product_ref = st.selectbox("Product Reference", product, key=51)
                c_submit = st.form_submit_button(label="PREDICT STATUS")

            # Input validation
            flag = 0
            invalid_field = ""
            for field, value in {"Quantity Tons": c_quantity_tons, "Thickness": c_thickness, "Width": c_width, "Customer ID": c_customer, "Selling Price": c_selling}.items():
                if not re.match(pattern, value):
                    flag = 1
                    invalid_field = field
                    break

        if c_submit and flag == 1:
            st.write(f"Please enter a valid number for {invalid_field}")

        if c_submit and flag == 0:
            new_sample_c = np.array([[np.log(float(c_quantity_tons)), np.log(float(c_selling)), c_application, np.log(float(c_thickness)), float(c_width), c_country, int(c_customer), int(c_product_ref), c_item_type]])
            new_sample_e = c_encode.transform(new_sample_c[:, [8]]).reshape(-1, 1)
            new_sample = np.concatenate((new_sample_c[:, [0, 1, 2, 3, 4, 5, 6, 7]], new_sample_e), axis=1)
            new_sample = c_scaler.transform(new_sample)
            c_predict = c_load.predict(new_sample)

            if c_predict == 1:
                st.write('## :green[The Status is Won]')
            else:
                st.write('## :red[The status is Lost]')

elif selected == "About":
        st.title("About the Project")
        st.write("""
    This project focuses on building and deploying a predictive model using the following technologies:
    
    - **Python Scripting**: Used for model development, automation, and deployment tasks.
    - **Data Preprocessing**: Cleaning, transforming, and preparing data for model training.
    - **Exploratory Data Analysis (EDA)**: Identifying patterns and trends in the data to inform model development.
    - **Machine Learning Model**: A predictive model trained to forecast outcomes based on input data.
    - **Streamlit Technology**: Deployed the model in an interactive web application to provide real-time predictions.
    """)
    

