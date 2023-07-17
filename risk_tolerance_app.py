import json

import numpy as np
import streamlit as st

from utils.inference import PredictRiskTolerance

with open("utils/value_mapping.json", "r") as file_handle:
    mapping_dict = json.load(file_handle)

# Set wide page layout
st.set_page_config(layout="wide")

# Title
st.title(":blue[Investor Risk Tolerance Prediction using Machine Learning]")
st.markdown(
    """
    Enter the data values on the left panel and get your risk tolerance!
    Access the source code on [GitHub](https://github.com/sanjayg96/risk_modelling/tree/master)

"""
)

expander = st.expander("About")
expander.write(
    """
* Random Forest Regressor is used to make the prediction.
* The data used for this project is the 2019 SCF data available [here](https://www.federalreserve.gov/econres/scfindex.htm).
"""
)


##################################################
# Sidebar
##################################################
side_bar = st.sidebar
side_bar.header("User Data")

# Total Assets
total_assets = side_bar.slider("Total Assets $", 0, 20_000_000, 70000)

# Total Liabilities
total_liabilities = side_bar.slider("Total Liabilities $", 0, 200_000, 5000)

networth = total_assets - total_liabilities

## Income
income = side_bar.slider("Total Annual Income $", 0, 3_000_000, 40000)

## Age
age = side_bar.slider("Age", 18, 95, 25)

## Education
education = side_bar.selectbox(
    "Education",
    (
        "No grades completed",
        "1st grade",
        "2nd grade",
        "3rd grade",
        "4th grade",
        "5th grade",
        "6th grade",
        "7th grade",
        "8th grade",
        "9th grade",
        "10th grade",
        "11th grade",
        "12th grade",
        "1 year of college",
        "2 years of college",
        "3 years of college",
        "4 years of college",
        "Graduate School",
    ),
    index=17,
)

## Marital Status
married = side_bar.selectbox("Marital Status", ("Single", "Married"))

## Number of kids
num_kids = side_bar.slider("Number of kids", 0, 7, 0)

## Occupational Category
occupation_category = side_bar.selectbox(
    "Occupational Category",
    (
        "Not working: Student/homemaker/retired",
        "Unemployed",
        "Work for someone else",
        "Self-Employed/Partnership",
    ),
    index=2,
)

## Occupational Classification
occupation_classification = side_bar.selectbox(
    "Occupational Classification",
    (
        "Managerial/Professional",
        "Technical/Sales/Services",
        "Others",
        "Not working",
    ),
)


##################################################

# Collect the data and get the prediction.
education = mapping_dict["education"][education]
married = mapping_dict["married"][married]
occupation_category = mapping_dict["occupation_category"][occupation_category]
occupation_classification = mapping_dict["occupation_classification"][
    occupation_classification
]

input_data = [
    age,
    education,
    married,
    occupation_category,
    occupation_classification,
    num_kids,
    networth,
    income,
]
risk_predictor = PredictRiskTolerance()
risk_tolerance = risk_predictor(input_data)
risk_tolerance = round(risk_tolerance * 100, 2)
risk_tolerance = np.clip(risk_tolerance, 0, 100)

if 0 <= risk_tolerance <= 30:
    risk_classification = "Conservative Investors"
elif 30 < risk_tolerance <= 60:
    risk_classification = "Moderate Investors"
else:
    risk_classification = "Aggressive Investors"

st.markdown("## Your Risk Tolerance: " + ":green[" + str(risk_tolerance) + "%" + "]")
st.write(
    f"""
    Risk tolerance is estimated based on investing patterns of people with similar demographics as yours.
    
    Thus, people like you have a similar risk tolerance score and could be classified as 
    **:green[{risk_classification}]**.

"""
)
