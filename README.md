# Investor Risk Tolerance Prediction using Machine Learning

[Click here](https://risktolerance-app.streamlit.app/) to access the deployed web app.

## About
Financial investments can be risky,  risk-free, or anywhere in between. How people choose to invest depends on their risk appetite or risk tolerance. People with higher risk tolerance have a propensity to make risky investments.

This project aims to predict the risk tolerance using demographic and financial information provided by the user. The prediction is done by a regression model - Random Forest Regressor.

## Data
The data used for this project is the [2019 Survey of Consumer Finances (SCF)](https://www.federalreserve.gov/econres/scfindex.htm).

This consumer data has hundreds of variables with information about their demographics, assets, debts, shopping patterns, expenditures etc. The data and variable descriptions given in the codebook is a text file. This can be hard to read and comprehend. For a simpler and easier reference you may visit [SCF Combined Extract Data](https://sda.berkeley.edu/data/scfcomb2013/Doc/hcbk.htm).

For the sake of simplicity and ease of use, only a few intuitive variables related to demographics and financials are considered from this data for modelling.

## Estimating Risk Tolerance
The data does not contain a variable called risk tolerance. It is estimated by looking at the fraction of risky investments. More precisely, risk tolerance within the scope of this project is defined and calculated as the fraction of total risky investments to that of the total investments. Total investments is the sum of risky investments and risk-free investments. Thus, the value of risk tolerance ranges from the values 0 to 1.

Risky investments include stocks, bonds, mutual funds etc. Risk-free investments include savings account, checking acount, FD etc.

## Modelling
Various regression models such as Linear Regression (with and without L1 and L2 regularization), Tree based Bagging and Boosting models were evaluated. Random Forest Regressor had the best performance and was chosen as the final model.

## Tools and libraries used

Purpose | Tools/Libraries
---|---
Interactive Python Development | Jupyter Notebook
Data manipulation | Pandas, JSON
Numerical computations | NumPy
Data visualizations | Matplotlib, Seaborn
Feature transforms and Modelling | Scikit-Learn
Deployment and Web app design | Streamlit

## Credits
This project was inspired by the risk tolerance case study presented in the book "Machine Learning and Data Science Blueprints for Finance" by the authors Hariom Tatsat, Sahil Puri, and Brad Lookabaugh. Do check out this book.
