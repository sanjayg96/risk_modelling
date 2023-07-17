import pickle
import compress_pickle

import pandas as pd
import streamlit as st


class PredictRiskTolerance:
    """
    Main class to predict the investor risk tolerance.

    Note: Instead of the usual "self" argument in the class methods, "_self" is used.
    This is done do tell Streamlit not to hash the argument "self".
    If this is not done, it'll throw the error - "UnhashableParamError"
    """

    def __init__(_self):
        """
        Iinitialize paths and variables.
        """

        _self.MODEL_PATH = r"model_weights/rf_regression.gz"
        _self.COLUMN_PATH = r"model_weights/rf_regression_columns.pkl"
        _self.OHE_PATH = r"model_weights/ohe.pkl"
        _self.categorical_variables = ["MARRIED", "OCCAT1", "OCCAT2"]

        return None

    def __call__(_self, input_data):
        """
        Call function to prepare data and predict risk tolerance.

        Args:
            input_data: list
                Input data received from the user.

        Returns:
            risk_tolerance: float
                Investor risk tolerance for the given input
        """
        final_model, column_list, ohe = _self.load_files()
        model_input = _self.prepare_data(input_data, column_list, ohe)
        risk_tolerance = _self.make_prediction(model_input, final_model)

        return risk_tolerance

    @st.cache_resource
    def load_files(_self):
        """
        Function to load the trained model and other saved files.

        Args:
            None

        Returns:
            final_model:
                scikit-learn trained model

            ohe:
                one-hot encoder object

            column_list:
                column order to prepare the data
        """
        with open(_self.MODEL_PATH, "rb") as file_handle:
            final_model = compress_pickle.load(file_handle)

        with open(_self.COLUMN_PATH, "rb") as file_handle:
            column_list = pickle.load(file_handle)

        with open(_self.OHE_PATH, "rb") as file_handle:
            ohe = pickle.load(file_handle)

        return final_model, column_list, ohe

    def prepare_data(_self, input_data, column_list, ohe):
        """
        Prepare data to feed into the model.

        Args:
            input_data: list
                Input data received from the user.

            column_list:
                column order to prepare the data

            ohe:
                one-hot encoder object

        Return:
            model_input: dataframe

        """
        data = pd.DataFrame([input_data], columns=column_list)
        encodings_data = pd.DataFrame(
            ohe.transform(data[_self.categorical_variables]).toarray()
        )
        encodings_data.columns = ohe.get_feature_names_out()

        data = data.drop(columns=_self.categorical_variables).reset_index(drop=True)
        model_input = pd.concat([data, encodings_data], axis=1)

        return model_input

    def make_prediction(_self, model_input, final_model):
        """
        Predict the risk tolerance.

        Args:
            model_input: dataframe
                Prepared data to make the predictions

            final_model:
                scikit-learn trained model

        Returns:
            risk_tolerance: float
                Investor risk tolerance for the given input
        """
        risk_tolerance = final_model.predict(model_input)[0]

        return risk_tolerance
