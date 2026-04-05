import joblib
import pandas as pd

MODEL_PATH = "models/predict_freight_model.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    load trained freight cost prediction model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.

    parameters
    ----------
    input_data : dict

    returns
    -------
    pd.Dataframe with predicted freight cost
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['Predicted_Freight'] = model.predict(input_df)
    return input_df

if __name__ == "__main__":
    # example inference run (local testing)
    sample_data = {
        "Dollars" : [18500, 9000, 5000, 11334]
    }
    prediction = predict_freight_cost(sample_data)
    print(prediction)