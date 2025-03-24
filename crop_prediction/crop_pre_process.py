import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler  # Required for loading scaler

# MODEL LOADING
MODEL_PATH = r"crop_prediction\crop_model.pkl"
SCALER_PATH = r"crop_prediction\scaler_crop_recommendation.pkl"

# Load the trained model
with open(MODEL_PATH, 'rb') as f:  # 'rb' means read binary mode
    model = pickle.load(f)

# Load the scaler
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

def crop_model(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):

    # Define column names (must match training data)
    feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    # New input data
    new_input = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])

    # Convert to DataFrame with column names
    new_input_df = pd.DataFrame(new_input, columns=feature_names)

    # Transform using the scaler
    scaled_input = scaler.transform(new_input_df)

    # Predict the crop
    predicted_crop = model.predict(scaled_input)[0]

    return predicted_crop

