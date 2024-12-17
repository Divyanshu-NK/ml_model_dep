import os
import pickle
import pandas as pd
from flask import Flask, request, render_template

# Initialize the Flask app
app = Flask(__name__)

# Load pre-trained model, OneHotEncoder, and StandardScaler
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/onehot_encoder.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

with open('models/standard_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the feature names
num_features = ['Farm_Area', 'Fertilizer_Used', 'Pesticide_Used', 'Yield', 'Water_Usage']
onehot_columns = ['Crop_Type', 'Irrigation_Type', 'Soil_Type', 'Season']

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None  # Default prediction value

    try:
        if request.method == 'POST':
            # Extract form data into a dictionary
            input_data = {
                'Crop_Type': request.form['Crop_Type'],
                'Farm_Area': float(request.form['Farm_Area']),
                'Irrigation_Type': request.form['Irrigation_Type'],
                'Fertilizer_Used': float(request.form['Fertilizer_Used']),
                'Pesticide_Used': float(request.form['Pesticide_Used']),
                'Yield': float(request.form['Yield']),
                'Soil_Type': request.form['Soil_Type'],
                'Season': request.form['Season'],
                'Water_Usage': float(request.form['Water_Usage'])
            }

            # Convert input dictionary to a DataFrame
            df = pd.DataFrame([input_data])

            # Separate numerical and categorical features
            df_num = df[num_features]
            df_cat = df[onehot_columns]

            # Debug: Print Original Data
            print("Original Numerical Features:\n", df_num)
            print("Original Categorical Features:\n", df_cat)

            # Apply OneHotEncoder to categorical columns
            encoded_cat = onehot_encoder.transform(df_cat).toarray()
            encoded_cat_df = pd.DataFrame(
                encoded_cat, 
                columns=onehot_encoder.get_feature_names_out(onehot_columns)
            )
            print("Encoded Categorical Features:\n", encoded_cat_df)

            # Scale numerical features
            scaled_num = scaler.transform(df_num)
            scaled_num_df = pd.DataFrame(scaled_num, columns=num_features)
            print("Scaled Numerical Features:\n", scaled_num_df)

            # Combine scaled numerical and encoded categorical features
            preprocessed_df = pd.concat([scaled_num_df, encoded_cat_df], axis=1)
            print("Final Preprocessed DataFrame:\n", preprocessed_df)

            # Verify input shape
            print("Model Input Shape:", preprocessed_df.shape)

            # Make prediction using the pre-trained model
            prediction = model.predict(preprocessed_df)[0]
            print("Model Prediction:", prediction)

    except Exception as e:
        # Handle errors gracefully
        prediction = f"Error: {str(e)}"
        print("Error:", e)

    # Render the HTML template with the prediction result
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True)
