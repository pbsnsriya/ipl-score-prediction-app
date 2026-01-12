from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import logging

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# Set up logging for better debugging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)

# Set file paths relative to the current working directory
model_path = os.path.join(os.getcwd(), 'model.tflite')
data_path = os.path.join(os.getcwd(), 'ipl_data.csv')

# Load the dataset and preprocess
try:
    ipl = pd.read_csv(data_path)
except FileNotFoundError:
    logging.error(f"Data file not found at {data_path}")
    raise

# Drop irrelevant columns and prepare the data
df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'mid', 'striker', 'non-striker'], axis=1)
X = df.drop(['total'], axis=1)
y = df['total']

# Encode categorical features
venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()
striker_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()

X['venue'] = venue_encoder.fit_transform(X['venue'])
X['bat_team'] = batting_team_encoder.fit_transform(X['bat_team'])
X['bowl_team'] = bowling_team_encoder.fit_transform(X['bowl_team'])
X['batsman'] = striker_encoder.fit_transform(X['batsman'])
X['bowler'] = bowler_encoder.fit_transform(X['bowler'])

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Load the TFLite model
try:
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    logging.error(f"Error loading the model: {e}")
    raise

@app.route('/')
def index():
    try:
        return render_template('index.html', venues=list(ipl['venue'].unique()),
                               batting_teams=list(ipl['bat_team'].unique()),
                               bowling_teams=list(ipl['bowl_team'].unique()),
                                batsmen=list(ipl['batsman'].unique()),
                                bowlers=list(ipl['bowler'].unique()))
    except Exception as e:
        logging.error(f"Error rendering the index page: {e}")
        return "An error occurred while loading the page."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = [
            venue_encoder.transform([data['venue']])[0],
            batting_team_encoder.transform([data['batting_team']])[0],
            bowling_team_encoder.transform([data['bowling_team']])[0],
            striker_encoder.transform([data['striker']])[0],
            bowler_encoder.transform([data['bowler']])[0]
        ]
        input_data = scaler.transform([input_data])
        
        # Prepare input for TFLite
        input_data = np.array(input_data, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        predicted_score = int(output_data[0][0])
        return jsonify({'predicted_score': predicted_score})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Error predicting score. Please check your input data.'})

if __name__ == '__main__':
    # Run the app
    app.run(debug=True)
