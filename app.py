# Import Flask, request, jsonify, pickle, and numpy
from flask import Flask, request, jsonify
import pickle
import numpy as np
# ADD THIS LINE to import CORS
from flask_cors import CORS

# Initialize the Flask application
app = Flask(__name__)
# ADD THIS LINE to enable CORS for your app
CORS(app)

# --- Load the Trained Model ---
model_filename = 'crop_model.pkl'
try:
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model file '{model_filename}' not found.")
    model = None


# --- Define the Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        features = [
            data['N'], data['P'], data['K'],
            data['temperature'], data['humidity'], data['ph'], data['rainfall']
        ]
    except KeyError as e:
        return jsonify({'error': f'Missing feature: {e}'}), 400

    final_features = np.array(features).reshape(1, -1)
    prediction = model.predict(final_features)

    return jsonify({'crop_recommendation': prediction[0]})


# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)