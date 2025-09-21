import requests
import json

# The URL of your running Flask API
url = 'http://127.0.0.1:5000/predict'

# Sample data for prediction
# This represents a typical environment for growing rice.
sample_data = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.87,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.9
}

# Send the POST request with the JSON data
try:
    response = requests.post(url, json=sample_data)
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

    # Print the prediction received from the API
    result = response.json()
    print("API Response:")
    print(json.dumps(result, indent=4))

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")