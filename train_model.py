import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# --- 1. Load the Dataset ---
try:
    df = pd.read_csv('Crop_recommendation.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'Crop_recommendation.csv' not found.")
    print("Please make sure the CSV file is in the same directory as this script.")
    exit()

# --- 2. Prepare the Data ---
# Separate features (X) and the target label (y)
X = df.drop('label', axis=1)
y = df['label']

print("\nFeatures (X):")
print(X.head())
print("\nLabels (y):")
print(y.head())


# --- 3. Split Data into Training and Testing Sets ---
# We'll use 80% of the data for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining data size: {X_train.shape[0]} samples")
print(f"Testing data size: {X_test.shape[0]} samples")


# --- 4. Train the Random Forest Model ---
# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

print("\nTraining the model...")
# Train the model on the training data
model.fit(X_train, y_train)
print("Model training complete!")


# --- 5. Evaluate the Model ---
# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")


# --- 6. Save the Trained Model ---
# We will save the model to a file using pickle
model_filename = 'crop_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"\nModel saved successfully as '{model_filename}'")