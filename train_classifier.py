import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os
import sys

data_file = 'augmented_medical_data_100k.csv'
if not os.path.exists(data_file):
    print(f"FATAL ERROR: Data file not found at '{data_file}'")
    print("Please make sure it's in the same folder as this script.")
    sys.exit(1)
    
print(f"Loading data from {data_file}...")
df = pd.read_csv(data_file)

y = df['breakdown_flag']

X = df.drop('breakdown_flag', axis=1)


categorical_features = ['device_name']
numeric_features = ['usage_hours', 'temperature', 'error_count']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

print("Preprocessing data (One-Hot Encoding 'device_name')...")
X_processed = preprocessor.fit_transform(X)

print("Splitting data into 80% train and 20% test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

print("Training the RandomForestClassifier...")
print("(This may take a minute with 100k rows...)")

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20)
model.fit(X_train, y_train)

print("Training complete.")

print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Breakdown (1)']))

new_model_filename = 'trained_breakdown_classifier.pkl'

pipeline_to_save = {
    'preprocessor': preprocessor,
    'model': model
}

joblib.dump(pipeline_to_save, new_model_filename)
print(f"\n--- Success! ---")
print(f"New model and preprocessor saved to '{new_model_filename}'")