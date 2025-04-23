import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv('../datasets/indian_liver_patient_dataset.csv')

# Rename target column for simplicity
df.rename(columns={"Dataset": "Liver_Problem"}, inplace=True)

# Fill missing values (basic)
df.fillna(df.mean(), inplace=True)

# Split features and label
X = df.drop("Liver_Problem", axis=1)
y = df["Liver_Problem"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open('../models/liver_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Liver model trained and saved!")
