import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Disable MLflow for GitHub Actions (causes permission issues)
# We'll keep it simple for CI/CD
os.environ['MLFLOW_TRACKING_URI'] = ''

# Load data
df = pd.read_csv('iris.csv')

# Prepare features and labels
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Convert species to numbers
le = LabelEncoder()
y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Model accuracy: {score:.4f}")

# Save model
os.makedirs('model', exist_ok=True)
pickle.dump(model, open('model/iris_model.pkl', 'wb'))
pickle.dump(le, open('model/label_encoder.pkl', 'wb'))

print("Training complete!")
print("Model saved to model/iris_model.pkl")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import mlflow
import mlflow.sklearn

# Load data
df = pd.read_csv('iris.csv')

# Prepare features and labels
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Convert species to numbers
le = LabelEncoder()
y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow tracking
mlflow.set_experiment("iris-classification")

with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    
    # Evaluate
    score = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", score)
    
    print(f"Model accuracy: {score}")
    
    # Save model
    os.makedirs('model', exist_ok=True)
    pickle.dump(model, open('model/iris_model.pkl', 'wb'))
    pickle.dump(le, open('model/label_encoder.pkl', 'wb'))
    
    mlflow.sklearn.log_model(model, "model")

print("Training complete!")
