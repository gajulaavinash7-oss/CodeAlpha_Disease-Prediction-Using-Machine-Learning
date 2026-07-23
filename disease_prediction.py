
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Breast Cancer Dataset
data = load_breast_cancer()

# Features and Target
X = data.data
y = data.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest Model
model = RandomForestClassifier(random_state=42)

# Train Model
model.fit(X_train, y_train)

# Predict Test Data
prediction = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, prediction)

print("========== Disease Prediction Project ==========")
print("Dataset:", "Breast Cancer Dataset")
print("Total Samples:", len(X))
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Number of Features:", X.shape[1])

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Predict One Patient
new_patient = X_test[0].reshape(1, -1)

result = model.predict(new_patient)

print("\nPrediction Result")

if result[0] == 0:
    print("Patient is predicted to have Cancer (Malignant)")
else:
    print("Patient is predicted to have No Cancer (Benign)")