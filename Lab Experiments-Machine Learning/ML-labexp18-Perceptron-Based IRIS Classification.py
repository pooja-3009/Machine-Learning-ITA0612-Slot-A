
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()

# Use two classes for binary classification
X = iris.data[:100, :2]
y = iris.target[:100]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Perceptron model
model = Perceptron(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Prediction
prediction = model.predict([[5.1, 3.5]])

print("Predicted Flower:",
      iris.target_names[prediction[0]])

print("Accuracy:",
      accuracy_score(y_test, model.predict(X_test)) * 100)
