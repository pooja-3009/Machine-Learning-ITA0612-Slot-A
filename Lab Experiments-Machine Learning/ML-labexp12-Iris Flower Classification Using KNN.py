
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Prediction
prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])

print("Predicted Flower:",
      iris.target_names[prediction[0]])

print("Accuracy:",
      accuracy_score(y_test, model.predict(X_test)) * 100)
