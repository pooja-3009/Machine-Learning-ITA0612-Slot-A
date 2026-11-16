
import numpy as np
from sklearn.mixture import GaussianMixture

# Input data
X = np.array([
    [1, 2], [1, 3], [2, 2],
    [8, 8], [9, 8], [8, 9],
    [1, 1], [9, 9]
])

# Apply Expectation Maximization
model = GaussianMixture(n_components=2, random_state=42)
model.fit(X)

# Predict clusters
labels = model.predict(X)

print("Input Data:")
print(X)

print("\nCluster Labels:")
print(labels)

print("\nCluster Means:")
print(model.means_)
