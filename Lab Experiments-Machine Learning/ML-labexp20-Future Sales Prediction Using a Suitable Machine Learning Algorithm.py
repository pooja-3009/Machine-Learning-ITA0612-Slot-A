
import pandas as pd
from sklearn.linear_model import LinearRegression

# Input dataset
data = {
    'Month': [1, 2, 3, 4, 5, 6],
    'Sales': [1000, 1200, 1400, 1600, 1800, 2000]
}

df = pd.DataFrame(data)

X = df[['Month']]
y = df['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict future sales
future = pd.DataFrame(
    [[7]],
    columns=['Month']
)

prediction = model.predict(future)

print("Future Month:", 7)
print("Predicted Sales:", round(prediction[0], 2))
