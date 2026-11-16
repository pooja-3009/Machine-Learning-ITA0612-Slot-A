
import pandas as pd
from sklearn.linear_model import LinearRegression

# Input dataset
data = {
    'Area': [1000, 1200, 1500, 1800, 2000, 2500],
    'Bedrooms': [2, 2, 3, 3, 4, 4],
    'Price': [2000000, 2400000, 3000000,
              3600000, 4000000, 5000000]
}

df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms']]
y = df['Price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Test input
test = pd.DataFrame(
    [[1600, 3]],
    columns=['Area', 'Bedrooms']
)

prediction = model.predict(test)

print("House Area:", 1600)
print("Bedrooms:", 3)
print("Predicted House Price:", round(prediction[0], 2))
