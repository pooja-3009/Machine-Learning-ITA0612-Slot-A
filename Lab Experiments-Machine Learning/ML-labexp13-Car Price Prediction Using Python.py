
import pandas as pd
from sklearn.linear_model import LinearRegression

# Input dataset
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020],
    'Mileage': [50000, 45000, 40000, 35000, 30000, 25000],
    'Price': [300000, 350000, 400000, 450000, 500000, 550000]
}

df = pd.DataFrame(data)

X = df[['Year', 'Mileage']]
y = df['Price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Test input
test = pd.DataFrame(
    [[2021, 20000]],
    columns=['Year', 'Mileage']
)

prediction = model.predict(test)

print("Car Year:", 2021)
print("Mileage:", 20000)
print("Predicted Car Price:", round(prediction[0], 2))
