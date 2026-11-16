
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Input dataset
data = {
    'RAM': [2, 3, 4, 6, 8, 12],
    'Battery': [2000, 2500, 3000, 3500, 4000, 5000],
    'Storage': [16, 32, 32, 64, 128, 256],
    'Price_Range': [0, 0, 1, 1, 2, 3]
}

df = pd.DataFrame(data)

X = df[['RAM', 'Battery', 'Storage']]
y = df['Price_Range']

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Test input
test = pd.DataFrame(
    [[8, 4000, 128]],
    columns=['RAM', 'Battery', 'Storage']
)

prediction = model.predict(test)

ranges = {
    0: "Low Cost",
    1: "Medium Cost",
    2: "High Cost",
    3: "Very High Cost"
}

print("Mobile Specifications:", test.values)
print("Predicted Price Range:", ranges[prediction[0]])
