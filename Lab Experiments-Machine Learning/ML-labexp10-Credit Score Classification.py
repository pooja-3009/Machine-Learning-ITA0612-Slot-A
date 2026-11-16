
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Input dataset
data = {
    'Income': [25000, 30000, 50000, 60000, 80000, 90000],
    'Loan': [20000, 25000, 30000, 20000, 25000, 30000],
    'Age': [22, 25, 30, 35, 40, 45],
    'Credit_Score': ['Low', 'Low', 'Medium',
                     'Medium', 'High', 'High']
}

df = pd.DataFrame(data)

X = df[['Income', 'Loan', 'Age']]
y = df['Credit_Score']

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Test input
test = pd.DataFrame(
    [[75000, 20000, 38]],
    columns=['Income', 'Loan', 'Age']
)

prediction = model.predict(test)

print("Input:", test.values)
print("Predicted Credit Score:", prediction[0])
