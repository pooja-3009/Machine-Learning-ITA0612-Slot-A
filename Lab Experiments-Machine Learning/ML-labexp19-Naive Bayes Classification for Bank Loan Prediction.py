
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Input dataset
data = {
    'Income': [20000, 25000, 30000, 40000, 50000, 60000],
    'Credit': ['Poor', 'Poor', 'Average',
               'Average', 'Good', 'Good'],
    'Loan': ['No', 'No', 'No', 'Yes', 'Yes', 'Yes']
}

df = pd.DataFrame(data)

# Encode categorical data
le = LabelEncoder()
df['Credit'] = le.fit_transform(df['Credit'])
df['Loan'] = le.fit_transform(df['Loan'])

X = df[['Income', 'Credit']]
y = df['Loan']

# Train model
model = GaussianNB()
model.fit(X, y)

# Test input
test = pd.DataFrame(
    [[55000, le.transform(['Good'])[0]]],
    columns=['Income', 'Credit']
)

prediction = model.predict(test)

print("Income: 55000")
print("Credit: Good")

if prediction[0] == 1:
    print("Loan Prediction: Approved")
else:
    print("Loan Prediction: Not Approved")
