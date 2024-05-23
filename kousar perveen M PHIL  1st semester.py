import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create a dataframe of 5 students
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Emma'],
    'marks': [85, 78, 92, 88, 75],
    'cgpa': [3.7, 3.5, 3.9, 3.8, 3.4],
    'percentage': [85, 78, 92, 88, 75]
}

df = pd.DataFrame(data)
                  

# Step 2: Split data into train and test sets
X = df[['marks', 'cgpa']]
y = df['percentage']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)