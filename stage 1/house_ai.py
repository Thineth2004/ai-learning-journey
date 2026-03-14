import numpy as np
from sklearn.linear_model import LinearRegression

# Features: size, bedrooms, age
X = np.array([
    [1000, 2, 10],
    [1500, 3, 5],
    [2000, 4, 2],
    [1200, 2, 8]
])

# House prices
y = np.array([200000, 350000, 500000, 250000])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# New house prediction
new_house = np.array([[1800, 3, 4]])

price = model.predict(new_house)

print("Predicted Price:", price[0])