import numpy as np

# Training data
X = np.array([1,2,3,4])
y = np.array([300,500,700,900])

# Initialize parameters
w = 0
b = 0

learning_rate = 0.01
epochs = 1000

n = len(X)

for epoch in range(epochs):

    y_pred = w * X + b

    error = y_pred - y

    dw = (2/n) * np.sum(error * X)
    db = (2/n) * np.sum(error)

    w = w - learning_rate * dw
    b = b - learning_rate * db

print("Weight:", w)
print("Bias:", b)

prediction = w*5 + b
print("Prediction for size 5:", prediction)