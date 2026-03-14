from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Load dataset 
digits = load_digits()

# Features (pixel values)
X = digits.data

# Labels (correct digits)
y = digits.target

# Split dataset 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = KNeighborsClassifier()

# Train model
model.fit(X_train, y_train)

# Test accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Predict one example
prediction = model.predict([X_test[7]])

print("Predicted:", prediction[0])
print("Actual:", y_test[7])

# Show the image
plt.imshow(X_test[7].reshape(8,8), cmap="gray")
plt.show()

