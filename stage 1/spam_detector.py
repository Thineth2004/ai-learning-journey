from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
messages = [
    "Win a free iPhone now",
    "Call me when you are free",
    "Congratulations you won a lottery",
    "Let's meet tomorrow",
]

labels = [1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Test prediction
test_message = ["You have won free cash"]
test_vector = vectorizer.transform(test_message)

prediction = model.predict(test_vector)

print("Spam" if prediction[0] == 1 else "Not Spam")