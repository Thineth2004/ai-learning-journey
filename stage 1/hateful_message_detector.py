from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Trainning data
messages = [
    "Hello, how are you",
    "Fuck off",
    "I'll rip your ass out",
    "Son of a bitch",
    "Fuck you",
    "Good Morning",
    "Good Evening",
    "Good Afternoon",
    "Good Night",
    "Have a nice day",
    "Relax",
    "How's your day going",
    "Calm down"
]

labels = [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] # 1 = positive message, 0 = negative message

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Test prediction
test_message = ["Fuck you nigga"]
test_vector = vectorizer.transform(test_message)

prediction = model.predict(test_vector)

print("Positive message" if prediction[0] == 1 else "Negative message")






