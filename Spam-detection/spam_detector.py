import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load Data
df = pd.read_csv("data/spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 2: Split Data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2)

# Step 3: Vectorize Messages
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 4: Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 5: Test Accuracy
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 6: Save Model
joblib.dump(model, 'model/spam_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

# Step 7: Predict on New Message
def predict_message(message):
    vect_msg = vectorizer.transform([message])
    result = model.predict(vect_msg)[0]
    print("Spam" if result else "Not Spam")

# Try it
predict_message("Congratulations! You've won a free ticket.")
