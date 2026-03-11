from flask import Flask, request, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords for Render server
nltk.download('stopwords')

app = Flask(__name__)

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = re.sub(r'<.*?>',' ',text)
    text = re.sub(r'[^a-zA-Z]',' ',text)
    text = text.lower()

    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]

    return " ".join(words)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET','POST'])
def predict():

    if request.method == 'POST':

        review = request.form['review']

        processed = preprocess_text(review)

        vector = vectorizer.transform([processed])

        prediction = model.predict(vector)

        return render_template("index.html", prediction=prediction[0])

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)