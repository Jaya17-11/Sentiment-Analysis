from flask import Flask, request, render_template
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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
    app.run(debug=True)