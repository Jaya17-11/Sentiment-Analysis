# Real Time Sentiment Analysis using movie reviews (ML Project)

  A Machine Learning web application that predicts whether a movie review is **Positive 😊 or Negative 😞**.
  
  The model is trained on the **IMDB Movie Review Dataset** and deployed using **Flask**.
  
  Users can enter a movie review and instantly get the predicted sentiment through a simple web interface.



## 🚀 Tech Stack

* Python
* scikit-learn
* pandas
* NLTK
* Flask
* HTML / CSS

---

## Dataset used
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## 🧠 Machine Learning Models Used

* Naive Bayes
* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest

The models were trained and evaluated, and the **best-performing model** was used for deployment.

---

## ⚙️ Project Workflow

1️⃣ **Data Collection**
Dataset collected from the IMDB movie review dataset.

2️⃣ **Data Preprocessing**

* Remove HTML tags
* Remove special characters
* Convert text to lowercase
* Remove stopwords
* Apply stemming

3️⃣ **Feature Extraction**

* TF-IDF Vectorization

4️⃣ **Model Training**

* Train multiple ML models
* Compare performance

5️⃣ **Model Evaluation**

* Accuracy
* Confusion Matrix
* ROC-AUC Score

  **Selected Model**
    Naive Bayes:	0.8575
    Logistic Regression:	0.8892
    SVM:	0.8853
    Random Forest:	0.8522
  
    Selected : Logistic Regression

6️⃣ **Deployment**

* Web interface created using Flask
* Users can input reviews and get real-time predictions.

---

## 📂 Project Structure

```
Sentiment-Analysis
│
├── app.py
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
│
├── templates
│   └── index.html
│
└── Sentiment_Analysis.ipynb
```

---

## ▶️ How to Run the Project

Clone the repository

```
git clone <repository-link>
```

Install dependencies

```
pip install -r requirements.txt
```

Run the Flask app

```
python app.py
```

Open in browser

```
http://127.0.0.1:5000
```

---

## ✨ Features

* Real-time sentiment prediction
* ML model comparison
* Clean web interface
* Emoji-based sentiment output



## Deployment link - Feel free to use
It may take sometime to load the application mostly 30-60 seconds sometimes 1-2 minutes to save server resources when not in use
https://sentiment-analysis-j91x.onrender.com




