# 📰 Fake News Detector using NLP & Machine Learning

This project detects whether a news article is **fake or real** using NLP and a Logistic Regression model.

## 🔍 Dataset
- Source: Kaggle - Fake and Real News Dataset
- Format: Combined `Fake.csv` and `True.csv`

## 🧪 Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, TfidfVectorizer
- Logistic Regression
- NLTK for stopwords cleaning

## 📁 Files
- `fake_news_model.joblib`: Trained ML model
- `tfidf_vectorizer.joblib`: TF-IDF vectorizer
- `Fake_News_Detector.ipynb`: Full training notebook

## 🚀 How to Use
```python
import joblib

model = joblib.load('fake_news_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

text = ["The President declared war on Mars today."]
vector = vectorizer.transform(text)
print(model.predict(vector))
