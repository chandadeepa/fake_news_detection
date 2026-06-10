import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample News Dataset
news_data = {
    "text": [
        "Government launches new education policy",
        "Scientists discover cure for cancer",
        "India wins cricket world cup",
        "New technology introduced in hospitals",
        "Aliens landed in Hyderabad yesterday",
        "Ghost seen driving car in Mumbai",
        "Drinking petrol cures fever instantly",
        "Moon will crash into Earth tomorrow"
    ],

    "label": [
        "REAL",
        "REAL",
        "REAL",
        "REAL",
        "FAKE",
        "FAKE",
        "FAKE",
        "FAKE"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(news_data)

# Text to numerical data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

# Labels
y = df["label"]

# Train ML Model
model = LogisticRegression()
model.fit(X, y)

# Streamlit UI
st.title("📰 Fake News Detection App")

st.write("Enter a news sentence below to check whether it is REAL or FAKE.")

user_news = st.text_area("Enter News")

if st.button("Check News"):

    if user_news.strip() == "":
        st.warning("Please enter some news text.")

    else:
        transformed_input = vectorizer.transform([user_news])

        prediction = model.predict(transformed_input)

        if prediction[0] == "REAL":
            st.success("✅ This News Looks REAL")

        else:
            st.error("❌ This News Looks FAKE")==