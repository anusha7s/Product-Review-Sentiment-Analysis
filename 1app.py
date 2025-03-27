import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from transformers import pipeline
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob

# ✅ set page config
st.set_page_config(page_title="sentiment analysis app", page_icon="😊", layout="wide")

# ✅ sidebar info
with st.sidebar:
    st.title("sentiment analysis app 😊")
    st.subheader("about")
    st.write("""
    ai-powered sentiment analysis for product reviews.

    *features:*
    - upload a dataset and analyze sentiment.
    - visualize sentiment trends with wordcloud & charts.
    - test real-time predictions.
    """)

# ✅ load hugging face sentiment analysis model
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_sentiment_pipeline()

# ✅ preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove special characters
        return text
    return ""

# ✅ tabs for sections
tabs = st.tabs(["📂 upload data", "🔍 sentiment analysis", "📊 visualization"])

# ✅ *tab 1: upload dataset*
with tabs[0]:
    st.subheader("📂 upload your dataset")
    uploaded_file = st.file_uploader("upload csv file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### dataset preview:")
        st.write(df.head())

        # ✅ ensure required columns exist
        if 'reviewText' not in df.columns:
            st.error("dataset must contain 'reviewText' column.")
        else:
            # ✅ textblob sentiment polarity
            df['reviewText'] = df['reviewText'].fillna("").astype(str)
            df['polarity'] = df['reviewText'].apply(lambda text: TextBlob(text).sentiment.polarity)
            df['review_len'] = df['reviewText'].apply(len)
            df['word_count'] = df['reviewText'].apply(lambda x: len(x.split()))

# ✅ *tab 2: sentiment prediction*
with tabs[1]:
    st.subheader("🔍 sentiment prediction")
    user_input = st.text_area("enter a review to analyze sentiment:")

    if st.button("analyze sentiment"):
        if user_input:
            prediction = sentiment_pipeline(user_input)[0]
            sentiment = "😊 positive" if prediction["label"] == "POSITIVE" else "😞 negative"
            confidence = round(prediction["score"] * 100, 2)

            st.success(f"predicted sentiment: {sentiment} ({confidence}% confidence)")
        else:
            st.warning("please enter a review for analysis.")

# ✅ *tab 3: visualization*
with tabs[2]:
    st.subheader("📊 sentiment visualization")

    if uploaded_file:
        try:
            # ✅ generate sentiment predictions for dataset
            df['sentiment_prediction'] = df['reviewText'].apply(lambda x: sentiment_pipeline(x)[0]['label'].lower())

            # ✅ wordcloud for positive reviews
            review_pos = df[df["sentiment_prediction"] == "positive"]["reviewText"].dropna()
            text = " ".join(review_pos)
            wordcloud = WordCloud(width=3000, height=2000, background_color='black', stopwords=STOPWORDS).generate(text)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

            # ✅ bar chart for sentiment counts
            sentiment_counts = df['sentiment_prediction'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=["red", "blue"], ax=ax)
            ax.set_xlabel("sentiment")
            ax.set_ylabel("count")
            ax.set_title("sentiment distribution")
            st.pyplot(fig)

            # ✅ histogram of review lengths
            fig, ax = plt.subplots()
            sns.histplot(df['review_len'], bins=30, kde=True, color='blue', ax=ax)
            ax.set_xlabel("review length")
            ax.set_ylabel("frequency")
            ax.set_title("distribution of review lengths")
            st.pyplot(fig)

            # ✅ pie chart for sentiment distribution
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=["red","blue"], startangle=140)
            ax.set_title("sentiment proportion")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"error generating visualizations: {e}")
