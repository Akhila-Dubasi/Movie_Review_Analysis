import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================
# Configuration
# ==========================

MAX_LEN = 200

st.set_page_config(
    page_title="Movie Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# ==========================
# Load Models
# ==========================

@st.cache_resource
def load_models():

    simple_rnn = tf.keras.models.load_model(
        "simple_rnn_model.h5"
    )

    lstm = tf.keras.models.load_model(
        "lstm_model.h5"
    )

    gru = tf.keras.models.load_model(
        "gru_model.h5"
    )

    return simple_rnn, lstm, gru

simple_rnn_model, lstm_model, gru_model = load_models()

# ==========================
# Load Tokenizer
# ==========================

@st.cache_resource
def load_tokenizer():

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return tokenizer

tokenizer = load_tokenizer()

# ==========================
# Preprocessing
# ==========================

def preprocess_text(text):

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    return padded

# ==========================
# Prediction Function
# ==========================

def predict_review(model, review):

    review_seq = preprocess_text(review)

    probability = model.predict(
        review_seq,
        verbose=0
    )[0][0]

    sentiment = (
        "Positive"
        if probability >= 0.5
        else "Negative"
    )

    confidence = (
        probability
        if probability >= 0.5
        else (1 - probability)
    )

    return sentiment, confidence, probability

# ==========================
# Header
# ==========================

st.title("🎬 Movie Review Sentiment Analysis System")

st.subheader(
    "Deep Learning Based Sentiment Classification"
)

st.markdown("---")

# ==========================
# Sidebar
# ==========================

st.sidebar.header("Model Selection")

selected_model = st.sidebar.radio(
    "Choose Model",
    (
        "SimpleRNN",
        "LSTM",
        "GRU"
    )
)

# ==========================
# Input Area
# ==========================

review = st.text_area(
    "Enter your movie review here...",
    height=200
)

# ==========================
# Analyze Button
# ==========================

if st.button("Analyze Review"):

    if review.strip() == "":

        st.warning(
            "Please enter a movie review."
        )

    else:

        model_map = {

            "SimpleRNN": simple_rnn_model,
            "LSTM": lstm_model,
            "GRU": gru_model
        }

        selected = model_map[selected_model]

        sentiment, confidence, probability = predict_review(
            selected,
            review
        )

        positive_prob = probability * 100
        negative_prob = (1 - probability) * 100

        # ==========================
        # Output Area
        # ==========================

        st.markdown("## Prediction Result")

        if sentiment == "Positive":

            st.success(
                f"✅ Sentiment: {sentiment}"
            )

        else:

            st.error(
                f"❌ Sentiment: {sentiment}"
            )

        st.info(
            f"Confidence: {confidence*100:.2f}%"
        )

        # ==========================
        # Probabilities
        # ==========================

        st.markdown("## Probability Distribution")

        prob_df = pd.DataFrame({

            "Class": [
                "Positive",
                "Negative"
            ],

            "Probability": [
                positive_prob,
                negative_prob
            ]
        })

        st.bar_chart(
            prob_df.set_index("Class")
        )

        # ==========================
        # Pie Chart
        # ==========================

        st.markdown("## Confidence Chart")

        fig, ax = plt.subplots(figsize=(5,5))

        ax.pie(
            [positive_prob, negative_prob],
            labels=["Positive", "Negative"],
            autopct="%1.1f%%"
        )

        ax.set_title("Prediction Confidence")

        st.pyplot(fig)

        # ==========================
        # Compare All Models
        # ==========================

        st.markdown("---")

        st.markdown(
            "## Compare Predictions From All Models"
        )

        comparison = []

        for model_name, model in model_map.items():

            s, c, p = predict_review(
                model,
                review
            )

            comparison.append([
                model_name,
                s,
                round(c * 100, 2)
            ])

        comparison_df = pd.DataFrame(

            comparison,

            columns=[
                "Model",
                "Predicted Sentiment",
                "Confidence (%)"
            ]
        )

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

        # ==========================
        # Confidence Comparison
        # ==========================

        st.markdown(
            "## Model Confidence Comparison"
        )

        chart_df = comparison_df.copy()

        chart_df = chart_df.set_index(
            "Model"
        )

        st.bar_chart(
            chart_df["Confidence (%)"]
        )