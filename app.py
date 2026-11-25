
import streamlit as st
import pickle

# Load trained model
with open("fake_news_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# Dashboard
st.title("🔍 Fake News Detection AI")
st.caption("Mini AI Tool | Machine Learning Project")

st.info("Enter a news headline or article below.")

news = st.text_area(
    "Enter News Here",
    placeholder="Type or paste your news..."
)

if st.button("Analyze News", use_container_width=True):

    if len(news.strip()) < 10:
        st.warning("Please enter at least 10 characters.")
    else:
        # Convert text
        news_vector = vectorizer.transform([news])

        # Prediction
        prediction = model.predict(news_vector)[0]
        probabilities = model.predict_proba(news_vector)[0]
        confidence = max(probabilities) * 100

        st.subheader("Analysis Result")

        if prediction == "FAKE":
            st.error("🚨 Prediction: FAKE NEWS")
        else:
            st.success("✅ Prediction: REAL NEWS")

        st.metric("Model Confidence", f"{confidence:.2f}%")

        st.warning(
            "This is an experimental ML prediction, "
            "not a verified fact-check. Always check reliable sources."
        )

st.divider()
st.caption("Developed using Python, Streamlit and Scikit-learn")
