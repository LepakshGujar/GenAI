from pathlib import Path
import pickle

import streamlit as st

PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
TRAIN_PATH = PROJECT_DIR / "train.txt"


@st.cache_data
def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run `python train.py` first to create model/model.pkl and model/vectorizer.pkl."
        )
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    with VECTORIZER_PATH.open("rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


@st.cache_data
def load_dataset():
    texts = []
    labels = []
    if TRAIN_PATH.exists():
        with TRAIN_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ";" not in line:
                    continue
                text, label = line.rsplit(";", 1)
                texts.append(text.strip())
                labels.append(label.strip())
    return texts, labels


def main():
    st.set_page_config(page_title="Emotion Classifier", page_icon="🧠")
    st.title("Emotion Classification Testing UI")
    st.write(
        "Enter a sentence below to test the trained emotion classifier. The model is trained on `train.txt` labels."
    )

    texts, labels = load_dataset()
    if texts:
        st.sidebar.markdown("### Dataset summary")
        st.sidebar.write(f"Examples: {len(texts)}")
        st.sidebar.write(f"Labels: {len(set(labels))}")
        st.sidebar.write(sorted(set(labels)))

    try:
        model, vectorizer = load_artifacts()
    except FileNotFoundError as err:
        st.error(str(err))
        return

    query = st.text_area("Test sentence", height=150)
    if st.button("Predict"):
        if not query.strip():
            st.warning("Please enter a sentence to predict.")
        else:
            features = vectorizer.transform([query])
            prediction = model.predict(features)[0]
            st.success(f"Predicted label: **{prediction}**")
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                labels_out = model.classes_
                st.write("Prediction probabilities:")
                for label, prob in sorted(zip(labels_out, probs), key=lambda x: -x[1]):
                    st.write(f"- {label}: {prob:.3f}")

    st.markdown("---")
    st.subheader("Sample training examples")
    for sample_text, sample_label in list(zip(texts[:6], labels[:6])):
        st.write(f"- {sample_text} → *{sample_label}*")


if __name__ == "__main__":
    main()
