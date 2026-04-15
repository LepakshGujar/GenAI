from pathlib import Path
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def load_data(path: Path):
    texts = []
    labels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ";" not in line:
                continue
            text, label = line.rsplit(";", 1)
            texts.append(text.strip())
            labels.append(label.strip())
    return texts, labels


def train_and_evaluate(texts, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return model, vectorizer, accuracy, report


def save_artifacts(model, vectorizer, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "vectorizer.pkl").open("wb") as f:
        pickle.dump(vectorizer, f)
    with (output_dir / "model.pkl").open("wb") as f:
        pickle.dump(model, f)


def main():
    data_path = Path(__file__).parent / "train.txt"
    model_dir = Path(__file__).parent / "model"

    texts, labels = load_data(data_path)
    if not texts:
        raise ValueError(f"No data loaded from {data_path}")

    model, vectorizer, accuracy, report = train_and_evaluate(texts, labels)
    save_artifacts(model, vectorizer, model_dir)

    print(f"Trained on {len(texts)} examples and saved model to {model_dir}")
    print(f"Accuracy on test split: {accuracy:.4f}\n")
    print("Classification report:\n")
    print(report)
    print("\nTo run the testing UI, use: streamlit run app.py")


if __name__ == "__main__":
    main()
