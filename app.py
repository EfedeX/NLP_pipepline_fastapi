import os
import joblib
import pandas as pd
import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

MODEL_PATH = "model.joblib"

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def spacy_preprocessor(text: str) -> str:
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)


def train_and_save() -> Pipeline:
    df = pd.read_csv("Restaurant_Reviews.tsv", sep="\t")
    X_train, X_test, y_train, y_test = train_test_split(
        df["Review"], df["Liked"], test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("vectorizacion", TfidfVectorizer(preprocessor=spacy_preprocessor)),
        ("modelado", MultinomialNB()),
    ])

    print("Entrenando modelo...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    print(classification_report(y_test, predictions))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")
    return pipeline


def load_pipeline() -> Pipeline:
    if os.path.exists(MODEL_PATH):
        print(f"Cargando modelo desde {MODEL_PATH}")
        return joblib.load(MODEL_PATH)
    return train_and_save()


pipeline_nlp = load_pipeline()

app = FastAPI(
    title="Restaurant Review Sentiment API",
    description="Predice si una reseña de restaurante es positiva o negativa.",
    version="1.0.0",
)


class ReviewRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    text: str
    prediction: int
    label: str


@app.get("/")
def root():
    return {"message": "API de análisis de sentimiento lista. Usa POST /predict"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ReviewRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="El campo 'text' no puede estar vacío.")

    pred = int(pipeline_nlp.predict([request.text])[0])
    return PredictionResponse(
        text=request.text,
        prediction=pred,
        label="Positiva" if pred == 1 else "Negativa",
    )


@app.post("/predict/batch")
def predict_batch(reviews: list[ReviewRequest]):
    if not reviews:
        raise HTTPException(status_code=400, detail="La lista de reseñas está vacía.")

    texts = [r.text for r in reviews]
    preds = pipeline_nlp.predict(texts)
    return [
        PredictionResponse(text=t, prediction=int(p), label="Positiva" if p == 1 else "Negativa")
        for t, p in zip(texts, preds)
    ]
