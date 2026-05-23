# Hybridge NLP

Two NLP projects using spaCy: a sentiment analysis REST API and a text preprocessing notebook.

---

## app.py — Restaurant Review Sentiment API

A FastAPI service that classifies restaurant reviews as positive or negative.

### How it works

On startup it loads (or trains) a scikit-learn pipeline:

1. **Preprocessing** — spaCy (`en_core_web_sm`) lemmatizes and strips stop words/punctuation from each review.
2. **Vectorization** — TF-IDF converts the cleaned text to numeric features.
3. **Classification** — Multinomial Naive Bayes predicts positive (`1`) or negative (`0`).

The trained pipeline is cached to `model.joblib` so it only trains once.

### Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Run

```bash
uvicorn app:app --reload
```

Interactive docs available at `http://localhost:8000/docs`.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Classify a single review |
| `POST` | `/predict/batch` | Classify a list of reviews |

**Single prediction example:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The food was absolutely amazing!"}'
```

```json
{"text": "The food was absolutely amazing!", "prediction": 1, "label": "Positiva"}
```

### Data

`Restaurant_Reviews.tsv` — tab-separated file with columns `Review` (text) and `Liked` (0/1).

---

## app_2.ipynb — PDF Text Preprocessing Notebook

A Jupyter notebook that extracts and cleans text from a Spanish PDF book using spaCy.

### How it works

| Step | What happens |
|------|-------------|
| **1. Extraction** | `pymupdf` reads `allan_poe.pdf` page by page into a single string |
| **2. Tokenization** | spaCy (`es_core_news_sm`) splits the text into tokens |
| **3. Stop word filtering** | Removes grammatical glue words, keeping only content words |
| **4. Lemmatization** | Reduces each word to its dictionary root (e.g. `fui` → `ir`) |

### Setup

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

### Run

```bash
jupyter notebook app_2.ipynb
```

Run cells top to bottom. Cell 1 depends on `doc` produced in Cell 0.
