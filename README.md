#  NLP App – Interaktywna analiza tekstu (Gradio + Transformers)

## Opis projektu

Aplikacja webowa do analizy tekstu w czasie rzeczywistym, wykorzystująca nowoczesne modele NLP z biblioteki Hugging Face.

Projekt umożliwia interaktywną eksplorację różnych zadań NLP poprzez interfejs stworzony w Gradio.

---
##  Funkcjonalności

###  1. Tokenizacja tekstu

* porównanie:

  * Word-level (NLTK)
  * Subword (BERT)
* liczba tokenów i ID tokenów

---

###  2. Analiza sentymentu

Funkcjonalność:

* klasyfikacja pozytywny / negatywny
* procentowa pewność predykcji

---

###  3. Similarity (podobieństwo semantyczne)

Funkcjonalność:

* porównanie dwóch zdań
* cosine similarity

---

###  4. Zero-shot classification

Funkcjonalność:

* klasyfikacja tekstu do dowolnych kategorii
* bez wcześniejszego trenowania

---

###  5. Streszczanie tekstu

Funkcjonalność:

* automatyczne streszczanie tekstu
* dynamiczne dopasowanie długości

---


## Uruchomienie

### 1. Klonowanie repo

```bash
git clone https://github.com/JedrekSz/nlp.git
```

---

### 2. Instalacja bibliotek



### 3. Uruchomienie aplikacji

```bash
python nlp_project.py
```

Po uruchomieniu aplikacja będzie dostępna w przeglądarce (lokalnie).

---

