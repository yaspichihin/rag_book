import spacy
import nltk
from nltk.corpus import wordnet
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


## Косинусное


def calc_cos_similarity(text1: str, text2: str) -> float:
    """Косинусное сходство на основе TF-IDF."""

    vectorizer = TfidfVectorizer(
        # Игнорировать распространенные английские слова
        stop_words="english",
        # Механизм взвешивания на основе обратной частоты документа
        use_idf=True,
        # Нормализация векторов
        norm="l2",
        # Используем униграммы и биграммы
        # Учитывает одиночные слова и парные соседние слова
        ngram_range=(1, 2),
        # Сублинейное TF масштабирование
        # Логарифмическое масштабирование частоты термина
        sublinear_tf=True,
        # Анализ текста на уровне слов
        analyzer="word",
    )
    tf_idf = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tf_idf[0:1], tf_idf[1:2])
    similarity = similarity[0][0]
    similarity = float(similarity)

    return similarity


## Расширенное


nltk.download("wordnet", quiet=True)
nlp = spacy.load("en_core_web_md")


def _get_synonyms(word: str) -> set[str]:
    """Возвращает множество синонимов слова через WordNet."""
    synonyms = {word}
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
    return synonyms


def _expand_text_with_synonyms(text: str) -> str:
    """Расширяет текст синонимами каждого слова."""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    expanded_tokens = []
    for token in tokens:
        expanded_tokens.append(token)
        expanded_tokens.extend(_get_synonyms(token))
    return " ".join(expanded_tokens)


def _calc_spacy_similarity(text1: str, text2: str) -> float:
    """Семантическое сходство через векторы spaCy."""
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)


def calc_enhanced_similarity(text1: str, text2: str) -> float:
    """
    Расширенная метрика сходства, объединяющая:
    - TF-IDF косинусное сходство на исходных текстах
    - TF-IDF косинусное сходство на текстах, расширенных синонимами
    - Семантическое сходство spaCy (word vectors)

    Итоговый результат — взвешенное среднее трёх метрик.
    """
    tfidf_score = calc_cos_similarity(text1, text2)
    expanded1 = _expand_text_with_synonyms(text1)
    expanded2 = _expand_text_with_synonyms(text2)
    synonym_score = calc_cos_similarity(expanded1, expanded2)
    spacy_score = _calc_spacy_similarity(text1, text2)
    weights = np.array([0.3, 0.4, 0.3])
    scores = np.array([tfidf_score, synonym_score, spacy_score])
    return float(np.dot(weights, scores))
