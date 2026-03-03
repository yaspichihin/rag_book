import spacy
import nltk
from nltk.corpus import wordnet
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("wordnet", quiet=True)
nlp = spacy.load("en_core_web_md")


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


def get_synonyms(word: str) -> set[str]:
    """Возвращает множество синонимов слова через WordNet."""
    synonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
    return synonyms


def preprocess_text(text: str) -> list[str]:
    doc = nlp(text.lower())
    lemmatized_words = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        lemmatized_words.append(token.lemma_)
    return lemmatized_words


def expand_with_synonyms(words: list[str]) -> list[str]:
    expanded_words = words.copy()
    for word in words:
        expanded_words.extend(get_synonyms(word))
    return expanded_words


def calc_enhanced_similarity(text1: str, text2: str) -> float:
    
    # Preprocess and tokenize texts
    words1 = preprocess_text(text1)
    words2 = preprocess_text(text2)

    # Expand with synonyms
    words1_expanded = expand_with_synonyms(words1)
    words2_expanded = expand_with_synonyms(words2)

    # Count word frequencies
    freq1 = Counter(words1_expanded)
    freq2 = Counter(words2_expanded)

    # Create a set of all unique words
    unique_words = set(freq1.keys()).union(set(freq2.keys()))

    # Create frequency vectors
    vector1 = [freq1[word] for word in unique_words]
    vector2 = [freq2[word] for word in unique_words]

    # Convert lists to numpy arrays
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    # Calculate cosine similarity
    cosine_similarity = (
        np.dot(vector1, vector2) 
        / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    )

    return cosine_similarity
    
