# С увеличением датасета увеличивается выполнение поиска по ключевым
# словам, осуществим переход к использованию индекса.

## Векторный поиск
import pandas as pd
from typing import Collection, Tuple
from llm import call_llm
from similarity import calc_cos_similarity, calc_enhanced_similarity
from simple_rag import augmented_input, db_records, llm_response
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def find_best_match(
    text_input: str,
    records: Collection[str],
) -> Tuple[int, str]:
    best_score = 0
    best_record = None

    for record in records:
        curr_score = calc_cos_similarity(text_input, record)
        if curr_score > best_score:
            best_score = curr_score
            best_record = record

    return best_score, best_record


from db import get_records
from display import print_text
from similarity import calc_enhanced_similarity

query = "define a rag store"
records = get_records()

best_similarity_score, best_similarity_record = find_best_match(
    query,
    records,
)

print_text(f"Best Similarity Score: {best_similarity_score:.3f}")
# Best Similarity Score: 0.126

print_text(f"Best Similarity Record: {best_similarity_record}")
# Best Similarity Record: A RAG vector store is a database or dataset ...

## Метрики

# Использовать будем те же метрики что и в simple rag
similarity_score = calc_enhanced_similarity(query, best_similarity_record)
print_text(f"Enhanced Similarity Score: {similarity_score:.3f}")
# Enhanced Similarity Score: 0.642


## Дополнительный ввод

augmented_input = query + ": " + best_similarity_record


## Генерация

llm_response = call_llm(augmented_input)
print_text(llm_response, width=60)


## Index-based search (Поиск на основе индекса)
# Ускорение извлечение документа при масштабировании документов.

def setup_vectorizer(records: list[str]):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        use_idf=True,
        norm="l2",
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(records)
    return vectorizer, tfidf_matrix


def find_best_match(query, vectorizer, tfidf_matrix):
    # Преобразование запроса в вектор TF-IDF
    query_tfidf = vectorizer.transform([query])
    # Вычисление сходства
    similarities = cosine_similarity(query_tfidf, tfidf_matrix)
    # Поиска максимального совпадения
    best_index = similarities.argmax()
    best_score = similarities[0, best_index]
    best_record = records[best_index]
    return best_score, best_record


vectorizer, tfidf_matrix = setup_vectorizer(records)
best_similarity_score, best_matching_record = find_best_match(
    query,
    vectorizer,
    tfidf_matrix,
)

print_text(f"Best Similarity Score: {best_similarity_score:.3f}")
# Best Similarity Score: 0.436

print_text(f"Best Matching Record: {best_matching_record}")
# Response:
# --------------------------------------------------------------------------------
# Best Matching Record: A RAG vector store is a database or dataset that contains
# vectorized data points.
# --------------------------------------------------------------------------------


## Enhanced Similarity
response = best_matching_record
print(query,": ", response)
# define a rag store :  A RAG vector store is a database or dataset that contains vectorized data points.

similarity_score = calc_enhanced_similarity(query, response)

print(f"Enhanced Similarity:, {similarity_score:.3f}")
# Enhanced Similarity:, 0.642



def setup_vectorizer(records):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(records)

    # Convert the TF-IDF matrix to a DataFrame for display purposes
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out(),
    )

    # Display the DataFrame
    print(tfidf_df)

    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = setup_vectorizer(db_records)
"""
     ability    access  accuracy  advancement  ...    without
0   0.000000  0.000000  0.000000     0.000000  ...   0.000000  
1   0.000000  0.000000  0.000000     0.000000  ...   0.000000  
2   0.000000  0.000000  0.000000     0.000000  ...   0.000000  
3   0.000000  0.000000  0.000000     0.000000  ...   0.000000  
4   0.000000  0.000000  0.000000     0.000000  ...   0.000000  
...      ...       ...       ...          ...  ...        ...
24  0.000000  0.000000  0.000000     0.000000  ...   0.000000  
25  0.000000  0.000000  0.228743     0.000000  ...   0.000000  
26  0.000000  0.000000  0.000000     0.173327  ...   0.000000  
27  0.000000  0.000000  0.000000     0.000000  ...   0.000000  
[28 rows x 297 columns]
"""

