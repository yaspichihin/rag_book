# Какой поиск выбрать:
# 1) По ключевым словам (Просто и быстро).
# 2) Векторный поиск (Для сематически богатых документов).
# 3) Поиск на основе индекса (Обеспечивает высокую скорость обработки больших датасетов).

# При этом можно сочетать все 3 способа:
# * Поиск по ключевым словам для имени файлов. 
# * Индексный поиск для группировки документов в индексируемое подмножество. 
# * Векторный поиск для перебора ограниченного числа документов в поиске наиболее подходящего.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from db import get_records
from display import print_text
from llm import call_llm


query = "define a rag store"
records = get_records()


class RetrievalComponent:

    VECTOR_METHODS = ['vector', 'indexed']
    
    def __init__(self, method='vector'):
        self.method = method
        if self.method in self.VECTOR_METHODS:
          self.vectorizer = TfidfVectorizer()
          self.tfidf_matrix = None
    
    def fit(self, records):
        self.document = records
        if self.method in self.VECTOR_METHODS:
            self.tfidf_matrix = self.vectorizer.fit_transform(records)
  
    def retrieve(self, query):
        match self.method:
            case 'keyword':
                return self.keyword_search(query)
            case 'vector':
                return self.vector_search(query)
            case 'indexed':
                return self.indexed_search(query)
    
    def keyword_search(self, query):
        best_score = 0
        best_record = None
        query_keywords = set(query.lower().split())
        for index, doc in enumerate(self.documents):
            doc_keywords = set(doc.lower().split())
            common_keywords = query_keywords.intersection(doc_keywords)
            score = len(common_keywords)
            if score > best_score:
                best_score = score
                best_record = self.documents[index]
        return best_record

    def vector_search(self, query):
        query_tfidf = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)
        best_index = similarities.argmax()
        return records[best_index]

    def indexed_search(self, query):
        # Assuming the tfidf_matrix is precomputed and stored
        query_tfidf = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)
        best_index = similarities.argmax()
        return records[best_index]




# Usage example

  # Choose from 'keyword', 'vector', 'indexed'
retrieval = RetrievalComponent(method='vector')
retrieval.fit(records)
best_matching_record = retrieval.retrieve(query)

print_text(best_matching_record)
# Response:
# --------------------------------------------------------------------------------
# A RAG vector store is a database or dataset that contains vectorized data
# points.
# --------------------------------------------------------------------------------


# Augmented Input

augmented_input=query+ " "+ best_matching_record
print_text(augmented_input)
# Response:
# --------------------------------------------------------------------------------
# define a rag store A RAG vector store is a database or dataset that contains
# vectorized data points.
# --------------------------------------------------------------------------------


llm_response = call_llm(augmented_input)
print_text(llm_response)
# Response:
# --------------------------------------------------------------------------------
# Arag vector store is a database or dataset that contains vectorized data points.
# ---