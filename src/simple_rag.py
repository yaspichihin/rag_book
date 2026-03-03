from typing import Tuple
from typing import Collection, Tuple

from similarity import calc_enhanced_similarity, calc_cos_similarity
from db import get_records
from display import print_text
from llm import call_llm



## Поиск и сопоставление ключевых слов

def find_best_match_keyword_search(
    query: str,
    db_records: Collection[str],
) -> Tuple[int, str]:
    best_score = 0
    best_record = None

    # Разделение запроса на ключевые слова
    query_keywords = set(query.lower().split())

    # Перебор каждой записи в базе данных
    for record in db_records:

        # Разделение записи на ключевые слова
        record_keywords = set(record.lower().split())

        # Подсчет количества общих ключевых слов
        common_keywords = query_keywords.intersection(record_keywords)
        current_score = len(common_keywords)

        if current_score > best_score:
            best_score = current_score
            best_record = record

    return best_score, best_record


query = "define a rag store"
db_records = get_records()

best_keyword_score, best_matching_record = find_best_match_keyword_search(
    query=query,
    db_records=db_records,
)

print("Best keyword score:", best_keyword_score)
# Best keyword score: 1

print_text(f"Best matching record: {best_matching_record}")
# Best matching record: Retrieval Augmented Generation (RAG) ...

## Метрики

### Косинусное сходство

score = calc_cos_similarity(query, best_matching_record)
print(f"Best Cosine Similarity Score: {score:.3f}")
# Best Cosine Similarity Score: 0.042

### Расширенное сходство

score = calc_enhanced_similarity(query, best_matching_record)
print(f"Best Enhanced Similarity Score: {score:.3f}")
# Best Enhanced Similarity Score: 0.325


## Дополнительный ввод

augmented_input = query + ": " + best_matching_record
print_text(f"Augmented input: {augmented_input}")

## Генерация

llm_response = call_llm(augmented_input)
print_text(llm_response)
