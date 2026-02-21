import argparse

from llm import call_llm
from db import get_records
from display import print_text


def get_args():
    parser = argparse.ArgumentParser(description="Получение аргументов")
    parser.add_argument('query', help="Запрос для LLM")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    records = get_records()
    response = call_llm(args.query)
    print_text(response)


if __name__ == "__main__":
    main()
