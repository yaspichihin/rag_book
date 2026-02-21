from openai import OpenAI

from config import BASE_URL, API_KEY, MODEL, TEMPERATURE


client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)


def call_llm(texts: list[str]) -> str:
    prompt = "Please elaborate on:\n" + "\n".join(texts)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an NLP expert. Always respond in Russian.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"
