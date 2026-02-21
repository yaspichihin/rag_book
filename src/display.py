import textwrap

def format_text(text: str, width: int = 80) -> str:
    return textwrap.fill(text, width=width)

def print_text(text: str) -> None:
    print("Response:")
    print(format_text(text))
  