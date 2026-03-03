import textwrap


def format_text(text: str, width: int) -> str:
    return textwrap.fill(text, width=width)


def print_text(text: str, width: int = 80) -> None:
    print("Response:")
    print("-" * width)
    print(format_text(text, width))
    print("-" * width)
