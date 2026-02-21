import types
import pytest

from llm import call_llm


def make_response(content: str):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )


@pytest.fixture
def llm_client(mocker):
    return mocker.patch("llm.client")


def test_call_llm_returns_stripped_response(llm_client):
    llm_client.chat.completions.create.return_value = make_response("  answer  ")
    assert call_llm(["hello"]) == "answer"


def test_call_llm_returns_error_on_exception(llm_client):
    llm_client.chat.completions.create.side_effect = RuntimeError("connection failed")
    assert call_llm(["hello"]) == "Error: connection failed"
