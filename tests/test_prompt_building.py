"""
Tests for pipeline.py::build_prompt_and_text — the base-model vs.
instruct-model prompt-construction branch.

No real tokenizer needed: `chat_template` is the only attribute
build_prompt_and_text reads off the tokenizer, and `apply_chat_template`
only needs to accept the same call signature transformers' real tokenizers
use, so a tiny fake stands in for both AutoTokenizer cases.
"""

from pipeline import build_prompt_and_text


class _FakeBaseTokenizer:
    """Mimics a base-model tokenizer (e.g. Pythia's): no chat template."""
    chat_template = None


class _FakeInstructTokenizer:
    """Mimics an instruct-tuned tokenizer (e.g. Qwen2.5-Instruct's)."""
    chat_template = "{{ messages }}"  # just needs to be truthy

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        return f"<|user|>{messages[0]['content']}<|assistant|>"


def test_base_model_uses_completion_style_prompt():
    prompt, text = build_prompt_and_text(_FakeBaseTokenizer(), "What is 2+2?", "4")
    assert prompt == "Question: What is 2+2?\nAnswer:"
    assert text == "Question: What is 2+2?\nAnswer: 4"
    assert text.startswith(prompt)


def test_instruct_model_uses_chat_template():
    prompt, text = build_prompt_and_text(_FakeInstructTokenizer(), "What is 2+2?", "4")
    assert prompt == "<|user|>What is 2+2?<|assistant|>"
    assert text == "<|user|>What is 2+2?<|assistant|>4"
    assert text.startswith(prompt)


def test_missing_chat_template_attribute_falls_back_to_base_style():
    class _NoAttrTokenizer:
        pass  # no chat_template attribute at all

    prompt, text = build_prompt_and_text(_NoAttrTokenizer(), "Q", "A")
    assert prompt == "Question: Q\nAnswer:"
    assert text == "Question: Q\nAnswer: A"
