"""Local question generation for document-grounded DocuQA workflows.

Question generation creates candidates only. Answers still come from the existing
retrieval + extractive QA pipeline and must pass exact source-span validation.
"""
from __future__ import annotations

import os
import re

DEFAULT_QUESTION_GENERATION_MODEL = os.getenv(
    "DOCUQA_QUESTION_GENERATION_MODEL",
    "valhalla/t5-small-e2e-qg",
)


def parse_generated_questions(text: str, limit: int = 10) -> list[str]:
    """Parse multi-question T5 output conservatively; discard malformed candidates."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if not isinstance(text, str) or not text.strip():
        return []

    decoded = re.sub(r"</?s>|<pad>|<unk>", " ", text, flags=re.IGNORECASE)
    decoded = re.sub(r"<sep>|<extra_id_\d+>", "\n", decoded, flags=re.IGNORECASE)
    seen: set[str] = set()
    questions: list[str] = []

    # The model may emit multiple questions in one sequence, separated by <sep>
    # or by whitespace. Accept only explicit question-mark-terminated strings.
    for match in re.finditer(r"([^?]{4,240}\?)", decoded):
        question = match.group(1).strip()
        question = re.sub(r"^\s*(?:[-*•]+\s*|\d+[.)\-:]\s*)", "", question)
        question = re.sub(r"\s+", " ", question).strip(" \t\r\n\"'“”")
        words = question.rstrip("?").split()
        if not 3 <= len(words) <= 30:
            continue
        if not question.endswith("?"):
            continue
        key = re.sub(r"[^a-z0-9]+", " ", question.casefold()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        questions.append(question)
        if len(questions) >= limit:
            break
    return questions


class LocalQuestionGenerator:
    """Load a small local sequence-to-sequence model for candidate generation."""

    def __init__(self, model_name: str = DEFAULT_QUESTION_GENERATION_MODEL) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local question generation requires torch and transformers."
            ) from exc

        self.model_name = model_name
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()

    def generate_questions(self, text: str, limit: int = 5) -> list[str]:
        """Generate question candidates from one source chunk, without answering them."""
        if not isinstance(text, str) or not text.strip():
            return []
        if limit < 1:
            raise ValueError("limit must be at least 1")

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {name: tensor.to(self._device) for name, tensor in encoded.items()}
        with self._torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=128,
                num_beams=4,
                no_repeat_ngram_size=2,
            )
        decoded = self._tokenizer.decode(generated[0], skip_special_tokens=False)
        return parse_generated_questions(decoded, limit=limit)
