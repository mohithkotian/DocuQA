"""Local extractive question answering with answer-span evidence validation."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Protocol, Sequence

from .retrieval import SearchResult

INSUFFICIENT_EVIDENCE = "Insufficient evidence in the provided documents."
from .config import DEFAULT_QA_MODEL


@dataclass(frozen=True)
class QAResponse:
    """Extracted answer plus auditable document evidence, or an abstention."""

    answer: str
    source: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    evidence_text: str | None = None
    answer_start: int | None = None
    answer_end: int | None = None
    qa_score: float | None = None
    retrieval_score: float | None = None
    supported: bool = False


@dataclass(frozen=True)
class _Candidate:
    answer: str
    context: str
    start: int
    end: int
    margin: float
    confidence: float


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]: ...


def _empty_response() -> QAResponse:
    return QAResponse(answer=INSUFFICIENT_EVIDENCE)


def _confidence_from_margin(margin: float) -> float:
    """Stable sigmoid used as an indicative, not calibrated, answer score."""
    if margin >= 0:
        return 1.0 / (1.0 + math.exp(-min(margin, 700.0)))
    exp_value = math.exp(max(margin, -700.0))
    return exp_value / (1.0 + exp_value)


class LocalExtractiveQAModel:
    """Load a Hugging Face extractive QA checkpoint and infer over retrieved chunks.

    Defaults to deepset/MiniLM SQuAD 2.0, a local 33.4M-parameter checkpoint
    trained with answerable and unanswerable QA examples. Downloading the model
    requires internet access on first use; model inference itself is local and does
    not call a hosted API or require an API key.
    """

    def __init__(self, model_name: str = DEFAULT_QA_MODEL, *, max_length: int = 384,
                 stride: int = 96, max_answer_tokens: int = 30,
                 no_answer_threshold: float = 0.0, device: str = "cpu") -> None:
        if max_length < 32:
            raise ValueError("max_length must be at least 32")
        if stride < 0 or stride >= max_length:
            raise ValueError("stride must be non-negative and smaller than max_length")
        if max_answer_tokens < 1:
            raise ValueError("max_answer_tokens must be at least 1")
        self.model_name = model_name
        self.max_length = max_length
        self.stride = stride
        self.max_answer_tokens = max_answer_tokens
        self.no_answer_threshold = no_answer_threshold
        self.device_name = device
        try:
            import torch
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers to use extractive QA") from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not self._tokenizer.is_fast:
            raise RuntimeError("QA requires a fast tokenizer for source-span evidence offsets")
        self._model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self._device = torch.device(device)
        self._model.to(self._device)
        self._model.eval()

    def _best_candidate(self, question: str, context: str) -> _Candidate | None:
        torch = self._torch
        if not context.strip():
            return None
        encoded = self._tokenizer(
            question.strip(), context,
            truncation="only_second",
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )
        offsets = encoded["offset_mapping"].tolist()
        model_inputs = {
            name: tensor.to(self._device)
            for name, tensor in encoded.items()
            if name in self._tokenizer.model_input_names and name != "offset_mapping"
        }
        with torch.inference_mode():
            output = self._model(**model_inputs)
        best: _Candidate | None = None
        start_logits = output.start_logits.detach().cpu()
        end_logits = output.end_logits.detach().cpu()
        for feature_index, feature_offsets in enumerate(offsets):
            sequence_ids = encoded.sequence_ids(feature_index)
            starts = start_logits[feature_index]
            ends = end_logits[feature_index]
            # SQuAD 2.0 checkpoints use the leading CLS token to represent no answer.
            null_score = float(starts[0] + ends[0])
            top_starts = torch.topk(starts, k=min(20, starts.shape[0])).indices.tolist()
            top_ends = torch.topk(ends, k=min(20, ends.shape[0])).indices.tolist()
            context_starts = [i for i, sequence in enumerate(sequence_ids)
                              if sequence == 1 and feature_offsets[i][1] > feature_offsets[i][0]]
            if not context_starts:
                continue
            first_context, last_context = context_starts[0], context_starts[-1]
            for start_token in top_starts:
                if start_token < first_context or start_token > last_context:
                    continue
                for end_token in top_ends:
                    if end_token < start_token or end_token > last_context:
                        continue
                    if end_token - start_token + 1 > self.max_answer_tokens:
                        continue
                    if any(sequence_ids[token] != 1 for token in range(start_token, end_token + 1)):
                        continue
                    char_start = int(feature_offsets[start_token][0])
                    char_end = int(feature_offsets[end_token][1])
                    if char_end <= char_start:
                        continue
                    answer = context[char_start:char_end]
                    if not answer.strip():
                        continue
                    answer_score = float(starts[start_token] + ends[end_token])
                    margin = answer_score - null_score
                    candidate = _Candidate(
                        answer=answer, context=context, start=char_start, end=char_end,
                        margin=margin, confidence=_confidence_from_margin(margin),
                    )
                    if best is None or candidate.margin > best.margin:
                        best = candidate
        return best

    @staticmethod
    def _validate(candidate: _Candidate) -> bool:
        """Require a verbatim non-empty answer span in the exact retrieved chunk."""
        if not candidate.answer.strip() or candidate.start < 0 or candidate.end <= candidate.start:
            return False
        return candidate.context[candidate.start:candidate.end] == candidate.answer

    @staticmethod
    def _calibrate_candidate(question: str, passage_text: str,
                             candidate: _Candidate) -> tuple[_Candidate, int]:
        """Evidence-bounded numeric calibration (Variant D).

        When a question requests a specific count/unit (e.g., 'how many overs')
        and the extracted candidate has a conflicting unit (e.g., 'one innings'),
        calibrate to the verbatim '<count> <unit>' span in the candidate's containing sentence,
        provided that sentence matches the question's entity cues.
        Returns the calibrated candidate (or original if not calibrated) and entity alignment hits.
        """
        if not candidate.answer.strip():
            return candidate, 0
        q_lower = question.lower().strip()
        match_q = re.search(r"\bhow many\s+([a-zA-Z\-]+)", q_lower)
        if not match_q:
            return candidate, 0
        target_unit = match_q.group(1).rstrip("s")
        if target_unit in candidate.answer.lower():
            return candidate, 0

        # Find containing sentence boundaries around candidate
        s_start = 0
        for m in re.finditer(r"(?<=[.!?\n])\s+", passage_text[:candidate.start]):
            s_start = m.end()
        m_end = re.search(r"[.!?\n]", passage_text[candidate.end:])
        s_end = candidate.end + m_end.start() if m_end else len(passage_text)
        sentence = passage_text[s_start:s_end]

        num_pattern = (
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twenty|fifty)\s+("
            + re.escape(target_unit)
            + r"[a-zA-Z]*)\b"
        )
        cand_match = re.search(num_pattern, sentence, flags=re.IGNORECASE)
        if not cand_match:
            return candidate, 0

        # Check entity alignment to avoid selecting a number from an entity-mismatched sentence
        q_words = set(re.findall(r"\b[a-zA-Z0-9\-]{3,}\b", q_lower)) - {
            "how", "many", "each", "team", "receive", "match", target_unit, target_unit + "s"
        }
        sent_words = set(re.findall(r"\b[a-zA-Z0-9\-]{3,}\b", sentence.lower()))
        hits = len(q_words & sent_words)
        if q_words and hits == 0:
            return candidate, 0

        calib_start = s_start + cand_match.start()
        calib_end = s_start + cand_match.end()
        calib_ans = passage_text[calib_start:calib_end]
        if not calib_ans.strip() or passage_text[calib_start:calib_end] != calib_ans:
            return candidate, 0

        new_candidate = _Candidate(
            answer=calib_ans,
            context=passage_text,
            start=calib_start,
            end=calib_end,
            margin=candidate.margin,
            confidence=candidate.confidence,
        )
        return new_candidate, hits

    def answer_from_passages(self, question: str,
                             passages: Sequence[SearchResult]) -> QAResponse:
        """Run span extraction over retrieved chunks and cite the best supported one."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not passages:
            return _empty_response()
        candidates: list[tuple[_Candidate, SearchResult, int]] = []
        for passage in passages:
            candidate = self._best_candidate(question, passage.chunk.text)
            if candidate is not None and self._validate(candidate):
                calib_cand, hits = self._calibrate_candidate(question, passage.chunk.text, candidate)
                if self._validate(calib_cand):
                    candidates.append((calib_cand, passage, hits))
                else:
                    candidates.append((candidate, passage, 0))
        supported = [item for item in candidates if item[0].margin > self.no_answer_threshold]
        if not supported:
            return _empty_response()
        candidate, passage, _ = max(supported, key=lambda item: (item[2], item[0].margin))
        return QAResponse(
            answer=candidate.answer,
            source=passage.source,
            page=passage.page,
            chunk_id=passage.chunk_id,
            evidence_text=passage.chunk.text,
            answer_start=candidate.start,
            answer_end=candidate.end,
            qa_score=candidate.confidence,
            retrieval_score=passage.score,
            supported=True,
        )


def retrieve_and_answer(question: str, retriever: Retriever,
                        qa_model: LocalExtractiveQAModel, top_k: int = 5) -> QAResponse:
    """Execute retrieval → extractive QA → source-span evidence validation."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    passages = retriever.search(question, top_k=top_k)
    return qa_model.answer_from_passages(question, passages)
