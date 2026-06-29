"""Self-contained scoring utilities for the CARAVELA benchmark.

CARAVELA (Cultural Awareness and Recognition Assessment for Visual Entity
Literacy and Analysis) is a native European Portuguese (pt-PT) benchmark of
culturally relevant visual entities, hosted at ``amalia-llm/CARAVELA``. It ships
three leaf tasks, each backed by one parquet file at the repo root:

    caravela_mcq        -> caravela_mcq.parquet        (4-option multiple choice)
    caravela_vqa        -> caravela_vqa.parquet        (short open-ended answers)
    caravela_reasoning  -> caravela_reasoning.parquet  (step-by-step reasoning)

Loaded by the ``!function utils.X`` YAML loader as a STANDALONE module, so it
uses absolute imports only and reuses the shared pt-PT helpers in
``_task_utils`` rather than re-implementing image decoding / MCQ extraction.

Each row exposes: ``image``, ``question``, ``answer``, ``entity``, ``area``,
``category`` (plus ``options_a..d`` for MCQ and ``reasoning_chain`` for
reasoning). Models are instructed to end their reply with ``ANSWER:`` followed
by the final answer; ``<think>...</think>`` reasoning is stripped upstream by
the harness (via the task's ``reasoning_tags``) before ``process_results``.
"""

import re

from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer
from lmms_eval.tasks._task_utils.pt_common import load_pil_image

# --- shared helpers ------------------------------------------------------- #


def caravela_doc_to_visual(doc):
    return [load_pil_image(doc["image"])]


def _extract_final_answer(text):
    """Return the text after the last ``ANSWER:`` marker, or the whole reply."""
    text = text.strip()
    idx = text.rfind("ANSWER:")
    if idx != -1:
        return text[idx + len("ANSWER:") :].strip()
    return text


def _normalize_text(text):
    """Lowercase, strip common pt-PT articles/prepositions and punctuation."""
    text = text.strip().lower()
    text = re.sub(r"\b(o|a|os|as|um|uma|uns|umas|de|do|da|dos|das|em|no|na|nos|nas)\b", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _result_text(results):
    """Normalise the harness result into a single string."""
    resp = results[0]
    return resp if isinstance(resp, str) else resp[0]


def _meta(doc):
    """Common per-sample metadata logged alongside every score."""
    return {
        "entity": doc.get("entity", ""),
        "area": doc.get("area", "general"),
        "category": doc.get("category", ""),
        "question": doc.get("question", ""),
    }


def caravela_aggregate_results(results):
    """Mean per-sample score as a percentage.

    Shared by all three leaves so they report the same metric name
    (``caravela_score``); this lets the ``caravela`` group pool them into a
    single weighted mean via its ``aggregate_metric_list``.
    """
    if not results:
        return 0.0
    return (sum(r["score"] for r in results) / len(results)) * 100.0


# --- MCQ ------------------------------------------------------------------ #


def caravela_mcq_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    options = f"A. {doc['options_a']}\nB. {doc['options_b']}\nC. {doc['options_c']}\nD. {doc['options_d']}"

    pre_prompt = ""
    post_prompt = "\nResponde apenas com a letra da opção correta."
    if lmms_eval_specific_kwargs:
        if lmms_eval_specific_kwargs.get("pre_prompt", ""):
            pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
        if lmms_eval_specific_kwargs.get("post_prompt", ""):
            post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    return f"{pre_prompt}{question}\n{options}{post_prompt}"


def caravela_mcq_process_results(doc, results):
    pred = extract_mcq_answer(_extract_final_answer(_result_text(results)), choices=["A", "B", "C", "D"])
    gt = doc["answer"].strip().upper()
    score = 1.0 if pred == gt else 0.0
    return {"caravela_score": {**_meta(doc), "score": score}}


# --- VQA ------------------------------------------------------------------ #


def caravela_vqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    pre_prompt = ""
    post_prompt = "\nResponde de forma curta e direta."
    if lmms_eval_specific_kwargs:
        if lmms_eval_specific_kwargs.get("pre_prompt", ""):
            pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
        if lmms_eval_specific_kwargs.get("post_prompt", ""):
            post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def caravela_vqa_process_results(doc, results):
    pred_norm = _normalize_text(_extract_final_answer(_result_text(results)))
    target_norm = _normalize_text(doc["answer"])

    exact = float(pred_norm == target_norm)
    contains = float(target_norm in pred_norm) if target_norm else 0.0
    score = max(exact, contains)
    return {"caravela_score": {**_meta(doc), "score": score}}


# --- Reasoning ------------------------------------------------------------ #


def caravela_reasoning_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    post_prompt = "\nExplica o teu raciocínio passo a passo."
    if lmms_eval_specific_kwargs and lmms_eval_specific_kwargs.get("post_prompt", ""):
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{question}{post_prompt}"


def caravela_reasoning_process_results(doc, results):
    pred = _normalize_text(_extract_final_answer(_result_text(results)))
    target = _normalize_text(doc["answer"])

    meta = {**_meta(doc), "reasoning_chain": doc.get("reasoning_chain", "")}
    target_words = {w for w in target.split() if len(w) > 3}
    pred_words = {w for w in pred.split() if len(w) > 3}
    if not target_words:
        score = 0.0
    else:
        overlap = target_words & pred_words
        precision = len(overlap) / len(pred_words) if pred_words else 0.0
        recall = len(overlap) / len(target_words)
        score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"caravela_score": {**meta, "score": score}}
