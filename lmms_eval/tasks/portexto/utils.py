"""Self-contained scoring utilities for the PorTEXTO task (``portexto_pt``).

This is the V2 lineage of the old ``texto_v2_pt`` task, renamed to PorTEXTO and
shipped as a single self-contained module (no separate utils + utils_v2 split).
It targets the dataset ``amalia-llm/PorTEXTO``.

Loaded by the ``!function utils.X`` YAML loader as a STANDALONE module, so it
uses absolute imports only.

METRIC NOTE: BLEU is computed as BLEU-1 (unigram) via ``sacrebleu``'s ``BLEU``
class, NOT the default corpus BLEU-4. ANLS is unchanged from the source.
"""

from sacrebleu.metrics import BLEU

from lmms_eval.tasks._task_utils.pt_common import load_pil_image

# --- doc helpers ---------------------------------------------------------- #


def texto_doc_to_visual(doc):
    return [load_pil_image(doc["image"])]


def texto_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


# --- ANLS ----------------------------------------------------------------- #


def _levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def _compute_anls(pred, references, threshold=0.5):
    values = []
    for answer in references:
        gt_answer = " ".join(answer.strip().lower().split())
        det_answer = " ".join(pred.strip().lower().split())
        dist = _levenshtein_distance(gt_answer, det_answer)
        length = max(len(answer.upper()), len(pred.upper()))
        values.append(0.0 if length == 0 else float(dist) / float(length))
    question_result = 1 - min(values)
    if question_result < threshold:
        question_result = 0
    return question_result


# --- BLEU helper (BLEU-1) ------------------------------------------------- #


def _corpus_bleu1(preds_formatted, refs_transposed):
    """Compute corpus-level BLEU-1 (unigram) score.

    Uses sacrebleu's ``BLEU`` class with ``max_ngram_order=1`` and
    ``effective_order=True`` instead of the default ``corpus_bleu`` (BLEU-4).
    """
    bleu = BLEU(max_ngram_order=1, effective_order=True)
    return bleu.corpus_score(preds_formatted, refs_transposed).score


# --- process_results (v2) ------------------------------------------------- #


def texto_v2_process_results(doc, results):
    try:
        pred = results[0].strip() if isinstance(results[0], str) else results[0][0].strip()
    except Exception:
        pred = ""

    target = doc["answer"]
    references = [target] if isinstance(target, str) else list(target)

    try:
        anls_score = _compute_anls(pred, references)
    except Exception:
        anls_score = 0.0

    return {
        "anls": anls_score,
        "bleu": (references, pred),
    }


# --- top-level aggregations ----------------------------------------------- #


def texto_anls_aggregation(items):
    return sum(items) / len(items)


def texto_bleu_aggregation(items):
    try:
        refs = list(zip(*items))[0]
        preds = list(zip(*items))[1]
        refs_formatted = []
        preds_formatted = []
        for ref, pred in zip(refs, preds):
            if isinstance(ref, str):
                refs_formatted.append([ref])
            else:
                refs_formatted.append(list(ref))
            preds_formatted.append(pred if isinstance(pred, str) else pred[0])
        max_ref_len = max(len(r) for r in refs_formatted)
        refs_transposed = []
        for i in range(max_ref_len):
            refs_transposed.append([r[i] if i < len(r) else r[0] for r in refs_formatted])
        return _corpus_bleu1(preds_formatted, refs_transposed)
    except Exception:
        return 0.0
