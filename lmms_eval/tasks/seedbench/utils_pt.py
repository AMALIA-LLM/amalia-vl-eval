"""Portuguese (pt-PT) overrides for the SEED-Bench task.

This module re-exports the upstream SEED-Bench helpers unchanged and overrides
only the doc-to-text builder (so it honours a Portuguese ``post_prompt``) and the
result parser (so it recognises Portuguese answer-prefix phrasings and normalises
case). The upstream ``utils.py`` is left byte-identical to lmms-eval, so the
English ``seedbench`` task is unaffected and the repo stays trivially syncable
with upstream.

The pt YAML references everything through this single ``utils_pt`` module.
"""

# Re-export the upstream helpers unchanged so the pt task is fully described by
# ``utils_pt`` + the pt YAML.
from lmms_eval.tasks.seedbench.utils import (  # noqa: F401
    seed_aggregation_result,
    seed_doc_to_visual,
)


def seed_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    question += "\n" + f"A. {doc['choice_a']}\n"
    question += f"B. {doc['choice_b']}\n"
    question += f"C. {doc['choice_c']}\n"
    question += f"D. {doc['choice_d']}"

    if lmms_eval_specific_kwargs and "post_prompt" in lmms_eval_specific_kwargs:
        return f"{question}{lmms_eval_specific_kwargs['post_prompt']}"

    return f"{question}\nAnswer with the option's letter from the given choices directly."


def seed_doc_to_text_mc(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]

    if lmms_eval_specific_kwargs and "mc_prompt" in lmms_eval_specific_kwargs:
        return f"{question}{lmms_eval_specific_kwargs['mc_prompt']}"

    return f"{question} Answer :"


def seed_process_result(doc, result):
    pred = result[0].strip()
    pred_lower = pred.lower().replace("\n", " ").strip()

    if pred_lower.startswith("opção ") or pred_lower.startswith("opcao "):
        pred = pred_lower[6:7]
    elif pred_lower.startswith("option "):
        pred = pred_lower[7:8]
    elif pred_lower.startswith("a resposta é ") or pred_lower.startswith("a resposta e "):
        pred = pred_lower[13:14]
    elif pred_lower.startswith("a resposta é a ") or pred_lower.startswith("a resposta e a"):
        pred = pred_lower[15:16]
    elif pred_lower.startswith("the answer is "):
        pred = pred_lower[14:15]
    elif pred_lower.startswith("resposta: "):
        pred = pred_lower[10:11]
    elif len(pred) > 1:
        pred = pred[0]

    pred = pred.upper()
    answer = doc["answer"].upper()
    data_type = doc["data_type"]

    return {f"seed_{data_type}": {"pred": pred, "answer": answer, "question_id": doc["question_id"]}, "seed_all": {"pred": pred, "answer": answer, "question_id": doc["question_id"]}}
