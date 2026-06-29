"""Portuguese (pt-PT) overrides for the MME task.

This module re-exports the upstream MME helpers it can use unchanged and
overrides only the functions that diverge for Portuguese: image byte-decoding,
the yes/no prediction parser (which must also recognise Portuguese answers),
the per-doc scoring (tolerant ground-truth parsing + ``translate_answer``) and
the aggregation (which skips incomplete pairs instead of asserting).

The upstream ``utils.py`` is left byte-identical to lmms-eval, so the English
``mme`` task is unaffected and the repo stays trivially syncable with upstream.

All imports are absolute because the ``!function`` loader imports this file as a
standalone module without package context.
"""

from collections import defaultdict

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.pt_common import load_pil_image, translate_answer

# Re-export the upstream helpers that need no change so the pt task is fully
# described by ``utils_pt`` + the pt YAML.
from lmms_eval.tasks.mme.utils import (  # noqa: F401
    eval_type_dict,
    mme_doc_to_text,
    replace_prompt,
)


def mme_doc_to_visual(doc):
    return [load_pil_image(doc["image"])]


def parse_pred_ans(pred_ans):
    """Brought from Otter Eval, extended to recognise Portuguese yes/no answers."""
    pred_ans = pred_ans.lower().strip().replace(".", "")
    pred_ans = translate_answer(pred_ans)
    pred_label = None

    if pred_ans in ["yes", "no", "sim", "não", "nao"]:
        pred_label = "yes" if pred_ans in ["yes", "sim"] else "no"
    elif len(pred_ans) == 1:
        if pred_ans in ["y", "s"]:
            pred_label = "yes"
        elif pred_ans in ["n"]:
            pred_label = "no"
        else:
            pred_label = "other"
    else:
        prefix_pred_ans = pred_ans[:4]
        if any(word in prefix_pred_ans for word in ["yes", "sim"]):
            pred_label = "yes"
        elif any(word in prefix_pred_ans for word in ["no", "não", "nao"]):
            pred_label = "no"
        else:
            pred_label = "other"
    return pred_label


def mme_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
    """
    pred = results[0]
    pred_ans = parse_pred_ans(pred)
    gt_ans = doc["answer"].lower().strip().replace(".", "")
    gt_ans = translate_answer(gt_ans)

    if gt_ans in ["sim", "s", "yes", "y"]:
        gt_ans = "yes"
    elif gt_ans in ["não", "nao", "n", "no"]:
        gt_ans = "no"

    assert gt_ans in ["yes", "no"], f"Unrecognized GT Answer: {gt_ans}"
    assert pred_ans in ["yes", "no", "other"]

    score = 1.0 if pred_ans == gt_ans else 0.0
    category = doc["category"]
    key_name = "mme_perception_score" if category in eval_type_dict["Perception"] else "mme_cognition_score"

    return {key_name: {"question_id": doc["question_id"], "category": category, "score": score}}


def mme_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = defaultdict(dict)
    for result in results:
        question_id = result["question_id"]
        score = result["score"]
        category = result["category"]
        if question_id not in category2score[category]:
            category2score[category][question_id] = []
        category2score[category][question_id].append(score)
    category2avg_score = {}
    for category, question2scores in category2score.items():
        total_score = 0
        for question_id, scores in question2scores.items():
            if len(scores) != 2:
                eval_logger.warning(f"question_id={question_id} in category={category} has {len(scores)} scores (expected 2), skipping")
                continue
            acc = sum(scores) / len(scores) * 100.0
            acc_plus = (sum(scores) == 2) * 100.0
            score = acc_plus + acc
            total_score += score
        num_valid = sum(1 for s in question2scores.values() if len(s) == 2)
        avg_score = total_score / num_valid if num_valid > 0 else 0.0
        category2avg_score[category] = avg_score
    for category, avg_score in category2avg_score.items():
        eval_logger.info(f"{category}: {avg_score:.2f}")
    total_score = sum(category2avg_score.values())
    return total_score
