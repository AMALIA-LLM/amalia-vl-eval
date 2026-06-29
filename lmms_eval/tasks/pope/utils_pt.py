"""Portuguese (pt-PT) overrides for the POPE task.

This module re-exports the upstream POPE aggregation helpers unchanged and
overrides only the functions that diverge for Portuguese: image byte-decoding,
the prompt builder (tolerant pre/post-prompt handling), and the per-doc scoring
(normalise yes/no + ``translate_answer`` + tolerant prediction parsing).

The upstream ``utils.py`` is left byte-identical to lmms-eval, so the English
``pope`` task is unaffected and the repo stays trivially syncable with upstream.

All imports are absolute because the ``!function`` loader imports this file as a
standalone module without package context.
"""

from lmms_eval.tasks._task_utils.pt_common import load_pil_image, translate_answer

# Re-export the upstream aggregation helpers unchanged so the pt task is fully
# described by ``utils_pt`` + the pt YAML.
from lmms_eval.tasks.pope.utils import (  # noqa: F401
    pope_aggregate_accuracy,
    pope_aggregate_f1_score,
    pope_aggregate_precision,
    pope_aggregate_recall,
    pope_aggregate_yes_ratio,
)


def pope_doc_to_visual(doc):
    # Assuming the 'doc' dictionary has a key 'image' with image data
    return [load_pil_image(doc["image"])]


def pope_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # Assuming the 'doc' dictionary has a key 'question' with the question text
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def pope_process_results(doc, results):

    def normalize_yes_no(ans):
        if ans in ["sim", "s", "yes", "y"]:
            return "yes"
        if ans in ["não", "nao", "n", "no"]:
            return "no"
        return ans

    pred = results[0].lower().strip()
    pred = normalize_yes_no(pred)
    pred = translate_answer(pred)

    if pred not in ["yes", "no"]:
        new_pred = results[0].lower().strip().split()[0].strip().replace(".", "").replace(",", "")
        new_pred = normalize_yes_no(new_pred)
        new_pred = translate_answer(new_pred)
        if new_pred in ["yes", "no"]:
            pred = new_pred

    gt_ans = doc["answer"].lower().strip()
    gt_ans = normalize_yes_no(gt_ans)
    gt_ans = translate_answer(gt_ans)

    assert gt_ans in ["yes", "no"], f"Unrecognized GT Answer in dataset: {doc['answer']}"

    score = 1.0 if pred == gt_ans else 0.0

    return {
        "pope_accuracy": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_precision": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_recall": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_f1_score": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_yes_ratio": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
    }
