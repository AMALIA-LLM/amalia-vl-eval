"""Portuguese (pt-PT) overrides for the MMStar task.

This module re-exports the upstream MMStar helpers that are unchanged and
overrides only the functions that diverge for the translated dataset:

* ``mmstar_doc_to_text`` — strips BOTH the English and Portuguese
  "please answer yes or no" boilerplate and applies the pre/post prompts.
* ``exact_match`` — recognises many additional Portuguese answer prefixes
  (e.g. "a resposta é", "opção", "resposta:") on top of the English ones.
* ``mmstar_process_results`` — redefined so it calls THIS module's pt
  ``exact_match`` (upstream ``mmstar_process_results`` references ``exact_match``
  as a module-level name, which would otherwise resolve to the upstream one).

The upstream ``utils.py`` is left byte-identical to lmms-eval, so the English
``mmstar`` task is unaffected and the repo stays trivially syncable with
upstream. The pt YAML references everything through this single ``utils_pt``
module.
"""

# Re-export the upstream functions the pt task uses unchanged so the pt task is
# fully described by ``utils_pt`` + the pt YAML.
from lmms_eval.tasks.mmstar.utils import (  # noqa: F401
    mmstar_aggregate_results,
    mmstar_doc_to_visual,
)


def mmstar_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()

    question = question.replace(" Please answer yes or no.", "").replace(" Por favor, responde sim ou não.", "")

    if lmms_eval_specific_kwargs:
        if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
            question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
        if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
            question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def exact_match(pred, gt):
    """Brought from MMStar and adapted for PT/EN"""
    answer = gt.lower().replace("\n", " ").strip()
    predict = pred.lower().replace("\n", " ").strip()
    try:
        if answer == predict[0]:
            return 1.0
        elif predict[0] == "(" and answer == predict[1]:
            return 1.0
        elif predict.startswith("option ") and answer == predict[7]:
            return 1.0
        elif predict.startswith("the answer is ") and answer == predict[14]:
            return 1.0
        elif predict.startswith("opção ") and answer == predict[6]:
            return 1.0
        elif predict.startswith("opcao ") and answer == predict[6]:
            return 1.0
        elif predict.startswith("a resposta é ") and answer == predict[13]:
            return 1.0
        elif predict.startswith("a resposta e ") and answer == predict[13]:
            return 1.0
        elif predict.startswith("a resposta e a ") and answer == predict[15]:
            return 1.0
        elif predict.startswith("a resposta é a ") and answer == predict[15]:
            return 1.0
        elif predict.startswith("resposta: ") and answer == predict[10]:
            return 1.0
    except Exception as e:
        return 0.0
    return 0.0


def mmstar_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = results[0]
    gt = doc["answer"]

    score = exact_match(pred, gt)
    category = doc["category"]
    l2_category = doc["l2_category"]
    return {category: {"question_id": doc["index"], "l2_category": l2_category, "score": score}, "average": {"question_id": doc["index"], "l2_category": l2_category, "score": score}}
