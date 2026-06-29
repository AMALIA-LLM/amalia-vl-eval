"""Portuguese (pt-PT) overrides for the ChartQA task.

This module re-exports the upstream ChartQA helpers unchanged and overrides only
the two functions that diverge for the translated dataset:

- ``chartqa_doc_to_visual`` — decodes images through the shared
  ``pt_common.load_pil_image`` helper (the translated HF parquet may store images
  as PIL, raw bytes, or ``{"bytes": ...}`` dicts).
- ``chartqa_process_results`` — strips trailing punctuation from the prediction
  before scoring.

The upstream ``utils.py`` is left byte-identical to lmms-eval, so the English
``chartqa`` task is unaffected and the repo stays trivially syncable with
upstream. The pt YAML references everything through this single ``utils_pt``
module.
"""

from lmms_eval.tasks._task_utils.pt_common import load_pil_image

# Re-export the upstream helpers unchanged so the pt task is fully described by
# ``utils_pt`` + the pt YAML. ``relaxed_correctness`` is imported so that the
# overridden ``chartqa_process_results`` below resolves it locally.
from lmms_eval.tasks.chartqa.utils import (  # noqa: F401
    chartqa_doc_to_text,
    relaxed_correctness,
)


def chartqa_doc_to_visual(doc):
    return [load_pil_image(doc["image"])]


def chartqa_process_results(doc, results):
    pred = results[0].strip()
    target = doc["answer"]
    if pred.endswith(".") and not target.strip().endswith("."):
        pred = pred[:-1]
    type = doc["type"]
    score = relaxed_correctness(pred, target)
    score = 1.0 if score else 0.0
    return_dict = {"relaxed_overall": score}
    if type == "human_test":
        return_dict["relaxed_human_split"] = score
    else:
        return_dict["relaxed_augmented_split"] = score
    return return_dict
