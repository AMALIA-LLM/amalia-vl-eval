"""Portuguese (pt-PT) overrides for the COCO Caption 2017 task.

The translated HF dataset stores images as raw bytes / ``{"bytes": ...}`` dicts
rather than decoded PIL images, and the prompt is supplied in Portuguese via the
task YAML's ``lmms_eval_specific_kwargs``. This module overrides only the two
helpers that diverge from upstream (``coco_doc_to_visual`` and
``coco_doc_to_text``) and re-exports the result-processing and aggregation
functions unchanged. The upstream ``utils.py`` is left byte-identical to
lmms-eval, so the English ``coco2017_cap_val`` task is unaffected.

The pt YAML references everything through this single ``utils_pt`` module.
"""

from lmms_eval.tasks._task_utils.pt_common import load_pil_image

# Re-export the unchanged upstream result-processing and aggregation helpers so
# the pt task is fully described by ``utils_pt`` + the pt YAML.
from lmms_eval.tasks.coco_cap.utils import (  # noqa: F401
    coco_bleu1,
    coco_bleu2,
    coco_bleu3,
    coco_bleu4,
    coco_cider,
    coco_meteor,
    coco_process_result,
    coco_rougel,
)


def coco_doc_to_visual(doc):
    """Decode the (possibly byte-encoded) image into an RGB PIL image."""
    return [load_pil_image(doc["image"])]


def coco_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Return the Portuguese caption prompt supplied via the task YAML."""
    if lmms_eval_specific_kwargs and "question" in lmms_eval_specific_kwargs:
        return lmms_eval_specific_kwargs["question"]
    return "Provide a one-sentence caption for the provided image."
