"""Portuguese (pt-PT) overrides for the RefCOCO bbox (captioning) task.

The translated HF dataset stores images as raw bytes / ``{"bytes": ...}`` dicts
rather than decoded PIL images, and the region prompt is supplied in Portuguese
via the task YAML's ``lmms_eval_specific_kwargs``. This module overrides only the
two helpers that diverge from upstream (``refcoco_bbox_doc_to_visual`` and
``refcoco_doc_to_text``) and re-exports the result-processing and aggregation
functions unchanged. The upstream ``utils.py`` is left byte-identical to
lmms-eval, so the English RefCOCO bbox tasks are unaffected.

The pt YAML references everything through this single ``utils_pt`` module.
"""

from PIL import ImageDraw

from lmms_eval.tasks._task_utils.pt_common import load_pil_image

# Re-export the unchanged upstream result-processing and aggregation helpers so
# the pt task is fully described by ``utils_pt`` + the pt YAML.
from lmms_eval.tasks.refcoco.utils import (  # noqa: F401
    refcoco_bleu1,
    refcoco_bleu2,
    refcoco_bleu3,
    refcoco_bleu4,
    refcoco_cider,
    refcoco_meteor,
    refcoco_process_result,
    refcoco_rougel,
)


def refcoco_bbox_doc_to_visual(doc):
    """Decode the (possibly byte-encoded) image and draw the target bbox on it."""
    bbox = doc["bbox"]
    image = load_pil_image(doc["image"])
    draw = ImageDraw.Draw(image)
    # Origin format (top x, top y, width, height)
    bbox_xy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    draw.rectangle(bbox_xy, outline="red")
    return [image.convert("RGB")]


def refcoco_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Return the Portuguese region-description prompt supplied via the YAML."""
    if lmms_eval_specific_kwargs and "question" in lmms_eval_specific_kwargs:
        return lmms_eval_specific_kwargs["question"]
    return "Provide a short description for this region."
