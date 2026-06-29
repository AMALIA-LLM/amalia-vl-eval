"""Portuguese (pt-PT) overrides for the OCRBench task.

This module re-exports the upstream OCRBench helpers unchanged and overrides
only ``ocrbench_doc_to_visual`` so that it decodes images through the shared
``pt_common.load_pil_image`` helper (the translated HF parquet may store images
as PIL, raw bytes, or ``{"bytes": ...}`` dicts).

The upstream ``utils.py`` is left byte-identical to lmms-eval, so the English
``ocrbench`` task is unaffected and the repo stays trivially syncable with
upstream. The pt YAML references everything through this single ``utils_pt``
module.
"""

from lmms_eval.tasks._task_utils.pt_common import load_pil_image

# Re-export the upstream helpers unchanged so the pt task is fully described by
# ``utils_pt`` + the pt YAML.
from lmms_eval.tasks.ocrbench.utils import (  # noqa: F401
    ocrbench_aggregate_accuracy,
    ocrbench_doc_to_text,
    ocrbench_process_results,
)


def ocrbench_doc_to_visual(doc):
    return [load_pil_image(doc["image"])]
