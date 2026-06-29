"""Portuguese (pt-PT) overrides for the InfographicVQA task.

This module re-exports the upstream InfoVQA helpers unchanged and provides the
ANLS scoring used by the translated dataset. The upstream ``utils.py`` is left
byte-identical to lmms-eval, so the English ``infovqa`` tasks are unaffected and
the repo stays trivially syncable with upstream. The pt YAML references
everything through this single ``utils_pt`` module.
"""

from lmms_eval.api.metrics import anls

# Re-export the upstream doc helpers unchanged so the pt task is fully described
# by ``utils_pt`` + the pt YAML.
from lmms_eval.tasks.infovqa.utils import (  # noqa: F401
    infovqa_doc_to_text,
    infovqa_doc_to_visual,
)


def infovqa_process_results(doc, results):
    pred = results[0].strip()
    answers = doc["answers"]
    if isinstance(answers, str):
        answers = [answers]
    if pred.endswith(".") and not any(a.strip().endswith(".") for a in answers):
        pred = pred[:-1]
    return anls(references=answers, predictions=[pred])
