"""Portuguese (pt-PT) overrides for the TextVQA task.

This module re-exports the upstream TextVQA helpers unchanged and overrides only
the image loader (so it handles the byte/dict image encodings used by the
translated HF parquet datasets, via the shared ``load_pil_image`` helper) and the
result parser (so it also records the ground-truth answers). The upstream
``utils.py`` is left byte-identical to lmms-eval, so the English ``textvqa``
tasks are unaffected and the repo stays trivially syncable with upstream.

The pt YAML references everything through this single ``utils_pt`` module.
"""

import statistics

from lmms_eval.tasks._task_utils.pt_common import load_pil_image
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor

# Re-export the upstream helpers unchanged so the pt task is fully described by
# ``utils_pt`` + the pt YAML.
from lmms_eval.tasks.textvqa.utils import (  # noqa: F401
    textvqa_aggregate_submissions,
    textvqa_doc_to_text,
)


def textvqa_doc_to_visual(doc):
    return [load_pil_image(doc["image"])]


def textvqa_process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    resAns = eval_ai_processor(result[0])
    accuracy = 0

    if "answers" in doc and doc["answers"] is not None:
        gtAcc = []

        for i in range(len(doc["answers"])):
            doc["answers"][i] = eval_ai_processor(doc["answers"][i])

        for i in range(len(doc["answers"])):
            otherGTAns = [doc["answers"][j] for j in range(len(doc["answers"])) if i != j]
            matchingAns = [item for item in otherGTAns if item == resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        accuracy = statistics.mean(gtAcc)

    return {
        "exact_match": accuracy,
        "gt_answers": doc.get("answers", []),
        "submission": {
            "question_id": doc["question_id"],
            "answer": resAns,
        },
    }
