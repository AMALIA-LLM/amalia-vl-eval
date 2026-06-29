"""Portuguese (pt-PT) overrides for the MathVision (test) task.

This module re-exports the upstream MathVision aggregation helper unchanged and
overrides only the functions that diverge for Portuguese:

* ``mathvision_doc_to_visual`` — decodes ``doc["decoded_image"]`` from any of the
  shapes the translated HF parquet stores images in (PIL | bytes | {"bytes":...}),
  via the centralised ``pt_common.load_pil_image`` helper.
* ``mathvision_doc_to_text`` — honours a ``query_prompt`` kwarg and uses the
  Portuguese "Opções:" label when a ``query_prompt`` is supplied.
* ``mathvision_process_results`` — recognises Portuguese answer-flag indicators
  ("Resposta:", "A resposta é", ...) in addition to the English ones.

The upstream ``utils.py`` is left byte-identical to lmms-eval, so the English
``mathvision_test`` task is unaffected and the repo stays trivially syncable with
upstream. The pt YAML references everything through this single ``utils_pt``
module.

NOTE: importing the upstream ``utils`` module triggers its module-level
``llm_judge`` server instantiation (upstream behaviour). The pt task uses the
rule-based ``mathvision_standard_eval`` path, not the judge.
"""

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.pt_common import load_pil_image

# Re-export the unchanged upstream aggregation helper so the pt task is fully
# described by ``utils_pt`` + the pt YAML.
from lmms_eval.tasks.mathvision.utils import (  # noqa: F401
    mathvision_aggregate_results_eval,
)

# Module-level helpers that ``mathvision_process_results`` depends on. These are
# imported in the upstream ``utils`` module from ``eval_utils``; re-import them
# here so the override below resolves them locally.
try:
    from lmms_eval.tasks.mathvision.eval_utils import (
        find_math_answer,
        is_equal,
        is_number,
    )
except ImportError as e:
    eval_logger.warning(f"Error importing eval_utils from lmms_eval.tasks.mathvision.eval_utils: {e}")


def mathvision_doc_to_visual(doc):
    return [load_pil_image(doc["decoded_image"])]


def mathvision_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question, choices = doc["question"], doc["options"]
    len_choices = len(choices)
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])

    mc_prompt = ""
    if lmms_eval_specific_kwargs is not None:
        mc_prompt = "\n" + lmms_eval_specific_kwargs["mc_prompt"]

    if "query_prompt" in lmms_eval_specific_kwargs:
        query_prompt = lmms_eval_specific_kwargs["query_prompt"]
    else:
        query_prompt = 'Please solve the problem step by step and put your answer in one "\\boxed{}".'
    if choices_str:
        if "query_prompt" in lmms_eval_specific_kwargs:  # the only run with query_prompt is mathvision_pt, which is in Portuguese, so we need to use the Portuguese
            query_prompt += f"{question}\nOpções: {choices_str}" + mc_prompt
        else:
            query_prompt += f"{question}\nChoices: {choices_str}" + mc_prompt
    else:
        query_prompt += question
    return query_prompt


def mathvision_process_results(doc, results):
    correct_list = []
    for pred in results:
        model_answer = pred.strip()

        gt_answer = str(doc["answer"])
        if len(doc["options"]) > 0:
            gt_answer_value = doc["options"][ord(gt_answer) - ord("A")]
        else:
            gt_answer_value = ""

        for c in "ABCDE":
            if model_answer.endswith(f" {c}.") or model_answer.endswith(f" ({c}).") or model_answer.startswith(f"{c}\n") or model_answer.startswith(f"({c})\n") or model_answer.startswith(f"({c}) {c}\n"):
                model_answer = c
        if is_number(model_answer.split("is ")[-1].rstrip(".")):
            model_answer = model_answer.split("is ")[-1].rstrip(".")
        if "oxed{" not in model_answer:
            for flag in ["the final answer is", "the answer is", "the correct answer is", "the answer should be", "Resposta:", "A resposta é", "A resposta correta é", "A resposta deve ser"]:
                raw_model_answer = model_answer
                model_answer = model_answer.split(flag)[-1].strip()
                if flag in raw_model_answer:
                    model_answer = model_answer.split("\n")[0].split(". ")[0]
                flag = flag.replace("the", "The").replace("a ", "A ")
                raw_model_answer = model_answer
                model_answer = model_answer.split(flag)[-1].strip()
                if flag in raw_model_answer:
                    model_answer = model_answer.split("\n")[0].split(". ")[0]
        elif model_answer.count("oxed{") > 1:
            model_answer = "\\boxed{" + model_answer.split("oxed{")[-1]

        model_answer = (
            find_math_answer(model_answer)
            .replace("(a)", "a")
            .replace("(b)", "b")
            .replace("(c)", "c")
            .replace("(d)", "d")
            .replace("(e)", "e")
            .replace("{a}", "a")
            .replace("{b}", "b")
            .replace("{c}", "c")
            .replace("{d}", "d")
            .replace("{e}", "e")
            .rstrip(".")
            .lstrip(":")
            .strip()
        )
        correct = is_equal(gt_answer, model_answer) or is_equal(gt_answer_value, model_answer)
        correct_list.append(correct)
    return {
        "mathvision_standard_eval": {
            # "question": doc["question"],
            # "answer": doc["answer"],
            "response": results,
            # "subject": doc["subject"],
            # "level": doc["level"],
            "scores": correct_list,
        },
    }
