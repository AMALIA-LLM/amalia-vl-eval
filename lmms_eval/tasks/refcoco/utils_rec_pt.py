"""Portuguese (pt-PT) overrides for the RefCOCO bbox REC (grounding) task.

The translated HF dataset stores images as ``{"bytes": ...}`` dicts rather than
decoded PIL images, the grounding prompt is supplied in Portuguese via the task
YAML's ``lmms_eval_specific_kwargs``, and modern models emit bounding boxes in a
variety of formats (Qwen2.5-VL JSON ``bbox_2d``, 0-1000 integer scales, comma
decimal separators, ``(...)`` parentheses, space separation). This module
overrides the helpers that diverge from upstream and re-exports the aggregation
function unchanged. The upstream ``utils_rec.py`` is left byte-identical to
lmms-eval, so the English RefCOCO REC tasks are unaffected.

Note on the dependency rule: upstream ``refcoco_bbox_rec_process_result`` calls
``parse_float_sequence_within`` as a module-level reference. Because we override
``parse_float_sequence_within`` here, we must also redefine
``refcoco_bbox_rec_process_result`` locally so it resolves to OUR parser rather
than the upstream one.

The pt YAML references everything through this single ``utils_rec_pt`` module.
"""

import io
import json
import re

from datasets import Dataset
from PIL import Image

# Re-export the unchanged upstream constant and aggregation helper so the pt task
# is fully described by ``utils_rec_pt`` + the pt YAML.
from lmms_eval.tasks.refcoco.utils_rec import (  # noqa: F401
    COCO_REC_METRICS,
    refcoco_bbox_rec_acc01,
    refcoco_bbox_rec_acc03,
    refcoco_bbox_rec_acc05,
    refcoco_bbox_rec_acc07,
    refcoco_bbox_rec_acc09,
    refcoco_bbox_rec_aggregation_result,
    refcoco_bbox_rec_center_acc,
    refcoco_bbox_rec_iou,
)


def refcoco_bbox_rec_preprocess_dataset(dataset: Dataset):
    # PIL image stored in dataset['image']
    # add `image_width` and `image_height` to the dataset
    # check if images are pillow or bytes
    def decode_image_and_add_dims(example):
        image = Image.open(io.BytesIO(example["image"]["bytes"]))
        return {"image": image, "image_width": image.width, "image_height": image.height}

    if isinstance(dataset[0]["image"], dict) and "bytes" in dataset[0]["image"]:
        dataset = dataset.map(decode_image_and_add_dims)
    else:
        dataset = dataset.map(lambda x: {"image_width": x["image"].width, "image_height": x["image"].height})

    # Original bbox format (top x, top y, width, height)
    # Convert to (top-left x, top-left y, bottom-right x, bottom-right y)
    # Normalize the bounding box coordinates to be between 0 and 1
    # using the image width and height
    dataset = dataset.map(
        lambda x: {
            "bbox": [
                x["bbox"][0] / x["image_width"],
                x["bbox"][1] / x["image_height"],
                (x["bbox"][0] + x["bbox"][2]) / x["image_width"],
                (x["bbox"][1] + x["bbox"][3]) / x["image_height"],
            ]
        }
    )

    # currently, the dataset has `answer` as a list of strings
    # each answer should be its own row
    # we will explode the dataset to have each answer be its own row
    # we need to do this because the evaluation script expects
    # each answer to be its own row
    exploded_rows = []
    for row in dataset:
        for answer in row["answer"]:
            new_row = row.copy()
            new_row["answer"] = answer
            exploded_rows.append(new_row)

    # Create a new dataset from the exploded rows
    new_dataset = Dataset.from_list(exploded_rows)

    return new_dataset


def refcoco_bbox_rec_doc_to_visual(doc):
    # Image is presented as is
    if isinstance(doc["image"], dict) and "bytes" in doc["image"]:
        image = Image.open(io.BytesIO(doc["image"]["bytes"]))
        return [image.convert("RGB")]
    else:
        image = doc["image"].convert("RGB")
        return [image.convert("RGB")]


def refcoco_bbox_rec_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    assert isinstance(doc["answer"], str), "Answer must be a string"
    if lmms_eval_specific_kwargs and "question" in lmms_eval_specific_kwargs:
        return f"{lmms_eval_specific_kwargs['question']}" + doc["answer"]
    else:
        return f"Bounding box coordinates are specified in the format [top-left x, top-left y, bottom-right x, bottom-right y]. All values are floating point numbers bounded between 0 and 1. Please provide the bounding box coordinate of the region this sentence describes: {doc['answer']}"


def parse_float_sequence_within(input_str):
    """
    Extract the first sequence of four floating-point numbers within square brackets from a string.

    Handles:
    - Standard bracket notation: [x1, y1, x2, y2]
    - Qwen2.5-VL JSON format: {"bbox_2d": [x1, y1, x2, y2], "label": "..."}
    - Coordinates in 0-1000 scale (auto-normalized to 0-1)

    Args:
    input_str (str): A string that may contain a sequence of four floats within square brackets.

    Returns:
    list: A list of four floats if the pattern is found, or a list of four zeros if the pattern is not found.
    """
    # Handle Qwen2.5-VL JSON bbox_2d format (possibly wrapped in markdown fences and/or a list)
    cleaned = re.sub(r"```(?:json)?\s*", "", input_str).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and len(parsed) > 0:
            parsed = parsed[0]
        if isinstance(parsed, dict) and "bbox_2d" in parsed:
            coords = parsed["bbox_2d"]
            if isinstance(coords, list) and len(coords) == 4:
                result = [float(c) for c in coords]
                if any(v > 1.0 for v in result):
                    result = [v / 1000.0 for v in result]
                return result
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    input_str = input_str.replace("(", "[").replace(")", "]")

    # Comma-separated format: [x1, y1, x2, y2] (commas may also be decimal separators)
    pattern = r"\[\s*(-?\d+(?:[.,]\d+)?),\s*(-?\d+(?:[.,]\d+)?),\s*(-?\d+(?:[.,]\d+)?),\s*(-?\d+(?:[.,]\d+)?)\s*\]"
    match = re.search(pattern, input_str)
    if match:
        result = [float(match.group(i).replace(",", ".")) for i in range(1, 5)]
        if any(v > 1.0 for v in result):
            result = [v / 1000.0 for v in result]
        return result

    # Space-separated format: [x1 y1 x2 y2] (Molmo2 style, may mix 0-1 floats and 0-1000 ints)
    space_pattern = r"\[\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\]"
    match = re.search(space_pattern, input_str)
    if match:
        result = [float(match.group(i)) for i in range(1, 5)]
        result = [v / 1000.0 if v > 1.0 else v for v in result]
        return result

    return [0, 0, 0, 0]


def refcoco_bbox_rec_process_result(doc, result):
    """
    Redefined locally (rather than re-exported) so that it calls OUR overridden
    ``parse_float_sequence_within`` above, not the upstream one.

    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = result[0] if len(result) > 0 else ""
    pred = parse_float_sequence_within(pred)
    ann_id = doc["question_id"]
    data_dict = {"answer": doc["answer"], "pred": pred, "ann_id": ann_id, "bbox": doc["bbox"]}
    return {f"refcoco_{metric}": data_dict for metric in COCO_REC_METRICS}
