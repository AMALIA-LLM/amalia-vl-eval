"""Shared helpers for the AMALIA-VL-Eval Portuguese (pt-PT) task variants.

These live here (rather than in core ``lmms_eval/utils.py``) so that the core
library and every upstream task module stay byte-identical to lmms-eval and the
repo remains trivially syncable with upstream. The pt-only ``utils_pt.py``
modules import from here.
"""

import io

from PIL import Image as PILImage


def translate_answer(answer):
    """Map Portuguese yes/no/other answers onto their English canonical form.

    Leaves any unrecognised answer untouched, so it is safe to call on
    already-English predictions.
    """
    answer = answer.lower().strip().replace(".", "")
    answer_map = {
        "sim": "yes",
        "não": "no",
        "nao": "no",
        "outro": "other",
        "yes": "yes",
        "no": "no",
        "other": "other",
    }
    return answer_map.get(answer, answer)


def load_pil_image(img_data):
    """Return an RGB :class:`PIL.Image` from any of the shapes the translated
    HF parquet datasets store images in (PIL image, raw bytes, or a
    ``{"bytes": ...}`` dict).
    """
    try:
        if isinstance(img_data, PILImage.Image):
            return img_data.convert("RGB")
        if isinstance(img_data, (bytes, bytearray)):
            return PILImage.open(io.BytesIO(img_data)).convert("RGB")
        if isinstance(img_data, dict) and "bytes" in img_data:
            return PILImage.open(io.BytesIO(img_data["bytes"])).convert("RGB")
    except OSError as e:
        raise ValueError(f"Invalid image data: {e}")
    raise TypeError(f"Unsupported image type: {type(img_data)}")
