"""Portuguese (pt-PT) overrides for the AI2D task.

This module re-exports the upstream AI2D helpers unchanged and overrides only
the answer-extraction filter so that it also recognises Portuguese answer
phrasings (e.g. "a resposta correta é: B"). The upstream ``utils.py`` is left
byte-identical to lmms-eval, so the English ``ai2d`` task is unaffected and the
repo stays trivially syncable with upstream.

The pt YAML references everything through this single ``utils_pt`` module.
"""

import re

from lmms_eval.filters.extraction import ExtendedRegexFilter

# Re-export the upstream doc helpers unchanged so the pt task is fully described
# by ``utils_pt`` + the pt YAML.
from lmms_eval.tasks.ai2d.utils import (  # noqa: F401
    ai2d_doc_to_target,
    ai2d_doc_to_text,
    ai2d_doc_to_visual,
)


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    """Extract the chosen option letter, tolerant of Portuguese phrasings.

    Extends the upstream behaviour (which only matched ``^\\s*([A-Z])\\.``) to
    also catch answers worded in Portuguese/English such as
    "a resposta é: B" or a bare standalone letter on its own line.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        filtered_resps = []

        for r, doc in zip(resps, docs):
            option_letter_regex = re.compile(
                r"(?:" r"^\s*([A-Z])\." r"|(?:resposta|answer|opção|opcao|letra)\s*(?:é|e|correta é|correta e)?\s*(?:a\s+)?:?\s*([A-Z])\b" r"|^\s*([A-Z])\s*$" r")",
                re.IGNORECASE,
            )

            filtered = []
            for resp in r:
                match = option_letter_regex.search(resp)
                if match:
                    letter = next(g for g in match.groups() if g is not None)
                    filtered.append(letter.upper())
                else:
                    filtered.append(resp)

            filtered_resps.append(filtered[0])

        return filtered_resps
