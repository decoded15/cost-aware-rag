import re

OPPOSITE_GROUPS = [
    {"increase", "decrease", "reduce", "lower", "raise", "extend", "shorten"},
    {"enable", "disable", "activate", "deactivate"},
    {"start", "stop"},
    {"add", "delete", "remove"},
    {"larger", "smaller", "bigger"},
    {"improve", "degrade", "worsen"},
    {"before", "after"},
    {"allow", "block"},
    {"connect", "disconnect"},
    {"open", "close"},
]


def extract_words(text: str) -> set[str]:
    """Return lowercase words from a question."""
    return set(re.findall(r"\b[a-z]+\b", text.lower()))


def has_conflicting_intent(
    incoming_question: str,
    cached_question: str,
) -> bool:
    """
    Conservatively reject questions containing opposite
    intent terms when their cached and incoming meanings differ.
    """
    incoming_words = extract_words(incoming_question)
    cached_words = extract_words(cached_question)

    for group in OPPOSITE_GROUPS:
        incoming_actions = incoming_words & group
        cached_actions = cached_words & group

        if incoming_actions and cached_actions:
            if incoming_actions != cached_actions:
                return True

    return False
