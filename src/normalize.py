import re
from collections import Counter

DIRECTION_WORDS = ["front-left", "front-right", "back-left", "back-right",
                   "left", "right", "front", "back", "above", "below", "behind"]


def normalize_answer(answer: str, category: str) -> str:
    """Normalize a model output for comparison against ground truth."""
    if answer is None:
        return ""
    a = answer.strip().lower().rstrip(".").rstrip(",")

    if category == "object_counting":
        nums = re.findall(r"\d+\.?\d*", a)
        if nums:
            try:
                v = float(nums[0])
                return str(int(v)) if v.is_integer() else str(v)
            except ValueError:
                return nums[0]
        return a

    if category == "relative_direction":
        for d in DIRECTION_WORDS:
            if d in a:
                return d
        return a

    if category == "relative_distance":
        # Often MC: "A", "B", "C", "D" or an object name
        m = re.match(r"^\(?([a-d])\)?[\.\):\s]", a)
        if m:
            return m.group(1)
        if len(a) == 1 and a in "abcd":
            return a
        return a

    return a


def is_correct(predicted: str, ground_truth: str, category: str,
               options: list | None = None) -> bool:
    """Compare normalized predicted vs. ground truth.

    For MC categories, accept either the letter or the option text.
    """
    p = normalize_answer(predicted, category)
    g = normalize_answer(str(ground_truth), category)

    if category == "object_counting":
        try:
            return abs(float(p) - float(g)) < 1e-6
        except (ValueError, TypeError):
            return p == g

    # MC handling
    if options:
        letters = ["a", "b", "c", "d", "e", "f"]
        # If GT looks like a letter, also accept matching option text
        if g in letters and len(options) > letters.index(g):
            gt_text = options[letters.index(g)].strip().lower()
            if p == g or p == gt_text or gt_text in p:
                return True
        # If GT is text, also accept the matching letter
        for i, opt in enumerate(options):
            opt_n = opt.strip().lower()
            if opt_n == g and (p == letters[i] or p == opt_n or opt_n in p):
                return True

    return p == g or (len(p) > 0 and len(g) > 0 and (p in g or g in p))


def consistency_score(answers: list[str], category: str) -> tuple[float, str]:
    """Fraction of answers matching the modal answer; returns (score, modal)."""
    if not answers:
        return 0.0, ""
    norm = [normalize_answer(a, category) for a in answers]
    counts = Counter(norm)
    modal, n_modal = counts.most_common(1)[0]
    return n_modal / len(norm), modal
