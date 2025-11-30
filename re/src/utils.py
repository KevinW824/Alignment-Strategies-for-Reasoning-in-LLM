import re
from math_grader import question_only_reward_fn


def extract_final(text: str):
    """Extract final answer from gsm8k answer"""
    m = re.search(r"####\s*([\d\.]+)", text.strip().splitlines()[-1])
    return m.group(1) if m else None


def extract_ground_truth_answer(answer_str: str) -> str:
    """Extract numerical answer, handling commas."""
    ANS_RE = re.compile(r"####\s*([\-0-9\.,]+)")
    match = ANS_RE.search(answer_str)
    if match:
        return match.group(1).strip().replace(",", "")
    # Fallback to original split logic
    if "####" in answer_str:
        return answer_str.split("####")[1].strip().replace(",", "")
    return answer_str.strip()


def answer_only_reward_fn(generated_text: str, gt: str) -> dict[str, float]:
    """
    Extracts the last numerical answer from a generated text.
    """
    matches = re.findall(r"(-?[\d,]*\d+(?:\.\d+)?)", generated_text)

    for candidate in reversed(matches):
        cleaned = candidate.replace(",", "")
        # Remove trailing dot if captured incorrectly (though regex attempts to handle decimals)
        if cleaned.endswith("."):
            cleaned = cleaned[:-1]

        if not cleaned:
            continue

        try:
            float(cleaned)
            is_correct = cleaned == gt
            return (
                {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}
                if is_correct
                else {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}
            )
        except ValueError:
            continue

    return question_only_reward_fn(generated_text, gt, True)
