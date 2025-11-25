import re


def extract_final(text: str):
    """Extract final answer from gsm8k answer"""
    m = re.search(r"####\s*([\d\.]+)", text.strip().splitlines()[-1])
    return m.group(1) if m else None
