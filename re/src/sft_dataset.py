import json
from torch.utils.data import Dataset
from typing import Optional, Dict


class SFTDataset(Dataset):
    """Dataset for SFT training with prompt-response pairs."""

    def __init__(self, data_path: str, max_examples: Optional[int] = None):
        """Load SFT dataset from JSONL file."""
        self.examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.examples.append(json.loads(line))
        if max_examples is not None:
            self.examples = self.examples[:max_examples]
        print(f"Loaded {len(self.examples)} SFT examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, str]:
        return self.examples[idx]


def format_prompt(question: str) -> str:
    return f"""[INST]
Solve this math problem step by step. Show your work clearly and end your solution with '#### X' where X is your final numerical answer with no units.

STRICT OUTPUT RULES (must follow):
- Use PLAIN TEXT ONLY (ASCII). Do NOT use LaTeX, math mode, dollar signs, \\boxed{{}}, \\( \\), \\[ \\], code fences, or any Markdown/TeX formatting.
- The LAST line must be EXACTLY: #### <number>
- Do not write any units, words, or punctuation on the last line — only the number after '#### '.

Question:
{question}
"""
