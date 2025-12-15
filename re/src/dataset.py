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
    return f"""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
"""
