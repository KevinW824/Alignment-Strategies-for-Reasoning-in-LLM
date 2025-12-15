import torch
import os
import json
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel  # type: ignore
from torch.utils.data import DataLoader, Dataset

from dataclasses import dataclass
from tqdm import tqdm

from sklearn.decomposition import PCA


@dataclass
class REConfig:

    # Model and data
    model_name: str = "Qwen/Qwen3-1.7B"
    # model_name: str = "Qwen/Qwen2.5-Math-1.5B"

    data_path: str = "./data/qwen3_17b_prompts_with_format.json"
    # data_path: str = "./data/qwen25_math15b_prompts_with_format.json"
    val_data_path: str = "../data/gsm8k/test.jsonl"

    output_dir: str = "../outputs/re"
    output_file: str = "contrastive_pca_vectors_qwen3_with_format_1.7B.pth"

    num_examples: Optional[int] = None  # None = use all
    batch_size: int = 4


class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def collate_fn(batch, tokenizer):
    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )
    return inputs


def get_hidden_states(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    config: REConfig,
    device: str,
    desc: str,
) -> List[torch.Tensor]:
    """Extract the final-token hidden states for every layer across the dataset."""
    dataset = PromptDataset(prompts)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=4,
        persistent_workers=True,
    )

    per_layer_hiddens = None
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=desc):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            hiddens = torch.stack([h[:, -1, :] for h in out.hidden_states])

            if per_layer_hiddens is None:
                per_layer_hiddens = [[] for _ in range(len(hiddens))]

            for l, h in enumerate(hiddens):
                per_layer_hiddens[l].append(h.cpu())  # move to CPU to save VRAM

    per_layer_hiddens = [torch.cat(layer_list, dim=0) for layer_list in per_layer_hiddens]  # type: ignore
    return per_layer_hiddens


def run_re_pipeline(config: REConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, output_hidden_states=True, trust_remote_code=True
    )
    model = model.to(device)  # type: ignore
    model.eval()

    # Load the new dataset
    with open(config.data_path, "r") as f:
        data = json.load(f)

    positive_prompts = data["positive_prompts"]
    negative_prompts = data["negative_prompts"]

    if config.num_examples is not None:
        positive_prompts = positive_prompts[: config.num_examples]
        negative_prompts = negative_prompts[: config.num_examples]

    pos_hiddens = get_hidden_states(
        model,
        tokenizer,
        positive_prompts,
        config,
        device,
        "Extracting positive states (H+)",
    )
    neg_hiddens = get_hidden_states(
        model,
        tokenizer,
        negative_prompts,
        config,
        device,
        "Extracting negative states (H-)",
    )

    n_pos = pos_hiddens[0].shape[0]
    n_neg = neg_hiddens[0].shape[0]
    n_samples = min(n_pos, n_neg)

    if n_pos != n_neg:
        pos_hiddens = [h[:n_samples] for h in pos_hiddens]
        neg_hiddens = [h[:n_samples] for h in neg_hiddens]

    control_vectors: List[torch.Tensor] = []

    # pos_hiddens and neg_hiddens are both List[Tensor] grouped per layer
    for _, (pos_acts, neg_acts) in enumerate(
        tqdm(
            zip(pos_hiddens, neg_hiddens), desc="PCA per layer", total=len(pos_hiddens)
        )
    ):

        diff_acts = (pos_acts - neg_acts).to(device)

        mean_diff_vec = diff_acts.mean(0)
        mean_norm = mean_diff_vec.norm().to(device)

        acts = diff_acts - diff_acts.mean(0, keepdim=True)

        pca = PCA(n_components=1)
        pca.fit(acts.cpu().numpy())  # PCA runs on CPU
        c = torch.tensor(pca.components_[0], dtype=torch.float32).to(device)

        c = c / c.norm() * mean_norm

        control_vectors.append(c)

    control_vectors_tensor = torch.stack(control_vectors)
    os.makedirs(config.output_dir, exist_ok=True)
    torch.save(control_vectors_tensor, f"{config.output_dir}/{config.output_file}")

    print(f"\nSaved PCA control vectors {config.output_dir}/{config.output_file}")
    print(f"Shape: {control_vectors_tensor.shape}")


if __name__ == "__main__":
    run_re_pipeline(config=REConfig())
