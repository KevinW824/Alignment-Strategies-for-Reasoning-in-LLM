import torch
import os
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel  # type: ignore
from torch.utils.data import DataLoader, Dataset
from torch import nn

from dataclasses import dataclass
from tqdm import tqdm

from sklearn.decomposition import PCA

from sft_dataset import SFTDataset, format_prompt


@dataclass
class REConfig:

    # Model and data
    # model_name: str = "Qwen/Qwen3-1.7B"
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"

    # Paths for positive and negative activations used in the contrastive setup
    pos_data_path: str = "../data/gsm8k/sft.jsonl"
    neg_data_path: str = "../data/gsm8k/negative_set.jsonl"

    val_data_path: str = "../data/gsm8k/test.jsonl"

    output_dir: str = "../outputs/re"
    output_file: str = "contrastive_pca_vectors_qwen2.5_math_1.5B.pth"

    num_examples: Optional[int] = None  # None = use all
    batch_size: int = 4


def collate_fn(batch, tokenizer):
    prompts = [format_prompt(item["prompt"]) for item in batch]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )
    return inputs


def get_hidden_states(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    data_path: str,
    config: REConfig,
    device: str,
    desc: str,
) -> List[torch.Tensor]:
    """Extract the final-token hidden states for every layer across the dataset."""
    dataset = SFTDataset(data_path=data_path, max_examples=config.num_examples)
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

    # Step 1: extract positive and negative activations
    pos_hiddens = get_hidden_states(
        model,
        tokenizer,
        config.pos_data_path,
        config,
        device,
        "Extracting positive states (H+)",
    )
    neg_hiddens = get_hidden_states(
        model,
        tokenizer,
        config.neg_data_path,
        config,
        device,
        "Extracting negative states (H-)",
    )

    # Step 2: align sample counts so each contrastive pair has the same size
    n_pos = pos_hiddens[0].shape[0]
    n_neg = neg_hiddens[0].shape[0]
    n_samples = min(n_pos, n_neg)

    if n_pos != n_neg:
        print(f"Warning: Mismatched sample sizes. (Pos: {n_pos}, Neg: {n_neg})")
        print(f"Using {n_samples} samples from each.")
        # Truncate to the smallest available count
        pos_hiddens = [h[:n_samples] for h in pos_hiddens]
        neg_hiddens = [h[:n_samples] for h in neg_hiddens]

    # Step 3: compute contrastive vectors (H+ - H-) and run PCA
    print("Calculating contrastive vectors (H+ - H-) and performing PCA...")
    control_vectors: List[torch.Tensor] = []

    # pos_hiddens and neg_hiddens are both List[Tensor] grouped per layer
    for layer_idx, (pos_acts, neg_acts) in enumerate(
        tqdm(
            zip(pos_hiddens, neg_hiddens), desc="PCA per layer", total=len(pos_hiddens)
        )
    ):

        # 1. Compute the scaling norm from the positive activations before centering
        mean_norm = pos_acts.norm(dim=1).mean().to(device)

        # 2. Compute the contrastive difference H(P+) - H(P-)
        diff_acts = (pos_acts - neg_acts).to(device)

        # 3. Center the differences before PCA
        acts = diff_acts - diff_acts.mean(0, keepdim=True)

        # 4. Run PCA
        pca = PCA(n_components=1)
        pca.fit(acts.cpu().numpy())  # PCA runs on CPU
        c = torch.tensor(pca.components_[0], dtype=torch.float32).to(device)

        # 5. Scale the PCA vector to match the positive activation norm
        c = c / c.norm() * mean_norm

        control_vectors.append(c)

    control_vectors_tensor = torch.stack(control_vectors)
    os.makedirs(config.output_dir, exist_ok=True)
    torch.save(control_vectors_tensor, f"{config.output_dir}/{config.output_file}")

    print(
        f"\nSaved CONTRASTIVE PCA control vectors → {config.output_dir}/{config.output_file}"
    )
    print(f"Shape: {control_vectors_tensor.shape}")


class ControlVectorInjector:
    def __init__(
        self,
        model: PreTrainedModel,
        control_vectors: torch.Tensor,
        alpha: float = 0.0,
        layers: Optional[List[int]] = None,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.device = device
        self.model = model
        self.alpha = alpha
        self.control_vectors = control_vectors.to(device)
        self.layers = (
            layers if layers is not None else list(range(0, control_vectors.shape[0]))
        )
        self.hooks = []
        self.register_hooks()

    def register_hooks(self):
        layers_module: nn.ModuleList
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers_module = self.model.model.layers  # type: ignore[attr-defined]
        elif hasattr(self.model, "layers"):
            layers_module = getattr(self.model, "layers")
        else:
            raise Exception("Layer cannot be located")

        for l in self.layers:
            if l < len(layers_module):
                handle = layers_module[l].register_forward_hook(self._forward_hook(l))  # type: ignore
                self.hooks.append(handle)

    def _forward_hook(self, layer_idx: int):
        def hook(module: nn.Module, inputs, output):
            if self.alpha == 0.0:
                return None
            if isinstance(output, torch.Tensor):
                vec = self.control_vectors[layer_idx].to(output.device).to(output.dtype)
                out = output.clone()
                out[:, -1, :] += self.alpha * vec
                return out
            elif isinstance(output, (tuple, list)):
                hidden = output[0]
                vec = self.control_vectors[layer_idx].to(hidden.device).to(hidden.dtype)
                hid = hidden.clone()
                hid[:, -1, :] += self.alpha * vec
                return type(output)((hid, *output[1:]))
            return None

        return hook

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


if __name__ == "__main__":
    run_re_pipeline(config=REConfig())
