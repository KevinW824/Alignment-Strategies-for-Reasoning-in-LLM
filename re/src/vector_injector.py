import torch
from torch import nn
from transformers import PreTrainedModel
from typing import Optional, List


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
        self.remove()

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
                return output

            vec = self.control_vectors[layer_idx]

            vec_reshaped = vec.reshape(1, 1, -1)

            if isinstance(output, torch.Tensor):
                vec_reshaped = vec_reshaped.to(output.device).to(output.dtype)

                out = output + (self.alpha * vec_reshaped)
                return out

            elif isinstance(output, (tuple, list)):
                hidden = output[0]

                vec_reshaped = vec_reshaped.to(hidden.device).to(hidden.dtype)

                hid = hidden + (self.alpha * vec_reshaped)
                return type(output)((hid, *output[1:]))

            return output

        return hook

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
