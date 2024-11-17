import torch
from torch import nn


class LoRAUpdate(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, device=None):
        super().__init__()
        self.A = nn.Parameter(data=torch.empty((in_features, rank), device=device))
        with torch.no_grad():
            nn.init.xavier_normal_(self.A)
        self.B = nn.Parameter(data=torch.zeros((rank, out_features), device=device))
        self.rank = rank
        self.alpha = alpha

    def forward(self, input):
        return (input @ self.A @ self.B) * self.alpha / self.rank


class LoRALayer(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int, alpha: int, device=None):
        super().__init__()
        self.original_layer = original_layer
        self.lora_update = LoRAUpdate(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            rank=rank,
            alpha=alpha,
            device=device,
        )

    def forward(self, input):
        return self.original_layer(input) + self.lora_update(input)

def wrap_bert_model_with_lora(model: nn.Module, bert_module_name: str, rank=1, alpha=1):
    for param in model.parameters():
        param.requires_grad = False
    
    for layer in model.__getattr__(bert_module_name).encoder.layer:
        layer.attention.self.query = LoRALayer(
            layer.attention.self.query,
            rank=rank,
            alpha=alpha,
        )
        layer.attention.self.key = LoRALayer(
            layer.attention.self.key,
            rank=rank,
            alpha=alpha,
        )
        layer.attention.self.value = LoRALayer(
            layer.attention.self.value,
            rank=rank,
            alpha=alpha,
        )
        layer.attention.output.dense = LoRALayer(
            layer.attention.output.dense,
            rank=rank,
            alpha=alpha,
        )

    return model
