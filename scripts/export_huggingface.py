"""Export Nexara models to a Hugging Face compatible repository format."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Nexara model to Hugging Face format.")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to input training checkpoint .pt file."
    )
    parser.add_argument(
        "--tokenizer", default="tokenizer/nexara-bpe.json", help="Path to BPE tokenizer json."
    )
    parser.add_argument(
        "--output-dir", default="exports/huggingface/Nexara-0.1", help="Output directory path."
    )
    return parser.parse_args()


# Source code forconfiguration_nexara.py
CONFIG_PY_CONTENT = """from transformers import PretrainedConfig

class NexaraConfig(PretrainedConfig):
    model_type = "nexara"
    
    def __init__(
        self,
        vocab_size=8192,
        max_sequence_length=256,
        n_layers=6,
        n_heads=8,
        embedding_dim=256,
        dropout=0.1,
        mlp_ratio=4.0,
        bias=False,
        rope_base=10000.0,
        tie_embeddings=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio
        self.bias = bias
        self.rope_base = rope_base
        self.tie_embeddings = tie_embeddings

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.n_heads
"""

# Source code for modeling_nexara.py
MODELING_PY_CONTENT = """import math
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_nexara import NexaraConfig


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first_half, second_half = x.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_sequence_length: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_sequence_length, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        embedding = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", embedding.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", embedding.sin()[None, None, :, :], persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = query.size(-2)
        cos = self.cos_cached[:, :, :sequence_length, :].to(dtype=query.dtype)
        sin = self.sin_cached[:, :, :sequence_length, :].to(dtype=query.dtype)
        return (
            (query * cos) + (_rotate_half(query) * sin),
            (key * cos) + (_rotate_half(key) * sin),
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: NexaraConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dropout_probability = config.dropout

        self.qkv_projection = nn.Linear(
            config.embedding_dim,
            3 * config.embedding_dim,
            bias=config.bias,
        )
        self.output_projection = nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
            bias=config.bias,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.rotary_embedding = RotaryEmbedding(
            head_dim=config.head_dim,
            max_sequence_length=config.max_sequence_length,
            base=config.rope_base,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = x.shape

        qkv = self.qkv_projection(x)
        qkv = qkv.view(
            batch_size,
            sequence_length,
            3,
            self.n_heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        query, key = self.rotary_embedding(query, key)

        if hasattr(F, "scaled_dot_product_attention"):
            attention = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.dropout_probability if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = (query @ key.transpose(-2, -1)) * scale
            mask = torch.triu(
                torch.ones(sequence_length, sequence_length, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))
            probabilities = F.softmax(scores, dim=-1)
            probabilities = self.attention_dropout(probabilities)
            attention = probabilities @ value

        attention = attention.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            embedding_dim,
        )
        return self.residual_dropout(self.output_projection(attention))


class FeedForward(nn.Module):
    def __init__(self, config: NexaraConfig) -> None:
        super().__init__()
        hidden_dim = int(config.embedding_dim * config.mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(config.embedding_dim, hidden_dim, bias=config.bias),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, config.embedding_dim, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: NexaraConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class NexaraForCausalLM(PreTrainedModel):
    config_class = NexaraConfig
    base_model_prefix = "transformer"
    
    def __init__(self, config: NexaraConfig) -> None:
        super().__init__(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
            
        self.post_init()
        
    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_embedding
        
    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.token_embedding = value
        
    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head
        
    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings
        
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        sequence_length = input_ids.size(1)
        x = self.dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
"""


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    out_dir = Path(args.output_dir)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_state = checkpoint["model_state_dict"]
    config = checkpoint.get("config", {})

    # Extract model and generation parameters
    model_config = config.get("model", {})
    gen_config = config.get("generation", {})

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export clean model state dict package as pytorch_model.bin
    weights_path = out_dir / "pytorch_model.bin"
    # Ensure CPU tensors
    model_state_cpu = {k: v.cpu() for k, v in model_state.items()}
    torch.save(model_state_cpu, weights_path)
    print(f"Exported weights to {weights_path}")

    # 2. Export config.json
    hf_config = {
        "architectures": ["NexaraForCausalLM"],
        "model_type": "nexara",
        "auto_map": {
            "AutoConfig": "configuration_nexara.NexaraConfig",
            "AutoModelForCausalLM": "modeling_nexara.NexaraForCausalLM",
        },
        "vocab_size": model_config.get("vocab_size", 8192),
        "max_sequence_length": model_config.get("max_sequence_length", 256),
        "n_layers": model_config.get("n_layers", 6),
        "n_heads": model_config.get("n_heads", 8),
        "embedding_dim": model_config.get("embedding_dim", 256),
        "dropout": model_config.get("dropout", 0.1),
        "mlp_ratio": model_config.get("mlp_ratio", 4.0),
        "bias": model_config.get("bias", False),
        "rope_base": model_config.get("rope_base", 10000.0),
        "tie_embeddings": model_config.get("tie_embeddings", True),
    }
    config_json_path = out_dir / "config.json"
    with config_json_path.open("w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)
    print(f"Exported config to {config_json_path}")

    # 3. Export generation_config.json
    hf_gen_config = {
        "temperature": gen_config.get("temperature", 0.8),
        "top_k": gen_config.get("top_k", 40),
        "top_p": gen_config.get("top_p", 0.95),
        "repetition_penalty": gen_config.get("repetition_penalty", 1.1),
        "max_new_tokens": gen_config.get("max_new_tokens", 128),
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    gen_json_path = out_dir / "generation_config.json"
    with gen_json_path.open("w", encoding="utf-8") as f:
        json.dump(hf_gen_config, f, indent=2)
    print(f"Exported generation config to {gen_json_path}")

    # 4. Copy Tokenizer to tokenizer.json
    shutil.copy(tokenizer_path, out_dir / "tokenizer.json")
    print(f"Copied tokenizer to {out_dir / 'tokenizer.json'}")

    # 5. Create tokenizer_config.json
    tokenizer_conf = {
        "add_bos_token": True,
        "add_eos_token": False,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>",
        "model_max_length": model_config.get("max_sequence_length", 256),
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    tok_json_path = out_dir / "tokenizer_config.json"
    with tok_json_path.open("w", encoding="utf-8") as f:
        json.dump(tokenizer_conf, f, indent=2)
    print(f"Exported tokenizer config to {tok_json_path}")

    # 6. Write custom Python configuration and modeling files
    (out_dir / "configuration_nexara.py").write_text(CONFIG_PY_CONTENT, encoding="utf-8")
    (out_dir / "modeling_nexara.py").write_text(MODELING_PY_CONTENT, encoding="utf-8")
    print("Created configuration_nexara.py and modeling_nexara.py custom scripts.")

    # 7. Write README.md metadata card
    readme_content = f"""---
license: mit
datasets:
- roneneldan/TinyStories
language:
- en
tags:
- text-generation
- custom-architecture
---

# Nexara-Nano-100M-Chat

Nexara-Nano-100M-Chat is a tiny Decoder-Only Transformer pretrained and supervised fine-tuned (SFT) from scratch. It is designed solely for experimental and educational purposes.

> [!WARNING]
> This model has only ~{sum(p.numel() for p in model_state.values() if p.ndim > 0):,} parameters and is trained on a small corpus. It is **not** guaranteed to perform well in production environments. Expect frequent hallucinations and limited factual knowledge.

## Model Architecture
* **Type**: Causal Decoder-Only Transformer (with Rotary Position Embeddings (RoPE), Pre-LN, Tied Embeddings, and GELU activation).
* **Parameters**: ~{sum(p.numel() for p in model_state.values() if p.ndim > 0):,}
* **Layers**: {model_config.get("n_layers", 6)}
* **Attention Heads**: {model_config.get("n_heads", 8)}
* **Embedding Dimension**: {model_config.get("embedding_dim", 256)}
* **Context Length**: {model_config.get("max_sequence_length", 256)} tokens
* **License**: MIT License

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your_username/Nexara-Nano-100M-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("your_username/Nexara-Nano-100M-Chat", trust_remote_code=True)

# Format SFT prompt template
prompt = "### System:\\nYou are Nexara, a helpful and polite AI assistant.\\n\\n### User:\\nWhat is your name?\\n\\n### Assistant:\\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""
    (out_dir / "README.md").write_text(readme_content, encoding="utf-8")
    print(f"Exported Hugging Face repository package successfully to {out_dir}")


if __name__ == "__main__":
    main()
