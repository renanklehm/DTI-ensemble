from __future__ import annotations

import gc
import hashlib
import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from .config import EncoderConfig
from .data import IndexedEmbeddings

LOGGER = logging.getLogger(__name__)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _fingerprint(values: Sequence[str], kind: str, config: EncoderConfig) -> str:
    digest = hashlib.sha256()
    settings = {
        "kind": kind,
        "model_name": config.model_name,
        "max_length": config.max_length,
        "pooling": config.pooling,
        "dtype": config.dtype,
        "values": len(values),
    }
    digest.update(json.dumps(settings, sort_keys=True).encode())
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _storage_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def _pool(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    method: str,
    special_tokens_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if method == "cls":
        return hidden[:, 0]
    valid_tokens = attention_mask
    if special_tokens_mask is not None:
        valid_tokens = valid_tokens * (1 - special_tokens_mask)
    mask = valid_tokens.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def load_or_compute_embeddings(
    raw_values: Sequence[str],
    kind: str,
    config: EncoderConfig,
    cache_root: str | Path,
    device: torch.device,
) -> IndexedEmbeddings:
    if kind not in {"smiles", "sequence"}:
        raise ValueError("kind must be smiles or sequence")
    normalize: Callable[[str], str]
    normalize = (lambda value: "".join(value.split()).upper()) if kind == "sequence" else str
    values = list(dict.fromkeys(normalize(value) for value in raw_values))
    if any(not value for value in values):
        raise ValueError(f"Empty {kind} values cannot be embedded")

    fingerprint = _fingerprint(values, kind, config)
    cache_dir = Path(cache_root) / f"{kind}_{fingerprint[:16]}"
    keys_path = cache_dir / "keys.parquet"
    tensor_path = cache_dir / "embeddings.pt"
    metadata_path = cache_dir / "metadata.json"
    if keys_path.exists() and tensor_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint:
            keys = pd.read_parquet(keys_path)[kind].astype(str).tolist()
            embeddings = torch.load(tensor_path, map_location="cpu", weights_only=True)
            if embeddings.ndim != 2 or len(keys) != embeddings.shape[0]:
                raise RuntimeError(f"Corrupt embedding cache: {cache_dir}")
            LOGGER.info("Reusing %s cache with %d vectors from %s", kind, len(keys), cache_dir)
            return IndexedEmbeddings(embeddings, {value: index for index, value in enumerate(keys)})

    LOGGER.info(
        "Encoding %d unique %s values with %s on %s",
        len(values),
        kind,
        config.model_name,
        device,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=config.trust_remote_code
    )
    model: nn.Module = AutoModel.from_pretrained(
        config.model_name, trust_remote_code=config.trust_remote_code
    ).to(device)
    model.eval()
    chunks: list[torch.Tensor] = []
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
    context = (lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)) if use_amp else nullcontext
    with torch.inference_mode():
        progress = tqdm(
            range(0, len(values), config.batch_size),
            desc=f"Embedding {kind}",
            unit="batch",
            dynamic_ncols=True,
        )
        for start in progress:
            batch = values[start : start + config.batch_size]
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            tokens = {name: tensor.to(device) for name, tensor in tokens.items()}
            special_tokens_mask = tokens.pop("special_tokens_mask", None)
            with context():
                output = model(**tokens)
                pooled = _pool(
                    output.last_hidden_state,
                    tokens["attention_mask"],
                    config.pooling,
                    special_tokens_mask,
                )
            chunks.append(pooled.to(device="cpu", dtype=_storage_dtype(config.dtype)))
            progress.set_postfix(items=min(start + config.batch_size, len(values)))
    embeddings = torch.cat(chunks, dim=0)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({kind: values}).to_parquet(keys_path, index=False)
    torch.save(embeddings, tensor_path)
    metadata_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "kind": kind,
                "count": len(values),
                "dimension": embeddings.shape[1],
                "model_name": config.model_name,
                "max_length": config.max_length,
                "pooling": config.pooling,
                "dtype": config.dtype,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    LOGGER.info("Saved %s embedding cache to %s", kind, cache_dir)
    return IndexedEmbeddings(embeddings, {value: index for index, value in enumerate(values)})
