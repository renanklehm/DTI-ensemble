# DTI Ensemble

A trainable late-fusion ensemble for GSDTI and DCGAT-DTI. It combines:

- a ChemBERTa embedding of each SMILES;
- an ESM-2 embedding of each protein sequence;
- the predicted label and active-class probability from both base models; and
- derived agreement, mean-confidence, and probability-disagreement signals.

The Transformer encoders are frozen feature extractors. Every unique molecule and
protein is embedded once, written to a content-addressed cache, and reused by the
fusion network. The trainable network independently projects both embeddings,
adds absolute-difference and elementwise-product interactions, projects the base
model signals, and predicts a single interaction logit.

## Setup

Python 3.11+ and `uv` are expected.

```powershell
uv sync --group dev
```

## Data alignment

`data/1_ready/df.parquet` is a long table with one `source` row per base-model
prediction. Repeated observations are supported. Within each
`(smiles, sequence, real_value, source)` group, rows receive an occurrence number;
GSDTI and DCGAT-DTI are inner-joined on the pair, truth, and occurrence. Any
occurrence missing from either model is dropped. The current file produces
117,960 paired examples.

Validate pairing and create an inspectable split table without loading either
Transformer:

```powershell
uv run python main.py inspect-data --config configs/default.yaml `
  --output artifacts/inspection/paired_splits.parquet
```

Available split strategies are:

- `stratified_random`: class-stratified observation split (default);
- `drug_cold`: no SMILES overlap across train/validation/test; and
- `target_cold`: no protein-sequence overlap across train/validation/test.

Use a cold split when the intended deployment includes unseen drugs or targets.

## Train

Edit [configs/default.yaml](configs/default.yaml), then run:

```powershell
uv run python main.py train --config configs/default.yaml
```

On the current CPU-only environment, full Transformer embedding generation will
be slow. Copy the repository to a CUDA environment and use the default config or
merge the desired values from [configs/gpu_large.yaml](configs/gpu_large.yaml).
The latter is a higher-capacity profile using ESM-2 150M and is intended for a GPU
with substantial VRAM.

Each run is stored under `artifacts/ensemble/<timestamp>/` and includes:

- `resolved_config.yaml`: the exact effective hyperparameters;
- `paired_splits.parquet`: durable row-level split assignments;
- `training.log`: timestamped preparation/training/evaluation logs;
- `metrics.csv`: per-epoch losses, learning rate, discrimination, calibration,
  and thresholded metrics;
- `last.pt`: resumable model, optimizer, scheduler, scaler, and early-stop state;
- `best.pt`: best validation checkpoint;
- `test_metrics.json`: ensemble and both base-model test metrics; and
- `test_predictions.parquet`: test probabilities and labels.

Embedding caches live under `artifacts/embedding_cache/`. Cache identity includes
the input values, encoder, maximum length, pooling method, and storage dtype.

Resume an interrupted run without changing its split:

```powershell
uv run python main.py train --config configs/default.yaml `
  --resume artifacts/ensemble/<run>/last.pt
```

Training uses weighted binary cross-entropy, AdamW, warmup plus cosine decay,
gradient clipping, optional accumulation and CUDA mixed precision, validation
threshold optimization, and patience-based early stopping. AUPRC is the default
selection metric.

## Inference

Inference accepts the same long source format or a wide CSV/Parquet containing:

```text
smiles, sequence,
gsdti_label, gsdti_probability,
dcgat_label, dcgat_probability
```

Run:

```powershell
uv run python main.py predict `
  --checkpoint artifacts/ensemble/<run>/best.pt `
  --input data/to_score.parquet `
  --output predictions.parquet
```

The result adds `ensemble_probability` and `ensemble_label`. The classification
threshold is the one optimized on the validation set and stored in the checkpoint.

## Tests

```powershell
uv run pytest -q
```

Tests cover repeated-row inner pairing, all split modes, validation failures,
fusion forward/backward behavior, CPU training, checkpointing, and resume state.
