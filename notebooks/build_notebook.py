"""Generates colab_train_and_full_inference.ipynb."""
import json
from pathlib import Path

def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}

def code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": source.splitlines(keepends=True)}

cells = []

cells.append(md("""# DTI Ensemble — train on authored subset + full-universe inference

Google Colab notebook (GPU runtime: **Runtime → Change runtime type → T4/L4 GPU**).

Pipeline:
1. **Setup** — mount Drive, install deps, load the `ensemble` package.
2. **Training data** — take your authored ground-truth subset (`smiles`, `sequence`, `real_value`),
   join the DCGAT-DTI and GSDTI base-model predictions for those pairs, and split 70/20/10.
3. **Train** — ChemBERTa + ESM-2 embeddings (cached) and the fusion network; reports test metrics
   against both base models.
4. **Full inference** — run the trained ensemble on **all 442,284,138 paired pairs**
   (112,914 substances × 3,917 targets present in both models), in chunks, writing parquet shards.

Artifacts (checkpoints, splits, prediction shards) are written under `OUTPUT_DIR` — point it at
Drive so nothing is lost when the runtime recycles.

> **Note on 24/364 unmapped ids:** DCGAT's grid covers 112,938 × 4,281 ids, but 24 substances and
> 364 targets have no mapping (they were excluded upstream). The 442M universe below is the
> intersection both models share, which is exactly the number quoted above.
"""))

cells.append(code("""# ============ 1. CONFIGURATION — edit this cell ============
from pathlib import Path

REPO_DIR = Path("/content/Ensemble")     # folder containing the `ensemble/` package
REPO_URL = ""                            # e.g. "https://github.com/you/Ensemble.git" — used only if REPO_DIR doesn't exist
MOUNT_DRIVE = True

# Raw prediction batches (folder of part-*.parquet each)
RAW_DIR = Path("/content/drive/MyDrive/nintx_data/0_raw")
GSDTI_DIR = RAW_DIR / "gsdti"
DCGAT_DIR = RAW_DIR / "dcgatdti"
META_DIR = GSDTI_DIR / "meta"            # substance_map.parquet / target_map.parquet

# Your authored ground truth: table with columns `smiles`, `sequence`, `real_value` (0/1)
GROUND_TRUTH_PATH = Path("/content/drive/MyDrive/nintx_data/groundtruth.parquet")

# Everything written here — put it on Drive so it survives runtime recycling
OUTPUT_DIR = Path("/content/drive/MyDrive/nintx_artifacts/colab_run")
CACHE_DIR = OUTPUT_DIR / "embedding_cache"

CONFIG_YAML = REPO_DIR / "configs" / "default.yaml"
USE_GPU_LARGE = False                    # True: ESM-2 150M + bigger fusion (needs >=24GB VRAM, see configs/gpu_large.yaml)

SPLIT_STRATEGY = "drug_cold"             # stratified_random | drug_cold | target_cold
TRAIN_FRACTION, TEST_FRACTION, VAL_FRACTION = 0.70, 0.20, 0.10
SEED = 42
EPOCHS = 50
PATIENCE = 7

INFER_BATCH_ROWS = 1_048_576             # rows per forward pass during full inference
WRITE_IDS = True                         # include Drug_ID / Target_ID columns in prediction shards"""))

cells.append(code("""# ============ 2. ENVIRONMENT SETUP ============
import sys, subprocess

if "google.colab" in sys.modules and MOUNT_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")

import os
if REPO_URL and not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
assert (REPO_DIR / "ensemble" / "pipeline.py").exists(), (
    f"ensemble package not found under {REPO_DIR} — set REPO_DIR or REPO_URL in the config cell"
)
sys.path.insert(0, str(REPO_DIR))

# IMPORTANT: install ONLY pure-Python packages here.
# NEVER run `pip install -e .` / install torch / torchvision / numpy on Colab:
# the repo's `>=` pins would replace Colab's CUDA-built torch and preinstalled numpy,
# which segfaults the kernel on `import torch`. The repo is used straight from disk
# via sys.path below — no editable install needed.
# Pin transformers <5: the generic AutoModel in 5.x serializes the downloaded config
# and passes it to the hub client as a repo id (HFValidationError). 4.x is unaffected;
# embeddings.py only uses AutoTokenizer/AutoModel, whose API is identical in 4.x.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==4.46.3", "tqdm", "pyarrow", "scikit-learn", "pyyaml"], check=True)

# Colab occasionally ships a torchvision whose compiled ops don't match the installed torch
# ("operator torchvision::nms does not exist"), which breaks *any* transformers import.
# We only use text encoders (AutoModel/AutoTokenizer), so if torchvision is broken we
# remove it: transformers then reports it as unavailable and works without it.
def _ensure_transformers_importable() -> None:
    try:
        import torchvision  # noqa: F401
    except Exception as exc:
        print(f"torchvision is broken ({type(exc).__name__}: {exc}); uninstalling it — "
              "the ensemble only needs text encoders, torchvision is not required")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchvision"], check=True)
        for name in [m for m in list(sys.modules)
                     if m == "transformers" or m.startswith(("transformers.", "torchvision", "tokenizers", "safetensors"))]:
            del sys.modules[name]
    import transformers
    print("transformers", transformers.__version__)

_ensure_transformers_importable()

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch", torch.__version__, "| device:", DEVICE,
      "|", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU ONLY — switch to a GPU runtime!")
assert DEVICE.type == "cuda", "This notebook is meant for a Colab GPU runtime."
# catches an environment poisoned by `pip install -e .`/manual torch upgrades:
# import succeeding but CUDA broken means the preinstalled torch build was replaced
if DEVICE.type == "cuda":
    _ = torch.zeros(8, device="cuda") + 1  # smoke-test an actual CUDA kernel launch
    print("CUDA kernel launch OK")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
print("Artifacts ->", OUTPUT_DIR)"""))

cells.append(code("""# ============ 3. ID MAPS (id -> smiles / sequence) ============
import pandas as pd

sub_map = pd.read_parquet(META_DIR / "substance_map.parquet")
tgt_map = pd.read_parquet(META_DIR / "target_map.parquet")
sub_map["smiles"] = sub_map["smiles"].astype(str)
tgt_map["sequence"] = tgt_map["sequence"].astype(str).str.replace(r"\\s+", "", regex=True).str.upper()

SMILES_TO_SUB = dict(zip(sub_map["smiles"], sub_map["substance_id"]))
SEQ_TO_TGT = dict(zip(tgt_map["sequence"], tgt_map["target_id"]))
DRUG_ID_OF = dict(zip(sub_map["substance_id"], sub_map["Drug_ID"]))
TARGET_ID_OF = dict(zip(tgt_map["target_id"], tgt_map["Target_ID"]))

print(f"{len(sub_map):,} substances | {len(tgt_map):,} targets | joint universe = {len(sub_map) * len(tgt_map):,} pairs")"""))

cells.append(code("""# ============ 4. BUILD THE TRAINING TABLE (ground truth + base-model signals) ============
import glob
import numpy as np
import pyarrow.parquet as pq

gt_path = str(GROUND_TRUTH_PATH)
if gt_path.endswith(".csv"):
    gt = pd.read_csv(gt_path)
else:
    gt = pd.read_parquet(gt_path)
gt = gt.rename(columns={"real_value": "target"})[["smiles", "sequence", "target"]]
gt["sequence"] = gt["sequence"].astype(str).str.replace(r"\\s+", "", regex=True).str.upper()
assert gt["target"].isin([0, 1]).all(), "real_value must be binary 0/1"
print(f"Ground truth rows: {len(gt):,}")

gt["substance_id"] = gt["smiles"].map(SMILES_TO_SUB)
gt["target_id"] = gt["sequence"].map(SEQ_TO_TGT)
n0 = len(gt)
gt = gt.dropna(subset=["substance_id", "target_id"]).astype({"substance_id": "int32", "target_id": "int32"})
print(f"Dropped {n0 - len(gt):,} rows whose smiles/sequence are outside the prediction universe")

needed_subs = gt["substance_id"].unique().tolist()
needed_tgts = gt["target_id"].unique().tolist()

def scan_signals(directory: str) -> dict:
    signals = {}
    for f in sorted(glob.glob(os.path.join(directory, "part-*.parquet"))):
        t = pq.read_table(f, filters=[("substance_id", "in", needed_subs),
                                      ("target_id", "in", needed_tgts)])
        if t.num_rows == 0:
            continue
        s = t["substance_id"].to_pylist(); j = t["target_id"].to_pylist()
        l = t["predicted_label"].to_pylist(); p = t["probability_active"].to_pylist()
        for row in zip(s, j, l, p):
            signals[(row[0], row[1])] = (row[2], row[3])
    return signals

print("Scanning GSDTI predictions ...")
gsdti_sig = scan_signals(str(GSDTI_DIR))
print("Scanning DCGAT-DTI predictions ...")
dcgat_sig = scan_signals(str(DCGAT_DIR))

for model, sig in (("gsdti", gsdti_sig), ("dcgat", dcgat_sig)):
    hit = [sig.get((s, t)) for s, t in zip(gt["substance_id"], gt["target_id"])]
    gt[f"{model}_label"] = [h[0] if h else np.nan for h in hit]
    gt[f"{model}_probability"] = [h[1] if h else np.nan for h in hit]
    print(f"{model}: matched {np.isfinite(gt[f'{model}_label']).sum():,}/{len(gt):,} pairs")

n0 = len(gt)
paired = gt.dropna(subset=["gsdti_label", "dcgat_label"]).reset_index(drop=True)
print(f"Paired training examples (both base models present): {len(paired):,} "
      f"({int(paired['target'].sum()):,} positive) — dropped {n0 - len(paired):,}")
paired = paired[["smiles", "sequence", "target", "gsdti_label", "gsdti_probability",
                 "dcgat_label", "dcgat_probability", "substance_id", "target_id"]]
paired.to_parquet(OUTPUT_DIR / "paired_training.parquet", index=False)
print("Saved ->", OUTPUT_DIR / "paired_training.parquet")"""))

cells.append(code("""# ============ 5. SPLITS (70/20/10) ============
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

from ensemble.config import load_config
from ensemble.data import create_splits

config = load_config(str(CONFIG_YAML))
if USE_GPU_LARGE:
    gpu_cfg = load_config(str(REPO_DIR / "configs" / "gpu_large.yaml"))
    config.smiles_encoder = gpu_cfg.smiles_encoder
    config.protein_encoder = gpu_cfg.protein_encoder
    config.model = gpu_cfg.model
    config.training.batch_size = gpu_cfg.training.batch_size
config.split.strategy = SPLIT_STRATEGY
# train fraction is implicit: 1 - validation - test
config.split.validation_fraction = VAL_FRACTION
config.split.test_fraction = TEST_FRACTION
config.split.seed = SEED
config.training.epochs = EPOCHS
config.training.patience = PATIENCE
config.runtime.device = str(DEVICE)

frame = create_splits(paired, config.split)
frame.to_parquet(OUTPUT_DIR / "paired_splits.parquet", index=False)
print(frame["split"].value_counts().to_dict())"""))

cells.append(code("""# ============ 6. EMBEDDINGS (ChemBERTa + ESM-2, cached on Drive) ============
from ensemble.embeddings import load_or_compute_embeddings

smiles_embeddings = load_or_compute_embeddings(
    frame["smiles"].unique().tolist(), "smiles", config.smiles_encoder, str(CACHE_DIR), DEVICE)
sequence_embeddings = load_or_compute_embeddings(
    frame["sequence"].unique().tolist(), "sequence", config.protein_encoder, str(CACHE_DIR), DEVICE)
print("smiles:", smiles_embeddings.values.shape, "| sequence:", sequence_embeddings.values.shape)"""))

cells.append(code("""# ============ 7. TRAIN ============
import json
import numpy as np
from ensemble.data import EnsembleDataset
from ensemble.metrics import binary_metrics
from ensemble.model import EnsembleModel
from ensemble.trainer import Trainer, make_loader, predict, seed_everything, write_metrics

seed_everything(SEED)
run_dir = OUTPUT_DIR / "ensemble_run"
run_dir.mkdir(parents=True, exist_ok=True)

subsets = {name: group.reset_index(drop=True) for name, group in frame.groupby("split")}
datasets = {name: EnsembleDataset(group, smiles_embeddings, sequence_embeddings)
            for name, group in subsets.items()}
loaders = {name: make_loader(dataset, config.training, shuffle=name == "train")
           for name, dataset in datasets.items()}

train_features = datasets["train"].base_features
model = EnsembleModel(
    smiles_dim=smiles_embeddings.values.shape[1],
    sequence_dim=sequence_embeddings.values.shape[1],
    config=config.model,
    base_mean=train_features.mean(dim=0),
    base_std=train_features.std(dim=0).clamp_min(1.0e-6),
)
positives = float(datasets["train"].targets.sum())
positive_weight = (len(datasets["train"]) - positives) / positives
print(f"positive weight: {positive_weight:.4f}")

trainer = Trainer(model, config.training, DEVICE, run_dir, positive_weight)
trainer.fit(loaders["train"], loaders["validation"])

targets, probabilities = predict(model, loaders["test"], DEVICE, "Testing best checkpoint")
test_metrics = binary_metrics(targets, probabilities, trainer.best_threshold)
baselines = {}
test_frame = subsets["test"]
for name, column in (("gsdti", "gsdti_probability"), ("dcgat", "dcgat_probability")):
    baselines[name] = binary_metrics(test_frame["target"].to_numpy(),
                                     test_frame[column].to_numpy(), 0.5)
report = {"ensemble": test_metrics, "baselines": baselines, "test_examples": len(test_frame)}
write_metrics(run_dir / "test_metrics.json", report)
print(json.dumps(report, indent=2))

predictions = test_frame.copy()
predictions["ensemble_probability"] = probabilities
predictions["ensemble_label"] = (probabilities >= trainer.best_threshold).astype(np.int8)
predictions.to_parquet(run_dir / "test_predictions.parquet", index=False)
BEST_CKPT = run_dir / "best.pt"
THRESHOLD = float(trainer.best_threshold)
print("Best checkpoint:", BEST_CKPT, "| threshold:", THRESHOLD)"""))

cells.append(md("""## Full inference on all 442,284,138 pairs

Both batches are dense, sorted grids over a shared id space, so each DCGAT prediction is found in
O(1) at row `substance_id * N_TARGETS + target_id` — no joins needed. The loop walks the GSDTI
shards, looks up the matching DCGAT signals, and runs batched GPU forward passes with the frozen
embedding tensors. One parquet shard is written per input part under
`OUTPUT_DIR/full_inference/`. Safe to interrupt/re-run — existing shards are skipped.
"""))

cells.append(code("""# ============ 8. FULL-UNIVERSE SETUP: DCGAT lookup arrays + full embeddings ============
import glob
import numpy as np
import torch

parts_d = sorted(glob.glob(os.path.join(DCGAT_DIR, "part-*.parquet")))
parts_g = sorted(glob.glob(os.path.join(GSDTI_DIR, "part-*.parquet")))
total_dcgat_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in parts_d)

last = pq.read_table(parts_d[-1], columns=["substance_id", "target_id"]).to_pydict()
N_TARGETS = int(np.max(pq.read_table(parts_d[0], columns=["target_id"])["target_id"].to_numpy())) + 1
N_SUBSTANCES = int(last["substance_id"][-1]) + 1
assert N_SUBSTANCES * N_TARGETS == total_dcgat_rows, (N_SUBSTANCES, N_TARGETS, total_dcgat_rows)
print(f"DCGAT dense grid: {N_SUBSTANCES:,} x {N_TARGETS:,} = {total_dcgat_rows:,}")

# O(1) lookup of DCGAT signals; label sentinel -1 marks rows absent from the grid
dcgat_label_arr = np.full(N_SUBSTANCES * N_TARGETS, -1, dtype=np.int8)
dcgat_prob_arr = np.zeros(N_SUBSTANCES * N_TARGETS, dtype=np.float16)
for f in parts_d:
    t = pq.read_table(f, columns=["substance_id", "target_id", "predicted_label", "probability_active"])
    k = t["substance_id"].to_numpy().astype(np.int64) * N_TARGETS + t["target_id"].to_numpy()
    dcgat_label_arr[k] = t["predicted_label"].to_numpy()
    dcgat_prob_arr[k] = t["probability_active"].to_numpy()
print("uncovered grid cells:", int((dcgat_label_arr < 0).sum()))

# Embed every substance/target in the joint universe (cache makes this cheap on re-runs)
universe_smiles = sub_map["smiles"].tolist()
universe_sequences = tgt_map["sequence"].tolist()
u_smiles_emb = load_or_compute_embeddings(universe_smiles, "smiles", config.smiles_encoder,
                                          str(CACHE_DIR), DEVICE)
u_seq_emb = load_or_compute_embeddings(universe_sequences, "sequence", config.protein_encoder,
                                       str(CACHE_DIR), DEVICE)

smiles_emb_idx = np.full(N_SUBSTANCES, -1, dtype=np.int64)
seq_emb_idx = np.full(N_TARGETS, -1, dtype=np.int64)
for sid, smi in zip(sub_map["substance_id"], sub_map["smiles"]):
    if smi in u_smiles_emb.index:
        smiles_emb_idx[sid] = u_smiles_emb.index[smi]
for tid, seq in zip(tgt_map["target_id"], tgt_map["sequence"]):
    if seq in u_seq_emb.index:
        seq_emb_idx[tid] = u_seq_emb.index[seq]
print("smiles embedded:", int((smiles_emb_idx >= 0).sum()), "| sequences embedded:", int((seq_emb_idx >= 0).sum()))"""))

cells.append(code("""# ============ 9. CHUNKED INFERENCE OVER ALL PAIRS ============
import pyarrow as pa

infer_dir = OUTPUT_DIR / "full_inference"
infer_dir.mkdir(parents=True, exist_ok=True)

# fresh model loaded from the best checkpoint (independent of the training state above)
ckpt = torch.load(BEST_CKPT, map_location=DEVICE, weights_only=False)
state = ckpt["model_state"]
infer_model = EnsembleModel(
    u_smiles_emb.values.shape[1], u_seq_emb.values.shape[1], config.model,
    state["base_mean"].cpu(), state["base_std"].cpu(),
).to(DEVICE)
infer_model.load_state_dict(state)
infer_model.eval()
threshold = float(ckpt.get("threshold", THRESHOLD))
smiles_tensor = u_smiles_emb.values.to(DEVICE)
seq_tensor = u_seq_emb.values.to(DEVICE)
use_amp = DEVICE.type == "cuda"

manifest = []
for i, f in enumerate(parts_g):
    shard = infer_dir / f"predictions_{i:05d}.parquet"
    if shard.exists():
        manifest.append(shard)
        continue
    t = pq.read_table(f)
    s = t["substance_id"].to_numpy().astype(np.int64)
    tg = t["target_id"].to_numpy().astype(np.int64)
    gsdti_label = t["predicted_label"].to_numpy()
    gsdti_prob = t["probability_active"].to_numpy().astype(np.float32)
    key = s * N_TARGETS + tg
    d_label = dcgat_label_arr[key]
    keep = (smiles_emb_idx[s] >= 0) & (seq_emb_idx[tg] >= 0) & (d_label >= 0)
    idx = np.nonzero(keep)[0]
    probs = np.full(len(s), np.nan, dtype=np.float32)

    si = smiles_emb_idx[s[idx]]
    ti = seq_emb_idx[tg[idx]]
    base = np.stack([gsdti_label[idx], gsdti_prob[idx],
                     d_label[idx].astype(np.float32),
                     dcgat_prob_arr[key[idx]].astype(np.float32)], axis=1)
    for start in range(0, len(idx), INFER_BATCH_ROWS):
        sl = slice(start, start + INFER_BATCH_ROWS)
        with torch.inference_mode(), torch.autocast("cuda", enabled=use_amp):
            se = smiles_tensor[torch.from_numpy(si[sl]).to(DEVICE)].float()
            te = seq_tensor[torch.from_numpy(ti[sl]).to(DEVICE)].float()
            bf = torch.from_numpy(base[sl]).float().to(DEVICE)
            logits = infer_model(se, te, bf)
            probs[idx[sl]] = torch.sigmoid(logits).float().cpu().numpy()

    columns = {
        "substance_id": pa.array(s, pa.int32()),
        "target_id": pa.array(tg, pa.int32()),
        "gsdti_probability": pa.array(gsdti_prob, pa.float32()),
        "dcgat_probability": pa.array(dcgat_prob_arr[key].astype(np.float32), pa.float32()),
        "ensemble_probability": pa.array(probs, pa.float32()),
        "ensemble_label": pa.array(np.where(np.isnan(probs), -1, (probs >= threshold)), pa.int8()),
    }
    if WRITE_IDS:
        columns["Drug_ID"] = pa.array([DRUG_ID_OF.get(v) for v in s.tolist()], pa.string())
        columns["Target_ID"] = pa.array([TARGET_ID_OF.get(v) for v in tg.tolist()], pa.string())
    pq.write_table(pa.table(columns), shard)
    manifest.append(shard)
    print(f"[{i + 1}/{len(parts_g)}] {shard.name}: {len(s):,} rows "
          f"({int(keep.sum()):,} scored) — {os.path.getsize(shard) / 1e6:.1f} MB")

print(f"Done. {len(manifest)} shards in {infer_dir}")"""))

cells.append(code("""# ============ 10. SUMMARY / MANIFEST ============
import glob
shards = sorted(glob.glob(os.path.join(infer_dir, "predictions_*.parquet")))
total = sum(pq.ParquetFile(f).metadata.num_rows for f in shards)
print(f"Shards: {len(shards)} | total predicted pairs: {total:,} / 442,284,138")
print("Run artifacts:", run_dir)
print("Prediction shards:", infer_dir)
# Optional: copy the whole run directory somewhere else
# !gsutil -m cp -r "{OUTPUT_DIR}" gs://your-bucket/nintx/"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

out = Path(__file__).with_name("colab_train_and_full_inference.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, len(cells), "cells")
