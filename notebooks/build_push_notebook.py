"""Generates push_predictions_to_neo4j.ipynb — one-time push of ensemble predictions."""
import json

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}

def code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": source.splitlines(keepends=True)}

cells = [
    md("""# Push ensemble predictions into Neo4j

One-time use: loads the shard parquets from `data/0_raw/ensemble`, keeps predictions with
probability > 0.8, deletes all existing `ACTIVITY_PREDICTION` relationships, and inserts the new
ones with `model = "ensemble"` and `probability`."""),

    code("""SHARD_DIR = "data/0_raw/ensemble"
THRESHOLD = 0.8
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "your-password"
DATABASE = "neo4j"
BATCH_SIZE = 10_000

import glob
shards = sorted(glob.glob(f"{SHARD_DIR}/predictions_*.parquet"))
print(f"{len(shards)} shards found")"""),

    code("""# load + filter (probability > 0.8; NaN rows fail the comparison and drop out)
import pandas as pd

frames = [pd.read_parquet(s, columns=["Drug_ID", "Target_ID", "ensemble_probability"])
          for s in shards]
preds = pd.concat(frames, ignore_index=True)
preds = preds[preds["ensemble_probability"] > THRESHOLD]
print(f"{len(preds):,} predictions above {THRESHOLD} (out of {sum(len(f) for f in frames):,} rows)")
preds.head()"""),

    code("""# delete existing relationships, then insert the new ones
# pip install neo4j  (uncomment if needed)
from neo4j import GraphDatabase

rows = [{"substance": d, "target": t, "probability": float(p)}
        for d, t, p in zip(preds["Drug_ID"], preds["Target_ID"], preds["ensemble_probability"])]

with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver:
    driver.verify_connectivity()
    with driver.session(database=DATABASE) as session:
        summary = session.run(
            "MATCH (s:Substance)-[r:ACTIVITY_PREDICTION]-(t:Target) DELETE r"
        ).consume()
        print("deleted:", summary.counters.relationships_deleted, "relationships")

        inserted = 0
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            session.run(
                "UNWIND $rows AS row "
                "MATCH (s:Substance {id: row.substance}), (t:Target {id: row.target}) "
                "CREATE (s)-[:ACTIVITY_PREDICTION {model: 'ensemble', probability: row.probability}]->(t)",
                rows=batch,
            ).consume()
            inserted += len(batch)
            print(f"inserted {inserted:,} / {len(rows):,}", end="\\r")

print(f"\\ndone: {inserted:,} ensemble ACTIVITY_PREDICTION relationships created")"""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

out = "push_predictions_to_neo4j.ipynb"
with open(out, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", out)
