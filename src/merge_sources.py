import pandas as pd

gsdti = pd.read_parquet("data/0_raw/gsdti.parquet")
gsdti["source"] = "GSDTI"
dcgat_dti = pd.read_csv("data/0_raw/dcgat-dti.csv", sep="|")
dcgat_dti = dcgat_dti[dcgat_dti["excluded_reason"].isna()]
dcgat_dti = dcgat_dti.rename(
    {"activity": "real_value", "predicted_label": "label", "probability_active": "probability"}, axis=1
).drop(
    ["excluded_reason", "probability_inactive", "split"], axis=1
)
dcgat_dti["source"] = "DCGAT-DTI"
df = pd.concat([gsdti, dcgat_dti])
df.to_parquet("data/1_ready/df.parquet")