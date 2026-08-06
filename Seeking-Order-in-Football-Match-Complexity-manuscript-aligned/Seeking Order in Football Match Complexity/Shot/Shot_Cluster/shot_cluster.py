# -*- coding: utf-8 -*-
"""Shot-location K-means clustering and K = 2–5 sensitivity analysis.

The prespecified main solution is K = 3 with seed 42. Clustering uses only the
standardized shot location `(location_x, location_y)`. Shot endpoints are retained
in the input table for other purposes but are never used in clustering.
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

INPUT_PATH = r""
SHEET_NAME = 0
OUTPUT_DIR = r""

K_VALUES = (2, 3, 4, 5)
MAIN_K = 3
MAIN_RANDOM_SEED = 42
MAIN_N_INIT = 50
STABILITY_SEEDS = tuple(range(20))
STABILITY_N_INIT = 10
SILHOUETTE_SAMPLE_SIZE = 5000


def read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False)
    return pd.read_excel(path, sheet_name=SHEET_NAME)


def sampled_silhouette(z, labels, seed):
    n = len(z)
    if n <= SILHOUETTE_SAMPLE_SIZE:
        return float(silhouette_score(z, labels))
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=SILHOUETTE_SAMPLE_SIZE, replace=False)
    return float(silhouette_score(z[idx], labels[idx]))


def main():
    if not INPUT_PATH or not OUTPUT_DIR:
        raise ValueError("Set INPUT_PATH and OUTPUT_DIR in the user-configuration section.")
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_table(INPUT_PATH)
    required = ["location_x", "location_y"]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Missing shot-location columns: {missing}")
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    complete = df[required].notna().all(axis=1)
    work = df.loc[complete, required].copy()
    if len(work) < max(K_VALUES):
        raise ValueError("Insufficient complete shot locations for K = 2–5.")
    z = StandardScaler().fit_transform(work.to_numpy(dtype=float))

    metrics_rows, size_rows, stability_rows = [], [], []
    main_labels_by_k = {}
    all_labels = df.copy()

    for k in K_VALUES:
        main_model = KMeans(n_clusters=k, n_init=MAIN_N_INIT, random_state=MAIN_RANDOM_SEED)
        main_labels = main_model.fit_predict(z).astype(int)
        main_labels_by_k[k] = main_labels
        all_labels[f"cluster_id_K{k}"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        all_labels.loc[complete, f"cluster_id_K{k}"] = main_labels

        counts = pd.Series(main_labels).value_counts().sort_index()
        for cluster_id, count in counts.items():
            size_rows.append({
                "K": k, "cluster_id": int(cluster_id), "N": int(count),
                "proportion": float(count / len(main_labels)),
                "centroid_x": float(work.loc[np.asarray(main_labels) == cluster_id, "location_x"].mean()),
                "centroid_y": float(work.loc[np.asarray(main_labels) == cluster_id, "location_y"].mean()),
            })

        seed_label_sets = []
        for seed in STABILITY_SEEDS:
            labels = KMeans(n_clusters=k, n_init=STABILITY_N_INIT, random_state=int(seed)).fit_predict(z)
            seed_label_sets.append(labels)
            stability_rows.append({
                "K": k, "seed": int(seed),
                "ARI_vs_seed42_reference": float(adjusted_rand_score(main_labels, labels)),
            })
        pairwise = [adjusted_rand_score(seed_label_sets[i], seed_label_sets[j])
                    for i, j in itertools.combinations(range(len(seed_label_sets)), 2)]
        metrics_rows.append({
            "K": k,
            "N": int(len(z)),
            "silhouette": sampled_silhouette(z, main_labels, MAIN_RANDOM_SEED + k),
            "calinski_harabasz": float(calinski_harabasz_score(z, main_labels)),
            "davies_bouldin": float(davies_bouldin_score(z, main_labels)),
            "inertia": float(main_model.inertia_),
            "min_cluster_N": int(counts.min()),
            "max_cluster_N": int(counts.max()),
            "mean_pairwise_ARI": float(np.mean(pairwise)),
            "sd_pairwise_ARI": float(np.std(pairwise, ddof=1)),
            "min_pairwise_ARI": float(np.min(pairwise)),
            "max_pairwise_ARI": float(np.max(pairwise)),
            "reference_seed": MAIN_RANDOM_SEED,
            "n_stability_seeds": len(STABILITY_SEEDS),
        })

    main_output = df.loc[complete].copy()
    main_output["cluster_id"] = main_labels_by_k[MAIN_K]
    main_output.to_csv(output_dir / "Shot_K3_main_for_DML.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(output_dir / "Shot_cluster_sensitivity_K2_K5.xlsx", engine="xlsxwriter") as writer:
        main_output.to_excel(writer, index=False, sheet_name="K3_main_for_DML")
        all_labels.to_excel(writer, index=False, sheet_name="all_K_labels")
        pd.DataFrame(metrics_rows).to_excel(writer, index=False, sheet_name="K_metrics")
        pd.DataFrame(size_rows).to_excel(writer, index=False, sheet_name="cluster_sizes")
        pd.DataFrame(stability_rows).to_excel(writer, index=False, sheet_name="seed_stability")
    print(f"[OK] Shot K=3 main data and K=2–5 sensitivity outputs written to {output_dir}")


if __name__ == "__main__":
    main()
