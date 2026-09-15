"""
Customer segmentation: K-Means clustering on RFM features, translation of
clusters into business-facing segment names, segment profiling, and a PCA
projection for visualization.

Extracted, unmodified in logic, from `notebooks/unsupervised_learning.ipynb`,
sections 2-6 of the "Customer Features, customer segmentation, prediction
model" part of the notebook (cells 57-62).
"""
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import ACTION_MAP, K_RANGE, MIN_K_FOR_BUSINESS_USE, RANDOM_STATE


def scale_rfm_features(rfm: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler, np.ndarray]:
    """Log-transform Frequency/Monetary (heavily right-skewed) then standardize.

    Mirrors:
        X = rfm[['Recency', 'Frequency', 'Monetary']].copy()
        X['Frequency'] = np.log1p(X['Frequency'])
        X['Monetary'] = np.log1p(X['Monetary'])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    """
    X = rfm[["Recency", "Frequency", "Monetary"]].copy()
    X["Frequency"] = np.log1p(X["Frequency"])
    X["Monetary"] = np.log1p(X["Monetary"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, scaler, X_scaled


def search_k(X_scaled: np.ndarray, k_range=K_RANGE, random_state: int = RANDOM_STATE):
    """Search a range of k for K-Means and return (inertias, silhouettes).

    Mirrors the elbow / silhouette search loop over k=3..8.
    """
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return inertias, silhouettes


def fit_best_kmeans(
    X_scaled: np.ndarray,
    k_range=K_RANGE,
    min_k_for_business_use: int = MIN_K_FOR_BUSINESS_USE,
    random_state: int = RANDOM_STATE,
) -> Tuple[KMeans, int, np.ndarray]:
    """Pick k>=min_k_for_business_use with the best silhouette score, fit K-Means.

    Mirrors:
        candidates = [k for k in k_range if k >= MIN_K_FOR_BUSINESS_USE]
        best_k = candidates[argmax(silhouette)]
        kmeans = KMeans(n_clusters=best_k, ...).fit_predict(X_scaled)

    The CRM brief asks to distinguish groups like VIP, at-risk, and
    one-time buyers -- that needs more resolution than the 3-cluster split
    silhouette alone tends to favor. We pick the best silhouette score
    within k=4..8, trading a little cohesion for buckets the CRM team can
    actually act on differently.
    """
    k_range = list(k_range)
    _, silhouettes = search_k(X_scaled, k_range=k_range, random_state=random_state)

    candidates = [k for k in k_range if k >= min_k_for_business_use]
    candidate_scores = [silhouettes[k_range.index(k)] for k in candidates]
    best_k = candidates[int(np.argmax(candidate_scores))]

    kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, best_k, labels


def business_segment(row: pd.Series) -> str:
    """Rule-based per-customer segment label from R/F/M scores + raw Frequency.

    Mirrors the notebook's `business_segment` function exactly. This
    vocabulary matches what a CRM team actually asks for.
    """
    r, f, m, freq = row["R_Score"], row["F_Score"], row["M_Score"], row["Frequency"]
    if freq == 1:
        return "One-Time Buyer"
    if r >= 4 and f >= 4 and m >= 4:
        return "VIP"
    if r >= 3 and f >= 3 and m >= 3:
        return "Loyal Customer"
    if r >= 4 and f <= 2:
        return "New Customer"
    if r <= 2 and f >= 4 and m >= 4:
        return "Win-Back Priority"  # high past value, gone quiet
    if r <= 2 and f >= 3:
        return "At Risk"
    if r <= 2 and f <= 2 and m <= 2:
        return "Lost / Let Go"
    return "Needs Attention"


def assign_segments(rfm: pd.DataFrame, labels: np.ndarray) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Attach Cluster + rule-based Segment labels to the RFM table.

    Follows the notebook's approach: each K-Means cluster is named after its
    most common rule-based segment, and names are kept unique across
    clusters so the CRM team never sees two buckets with the same name.

    One deliberate fix versus the notebook
    --------------------------------------
    The notebook resolves name collisions by *dropping* the later cluster
    from the mapping:

        seen = set()
        unique_cluster = {}
        for k, v in cluster_to_label.items():
            if v not in seen:
                unique_cluster[k] = v
                seen.add(v)
        rfm['Segment'] = rfm['Cluster'].map(unique_cluster)

    When two clusters share a modal rule segment (which happens on the full
    dataset -- e.g. two clusters are both modally "VIP"), the second cluster
    maps to NaN and every customer in it ends up with no segment at all. On
    this data that silently leaves ~23% of customers unlabeled, which would
    make the segment profile, the action list and the app misleading.

    Here, a cluster whose modal name is already taken falls back to its
    next-most-common rule segment that is still free, and only if every
    candidate is taken does it get a numbered suffix. The naming logic is
    otherwise identical, and the raw per-customer `RuleSegment` column is
    kept so the original rule-based labels remain available.
    """
    rfm = rfm.copy()
    rfm["Cluster"] = labels
    rfm["RuleSegment"] = rfm.apply(business_segment, axis=1)

    # Rule-segment frequencies per cluster, most common first.
    ranked = (
        rfm.groupby("Cluster")["RuleSegment"]
        .agg(lambda x: list(x.value_counts().index))
        .to_dict()
    )

    # Assign the biggest clusters first so the dominant group keeps the
    # name it most deserves.
    cluster_sizes = rfm["Cluster"].value_counts()

    unique_cluster: Dict[int, str] = {}
    taken = set()
    for cluster in cluster_sizes.index:
        candidates = ranked[cluster]
        label = next((c for c in candidates if c not in taken), None)
        if label is None:
            # Every rule segment present in this cluster is already used.
            base = candidates[0]
            suffix = 2
            while f"{base} ({suffix})" in taken:
                suffix += 1
            label = f"{base} ({suffix})"
        unique_cluster[cluster] = label
        taken.add(label)

    unique_cluster = dict(sorted(unique_cluster.items()))
    rfm["Segment"] = rfm["Cluster"].map(unique_cluster)

    assert rfm["Segment"].notna().all(), "every customer must land in a named segment"
    return rfm, unique_cluster


def action_for(segment: str) -> str:
    """Look up the recommended CRM action for a segment name.

    Handles the numbered-suffix fallback from `assign_segments` (e.g.
    "VIP (2)" resolves to the "VIP" action) so no segment is ever left
    without guidance for the CRM team.
    """
    if segment in ACTION_MAP:
        return ACTION_MAP[segment]
    base = re.sub(r"\s*\(\d+\)$", "", str(segment))
    return ACTION_MAP.get(base, "Monitor: review this segment manually")


def segment_profile(rfm: pd.DataFrame) -> pd.DataFrame:
    """Aggregate segment-level stats + recommended CRM action per segment.

    Mirrors the notebook's "SEGMENT PROFILES" + "RECOMMENDED CRM ACTION"
    cells.
    """
    profile = rfm.groupby("Segment").agg(
        Customers=("Customer ID", "count"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetary=("Monetary", "mean"),
        TotalRevenue=("Monetary", "sum"),
    ).sort_values("TotalRevenue", ascending=False)

    profile["PctCustomers"] = profile["Customers"] / len(rfm) * 100
    profile["PctRevenue"] = profile["TotalRevenue"] / rfm["Monetary"].sum() * 100
    profile["RecommendedAction"] = profile.index.map(action_for)
    return profile


def add_pca_projection(rfm: pd.DataFrame, X_scaled: np.ndarray, random_state: int = RANDOM_STATE) -> Tuple[pd.DataFrame, PCA]:
    """Add a 2D PCA projection of the scaled RFM features for visualization.

    Mirrors:
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        rfm['PCA1'], rfm['PCA2'] = X_pca[:, 0], X_pca[:, 1]
    """
    rfm = rfm.copy()
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    rfm["PCA1"] = X_pca[:, 0]
    rfm["PCA2"] = X_pca[:, 1]
    return rfm, pca
