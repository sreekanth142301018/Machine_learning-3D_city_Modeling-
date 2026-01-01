
import os
import gc
import joblib
import numpy as np
import pandas as pd
import laspy
from shapely.geometry import MultiPoint

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.metrics import (
    recall_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
    make_scorer,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
LAS_FILE = "data/for_noof_flors.las"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_POINTS = 600_000
AVG_FLOOR_HEIGHT = 3.0      # average floor height (tune this if needed)
MIN_FLOOR_HEIGHT = 2.7      # physical lower bound per floor
MAX_FLOOR_HEIGHT = 4.0      # physical upper bound per floor

BIN_SIZE = 0.25
WINDOW_DENSITY_THRESH = 0.45
VENT_MIN_GAP = 0.35
VENT_MAX_GAP = 2.2
MIN_CLUSTER_POINTS = 200
RANDOM_STATE = 42

MAX_FLOORS_CLASS = 12       # upper limit for label/predictions

np.random.seed(RANDOM_STATE)

# ----------------------------
# HELPERS
# ----------------------------
def load_las_sample(las_path, max_points=MAX_POINTS):
    if not os.path.exists(las_path):
        raise FileNotFoundError(las_path)
    las = laspy.read(las_path)
    pts = np.vstack((las.x, las.y, las.z)).astype(np.float32).T

    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        cls = las.classification[idx] if hasattr(las, "classification") else None
    else:
        cls = las.classification if hasattr(las, "classification") else None

    del las
    gc.collect()
    return pts, cls


def remove_ground(points, cls):
    # use LAS classification if available
    if cls is not None and len(cls) == len(points):
        mask = cls != 2      # class 2 = ground
        if np.sum(mask) > 50:
            return points[mask]
    # fallback: 5th percentile
    zthr = np.percentile(points[:, 2], 5)
    return points[points[:, 2] > zthr]


def adaptive_dbscan_eps(X2d, k=6):
    if len(X2d) <= 1:
        return 2.0
    nbrs = NearestNeighbors(n_neighbors=min(k, len(X2d) - 1)).fit(X2d)
    distances, _ = nbrs.kneighbors(X2d)
    kdist = np.median(distances[:, -1])
    return max(1.5, kdist * 3.0)


def vertical_profile(z_values, bin_size=BIN_SIZE):
    zmin, zmax = z_values.min(), z_values.max()
    if zmax <= zmin:
        bins = np.array([zmin, zmin + bin_size])
    else:
        bins = np.arange(zmin, zmax + bin_size, bin_size)

    hist, edges = np.histogram(z_values, bins=bins)
    heights = edges[:-1] + bin_size / 2.0
    norm = hist / float(hist.max()) if hist.max() > 0 else hist.astype(float)
    return heights, hist, norm


def detect_window_gaps(heights, norm_hist):
    low_idx = np.where(norm_hist < WINDOW_DENSITY_THRESH)[0]
    gaps = []
    if len(low_idx) == 0:
        return gaps

    start = low_idx[0]
    prev = low_idx[0]
    for i in low_idx[1:]:
        if i - prev > 1:
            s_z = heights[start]
            e_z = heights[prev] + BIN_SIZE
            gap_h = e_z - s_z
            if VENT_MIN_GAP <= gap_h <= VENT_MAX_GAP:
                gaps.append((s_z, e_z, gap_h))
            start = i
        prev = i

    # last group
    s_z = heights[start]
    e_z = heights[prev] + BIN_SIZE
    gap_h = e_z - s_z
    if VENT_MIN_GAP <= gap_h <= VENT_MAX_GAP:
        gaps.append((s_z, e_z, gap_h))

    return gaps


def cluster_area(points2d):
    try:
        return MultiPoint(points2d).convex_hull.area
    except Exception:
        return 0.0


def height_based_floor_estimate(height):
    """Main rule: floors from total height, with physical bounds."""
    if height <= 0:
        return 1

    rough = height / AVG_FLOOR_HEIGHT
    floors = int(round(rough))
    floors = max(1, floors)

    # physical limits: each floor between MIN_FLOOR_HEIGHT and MAX_FLOOR_HEIGHT
    min_possible = max(1, int(np.floor(height / MAX_FLOOR_HEIGHT)))
    max_possible = max(1, int(np.ceil(height / MIN_FLOOR_HEIGHT)))
    floors = int(np.clip(floors, min_possible, max_possible))
    return floors


def combine_height_and_windows(height_floors, n_gaps):
    """
    Use window gaps as soft hint:
    - If gaps close to height_floors => average.
    - Else ignore gaps (trust height).
    Typically for G+N buildings: n_gaps ~ floors - 1 or similar.
    """
    if n_gaps <= 0:
        return height_floors
    if abs(n_gaps - height_floors) <= 1:
        return int(round((height_floors + n_gaps) / 2.0))
    else:
        return height_floors


def clamp_prediction_by_height(pred_floors, height):
    """Clamp RF prediction using same physical constraints."""
    pred_floors = max(1, int(round(pred_floors)))
    if height <= 0:
        return pred_floors
    min_possible = max(1, int(np.floor(height / MAX_FLOOR_HEIGHT)))
    max_possible = max(1, int(np.ceil(height / MIN_FLOOR_HEIGHT)))
    pred_floors = int(np.clip(pred_floors, min_possible, max_possible))
    return min(pred_floors, MAX_FLOORS_CLASS)


# ----------------------------
# MAIN PIPELINE
# ----------------------------
print("Loading LAS...")
points, cls = load_las_sample(LAS_FILE)
non_ground = remove_ground(points, cls)
print(f"Non-ground points: {len(non_ground)}")

# --- Clustering ---
print("Clustering footprints...")
eps = adaptive_dbscan_eps(non_ground[:, :2])
db = DBSCAN(eps=eps, min_samples=60).fit(non_ground[:, :2])
labels = db.labels_
unique_labels = [l for l in set(labels) if l != -1]
print(f"Detected {len(unique_labels)} clusters")

# --- Feature Extraction ---
rows = []
for lbl in unique_labels:
    cluster = non_ground[labels == lbl]
    if len(cluster) < MIN_CLUSTER_POINTS:
        continue

    z = cluster[:, 2]
    heights, hist, norm = vertical_profile(z)
    gaps = detect_window_gaps(heights, norm)

    z5, z95 = np.percentile(z, [5, 95])
    height = z95 - z5
    area = cluster_area(cluster[:, :2])
    density = len(cluster) / max(area, 1.0)
    low_frac = float(np.mean(norm < WINDOW_DENSITY_THRESH)) if len(norm) > 0 else 0.0
    n_gaps = len(gaps)

    floors_h = height_based_floor_estimate(height)
    floors_combined = combine_height_and_windows(floors_h, n_gaps)
    floors_combined = int(np.clip(floors_combined, 1, MAX_FLOORS_CLASS))

    rows.append(
        {
            "cluster_id": int(lbl),
            "n_points": int(len(cluster)),
            "height": float(height),
            "area": float(area),
            "density": float(density),
            "low_frac": low_frac,
            "n_gaps": int(n_gaps),
            "floor_est_height": floors_h,
            "floor_label": floors_combined,
        }
    )

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError("No clusters found. Try reducing MIN_CLUSTER_POINTS.")

# Sort by area (biggest clusters first – these are usually main buildings)
df = df.sort_values("area", ascending=False).reset_index(drop=True)

# Give human-friendly building IDs: 1, 2, 3, ...
df["building_id"] = df.index + 1

print("All clusters (sorted by area):")
print(
    df[
        [
            "building_id",
            "cluster_id",
            "height",
            "area",
            "floor_est_height",
            "floor_label",
        ]
    ].head()
)

print("Initial cluster features:\n", df[["cluster_id", "height", "floor_est_height", "floor_label"]])

# --- Feature Matrix ---
FEATS = ["height", "area", "density", "low_frac", "n_gaps"]
X = df[FEATS].values
y = df["floor_label"].astype(int).values

# ----------------------------
# Hyperparameter Tuning (RandomForest)
# ----------------------------
print("Tuning RandomForest...")

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 8, 10, None],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 3],
}

rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")

# handle class counts for CV
unique_classes, counts = np.unique(y, return_counts=True)
min_count = counts.min()

if min_count >= 2 and len(unique_classes) > 1:
    n_splits = min(5, int(min_count))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
else:
    # fall back to plain KFold when some class has only 1 member
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    print("⚠️ Using KFold (not stratified) because some classes are too rare.")

# custom scorer: macro recall with zero_division=0 (no warnings)
recall_macro_scorer = make_scorer(
    recall_score,
    average="macro",
    zero_division=0,
)

grid = GridSearchCV(
    rf,
    param_grid,
    scoring=recall_macro_scorer,
    cv=cv,
    n_jobs=-1,
)
grid.fit(X, y)
best_model = grid.best_estimator_
print("Best RF Params:", grid.best_params_)

# --- Predict + post-process ---
y_pred_raw = best_model.predict(X)

floors_final = []
for pred, h in zip(y_pred_raw, df["height"].values):
    floors_final.append(clamp_prediction_by_height(pred, h))

df["floors_pred"] = floors_final

# --- Evaluation metrics on the training set ---
y_true = df["floor_label"].astype(int).values      # pseudo/true labels
y_pred = df["floors_pred"].astype(int).values      # final RF + constraints

acc = accuracy_score(y_true, y_pred)
recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

print("\n=== Training-set performance (final constrained predictions) ===")
print(f"Accuracy        : {acc:.3f}")
print(f"Macro Recall    : {recall_macro:.3f}")
print("\nClassification report:")
print(classification_report(y_true, y_pred, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))

# --- Estimate per-floor usable area ---
df["usable_area_per_floor"] = (df["area"] * (1.0 - df["low_frac"])) / np.maximum(
    df["floors_pred"], 1
)

# --- Save Outputs ---
out_csv = os.path.join(OUTPUT_DIR, "3.csv")
df.to_csv(out_csv, index=False)
joblib.dump(best_model, os.path.join(OUTPUT_DIR, "rf_floor_model.pkl"))
print(f"✅ Saved results to: {out_csv}")

# --- Quick sanity check: height / floors_pred (should be 2.7–4.0 m range mostly)
df["height_per_floor"] = df["height"] / df["floors_pred"]
print("\nSample (height, floors_pred, height_per_floor):")
print(df[["cluster_id", "height", "floors_pred", "height_per_floor"]].head())

# --- Scatter plot height vs predicted floors ---
plt.figure(figsize=(8, 6))
plt.scatter(df["height"], df["floors_pred"], s=80)
for _, r in df.iterrows():
    plt.text(r["height"], r["floors_pred"], str(r["cluster_id"]), fontsize=8)
plt.xlabel("Building Height (m)")
plt.ylabel("Predicted Floors")
plt.title("Height vs Predicted Floors (RF + physical constraints)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "height_vs_predicted_floors_final.png"), dpi=200)
print("✅ Saved height vs floors plot.")

# =========================
#   TOP-4 / TOP-5 EXPORTS
# =========================

# top 4 buildings by area (for main analysis)
df_top4 = df.sort_values("area", ascending=False).head(4).copy()
df_top4["building_id"] = ["Building 1", "Building 2", "Building 3", "Building 4"]

top4_csv = os.path.join(OUTPUT_DIR, "top4_buildings.csv")
df_top4.to_csv(top4_csv, index=False)
print(f"✅ Saved Top 4 Buildings CSV: {top4_csv}")

# top 5 buildings by area (for W histograms & scatter)
df_top5 = df.sort_values("area", ascending=False).head(5).copy()
print("\nTop 5 buildings selected for histogram:")
print(df_top5[["cluster_id", "height", "area", "n_gaps"]])

# --- Histogram: window bands (W = n_gaps) for Top 5 buildings ---
plt.figure(figsize=(8, 5))
max_gaps = int(df_top5["n_gaps"].max())
bins = np.arange(-0.5, max_gaps + 1.5, 1)  # centers at 0,1,2,...

plt.hist(df_top5["n_gaps"], bins=bins)
plt.xticks(range(0, max_gaps + 1))
plt.xlabel("Number of Window Bands (W)")
plt.ylabel("Number of Buildings")
plt.title("Window Bands Histogram for Top 5 LiDAR Building Clusters")
plt.grid(True)
plt.tight_layout()

hist_path = os.path.join(OUTPUT_DIR, "window_bands_top5_histogram.png")
plt.savefig(hist_path, dpi=200)
print(f"✅ Top 5 window bands histogram saved at: {hist_path}")

# --- Scatter Plot: Gaps (Window Bands) vs Height for Top 5 ---
plt.figure(figsize=(8, 5))
plt.scatter(df_top5["n_gaps"], df_top5["height"], s=100, c="red")
plt.xlabel("Number of Window Bands (W = n_gaps)")
plt.ylabel("Building Height (m)")
plt.title("Gaps vs Height for Top 5 Buildings")
plt.grid(True)
plt.tight_layout()

scatter_path = os.path.join(OUTPUT_DIR, "gaps_vs_height_top5.png")
plt.savefig(scatter_path, dpi=200)
print(f"✅ Gaps vs Height plot saved at: {scatter_path}")

print("Done.")
