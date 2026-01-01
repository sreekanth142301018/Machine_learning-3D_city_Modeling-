


import os
import gc
import numpy as np
import pandas as pd
import laspy
from shapely.geometry import MultiPoint
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import xml.etree.ElementTree as ET

# ---------------- CONFIG ----------------
LAS_FILE     = "data/for_noof_flors.las"
BUILDING_CSV = "output/top4_buildings.csv"   # from floor_detection script
OUTPUT_GML   = "output/city_model_lod2_with_windows.gml"

# keep this much LiDAR max in memory for geometry
MAX_POINTS          = 600_000        # was 4,000,000 (too heavy)
MIN_CLUSTER_POINTS  = 200
MAX_HULL_POINTS     = 50_000        # max points per cluster for convex hull
RANDOM_STATE        = 42
np.random.seed(RANDOM_STATE)

# window geometry parameters (how big windows are on walls)
WINDOW_WIDTH_FRAC   = 0.3    # fraction of wall length
WINDOW_VERT_MARGIN  = 0.2    # 20% vertical margin inside window band

# ---------- HELPERS ----------
def load_las_sample(las_path, max_points=MAX_POINTS):
    """Load LAS and subsample if needed to avoid memory explosion."""
    if not os.path.exists(las_path):
        raise FileNotFoundError(las_path)
    las = laspy.read(las_path)
    pts = np.vstack((las.x, las.y, las.z)).astype(np.float32).T

    cls = None
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        if hasattr(las, "classification"):
            cls = np.array(las.classification)[idx]
    else:
        if hasattr(las, "classification"):
            cls = np.array(las.classification)

    del las
    gc.collect()
    return pts, cls


def remove_ground(points, cls):
    """Simple ground removal using LAS class or 5th percentile."""
    if cls is not None and len(cls) == len(points):
        mask = cls != 2   # 2 = ground
        if np.sum(mask) > 50:
            return points[mask]
    zthr = np.percentile(points[:, 2], 5)
    return points[points[:, 2] > zthr]


def adaptive_dbscan_eps(X2d, k=6):
    """Estimate DBSCAN eps from k-NN distances."""
    if len(X2d) <= k:
        return 2.5
    nbrs = NearestNeighbors(n_neighbors=min(k, len(X2d) - 1)).fit(X2d)
    distances, _ = nbrs.kneighbors(X2d)
    kdist = np.median(distances[:, -1])
    return max(1.5, 3.0 * kdist)


def footprint_corners(points2d):
    """
    Get convex hull corners for building footprint.
    Subsample if too many points to keep hull fast.
    """
    if len(points2d) < 3:
        return None

    pts = points2d
    if len(pts) > MAX_HULL_POINTS:
        idx = np.random.choice(len(pts), MAX_HULL_POINTS, replace=False)
        pts = pts[idx]

    try:
        hull = MultiPoint(pts).convex_hull
        if hull.is_empty:
            return None
        coords = np.array(hull.exterior.coords)   # last == first
        return coords[:-1]                        # drop repeated last
    except Exception as e:
        print(f"Convex hull failed for cluster: {e}")
        return None


def poslist_from_ring(coords_xy, z):
    """
    Build gml:posList string from 2D ring + constant z.
    """
    coords_xy = list(coords_xy)
    if len(coords_xy) == 0:
        return ""
    # coords may be numpy arrays; use allclose for robust equality check
    if not np.allclose(coords_xy[0], coords_xy[-1]):
        coords_xy.append(coords_xy[0])

    vals = []
    for x, y in coords_xy:
        vals.extend([str(x), str(y), str(z)])
    return " ".join(vals)


# ---------- LOAD BUILDING CSV ----------
if not os.path.exists(BUILDING_CSV):
    raise FileNotFoundError(f"Missing building CSV: {BUILDING_CSV}")

bdf = pd.read_csv(BUILDING_CSV)

# expected columns: cluster_id, height, base_z, floors_est, window_bands
# We only REQUIRE cluster_id and height.
# base_z and floors_est are optional (we fall back if missing).
required_cols = ["cluster_id", "height"]
for c in required_cols:
    if c not in bdf.columns:
        raise ValueError(f"Column '{c}' missing in {BUILDING_CSV}")

building_map = {}
for _, row in bdf.iterrows():
    cid = int(row["cluster_id"])
    height = float(row["height"])

    # base_z might not be in top4_buildings.csv → default to 0.0
    if "base_z" in bdf.columns:
        base_z = float(row["base_z"])
    else:
        base_z = 0.0   # put all buildings on ground (z = 0)

    # floors_est might not exist → use floors_pred instead
    if "floors_est" in bdf.columns:
        floors = int(row["floors_est"])
    else:
        floors = int(row["floors_pred"])

    # window_bands may not exist → no windows
    wb_str = str(row["window_bands"]) if "window_bands" in bdf.columns else ""
    bands = []
    if wb_str.strip() != "":
        for part in wb_str.split(";"):
            try:
                s, e = part.split(":")
                bands.append((float(s), float(e)))
            except Exception:
                continue

    building_map[cid] = {
        "height": height,
        "base_z": base_z,
        "floors": floors,
        "window_bands": bands,
    }


if not building_map:
    raise RuntimeError("No buildings in CSV to build CityGML.")

print("Loaded building info for cluster_ids:", sorted(building_map.keys()))

# ---------- CLUSTER LAS (GEOMETRY) ----------
print("Loading LAS for geometry...")
points, cls = load_las_sample(LAS_FILE, max_points=MAX_POINTS)
non_ground = remove_ground(points, cls)
print(f"Non-ground points after ground removal: {len(non_ground)}")

if len(non_ground) == 0:
    raise RuntimeError("No non-ground points found; check LAS / CSF step.")

print("Running DBSCAN on non-ground points (this may take some time)...")
eps = adaptive_dbscan_eps(non_ground[:, :2], k=8)
print(f"  -> using eps = {eps:.2f}")

db = DBSCAN(eps=eps, min_samples=60).fit(non_ground[:, :2])
labels = db.labels_
unique_labels = [l for l in set(labels) if l != -1]
print(f"Detected {len(unique_labels)} clusters in LAS.")

# Only build geometry for cluster_ids that appear in building_map
geom_map = {}
for lbl in unique_labels:
    if lbl not in building_map:
        continue
    cluster = non_ground[labels == lbl]
    if len(cluster) < MIN_CLUSTER_POINTS:
        continue

    corners = footprint_corners(cluster[:, :2])
    if corners is None or len(corners) < 3:
        print(f"Cluster {lbl}: not enough corners, skipping.")
        continue

    geom_map[int(lbl)] = {
        "corners_xy": corners
    }

missing = set(building_map.keys()) - set(geom_map.keys())
if missing:
    print("⚠ WARNING: These cluster_ids exist in CSV but not in geometry map:", sorted(missing))

print("Geometry available for cluster_ids:", sorted(geom_map.keys()))

# ---------- PREPARE CityGML ROOT ----------
NS_GML  = "http://www.opengis.net/gml"
NS_CORE = "http://www.opengis.net/citygml/2.0"
NS_BLDG = "http://www.opengis.net/citygml/building/2.0"
NS_XSI  = "http://www.w3.org/2001/XMLSchema-instance"

ET.register_namespace("gml", NS_GML)
ET.register_namespace("core", NS_CORE)
ET.register_namespace("bldg", NS_BLDG)
ET.register_namespace("xsi", NS_XSI)

root = ET.Element(
    f"{{{NS_CORE}}}CityModel",
    attrib={
        f"{{{NS_XSI}}}schemaLocation":
            "http://www.opengis.net/citygml/2.0 "
            "http://schemas.opengis.net/citygml/2.0/cityGMLBase.xsd"
    }
)

def add_surface_with_poslist(parent_bldg, surface_tag, poslist_text):
    """
    Create GroundSurface / RoofSurface (no windows).
    Returns the created <bldg:...Surface> element.
    """
    bounded_by = ET.SubElement(parent_bldg, f"{{{NS_BLDG}}}boundedBy")
    surf       = ET.SubElement(bounded_by, f"{{{NS_BLDG}}}{surface_tag}")
    lod2       = ET.SubElement(surf, f"{{{NS_BLDG}}}lod2MultiSurface")
    ms         = ET.SubElement(lod2, f"{{{NS_GML}}}MultiSurface")
    sm         = ET.SubElement(ms, f"{{{NS_GML}}}surfaceMember")
    poly       = ET.SubElement(sm, f"{{{NS_GML}}}Polygon")
    ext        = ET.SubElement(poly, f"{{{NS_GML}}}exterior")
    lr         = ET.SubElement(ext, f"{{{NS_GML}}}LinearRing")
    pos        = ET.SubElement(lr, f"{{{NS_GML}}}posList", attrib={"srsDimension": "3"})
    pos.text   = poslist_text
    return surf


def add_wall_surface_with_windows(parent_bldg, x1, y1, x2, y2,
                                  base_z, roof_z, window_bands):
    """
    Creates one WallSurface quad and optional Window openings.
    window_bands is a list of (z_start, z_end).
    """
    # Wall polygon (rectangle)
    wall_coords = [
        (x1, y1, base_z),
        (x2, y2, base_z),
        (x2, y2, roof_z),
        (x1, y1, roof_z),
        (x1, y1, base_z),
    ]
    wall_pos_vals = []
    for (x, y, z) in wall_coords:
        wall_pos_vals.extend([str(x), str(y), str(z)])
    wall_poslist = " ".join(wall_pos_vals)

    # <bldg:WallSurface>
    bounded_by = ET.SubElement(parent_bldg, f"{{{NS_BLDG}}}boundedBy")
    wall_surf  = ET.SubElement(bounded_by, f"{{{NS_BLDG}}}WallSurface")
    lod2       = ET.SubElement(wall_surf, f"{{{NS_BLDG}}}lod2MultiSurface")
    ms         = ET.SubElement(lod2, f"{{{NS_GML}}}MultiSurface")
    sm         = ET.SubElement(ms, f"{{{NS_GML}}}surfaceMember")
    poly       = ET.SubElement(sm, f"{{{NS_GML}}}Polygon")
    ext        = ET.SubElement(poly, f"{{{NS_GML}}}exterior")
    lr         = ET.SubElement(ext, f"{{{NS_GML}}}LinearRing")
    pos        = ET.SubElement(lr, f"{{{NS_GML}}}posList", attrib={"srsDimension": "3"})
    pos.text   = wall_poslist

    # If no window bands for this building, done
    if not window_bands:
        return

    # direction & length of wall (for placing windows)
    dx = x2 - x1
    dy = y2 - y1
    wall_len = np.sqrt(dx * dx + dy * dy)
    if wall_len == 0:
        return

    win_width = WINDOW_WIDTH_FRAC * wall_len
    half_w = win_width / 2.0

    # unit direction vector along wall
    ux = dx / wall_len
    uy = dy / wall_len

    # wall center
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    for (z_start, z_end) in window_bands:
        dz = z_end - z_start
        if dz <= 0:
            continue
        z_bottom = z_start + WINDOW_VERT_MARGIN * dz
        z_top    = z_end   - WINDOW_VERT_MARGIN * dz
        if z_top <= z_bottom:
            continue

        # horizontal extents along wall
        left_x  = cx - ux * half_w
        left_y  = cy - uy * half_w
        right_x = cx + ux * half_w
        right_y = cy + uy * half_w

        win_coords = [
            (left_x,  left_y,  z_bottom),
            (right_x, right_y, z_bottom),
            (right_x, right_y, z_top),
            (left_x,  left_y,  z_top),
            (left_x,  left_y,  z_bottom),
        ]
        win_pos_vals = []
        for (x, y, z) in win_coords:
            win_pos_vals.extend([str(x), str(y), str(z)])
        win_poslist = " ".join(win_pos_vals)

        # <bldg:opening><bldg:Window> geometry
        opening = ET.SubElement(wall_surf, f"{{{NS_BLDG}}}opening")
        window  = ET.SubElement(opening,   f"{{{NS_BLDG}}}Window")
        w_lod   = ET.SubElement(window,    f"{{{NS_BLDG}}}lod3MultiSurface")
        w_ms    = ET.SubElement(w_lod,     f"{{{NS_GML}}}MultiSurface")
        w_sm    = ET.SubElement(w_ms,      f"{{{NS_GML}}}surfaceMember")
        w_poly  = ET.SubElement(w_sm,      f"{{{NS_GML}}}Polygon")
        w_ext   = ET.SubElement(w_poly,    f"{{{NS_GML}}}exterior")
        w_lr    = ET.SubElement(w_ext,     f"{{{NS_GML}}}LinearRing")
        w_pos   = ET.SubElement(w_lr,      f"{{{NS_GML}}}posList", attrib={"srsDimension": "3"})
        w_pos.text = win_poslist


# ---------- BUILD BUILDINGS ----------
bldg_count = 0

for cid in sorted(building_map.keys()):
    if cid not in geom_map:
        print(f"Skipping cluster {cid}: no geometry (corners) found.")
        continue

    props = building_map[cid]
    geom  = geom_map[cid]

    height       = props["height"]
    base_z       = props["base_z"]
    floors       = props["floors"]
    window_bands = props["window_bands"]

    roof_z = base_z + height
    corners_xy = geom["corners_xy"]

    if corners_xy is None or len(corners_xy) < 3:
        print(f"Skipping cluster {cid}: invalid corners.")
        continue

    bldg_count += 1
    bldg_id = f"Bldg_{cid}"

    com = ET.SubElement(root, f"{{{NS_CORE}}}cityObjectMember")
    bldg = ET.SubElement(
        com,
        f"{{{NS_BLDG}}}Building",
        attrib={f"{{{NS_GML}}}id": bldg_id},
    )

    # basic attributes
    name_el = ET.SubElement(bldg, f"{{{NS_GML}}}name")
    name_el.text = f"Building_{cid}"

    storeys_el = ET.SubElement(bldg, f"{{{NS_BLDG}}}storeysAboveGround")
    storeys_el.text = str(floors)

    mh_el = ET.SubElement(
        bldg,
        f"{{{NS_BLDG}}}measuredHeight",
        attrib={"uom": "m"},
    )
    mh_el.text = f"{height:.2f}"

    # GroundSurface
    ground_poslist = poslist_from_ring(corners_xy, base_z)
    add_surface_with_poslist(bldg, "GroundSurface", ground_poslist)

    # RoofSurface
    roof_poslist = poslist_from_ring(corners_xy, roof_z)
    add_surface_with_poslist(bldg, "RoofSurface", roof_poslist)

    # WallSurfaces + windows
    ring = list(corners_xy)
    if len(ring) < 3:
        continue
    # close ring if needed
    if not np.allclose(ring[0], ring[-1]):
        ring.append(ring[0])

    for i in range(len(ring) - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        # same window bands on every wall (simple assumption)
        add_wall_surface_with_windows(
            bldg, x1, y1, x2, y2, base_z, roof_z, window_bands
        )

print(f"\nCreated {bldg_count} LOD2 buildings with windows.")

# ---------- WRITE GML ----------
tree = ET.ElementTree(root)
os.makedirs(os.path.dirname(OUTPUT_GML), exist_ok=True)
tree.write(OUTPUT_GML, encoding="utf-8", xml_declaration=True)
print(f"✅ CityGML LOD2 with windows saved to: {OUTPUT_GML}")
