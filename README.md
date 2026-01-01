# LiDAR-Based Building Floor Estimation and CityGML Generation

## 1. Project Overview
This project presents a complete pipeline for automatically estimating the number of floors in buildings using LiDAR point cloud data and generating a CityGML LOD2 3D building model. The workflow starts from raw LiDAR preprocessing and ends with a semantic 3D city model suitable for urban analysis and visualization.

---

## 2. Input Data
- **LiDAR point cloud** in `.las` format
- LiDAR data may include ground, buildings, vegetation, and other objects

---

## 3. Preprocessing (Ground vs Non-Ground Separation)
1. The raw LiDAR file is loaded and visually inspected.
2. Ground points are separated from non-ground points using:
   - LAS classification (if available), or
   - Height-based filtering using percentile statistics.
3. Ground points are removed, retaining only above-ground objects such as buildings.

**Output:** Non-ground LiDAR points.

---

## 4. Building Clustering (Footprint Detection)
1. Non-ground points are projected onto the XY plane.
2. **DBSCAN clustering** is applied to group points belonging to individual buildings.
3. Noise points and very small clusters are discarded.
4. Each valid cluster represents one building footprint.

**Why DBSCAN?**
- Does not require a predefined number of buildings
- Robust to noise and varying point density

**Output:** Individual building clusters.

---

## 5. Feature Extraction
For each detected building cluster:
- Building height is computed using robust percentiles (e.g., 5th–95th percentile) to remove LiDAR noise.
- Footprint area is computed using the convex hull.
- Point density and vertical distribution are calculated.
- A vertical height histogram is generated to analyze point density across elevation.

**Extracted features include:**
- Total height
- Footprint area
- Point density
- Vertical density profile

---

## 6. Window Band Detection
1. Vertical height histograms are computed with a small bin size (e.g., 0.25 m).
2. Low-density gaps in the histogram are detected.
3. These gaps represent **window bands or floor separations**, while small gaps caused by ventilators are ignored using physical constraints.

**Output:** Number of detected window bands per building.

---

## 7. Floor Estimation Logic
Floor count is estimated by combining:
- **Height-based estimate**  
H = round(total_height / average_floor_height)
- **Window-band signal (W)** from detected vertical gaps

### Fusion rule:
- If |W − H| ≤ 1 → floors = average(H, W)
- Else → floors = H

Physical constraints are applied to ensure realistic floor heights.

**Result:** Robust and realistic floor count per building.

---

## 8. Machine Learning Refinement (Optional)
1. Extracted features are used to train a **Random Forest classifier**.
2. Hyperparameter tuning is performed using **GridSearchCV**.
3. The model prioritizes recall to avoid underestimating floors.
4. Final predictions are constrained using physical height limits.

**Output:** Final predicted number of floors.

---

## 9. Selection of Main Buildings
- Buildings are sorted by footprint area.
- The largest buildings (e.g., top 4 or top 5) are selected for CityGML modeling.

---

## 10. CityGML LOD2 Generation
1. Building footprints are converted into polygon geometry.
2. Buildings are extruded from ground level (z = 0) to roof height.
3. CityGML LOD2 components are created:
 - GroundSurface
 - WallSurface
 - RoofSurface
4. Semantic attributes such as:
 - Number of floors
 - Measured height  
 are added to each building.

**Output:** CityGML LOD2 `.gml` file.

---

## 11. Outputs
- CSV files with building features and floor estimates
- Trained Random Forest model (`.pkl`)
- Visualization plots (height vs floors, window gap histograms)
- CityGML LOD2 3D building model

---

## 12. Applications
- Urban 3D city modeling
- Smart city planning
- Building information extraction
- LiDAR-based geospatial analysis

---

## 13. Limitations
- Roof geometry is simplified due to limited LiDAR roof detail
- Window detection is indirect and may be affected by noise
- Accuracy depends on LiDAR point density and data quality
- Complex or curved footprints are approximated using convex hulls

---

## 14. Author
**Sreekanth**

