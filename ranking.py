import json
import requests
import numpy as np
import pandas as pd
import folium
import folium.plugins
from scipy.spatial import cKDTree
from shapely.geometry import shape, Point

BASE = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action"

# =============================================================================
# DATASET INVENTORY & ROLES
# =============================================================================
# Four scoring dimensions drive the composite priority score.
# Each dataset contributes to exactly one dimension.
#
# DIMENSION 1 — Traffic load (weight 0.35)
#   "Traffic Signal Vehicle and Pedestrian Volumes"  [6afa3b1f-...]
#   Daily vehicle counts with heavy-truck percentage. Trucks damage pavement
#   ~16× more per pass than cars (4th-power axle law), so wear-adjusted volume
#   is total_vehicle × (1 + 3 × heavy_pct). Pedestrian count is retained for
#   the map popup but does not feed the score directly.
#
# DIMENSION 2 — Collision severity (weight 0.35)
#   "Motor Vehicle Collisions Involving Killed or Seriously Injured Persons"
#   [9c9a9b60-...]
#   Historical record of injury-level and fatal collisions within 75 m of each
#   intersection. Fatal collisions are weighted 5× to reflect severity, not just
#   frequency. This dimension captures latent danger independent of traffic volume.
#
# DIMENSION 3 — Institution impact factor (weight 0.20)
#   Measures how much social harm road degradation causes by gauging what
#   institutions depend on nearby roads. Split across four sub-components:
#
#   3a. Emergency services — 35% of impact factor (radius 500 m)
#       "Fire Station Locations"     [9d1b7352-...]
#       "Police Facility Locations"  [4afc3c66-...]
#       Emergency vehicles require trafficable roads to meet response-time
#       targets. A failed road surface near a fire hall or division can add
#       minutes to response, directly affecting life-safety outcomes.
#
#   3b. Vulnerable population care — 30% of impact factor (radius 400 m)
#       "Long Term Care Locations (City-Operated)"  [8316c8fb-...]
#       LTC residents cannot self-evacuate and depend on reliable road access
#       for ambulance response, medical supply delivery, and family visits.
#       Score is bed-count weighted: a 200-bed facility warrants more road
#       investment than a 20-bed facility on the same block.
#
#   3c. Education & child care — 20% of impact factor (radius 400 m)
#       "School Locations – All Types"      [02ef7447-...]
#       "Licensed Child Care Centres"       [69403dde-...]
#       Schools and child care centres generate concentrated child pedestrian
#       activity at peak hours. Road surface failures (potholes, cracking) near
#       these sites increase injury risk for children crossing on foot or by
#       bicycle. Childcare centres are weighted 0.6× relative to schools to
#       reflect smaller catchment areas.
#
#   3d. Community access — 15% of impact factor (radius 400 m)
#       "Library Branch General Information"   [7420a950-...]
#       "Parks and Recreation Facilities"      [e8cd0f4d-..., Community Centre only]
#       Libraries and community centres serve as mobility anchors: they attract
#       seniors, youth in programming, and residents without private vehicles.
#       Degraded road access disproportionately affects those who walk, use
#       mobility aids, or rely on accessible transit stops near these sites.
#
# DIMENSION 4 — Population density (weight 0.10)
#   "Neighbourhood Profiles"  [7f8eee5e-...]
#   "Neighbourhood Boundary File"  [5e6095fc-...]
#   Ambient demand multiplier. Denser neighbourhoods have more road interactions
#   per unit area, so equivalent road degradation affects more residents.
# =============================================================================


# --- helpers ---

def fetch_all(resource_id, batch=5000):
    records, offset = [], 0
    while True:
        r = requests.get(
            f"{BASE}/datastore_search",
            params={"resource_id": resource_id, "limit": batch, "offset": offset},
            timeout=30,
        )
        result = r.json()["result"]
        records.extend(result["records"])
        if len(records) >= result["total"]:
            break
        offset += len(result["records"])
    return records


def parse_point_geom(g):
    if isinstance(g, str):
        g = json.loads(g)
    lon, lat = g["coordinates"]
    return lat, lon


def to_local_xy(lats, lons, ref_lat=43.7, ref_lon=-79.4):
    x = (lons - ref_lon) * np.cos(np.radians(ref_lat)) * 111_320
    y = (lats - ref_lat) * 110_540
    return np.column_stack([x, y])


def normalize(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr, dtype=float)


def score_to_hex(score):
    r = int(220 * score)
    g = int(200 * (1 - score))
    return f"#{r:02x}{g:02x}20"


def geom_to_latlon(df, geom_col="geometry"):
    coords = df[geom_col].apply(parse_point_geom)
    df = df.copy()
    df["lat"] = coords.apply(lambda x: x[0])
    df["lon"] = coords.apply(lambda x: x[1])
    return df.dropna(subset=["lat", "lon"]).reset_index(drop=True)


# =============================================================================
# FETCH — Dimension 1: Traffic load
# =============================================================================

print("Fetching traffic counts…")
traffic_df = pd.DataFrame(fetch_all("6afa3b1f-f6a5-4235-8bd6-7568411c19f4"))
traffic_df["lat"] = pd.to_numeric(traffic_df["latitude"], errors="coerce")
traffic_df["lon"] = pd.to_numeric(traffic_df["longitude"], errors="coerce")
traffic_df["total_vehicle"] = pd.to_numeric(traffic_df["total_vehicle"], errors="coerce")
traffic_df["total_pedestrian"] = pd.to_numeric(traffic_df["total_pedestrian"], errors="coerce")
traffic_df["total_heavy_pct"] = pd.to_numeric(traffic_df["total_heavy_pct"], errors="coerce")
traffic_df = traffic_df.dropna(subset=["lat", "lon", "total_vehicle"]).reset_index(drop=True)

# =============================================================================
# FETCH — Dimension 2: Collision severity
# =============================================================================

print("Fetching collision data…")
collision_df = pd.DataFrame(fetch_all("9c9a9b60-95c1-4541-ad44-15c4a643aff9"))
collision_df["lat"] = pd.to_numeric(collision_df["latitude"], errors="coerce")
collision_df["lon"] = pd.to_numeric(collision_df["longitude"], errors="coerce")
collision_df = collision_df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
collision_df["is_fatal"] = collision_df["acclass"].str.contains("Fatal", na=False).astype(int)

# =============================================================================
# FETCH — Dimension 3a: Emergency services
# =============================================================================

print("Fetching fire station locations…")
fire_df = geom_to_latlon(pd.DataFrame(fetch_all("9d1b7352-32ce-4af2-8681-595ce9e47b6e")))

print("Fetching police facility locations…")
police_df = geom_to_latlon(pd.DataFrame(fetch_all("4afc3c66-5614-466a-b714-e8d6336fc6d3")))

# =============================================================================
# FETCH — Dimension 3b: Vulnerable population care
# =============================================================================

print("Fetching long-term care locations…")
ltc_raw = pd.DataFrame(fetch_all("8316c8fb-1d08-45dc-b46f-257498ba6403"))
ltc_raw["beds"] = pd.to_numeric(ltc_raw["BEDS"], errors="coerce").fillna(0)
ltc_df = geom_to_latlon(ltc_raw)

# =============================================================================
# FETCH — Dimension 3c: Education & child care
# =============================================================================

print("Fetching school locations…")
school_raw = pd.DataFrame(fetch_all("02ef7447-54d9-4aa7-b76d-8ef8138ac546"))
school_df = geom_to_latlon(school_raw)

print("Fetching licensed childcare centres…")
childcare_raw = pd.DataFrame(fetch_all("69403dde-a0b3-491d-9451-2338806a3bf0"))
childcare_raw["capacity"] = pd.to_numeric(childcare_raw["TOTSPACE"], errors="coerce").fillna(0)
childcare_df = geom_to_latlon(childcare_raw)

# =============================================================================
# FETCH — Dimension 3d: Community access
# =============================================================================

print("Fetching library branch locations…")
lib_raw = pd.DataFrame(fetch_all("7420a950-e62b-41da-826c-32d31c46e8f8"))
lib_df = lib_raw.copy()
lib_df["lat"] = pd.to_numeric(lib_df["Lat"], errors="coerce")
lib_df["lon"] = pd.to_numeric(lib_df["Long"], errors="coerce")
lib_df = lib_df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

print("Fetching community centres (from Parks & Recreation facilities)…")
rec_raw = pd.DataFrame(fetch_all("e8cd0f4d-4910-42a0-81f9-cf8c2218753a"))
cc_df = geom_to_latlon(rec_raw[rec_raw["TYPE"] == "Community Centre"].copy())

# =============================================================================
# FETCH — Dimension 4: Population density
# =============================================================================

print("Fetching neighbourhood geometries…")
nbhd_df = pd.DataFrame(fetch_all("5e6095fc-1bef-4776-887c-28d37f722c51"))
nbhd_shapes = []
for _, row in nbhd_df.iterrows():
    try:
        poly = shape(json.loads(row["geometry"]))
        nbhd_shapes.append((row["AREA_NAME"], poly))
    except Exception:
        pass

print("Fetching neighbourhood profiles…")
profile_df = pd.DataFrame(fetch_all("7f8eee5e-85fb-415c-aef3-c3bd4998445f"))
nbhd_cols = [
    c for c in profile_df.columns
    if c not in ["_id", "Category", "Topic", "Data Source", "Characteristic", "City of Toronto"]
]
density_row = profile_df[
    profile_df["Characteristic"].str.contains("Population density per square", na=False)
].head(1)
if not density_row.empty:
    nbhd_density = (
        density_row[nbhd_cols].iloc[0]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
else:
    nbhd_density = pd.Series(dtype=float)


# =============================================================================
# SPATIAL INDICES
# =============================================================================

print("Building spatial indices…")
int_xy       = to_local_xy(traffic_df["lat"].values,  traffic_df["lon"].values)
collision_xy = to_local_xy(collision_df["lat"].values, collision_df["lon"].values)
school_xy    = to_local_xy(school_df["lat"].values,    school_df["lon"].values)
childcare_xy = to_local_xy(childcare_df["lat"].values, childcare_df["lon"].values)
lib_xy       = to_local_xy(lib_df["lat"].values,       lib_df["lon"].values)
cc_xy        = to_local_xy(cc_df["lat"].values,        cc_df["lon"].values)
fire_xy      = to_local_xy(fire_df["lat"].values,      fire_df["lon"].values)
police_xy    = to_local_xy(police_df["lat"].values,    police_df["lon"].values)
ltc_xy       = to_local_xy(ltc_df["lat"].values,       ltc_df["lon"].values)

int_tree       = cKDTree(int_xy)
collision_tree = cKDTree(collision_xy)
school_tree    = cKDTree(school_xy)
childcare_tree = cKDTree(childcare_xy)
lib_tree       = cKDTree(lib_xy)
cc_tree        = cKDTree(cc_xy)
fire_tree      = cKDTree(fire_xy)
police_tree    = cKDTree(police_xy)
ltc_tree       = cKDTree(ltc_xy)

nbhd_centroids_xy = to_local_xy(
    np.array([p.centroid.y for _, p in nbhd_shapes]),
    np.array([p.centroid.x for _, p in nbhd_shapes]),
)
nbhd_centroid_tree = cKDTree(nbhd_centroids_xy)


# =============================================================================
# SCORE 1 — Traffic load
# =============================================================================

heavy_pct = traffic_df["total_heavy_pct"].fillna(0).values / 100
traffic_score_raw = traffic_df["total_vehicle"].values * (1 + 3 * heavy_pct)


# =============================================================================
# SCORE 2 — Collision severity
# =============================================================================

print("Computing collision scores…")
near_collisions  = int_tree.query_ball_tree(collision_tree, r=75)
collision_counts = np.array([len(idx) for idx in near_collisions], dtype=float)
fatal_counts     = np.array([
    int(collision_df.iloc[idx]["is_fatal"].sum()) if idx else 0
    for idx in near_collisions
], dtype=float)
collision_score_raw = collision_counts + 4 * fatal_counts


# =============================================================================
# SCORE 3 — Institution impact factor
# =============================================================================

print("Computing institution impact factor…")

# 3a — Emergency services (fire + police, 500 m)
near_fire   = int_tree.query_ball_tree(fire_tree,   r=500)
near_police = int_tree.query_ball_tree(police_tree, r=500)
n_emergency = np.array([len(f) + len(p) for f, p in zip(near_fire, near_police)], dtype=float)

# 3b — Vulnerable population care: sum of LTC beds within 400 m
near_ltc    = int_tree.query_ball_tree(ltc_tree, r=400)
ltc_beds    = np.array([
    float(ltc_df.iloc[idx]["beds"].sum()) if idx else 0.0
    for idx in near_ltc
], dtype=float)

# 3c — Education: schools + childcare (400 m)
# Childcare centres are weighted 0.6 relative to schools: similar vulnerability,
# smaller catchment and lower volumes.
near_schools   = int_tree.query_ball_tree(school_tree,    r=400)
near_childcare = int_tree.query_ball_tree(childcare_tree, r=400)
n_schools      = np.array([len(idx) for idx in near_schools],   dtype=float)
n_childcare    = np.array([len(idx) for idx in near_childcare], dtype=float)
education_raw  = n_schools + 0.6 * n_childcare

# 3d — Community access: libraries + community centres (400 m)
near_lib = int_tree.query_ball_tree(lib_tree, r=400)
near_cc  = int_tree.query_ball_tree(cc_tree,  r=400)
n_lib    = np.array([len(idx) for idx in near_lib], dtype=float)
n_cc     = np.array([len(idx) for idx in near_cc],  dtype=float)
community_raw = n_lib + n_cc

# Weighted impact factor (each sub-component normalized 0-1 first)
impact_score = (
    0.35 * normalize(n_emergency)    +   # emergency services
    0.30 * normalize(ltc_beds)       +   # vulnerable population care
    0.20 * normalize(education_raw)  +   # education & child care
    0.15 * normalize(community_raw)      # community access
)


# =============================================================================
# SCORE 4 — Population density
# =============================================================================

print("Assigning neighbourhoods…")
_, cand_idx = nbhd_centroid_tree.query(int_xy, k=5)
assigned_density = np.zeros(len(traffic_df))
for i, (lat, lon) in enumerate(zip(traffic_df["lat"].values, traffic_df["lon"].values)):
    pt = Point(lon, lat)
    for j in cand_idx[i]:
        name, poly = nbhd_shapes[j]
        if poly.contains(pt):
            assigned_density[i] = nbhd_density.get(name, 0)
            break


# =============================================================================
# COMPOSITE SCORE
# =============================================================================

priority = (
    0.35 * normalize(traffic_score_raw)   +
    0.35 * normalize(collision_score_raw) +
    0.20 * impact_score                   +
    0.10 * normalize(assigned_density)
)

traffic_df["score_traffic"]   = normalize(traffic_score_raw)
traffic_df["score_collision"] = normalize(collision_score_raw)
traffic_df["score_impact"]    = impact_score
traffic_df["score_density"]   = normalize(assigned_density)
traffic_df["priority_score"]  = priority
traffic_df["collision_count"] = collision_counts.astype(int)
traffic_df["fatal_count"]     = fatal_counts.astype(int)
traffic_df["n_schools"]       = n_schools.astype(int)
traffic_df["n_childcare"]     = n_childcare.astype(int)
traffic_df["n_libraries"]     = n_lib.astype(int)
traffic_df["n_community_centres"] = n_cc.astype(int)
traffic_df["n_emergency_services"] = n_emergency.astype(int)
traffic_df["ltc_beds_nearby"] = ltc_beds.astype(int)


# =============================================================================
# OUTPUT
# =============================================================================

out_cols = [
    "location_name", "lat", "lon", "centreline_id", "centreline_type",
    "total_vehicle", "total_pedestrian",
    "collision_count", "fatal_count",
    "n_schools", "n_childcare", "n_libraries", "n_community_centres",
    "n_emergency_services", "ltc_beds_nearby",
    "score_traffic", "score_collision", "score_impact", "score_density",
    "priority_score",
]
scored = traffic_df[out_cols].sort_values("priority_score", ascending=False).reset_index(drop=True)
scored.to_csv("intersection_scores.csv", index=False)
print(f"\nSaved {len(scored)} locations → intersection_scores.csv")

print("\nTop 15 priority intersections:")
print(
    scored.head(15)[[
        "location_name", "priority_score",
        "score_traffic", "score_collision", "score_impact",
    ]].to_string(index=False)
)

print("\nInstitution dataset summary:")
print(f"  Fire stations:      {len(fire_df)}")
print(f"  Police facilities:  {len(police_df)}")
print(f"  LTC facilities:     {len(ltc_df)}  ({int(ltc_df['beds'].sum())} total beds)")
print(f"  Schools:            {len(school_df)}")
print(f"  Childcare centres:  {len(childcare_df)}")
print(f"  Libraries:          {len(lib_df)}")
print(f"  Community centres:  {len(cc_df)}")


# =============================================================================
# MAP
# =============================================================================

print("\nBuilding map…")
m = folium.Map(location=[43.7, -79.4], zoom_start=12, tiles="CartoDB positron")

heat_data = scored[["lat", "lon", "priority_score"]].values.tolist()
folium.plugins.HeatMap(
    heat_data,
    min_opacity=0.3,
    max_val=1.0,
    radius=22,
    blur=18,
    gradient={0.0: "green", 0.4: "yellow", 0.7: "orange", 1.0: "red"},
).add_to(m)

top100 = scored.head(100)
for rank, (_, row) in enumerate(top100.iterrows(), 1):
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=6,
        color=score_to_hex(row["priority_score"]),
        fill=True,
        fill_opacity=0.85,
        weight=1,
        popup=folium.Popup(
            f"<b>#{rank} {row['location_name']}</b><br>"
            f"Priority: <b>{row['priority_score']:.3f}</b><br>"
            f"Daily vehicles: {int(row['total_vehicle']):,}<br>"
            f"Collisions nearby: {int(row['collision_count'])} ({int(row['fatal_count'])} fatal)<br>"
            f"Impact factor: {row['score_impact']:.3f} "
            f"(emergency: {int(row['n_emergency_services'])}, "
            f"LTC beds: {int(row['ltc_beds_nearby'])}, "
            f"schools: {int(row['n_schools'])}, "
            f"childcare: {int(row['n_childcare'])}, "
            f"libraries: {int(row['n_libraries'])}, "
            f"community centres: {int(row['n_community_centres'])})",
            max_width=320,
        ),
    ).add_to(m)

m.save("road_priority_map.html")
print("Map saved → road_priority_map.html")
