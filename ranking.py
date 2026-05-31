import json
import requests
import numpy as np
import pandas as pd
import folium
import folium.plugins
from scipy.spatial import cKDTree
from shapely.geometry import shape, Point

BASE = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action"

# ── Data sources — traceable provenance ───────────────────────────────────────
# Resource ids are resolved live against the CKAN API (print_provenance) so we
# never silently depend on a stale hardcoded UUID. Public page for each dataset is
# https://open.toronto.ca/dataset/<slug>/ ; licence is the Open Government Licence – Toronto.
SOURCES = {
    "traffic":      {"id": "6afa3b1f-f6a5-4235-8bd6-7568411c19f4", "role": "Dim 1 · traffic load"},
    "ksi":          {"id": "9c9a9b60-95c1-4541-ad44-15c4a643aff9", "role": "Dim 2 · collisions (KSI)"},
    "fire":         {"id": "9d1b7352-32ce-4af2-8681-595ce9e47b6e", "role": "Dim 3a · fire stations"},
    "police":       {"id": "4afc3c66-5614-466a-b714-e8d6336fc6d3", "role": "Dim 3a · police facilities"},
    "ltc":          {"id": "8316c8fb-1d08-45dc-b46f-257498ba6403", "role": "Dim 3b · long-term care"},
    "school":       {"id": "02ef7447-54d9-4aa7-b76d-8ef8138ac546", "role": "Dim 3c · schools"},
    "childcare":    {"id": "69403dde-a0b3-491d-9451-2338806a3bf0", "role": "Dim 3c · child care"},
    "library":      {"id": "7420a950-e62b-41da-826c-32d31c46e8f8", "role": "Dim 3d · libraries"},
    "recreation":   {"id": "e8cd0f4d-4910-42a0-81f9-cf8c2218753a", "role": "Dim 3d · parks & rec"},
    "nbhd_geom":    {"id": "5e6095fc-1bef-4776-887c-28d37f722c51", "role": "Dim 4 · nbhd boundaries"},
    "nbhd_profile": {"id": "7f8eee5e-85fb-415c-aef3-c3bd4998445f", "role": "Dim 4 · nbhd profiles"},
    "centreline":   {"id": "ad296ebf-fca6-4e67-b3ce-48040a20e6cd", "role": "map · Toronto Centreline"},
}
src = lambda key: SOURCES[key]["id"]


def print_provenance():
    print("Data provenance (resolved from CKAN API):")
    for key, meta in SOURCES.items():
        try:
            info = requests.get(f"{BASE}/resource_show", params={"id": meta["id"]}, timeout=20).json()["result"]
            pkg = requests.get(f"{BASE}/package_show", params={"id": info["package_id"]}, timeout=20).json()["result"]
            url = f"https://open.toronto.ca/dataset/{pkg.get('name', '')}/"
            print(f"  {key:12} {meta['role']:26} {url}  [{pkg.get('license_id') or 'n/a'}; {info.get('format')}]")
        except Exception as e:
            print(f"  {key:12} {meta['role']:26} (provenance lookup failed: {type(e).__name__})")

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

print_provenance()

print("Fetching traffic counts…")
traffic_df = pd.DataFrame(fetch_all(src("traffic")))
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
collision_df = pd.DataFrame(fetch_all(src("ksi")))
collision_df["lat"] = pd.to_numeric(collision_df["latitude"], errors="coerce")
collision_df["lon"] = pd.to_numeric(collision_df["longitude"], errors="coerce")
collision_df = collision_df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
# acclass is one of "Fatal Injury" / "Non-Fatal Injury" / "Property Damage Only".
# Use an exact match — str.contains("Fatal") also matches "Non-Fatal Injury"
# (substring), which would flag every injury collision as fatal.
collision_df["is_fatal"] = (
    collision_df["acclass"].str.strip().str.lower().eq("fatal injury").astype(int)
)

# =============================================================================
# FETCH — Dimension 3a: Emergency services
# =============================================================================

print("Fetching fire station locations…")
fire_df = geom_to_latlon(pd.DataFrame(fetch_all(src("fire"))))

print("Fetching police facility locations…")
police_df = geom_to_latlon(pd.DataFrame(fetch_all(src("police"))))

# =============================================================================
# FETCH — Dimension 3b: Vulnerable population care
# =============================================================================

print("Fetching long-term care locations…")
ltc_raw = pd.DataFrame(fetch_all(src("ltc")))
ltc_raw["beds"] = pd.to_numeric(ltc_raw["BEDS"], errors="coerce").fillna(0)
ltc_df = geom_to_latlon(ltc_raw)

# =============================================================================
# FETCH — Dimension 3c: Education & child care
# =============================================================================

print("Fetching school locations…")
school_raw = pd.DataFrame(fetch_all(src("school")))
school_df = geom_to_latlon(school_raw)

print("Fetching licensed childcare centres…")
childcare_raw = pd.DataFrame(fetch_all(src("childcare")))
childcare_raw["capacity"] = pd.to_numeric(childcare_raw["TOTSPACE"], errors="coerce").fillna(0)
childcare_df = geom_to_latlon(childcare_raw)

# =============================================================================
# FETCH — Dimension 3d: Community access
# =============================================================================

print("Fetching library branch locations…")
lib_raw = pd.DataFrame(fetch_all(src("library")))
lib_df = lib_raw.copy()
lib_df["lat"] = pd.to_numeric(lib_df["Lat"], errors="coerce")
lib_df["lon"] = pd.to_numeric(lib_df["Long"], errors="coerce")
lib_df = lib_df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

print("Fetching community centres (from Parks & Recreation facilities)…")
rec_raw = pd.DataFrame(fetch_all(src("recreation")))
cc_df = geom_to_latlon(rec_raw[rec_raw["TYPE"] == "Community Centre"].copy())

# =============================================================================
# FETCH — Dimension 4: Population density
# =============================================================================

print("Fetching neighbourhood geometries…")
nbhd_df = pd.DataFrame(fetch_all(src("nbhd_geom")))
nbhd_shapes = []
for _, row in nbhd_df.iterrows():
    try:
        poly = shape(json.loads(row["geometry"]))
        nbhd_shapes.append((row["AREA_NAME"], poly))
    except Exception:
        pass

print("Fetching neighbourhood profiles…")
profile_df = pd.DataFrame(fetch_all(src("nbhd_profile")))
nbhd_cols = [
    c for c in profile_df.columns
    if c not in ["_id", "Category", "Topic", "Data Source", "Characteristic", "City of Toronto"]
]
density_row = profile_df[
    profile_df["Characteristic"].str.contains("Population density per square", na=False)
].head(1)
if not density_row.empty:
    # Density values arrive as thousands-separated strings ("3,929"); pd.to_numeric
    # without stripping the commas coerces every value to NaN, which silently zeroed
    # out the entire density dimension (its 0.10 weight contributed nothing).
    nbhd_density = pd.to_numeric(
        density_row[nbhd_cols].iloc[0].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).dropna()
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

# KSI is PERSON-level (~2.7 rows per collision), so dedupe collision_id to count
# distinct COLLISIONS, not person-involvements. Fatal = distinct collisions whose
# acclass is "Fatal Injury" (is_fatal is constant within a collision_id).
def _coll_counts(idx):
    if not idx:
        return 0.0, 0.0
    sub = collision_df.iloc[idx]
    return (float(sub["collision_id"].nunique()),
            float(sub.loc[sub["is_fatal"] == 1, "collision_id"].nunique()))

_cc = [_coll_counts(idx) for idx in near_collisions]
collision_counts = np.array([c for c, _ in _cc], dtype=float)
fatal_counts     = np.array([f for _, f in _cc], dtype=float)
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

import matplotlib              # continuous colour gradient for the road lines
import branca.colormap as bcm  # gradient legend

# Render risk as a continuous gradient painted along the centreline network.
# Each road segment inherits the priority score of its NEAREST scored intersection,
# extrapolating the point-based score across the whole network (the "gradient").
# Deterioration risk: the validated predictionmap "CrackWatch TO" model output,
# calibrated forward-180-day pothole risk per Centreline SEGMENT (keyed CENTRELINE_ID).
RISK_MAP_PATH = "../ground-truth/predictionmap/risk_map.csv"
W_PRIORITY, W_RISK = 0.5, 0.5   # segment-composite blend: community priority vs deterioration risk

print("\nFetching centreline network for the map…")
tcl = pd.DataFrame(fetch_all(src("centreline")))

# Keep road-carrying centrelines; drop rivers/trails/rail/hydro/shoreline/etc.
_NON_ROAD = ("river", "creek", "trail", "walkway", "cycle", "hydro",
             "ferry", "geostat", "rail", "shoreline")
_desc = tcl["FEATURE_CODE_DESC"].fillna("").str.lower()
tcl = tcl[~_desc.apply(lambda d: any(k in d for k in _NON_ROAD))].copy()


def _line_coords(g):
    if isinstance(g, str):
        g = json.loads(g)
    return g["coordinates"] if g.get("type") == "LineString" else None


tcl["coords"] = tcl["geometry"].apply(_line_coords)
tcl = tcl[tcl["coords"].apply(lambda c: c is not None and len(c) >= 2)].reset_index(drop=True)
tcl["CENTRELINE_ID"] = pd.to_numeric(tcl["CENTRELINE_ID"], errors="coerce")

# Nearest scored intersection for each segment (query at the segment midpoint)
mids   = np.array([c[len(c) // 2] for c in tcl["coords"]])   # [lon, lat]
seg_xy = to_local_xy(mids[:, 1], mids[:, 0])
_, nn  = int_tree.query(seg_xy, k=1)                          # index into traffic_df / priority

def _nrm(a):
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

our_norm = _nrm(priority[nn])   # our community/traffic/collision composite, 0-1

# Join the validated deterioration risk (proba_high) by Centreline SEGMENT id.
try:
    _rm = pd.read_csv(RISK_MAP_PATH)
    _risk = _rm.groupby("centreline_id")["proba_high"].mean().to_dict()
    risk = np.array([_risk.get(int(c), np.nan) if pd.notna(c) else np.nan
                     for c in tcl["CENTRELINE_ID"]], dtype=float)
    _m = int(np.isfinite(risk).sum())
    print(f"Deterioration risk joined: {_m:,}/{len(tcl):,} segments ({_m / len(tcl):.0%})  <- {RISK_MAP_PATH}")
except FileNotFoundError:
    risk = np.full(len(tcl), np.nan)
    print(f"  (predictionmap risk_map not found at {RISK_MAP_PATH}; colouring by priority only)")

# Segment composite: blend deterioration risk with our priority where risk exists.
combined = np.where(np.isfinite(risk), W_PRIORITY * our_norm + W_RISK * risk, our_norm)
seg_norm = _nrm(combined)
seg_pct = seg_norm.argsort().argsort() / max(len(seg_norm) - 1, 1) * 100  # 0-100 percentile rank
s_lo, s_hi = 0.0, 1.0

cmap = matplotlib.colormaps["RdYlGn_r"]   # green (low) → yellow → orange → red (high)


def _hex(v):
    r, g, b, _ = cmap(float(v))
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


seg_hex = [_hex(v) for v in seg_norm]
names, colls = traffic_df["location_name"].values, traffic_df["collision_count"].values
fatals, vehs = traffic_df["fatal_count"].values, traffic_df["total_vehicle"].values
seg_names = tcl["LINEAR_NAME_FULL"].fillna("").astype(str).values
cids = tcl["CENTRELINE_ID"].values

print("Building map…")
m = folium.Map(location=[43.7, -79.4], zoom_start=12, tiles="CartoDB dark_matter")

features = []
for i, coords in enumerate(tcl["coords"]):
    j = int(nn[i])
    features.append({
        "type": "Feature",
        "geometry": {"type": "LineString",
                     "coordinates": [[round(x, 5), round(y, 5)] for x, y in coords]},
        "properties": {
            "color": seg_hex[i],
            "centreline_id": int(cids[i]) if pd.notna(cids[i]) else None,
            "street": seg_names[i] or str(names[j]),
            "priority": round(float(seg_norm[i]), 3),
            "priority_pct": int(round(seg_pct[i])),
            "risk_high": (round(float(risk[i]), 3) if np.isfinite(risk[i]) else None),
            "nearest": str(names[j]),
            "collisions": int(colls[j]),
            "fatals": int(fatals[j]),
            "vehicles": int(vehs[j]),
        },
    })

folium.GeoJson(
    {"type": "FeatureCollection", "features": features},
    style_function=lambda f: {"color": f["properties"]["color"], "weight": 2.5, "opacity": 0.85},
    highlight_function=lambda f: {"weight": 5, "opacity": 1.0},
    tooltip=folium.GeoJsonTooltip(
        fields=["street", "priority_pct", "vehicles"],
        aliases=["Street", "Priority (percentile)", "Daily vehicles"],
    ),
    name="Road risk (gradient)",
).add_to(m)

legend = bcm.LinearColormap(
    [_hex(t) for t in np.linspace(0, 1, 8)], vmin=s_lo, vmax=s_hi,
    caption="Segment priority — community impact blended with deterioration risk (0-1)",
)
legend.add_to(m)

m.save("road_priority_map.html")
print(f"Map saved → road_priority_map.html  ({len(features):,} coloured road segments)")

# ── Export a compact per-segment GeoJSON for the Cracked City app ─────────────
# Popup shows Street · Priority (percentile) · Daily vehicles; `priority` (0-1) is
# kept only so the map can colour the line. The app colours/labels from these.
from shapely.geometry import LineString as _LS
def _simp(coords, tol=0.0001):   # Douglas-Peucker (~11 m) — drops collinear vertices
    if len(coords) <= 2:
        return [[round(x, 5), round(y, 5)] for x, y in coords]
    return [[round(x, 5), round(y, 5)] for x, y in _LS(coords).simplify(tol).coords]

app_fc = {"type": "FeatureCollection", "features": [
    {"type": "Feature",
     "geometry": {"type": "LineString", "coordinates": _simp(f["geometry"]["coordinates"])},
     "properties": {k: f["properties"][k] for k in ("street", "priority_pct", "vehicles")}}
    for f in features]}
with open("road_priority_segments.geojson", "w") as _fh:
    json.dump(app_fc, _fh, separators=(",", ":"))   # compact: ~9.4 MB, under HF's 10 MB non-LFS limit
print(f"App artifact saved → road_priority_segments.geojson  ({len(features):,} segments)")
