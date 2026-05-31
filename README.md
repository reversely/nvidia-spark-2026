# Road Priority Algorithm

Analyzes and visualizes road-repair priority across Toronto using City of Toronto open data.

Two analyses over City of Toronto open data:

1. **Intersection priority ranking** ([`ranking.py`](ranking.py)) — scores ~6,400 signalized
   intersections 0–1 on how badly the surrounding road network needs investment, blending
   traffic load, collision history, nearby-institution dependence, and population density.
2. **Economic impact of the pothole backlog** ([`economic_impact.py`](economic_impact.py)) —
   builds a ward × month panel of 311 road-surface complaints vs. KSI crashes, tests whether
   backlog predicts crashes (it doesn't, significantly), and values poor roads through the
   defensible vehicle-operating-cost channel.

All inputs are pulled live from the City of Toronto CKAN API (Open Government Licence – Toronto);
resource IDs are resolved at runtime so the pipeline never depends on a stale UUID.

## Data sources

| Dataset | Resource ID | Used by | Public page |
|---------|-------------|---------|-------------|
| Traffic Signal Vehicle & Pedestrian Volumes | `6afa3b1f-f6a5-4235-8bd6-7568411c19f4` | ranking — Dim 1 | [open.toronto.ca](https://open.toronto.ca/dataset/traffic-signal-vehicle-and-pedestrian-volumes/) |
| Motor Vehicle Collisions (KSI) | `9c9a9b60-95c1-4541-ad44-15c4a643aff9` | ranking — Dim 2; econ | [open.toronto.ca](https://open.toronto.ca/dataset/motor-vehicle-collisions-involving-killed-or-seriously-injured-persons/) |
| Fire Station Locations | `9d1b7352-32ce-4af2-8681-595ce9e47b6e` | ranking — Dim 3a | [open.toronto.ca](https://open.toronto.ca/dataset/fire-station-locations/) |
| Police Facility Locations | `4afc3c66-5614-466a-b714-e8d6336fc6d3` | ranking — Dim 3a | [open.toronto.ca](https://open.toronto.ca/dataset/police-facility-locations/) |
| Long-Term Care Locations (City-Operated) | `8316c8fb-1d08-45dc-b46f-257498ba6403` | ranking — Dim 3b | [open.toronto.ca](https://open.toronto.ca/dataset/long-term-care-locations-city-operated/) |
| School Locations (All Types) | `02ef7447-54d9-4aa7-b76d-8ef8138ac546` | ranking — Dim 3c | [open.toronto.ca](https://open.toronto.ca/dataset/school-locations-all-types/) |
| Licensed Child Care Centres | `69403dde-a0b3-491d-9451-2338806a3bf0` | ranking — Dim 3c | [open.toronto.ca](https://open.toronto.ca/dataset/licensed-child-care-centres/) |
| Library Branch General Information | `7420a950-e62b-41da-826c-32d31c46e8f8` | ranking — Dim 3d | [open.toronto.ca](https://open.toronto.ca/dataset/library-branch-general-information/) |
| Parks & Recreation Facilities (Community Centres) | `e8cd0f4d-4910-42a0-81f9-cf8c2218753a` | ranking — Dim 3d | [open.toronto.ca](https://open.toronto.ca/dataset/parks-and-recreation-facilities/) |
| Neighbourhood Profiles | `7f8eee5e-85fb-415c-aef3-c3bd4998445f` | ranking — Dim 4 | [open.toronto.ca](https://open.toronto.ca/dataset/neighbourhood-profiles/) |
| Neighbourhood Boundary File | `5e6095fc-1bef-4776-887c-28d37f722c51` | ranking — Dim 4 | [open.toronto.ca](https://open.toronto.ca/dataset/neighbourhoods/) |
| Toronto Centreline (TCL) | `ad296ebf-fca6-4e67-b3ce-48040a20e6cd` | ranking — map | [open.toronto.ca](https://open.toronto.ca/dataset/toronto-centreline-tcl/) |
| 311 Service Requests (Customer Initiated) | slug `311-service-requests-customer-initiated` (yearly ZIPs) | econ | [open.toronto.ca](https://open.toronto.ca/dataset/311-service-requests-customer-initiated/) |
| City Wards (25-ward) | `737b29e0-8329-4260-b6af-21555ab24f28` | econ | [open.toronto.ca](https://open.toronto.ca/dataset/city-wards/) |

External (non-CKAN): **CrackWatch TO** deterioration risk — called from the dyno inference model
(map blend) — and **Ontario MTO Vehicle Population Data** ([data.ontario.ca](https://data.ontario.ca/dataset/vehicle-population-data), econ denominator).

---

## 1. Intersection priority ranking

`ranking.py` → [`intersection_scores.csv`](intersection_scores.csv),
[`road_priority_map.html`](road_priority_map.html), `road_priority_segments.geojson`.

### Composite score

Each intersection's `priority_score` (0–1) is a weighted blend of four normalized dimensions
([`ranking.py:396-401`](ranking.py#L396-L401)):

```python
priority = (
    0.35 * normalize(traffic_score_raw)   +   # Dim 1 — traffic load
    0.35 * normalize(collision_score_raw) +   # Dim 2 — collision severity
    0.20 * impact_score                   +   # Dim 3 — institution impact
    0.10 * normalize(assigned_density)        # Dim 4 — population density
)
```

Every component is min-max normalized to 0–1 *before* being weighted.

### Datasets and weights

| Dim | Weight | Dataset (CKAN) | Fields used | Computation |
|-----|--------|----------------|-------------|-------------|
| **1. Traffic load** | **0.35** | Traffic Signal Vehicle & Pedestrian Volumes | `total_vehicle`, `total_heavy_pct` | `total_vehicle × (1 + 3·heavy_pct)` — heavy trucks damage pavement ~4× (4th-power axle law). Pedestrian counts are kept for map popups but **not** scored. |
| **2. Collision severity** | **0.35** | Motor Vehicle Collisions (Killed or Seriously Injured) | `latitude`, `longitude`, `acclass` | Unique collisions within **75 m**, fatal weighted **5×**: `collisions + 4·fatals`. KSI is person-level, so `collision_id` is deduped to count distinct collisions. |
| **3. Institution impact** | **0.20** | *(four sub-components, see below)* | — | weighted blend of normalized sub-scores |
| **4. Population density** | **0.10** | Neighbourhood Profiles + Neighbourhood Boundary File | persons / km² | density of the neighbourhood polygon each intersection falls within |

**Dimension 3 — institution impact factor** is itself a weighted blend
([`ranking.py:368-373`](ranking.py#L368-L373)):

```python
impact_score = (
    0.35 * normalize(n_emergency)    +   # 3a fire + police within 500 m
    0.30 * normalize(ltc_beds)       +   # 3b LTC beds within 400 m (bed-weighted)
    0.20 * normalize(education_raw)  +   # 3c schools + 0.6·childcare within 400 m
    0.15 * normalize(community_raw)      # 3d libraries + community centres within 400 m
)
```

| Sub | Dataset(s) | Radius | Effective weight | Notes |
|-----|-----------|--------|------------------|-------|
| **3a Emergency services** | Fire Station Locations + Police Facility Locations | 500 m | 0.20 × 0.35 = **0.07** | count of facilities nearby |
| **3b Vulnerable population** | Long-Term Care Locations (City-Operated) | 400 m | 0.20 × 0.30 = **0.06** | sum of `BEDS` — a 200-bed home outweighs a 20-bed one |
| **3c Education & child care** | School Locations (All Types) + Licensed Child Care Centres | 400 m | 0.20 × 0.20 = **0.04** | schools 1×, childcare 0.6× (smaller catchment) |
| **3d Community access** | Library Branches + Parks & Rec (Community Centres only) | 400 m | 0.20 × 0.15 = **0.03** | count of libraries + community centres |

### Output columns (`intersection_scores.csv`)

`priority_score`, the four component sub-scores (`score_traffic`, `score_collision`,
`score_impact`, `score_density`), plus raw reference counts (`collision_count`, `fatal_count`,
`n_schools`, `n_childcare`, `n_libraries`, `n_commuhynity_centres`, `n_emergency_services`,
`ltc_beds_nearby`, `total_vehicle`, `total_pedestrian`).

### The map blends in a 5th source

`road_priority_map.html` does **not** simply paint `priority_score`. Each centreline road
segment inherits its nearest intersection's score, then blends in a deterioration-risk model
([`ranking.py:465-512`](ranking.py#L465-L512)):

```python
W_PRIORITY, W_RISK = 0.5, 0.5
combined = 0.5 * our_priority + 0.5 * deterioration_risk   # where risk exists
```

`deterioration_risk` (`proba_high`) is called from the dyno inference model (the "CrackWatch TO"
forward-180-day pothole-risk model, keyed by `CENTRELINE_ID`). So the **map weighting differs
from the intersection-score weighting**.

---

## 2. Economic impact of the pothole backlog

`economic_impact.py` (consolidated from `pothole_economic_impact.ipynb`) →
`ward_economic_exposure.geojson`. This is a **separate** pipeline from the ranking.

Ward × month panel built from three streams:

1. **311 Service Requests (Customer Initiated)** — pothole / road-surface complaints, filtered to
   Transportation Services, keyword-matched, by ward × month. Restricted to the 25-ward era
   (post-2018) for boundary consistency.
2. **KSI Collisions** — same dataset as Dim 2 above, spatially joined to wards and aggregated to
   ward × month.
3. **`intersection_scores.csv`** — the ranking output, aggregated to a ward-level road-risk
   baseline (mean priority + share of top-quartile "high-risk" intersections).

### Findings & valuation

- A fixed-effects OLS (ward FE + month FE, HC3 SE) regresses crashes on 1–3-month lagged backlog.
  **The lagged-backlog coefficient is not statistically significant** — the 311→crash link is null.
- **Headline figure** therefore runs through the **vehicle-operating-cost channel** (CAA/CPCS):
  ~**$126/vehicle/yr** × Toronto's registered passenger vehicles (Ontario MTO data) ≈ **$150M/yr**.
- The crash-attributed dollar figure is computed but flagged **illustrative only**.
- The ranking enters here as a **deferred-maintenance multiplier**: wards in the top quartile of
  high-risk-intersection share get a **4.5×** multiplier on attributable crash cost
  ([`economic_impact.py:843-849`](economic_impact.py#L843-L849)).

Collision costs (Transport Canada, 2010 CAD): fatal **$8,149,776**, major/serious **$1,012,202**,
minor **$37,489**, minimal/PDO **$9,780**.

---

## Running

```bash
.venv/bin/python ranking.py            # build intersection scores + map
.venv/bin/python economic_impact.py    # build ward economic-exposure analysis
.venv/bin/python explore-data.py       # inspect the candidate source datasets
```

Dependencies: `requests`, `numpy`, `pandas`, `geopandas`, `shapely`, `scipy`, `folium`,
`branca`, `matplotlib`, `seaborn`, `statsmodels` (see `pyproject.toml` / `uv.lock`).
