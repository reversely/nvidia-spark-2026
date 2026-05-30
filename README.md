# Road Priority Algorithm

Analyzes and visualizes road priority data using Toronto open datasets.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Installation

```bash
uv sync
```

## Usage

```bash
uv run python datasets/datalist.py
```

## Dependencies

| Package | Purpose |
|---|---|
| requests | Fetch data from Toronto Open Data API |
| pandas / numpy | Data processing |
| scipy | Spatial and statistical computation |
| shapely / geopandas | Geospatial analysis |
| folium | Interactive map generation |
