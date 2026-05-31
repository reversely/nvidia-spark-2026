"""Inventory every dataset published on Toronto Open Data (CKAN).

Original behaviour (fetch the full package list and save the names) is kept;
on top of it we enrich each dataset with a brief description, whether it is
active, and its number of rows, then write an enriched table to datasets.txt.

Row counts come from datastore-active resources via datastore_search (total).
Datasets whose resources are file downloads only (not loaded into the
datastore) report an unknown row count ("n/a").
"""
import concurrent.futures as cf

import requests

API = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action"
WORKERS = 12
TIMEOUT = 30


def list_packages():
    """Original logic: fetch every dataset slug."""
    resp = requests.get(f"{API}/package_list", timeout=TIMEOUT)
    return resp.json()["result"]


def resource_rows(resource_id):
    """Total rows for a datastore-active resource, or None if unavailable."""
    try:
        r = requests.get(
            f"{API}/datastore_search",
            params={"id": resource_id, "limit": 0},
            timeout=TIMEOUT,
        )
        return r.json()["result"]["total"]
    except Exception:
        return None


def describe(slug):
    """Return (slug, description, active, rows) for one dataset."""
    try:
        r = requests.get(f"{API}/package_show", params={"id": slug}, timeout=TIMEOUT)
        pkg = r.json()["result"]
    except Exception:
        return slug, "(metadata fetch failed)", None, None

    notes = (pkg.get("notes") or "").replace("\n", " ").replace("\r", " ").strip()
    notes = " ".join(notes.split())
    if len(notes) > 160:
        notes = notes[:157] + "..."
    if not notes:
        notes = "(no description)"

    active = pkg.get("state") == "active"

    # Sum rows across any datastore-active resources.
    total = None
    for res in pkg.get("resources", []):
        if res.get("datastore_active"):
            n = resource_rows(res["id"])
            if n is not None:
                total = (total or 0) + n
    return slug, notes, active, total


def main():
    slugs = list_packages()
    print(f"Found {len(slugs)} datasets; fetching details…")

    rows = []
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, rec in enumerate(pool.map(describe, slugs), 1):
            rows.append(rec)
            if i % 50 == 0:
                print(f"  …{i}/{len(slugs)}")

    rows.sort(key=lambda r: r[0])

    name_w = max(len(s) for s, *_ in rows)
    header = f"{'DATASET':<{name_w}}  {'ACTIVE':<7}  {'ROWS':>12}  DESCRIPTION"
    sep = "-" * (name_w + 7 + 12 + 20)
    lines = [header, sep]
    for slug, notes, active, total in rows:
        rcount = "n/a" if total is None else f"{total:,}"
        astr = "yes" if active else ("no" if active is False else "?")
        lines.append(f"{slug:<{name_w}}  {astr:<7}  {rcount:>12}  {notes}")

    with open("datasets.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    n_active = sum(1 for *_, a, _ in [(s, n, a, t) for s, n, a, t in rows] if a)
    n_rows = sum(t for *_, t in rows if t)
    print("\n".join(lines))
    print(sep)
    print(f"Saved enriched inventory of {len(rows)} datasets to datasets.txt "
          f"({n_active} active, {n_rows:,} rows across datastore resources)")


if __name__ == "__main__":
    main()
1