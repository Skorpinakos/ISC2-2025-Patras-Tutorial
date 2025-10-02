"""
Requires: requests, pandas, matplotlib
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Hardcoded query parameters (edit these) ----------------
BASE_URL = "http://labserver.sense-campus.gr:9090/ngsi-ld/v1/temporal/entities"
TYPE = "AirQualityObserved"
TYPE = "CrowdFlowObserved"
TIMEREL = "between"
TIME = "2025-09-29T00:00:00Z"
END_TIME = "2025-10-02T12:00:00Z"
LIMIT = None 



# ---------------- Filtering options (after fetch) ----------------
INCLUDE_IDS = {                                                                                   #we keep some of the entities to avoid clutter and non consistent stations
    "urn:ngsi-ld:ice-ht:Patras:101609",
    "urn:ngsi-ld:ice-ht:Patras:101589",
    "urn:ngsi-ld:ice-ht:Patras:1566",
    "urn:ngsi-ld:patras:estia:crowd:waitingarea",
    "urn:ngsi-ld:patras:estia:crowd:restaurant"
}
# Keep only these attribute names (by simple tail name). Empty set -> keep all.

KEEP_ATTRS = {"pm25","temperature","relativeHumidity","peopleCount"}

# ---------------- Helpers ----------------
META = {"id", "type", "@context", "dateObserved"}

def simple_name(attr_key: str) -> str:
    """Return a human-friendly tail for an IRI/curie (last of '/', '#', or ':')."""
    return str(attr_key).rsplit("/", 1)[-1].rsplit("#", 1)[-1].rsplit(":", 1)[-1]

def looks_like_series(v) -> bool:
    if not isinstance(v, dict) or v.get("type") != "Property":
        return False
    vals = v.get("values")
    if not isinstance(vals, list) or not vals:
        return False
    f = vals[0]
    return isinstance(f, dict) and (
        ("values" in f and isinstance(f["values"], list))
        or ("observedAt" in f and "value" in f)
    )

def find_all_series_attrs(ent: dict):
    return [k for k, v in ent.items() if k not in META and looks_like_series(v)]

def to_df(ent: dict, attr: str) -> pd.DataFrame:
    rows = []
    for block in ent.get(attr, {}).get("values", []):
        if isinstance(block, dict) and "values" in block and isinstance(block["values"], list):
            for p in block["values"]:
                if isinstance(p, dict) and "observedAt" in p and "value" in p:
                    try:
                        rows.append((p["observedAt"], float(p["value"])))
                    except (TypeError, ValueError):
                        pass
        elif isinstance(block, dict) and "observedAt" in block and "value" in block:
            try:
                rows.append((block["observedAt"], float(block["value"])))
            except (TypeError, ValueError):
                pass

    if not rows:
        return pd.DataFrame(columns=["timestamp", "value"])

    df = pd.DataFrame(rows, columns=["timestamp", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df

def include_entity(eid: str) -> bool:
    return (not INCLUDE_IDS) or (eid in INCLUDE_IDS)

# ---------------- Main ----------------
def main():
    params = {
        "type": TYPE,
        "timerel": TIMEREL,
        "time": TIME,
        "endTime": END_TIME,
    }
    if LIMIT is not None:
        params["limit"] = str(LIMIT)

    headers = {"Accept": "application/json"}

    r = requests.get(BASE_URL, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    payload = r.json()

    if not isinstance(payload, list) or not payload:
        raise RuntimeError("No entities returned from broker.")

    # Filter entities by INCLUDE_IDS (include-only; empty set means keep all)
    payload = [ent for ent in payload if include_entity(str(ent.get("id", "")))]
    if not payload:
        raise RuntimeError("All entities excluded by INCLUDE_IDS; nothing to plot.")

    # Collect ALL series-like attributes, then apply KEEP_ATTRS (by simple name)
    all_attrs = sorted({a for ent in payload for a in find_all_series_attrs(ent)})
    if KEEP_ATTRS:
        keep_lower = {s.lower() for s in KEEP_ATTRS}
        attrs = [a for a in all_attrs if simple_name(a).lower() in keep_lower]
    else:
        attrs = all_attrs

    if not attrs:
        raise RuntimeError("No plottable attributes after KEEP_ATTRS filtering.")

    # Plot: one subplot per attribute, one line per entity
    n = len(attrs)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(10, max(2.5 * n, 3.5)))
    if n == 1:
        axes = [axes]

    for ax, attr in zip(axes, attrs):
        any_line = False
        for ent in payload:
            if attr not in ent:
                continue
            df = to_df(ent, attr)
            if df.empty:
                continue
            ax.plot(df["timestamp"], df["value"], label=str(ent.get("id", "")))
            any_line = True

        ax.set_title(simple_name(attr))
        ax.set_ylabel("Value")
        if any_line and len(payload) > 1:
            ax.legend(fontsize="small")

    axes[-1].set_xlabel("Time")
    fig.suptitle("NGSI-LD Temporal Series (separate panels)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

if __name__ == "__main__":
    main()
