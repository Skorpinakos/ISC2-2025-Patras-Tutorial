#!/usr/bin/env python3
"""
NGSI-LD: fetch REAL temporal series, filter, send them to batch_forecast,
and plot REAL (history) vs FORECAST (server response) on separate subplots.

Requires: requests, pandas, matplotlib
"""

import os, json
from pathlib import Path
from typing import List, Tuple, Any, Dict, Iterable
import requests
import pandas as pd
import matplotlib.pyplot as plt



# ---- Dev helpers --------------------------------------------------------------
# If set, skip calling the forecast API and use a cached 'forecast_response.json'
SKIP_FORECAST_CALL_ENV = "SKIP_FORECAST"  # "1" to skip POST and load from file
FORECAST_CACHE_FILE = "forecast_response.json"

# If set, skip calling the historical API and use a cached 'historical_response.json'
SKIP_HIST_CALL_ENV = "SKIP_HIST"  # "1" to skip GET and load from file
HIST_CACHE_FILE = "historical_response.json"

# =============================================================================
# Utilities
# =============================================================================

META = {"id", "type", "@context", "dateObserved"}

def attr_tail(iri: str) -> str:
    """Tail of IRI/curie (last of '/', '#', or ':')."""
    return str(iri).rsplit("/", 1)[-1].rsplit("#", 1)[-1].rsplit(":", 1)[-1]

def looks_like_series(v: Any) -> bool:
    if not isinstance(v, dict) or v.get("type") != "Property":
        return False
    vals = v.get("values")
    if not isinstance(vals, list) or not vals:
        return False
    f = vals[0]
    return isinstance(f, dict) and (
        ("values" in f and isinstance(f["values"], list)) or
        ("observedAt" in f and "value" in f)
    )

def find_series_attrs(ent: dict) -> List[str]:
    return [k for k, v in ent.items() if k not in META and looks_like_series(v)]

def to_df(ent: dict, attr: str) -> pd.DataFrame:
    rows: List[Tuple[str, float]] = []
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

def collect_all_attrs(entities: Iterable[dict]) -> set:
    s = set()
    for ent in entities:
        for a in find_series_attrs(ent):
            s.add(a)
    return s

# =============================================================================
# Historical (REAL) data fetch + filtering
# =============================================================================

def fetch_historical() -> List[dict]:
    if os.getenv(SKIP_HIST_CALL_ENV) == "1":
        data = json.loads(Path(HIST_CACHE_FILE).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise TypeError("Cached historical_response.json must be a list.")
        return data

    params = {
        "type": HIST_TYPE,
        "timerel": HIST_TIMEREL,
        "time": HIST_TIME,
        "endTime": HIST_END_TIME,
    }
    if HIST_LIMIT is not None:
        params["limit"] = str(HIST_LIMIT)

    headers = {"Accept": "application/json"}
    r = requests.get(HIST_BASE_URL, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    payload = r.json()
    Path(HIST_CACHE_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not isinstance(payload, list) or not payload:
        raise RuntimeError("No entities returned from historical broker.")
    return payload

def filter_entities_and_attrs(entities: List[dict]) -> Tuple[List[dict], List[str]]:
    # Keep only whitelisted entity IDs
    ents = [e for e in entities if include_entity(str(e.get("id", "")))]
    if not ents:
        raise RuntimeError("All entities excluded by INCLUDE_IDS; nothing to plot.")

    # Discover attributes, then apply KEEP_ATTRS by tail-name (case-insensitive)
    all_attrs = sorted(collect_all_attrs(ents))
    if KEEP_ATTRS:
        keep_lower = {s.lower() for s in KEEP_ATTRS}
        attrs = [a for a in all_attrs if attr_tail(a).lower() in keep_lower]
    else:
        attrs = all_attrs

    if not attrs:
        raise RuntimeError("No plottable attributes after KEEP_ATTRS filtering.")
    return ents, attrs

# =============================================================================
# Build forecast payload FROM real data (nested batches as server expects)
# =============================================================================

def df_to_nested_batches(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert a DataFrame (timestamp,value) to the NESTED structure expected by the
    /ngsi-ld/batch_forecast endpoint:
      "values": [ { "type":"Property", "values":[ { "metadata":{}, "value":X, "observedAt":ISO } ] }, ... ]
    One point per outer batch (simple & valid per the server's parser).
    """
    out: List[Dict[str, Any]] = []
    for ts, val in df[["timestamp", "value"]].itertuples(index=False):
        iso = pd.Timestamp(ts).tz_convert("UTC").isoformat().replace("+00:00", "Z")
        out.append({
            "type": "Property",
            "values": [{
                "metadata": {},
                "value": float(val),
                "observedAt": iso
            }]
        })
    return out

def build_forecast_payload(real_entities: List[dict], attrs: List[str]) -> List[dict]:
    """
    Construct payload for /ngsi-ld/batch_forecast using the real entities'
    attribute keys and series values found in the fetched historical window,
    with the nested 'values' shape required by the server.
    """
    payload: List[dict] = []
    for ent in real_entities:
        new_ent: Dict[str, Any] = {"id": ent.get("id", ""), "type": ent.get("type", HIST_TYPE)}
        for attr in attrs:
            if attr not in ent:
                continue
            df = to_df(ent, attr)
            if df.empty:
                continue
            new_ent[attr] = {
                "type": "Property",
                "values": df_to_nested_batches(df)  # NESTED batches (one point per batch)
            }
        # Keep only entities that ended up with at least one attribute
        if any(k not in META for k in new_ent.keys()):
            payload.append(new_ent)
    if not payload:
        raise RuntimeError("Forecast payload ended up empty (no series found).")

    with open("payload_to_forecaster.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload

# =============================================================================
# Forecast call
# =============================================================================

def call_forecast(payload: List[dict]) -> List[dict]:
    if os.getenv(SKIP_FORECAST_CALL_ENV) == "1":
        data = json.loads(Path(FORECAST_CACHE_FILE).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise TypeError("Cached forecast_response.json must be a list.")
        return data

    url = f"{FC_BASE}{FC_PATH}"
    headers = {"Content-Type": "application/json", "X-API-KEY": FC_API_KEY}
    r = requests.post(url, headers=headers, params=FC_PARAMS, json=payload, timeout=60)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        # Helpful when debugging: keep the server's error text
        print("Forecast error:", r.status_code, r.text)
        raise
    resp = r.json()
    Path(FORECAST_CACHE_FILE).write_text(json.dumps(resp, indent=2), encoding="utf-8")

    if not isinstance(resp, list) or not resp or not all(isinstance(x, dict) for x in resp):
        raise TypeError("Expected a list of NGSI-LD entity dicts in forecast response.")
    return resp

# =============================================================================
# Plotting
# =============================================================================

def plot_all(real_ents: List[dict], forecast_ents: List[dict], attrs: List[str]) -> None:
    # union of attrs in case forecast adds new ones (rare)
    all_attrs = sorted(set(attrs) | collect_all_attrs(forecast_ents))
    n = len(all_attrs)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(11, max(2.6 * n, 3.6)))
    if n == 1:
        axes = [axes]

    for ax, attr in zip(axes, all_attrs):
        plotted = False

        # Plot REAL (historical) data
        for ent in real_ents:
            if attr not in ent:
                continue
            df = to_df(ent, attr)
            if df.empty:
                continue
            ax.plot(df["timestamp"], df["value"], linestyle="-", linewidth=1.2,
                    label=f"{ent.get('id','')}: real")
            plotted = True

        # Plot FORECAST (server response)
        for ent in forecast_ents:
            if attr not in ent:
                continue
            df = to_df(ent, attr)
            if df.empty:
                continue
            ax.plot(df["timestamp"], df["value"], linestyle="--", linewidth=1.3,
                    label=f"{ent.get('id','')}: forecast")
            plotted = True

        ax.set_title(attr_tail(attr))
        ax.set_ylabel("Value")
        if plotted:
            ax.legend(fontsize="small")

    axes[-1].set_xlabel("Time")
    fig.suptitle("REAL vs FORECAST time series (per attribute)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()



# =============================================================================
# Configuration
# =============================================================================

# ---- Real (historical) fetch -------------------------------------------------
HIST_BASE_URL = "http://labserver.sense-campus.gr:9090/ngsi-ld/v1/temporal/entities"
HIST_TYPE = "AirQualityObserved"
#HIST_TYPE = "CrowdFlowObserved"
HIST_TIMEREL = "between"
HIST_TIME = "2025-09-10T00:00:00Z"
HIST_END_TIME = "2025-10-01T03:00:00Z"
HIST_LIMIT = None  # e.g., 1000 (stringified if not None)

# ---- Filtering on entities/attributes (applied AFTER historical fetch) -------
INCLUDE_IDS = {                                                          #we keep some of the entities to avoid clutter and non consistent stations
    "urn:ngsi-ld:ice-ht:Patras:101609",
    "urn:ngsi-ld:ice-ht:Patras:101589",
    "urn:ngsi-ld:ice-ht:Patras:1566",
    "urn:ngsi-ld:patras:estia:crowd:waitingarea",
    "urn:ngsi-ld:patras:estia:crowd:restaurant"
}
KEEP_ATTRS = {"pm25", "temperature", "relativeHumidity", "peopleCount"}  #so we can ignore non numeric values (although the api can process them we dont need to see thme "plotted")

# ---- Forecast request ---------------------------------------------------------
FC_BASE = "http://labserver.sense-campus.gr:9013"
FC_PATH = "/ngsi-ld/batch_forecast"
FC_API_KEY = "isc2_2025"
FC_PARAMS = {
    "timerel": "between",
    "time": "2025-10-01T03:00:00Z",
    "endTime": "2025-10-02T18:00:00Z",
    "interval_seconds": 10 * 60,
}

# =============================================================================
# Main
# =============================================================================

def main():
    print(f"[1/3] Fetching REAL data: {HIST_BASE_URL}")
    hist = fetch_historical()

    print(f"[2/3] Filtering entities/attributes")
    hist_filtered, attrs = filter_entities_and_attrs(hist)

    print(f"[3/3] Building forecast payload from REAL data and calling forecast: {FC_BASE}{FC_PATH}")
    payload = build_forecast_payload(hist_filtered, attrs)
    forecast_resp = call_forecast(payload)

    print("Plotting REAL vs FORECAST ...")
    plot_all(hist_filtered, forecast_resp, attrs)

if __name__ == "__main__":
    main()
