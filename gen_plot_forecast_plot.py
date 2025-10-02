#!/usr/bin/env python3
"""NGSI-LD client: plot ALL time-series on separate subplots (one window),
overlaying the original REQUEST payload and the server RESPONSE together.
"""

import os, json
from pathlib import Path
from typing import List, Tuple, Any, Dict
import requests, pandas as pd, matplotlib.pyplot as plt

# --- Config from env ----------------------------------------------------------
BASE = "http://labserver.sense-campus.gr:9013"
PATH = "/ngsi-ld/batch_forecast"
APIK = "isc2_2025"



# --- Payload generation -------------------------------------------------------
from payload_faker import generate_ngsi_payload

payload = generate_ngsi_payload(
    attrs_by_entity={
        "urn:ngsi-ld:Sensor:1": {
            "https://example.org/Temperature": {"median": 25.0, "amplitude": 6.0, "noise": 1.9},
            "https://example.org/humidity":    {"median": 55.0, "amplitude": 10.0, "noise": 2.2},
        },
        "urn:ngsi-ld:Sensor:2": {
            "https://example.org/Temperature": {"median": 28.5, "amplitude": 5.0, "noise": 1.6},
        },
    },
    start_iso="2025-09-25T00:00:00Z",
    end_iso="2025-10-05T00:00:00Z",
    seconds_per_sample=30*60,
    seed=1066
)

META = {"id","type","@context","dateObserved"}

# --- Request details ----------------------------------------------------------
headers = {"Content-Type": "application/json", "X-API-KEY": APIK}
params  = {
    "timerel":"between",
    "time":"2025-10-05T00:00:00Z",
    "endTime":"2025-10-08T18:00:00Z",
    "interval_seconds":3*60*60
}

# --- Helpers ------------------------------------------------------------------
def looks_like_series(v: Any) -> bool:
    if not isinstance(v, dict) or v.get("type") != "Property": return False
    vals = v.get("values")
    if not isinstance(vals, list) or not vals: return False
    f = vals[0]
    return isinstance(f, dict) and (
        ("values" in f and isinstance(f["values"], list)) or
        ("observedAt" in f and "value" in f)
    )

def find_attrs(ent: dict) -> List[str]:
    return [k for k,v in ent.items() if k not in META and looks_like_series(v)]

def to_df(ent: dict, attr: str) -> pd.DataFrame:
    rows: List[Tuple[str, float]] = []
    for block in ent.get(attr, {}).get("values", []):
        if isinstance(block, dict) and "values" in block and isinstance(block["values"], list):
            for p in block["values"]:
                if isinstance(p, dict) and "observedAt" in p and "value" in p:
                    rows.append((p["observedAt"], float(p["value"])))
        elif isinstance(block, dict) and "observedAt" in block and "value" in block:
            rows.append((block["observedAt"], float(block["value"])))
    if not rows:
        raise ValueError(f"No (observedAt,value) pairs under {attr}")
    df = pd.DataFrame(rows, columns=["timestamp","value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp")

def fetch() -> List[dict]:
    if os.getenv("SKIP_CALL") == "1":
        return json.loads(Path("api_response.json").read_text(encoding="utf-8"))
    url = f"{BASE}{PATH}"
    r = requests.post(url, headers=headers, params=params, json=payload, timeout=60)
    r.raise_for_status()
    resp = r.json()
    Path("api_response.json").write_text(json.dumps(resp, indent=2), encoding="utf-8")
    return resp

def collect_all_attrs(entities: List[dict]) -> set:
    s = set()
    for ent in entities:
        s.update(find_attrs(ent))
    return s

def attr_tail(iri: str) -> str:
    return iri.rsplit("/",1)[-1].rsplit("#",1)[-1].rsplit(":",1)[-1]

# --- Main ---------------------------------------------------------------------
def main():
    print(f"Calling: POST {BASE}{PATH}")
    resp = fetch()
    if not isinstance(resp, list) or not resp or not all(isinstance(x, dict) for x in resp):
        raise TypeError("Expected: list of NGSI-LD entity dicts in response.")

    # Union of attributes across BOTH request payload and response
    attrs = sorted(collect_all_attrs(payload) | collect_all_attrs(resp))
    if not attrs:
        raise RuntimeError("No plottable attributes found in request+response.")

    # Prepare subplots (one per attribute)
    n = len(attrs)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(10, max(2.5*n, 3.5)))
    if n == 1: axes = [axes]

    # Plot loop
    for ax, attr in zip(axes, attrs):
        plotted_any = False

        # 1) Plot RESPONSE (server) – solid line
        for ent in resp:
            if attr not in ent: continue
            try:
                df = to_df(ent, attr)
            except Exception as e:
                print(f"[resp] Skip {attr} for {ent.get('id')}: {e}")
                continue
            ax.plot(df["timestamp"], df["value"], label=f"{ent.get('id','')}: response")
            plotted_any = True

        # 2) Plot REQUEST (original payload) – dashed line with markers
        for ent in payload:
            if attr not in ent: continue
            try:
                df_in = to_df(ent, attr)
            except Exception as e:
                print(f"[req ] Skip {attr} for {ent.get('id')}: {e}")
                continue
            # stylistic difference to distinguish request from response
            ax.plot(df_in["timestamp"], df_in["value"],
                    markersize=3, label=f"{ent.get('id','')}: request")
            plotted_any = True

        # Titles/labels
        ax.set_title(attr_tail(attr))
        ax.set_ylabel("Value")
        if plotted_any:
            ax.legend(fontsize="small")

    axes[-1].set_xlabel("Time")
    fig.suptitle("Prediction Time Series (response vs. original request)", y=0.995)
    fig.tight_layout(rect=[0,0,1,0.97])
    plt.show()

if __name__ == "__main__":
    main()
