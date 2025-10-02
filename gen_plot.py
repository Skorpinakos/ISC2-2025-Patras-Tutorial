
"""Generate fake NGSI-LD time-series and plot them (separate subplots)."""

from typing import Any, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt


from payload_faker import generate_ngsi_payload

META = {"id", "type", "@context", "dateObserved"}

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

def find_attrs(ent: dict) -> List[str]:
    return [k for k, v in ent.items() if k not in META and looks_like_series(v)]

def to_df(ent: dict, attr: str) -> pd.DataFrame:
    rows: List[Tuple[str, float]] = []
    for block in ent.get(attr, {}).get("values", []):
        if isinstance(block, dict) and "values" in block and isinstance(block["values"], list):
            for p in block["values"]:
                if isinstance(p, dict) and "observedAt" in p and "value" in p:
                    rows.append((p["observedAt"], float(p["value"])))
        elif isinstance(block, dict) and "observedAt" in block and "value" in block:
            rows.append((block["observedAt"], float(block["value"])))
    df = pd.DataFrame(rows, columns=["timestamp", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp")

def main():
    
    payload = generate_ngsi_payload(
        attrs_by_entity={
            "urn:ngsi-ld:Sensor:1": {
                "https://example.org/Temperature": {"median": 25.0, "amplitude": 6.0, "noise": 1.9},
                "https://example.org/humidity":    {"median": 55.0, "amplitude": 10.0, "noise": 5.2},
            },
            "urn:ngsi-ld:Sensor:2": {
                "https://example.org/Temperature": {"median": 28.5, "amplitude": 5.0, "noise": 2.6},
            },
        },
        start_iso="2025-10-01T00:00:00Z",
        end_iso="2025-10-10T06:00:00Z",
        seconds_per_sample=60*60,
        seed=1066,
    )
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("Faker returned no entities.")

    # ---- Gather attributes across entities ----------------------------------
    attrs = sorted({a for ent in payload for a in find_attrs(ent)})
    if not attrs:
        raise RuntimeError("No plottable attributes found in fake payload.")

    # ---- Plot: one subplot per attribute ------------------------------------
    n = len(attrs)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(10, max(2.5 * n, 3.5)))
    if n == 1:
        axes = [axes]

    for ax, attr in zip(axes, attrs):
        for ent in payload:
            if attr not in ent:
                continue
            df = to_df(ent, attr)
            ax.plot(df["timestamp"], df["value"], label=str(ent.get("id", "")))
        title = attr.rsplit("/", 1)[-1].rsplit("#", 1)[-1].rsplit(":", 1)[-1]
        ax.set_title(title)
        ax.set_ylabel("Value")
        if len(payload) > 1:
            ax.legend(fontsize="small")

    axes[-1].set_xlabel("Time")
    fig.suptitle("Fake NGSI-LD Time Series (separate panels)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

if __name__ == "__main__":
    main()
