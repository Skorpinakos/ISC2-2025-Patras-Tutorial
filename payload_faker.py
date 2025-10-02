from datetime import datetime, timedelta, timezone
import math, random
from typing import List, Dict, Any, Optional, Union

ISO_ZFMT = "%Y-%m-%dT%H:%M:%SZ"
SECONDS_PER_DAY = 86400.0

def _to_iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime(ISO_ZFMT)

def _parse_iso_z(s: str) -> datetime:
    return datetime.strptime(s, ISO_ZFMT).replace(tzinfo=timezone.utc)

def _make_timestamps_range(
    *, start_iso: str, end_iso: str, seconds_per_sample: Union[int, float], inclusive_end: bool = False
) -> List[str]:
    secs = float(seconds_per_sample)
    if secs <= 0:
        raise ValueError("seconds_per_sample must be > 0.")
    start = _parse_iso_z(start_iso)
    end = _parse_iso_z(end_iso)
    if end <= start:
        raise ValueError("end_iso must be after start_iso.")
    step = timedelta(seconds=secs)
    t, out = start, []
    if inclusive_end:
        while t <= end:
            out.append(_to_iso_z(t))
            t += step
    else:
        while t < end:
            out.append(_to_iso_z(t))
            t += step
    if not out:
        out.append(_to_iso_z(start))
    return out

def _daily_periodic_value(ts: datetime, *, median: float, amplitude: float, noise: float) -> float:
    """Sine with 24h UTC period + Gaussian noise."""
    seconds_into_day = (ts - ts.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    phase = 2 * math.pi * (seconds_into_day / SECONDS_PER_DAY)
    base = median + amplitude * math.sin(phase)
    return base + (random.gauss(0.0, noise) if noise > 0 else 0.0)

def generate_ngsi_payload(
    attrs_by_entity: Dict[str, Dict[str, Dict[str, float]]],
    *,
    start_iso: str,
    end_iso: str,
    seconds_per_sample: Union[int, float],
    entity_type: str = "Sensor",
    context: Optional[List[str]] = None,
    inclusive_end: bool = False,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    attrs_by_entity:
      {
        "<entity_id>": {
          "<attr_iri>": {"median": <float>, "amplitude": <float>, "noise": <float>},
          ...
        },
        ...
      }
    """
    if seed is not None:
        random.seed(seed)
    if context is None:
        context = ["https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"]

    timestamps = _make_timestamps_range(
        start_iso=start_iso,
        end_iso=end_iso,
        seconds_per_sample=seconds_per_sample,
        inclusive_end=inclusive_end
    )
    dt_list = [_parse_iso_z(ts) for ts in timestamps]

    payload: List[Dict[str, Any]] = []
    for eid, attrs in attrs_by_entity.items():
        entity: Dict[str, Any] = {
            "id": eid,
            "type": entity_type,
            "@context": context,
            "dateObserved": {"type": "Property", "values": timestamps},
        }

        for attr_iri, spec in attrs.items():
            median = float(spec.get("median", 0.0))
            amplitude = float(spec.get("amplitude", 0.0))
            noise = float(spec.get("noise", 0.0))

            series = [
                _daily_periodic_value(dt, median=median, amplitude=amplitude, noise=noise)
                for dt in dt_list
            ]
            values_block = [
                {"values": [{"observedAt": timestamps[i], "value": series[i]}]}
                for i in range(len(timestamps))
            ]
            entity[attr_iri] = {"type": "Property", "values": values_block}

        payload.append(entity)

    return payload
