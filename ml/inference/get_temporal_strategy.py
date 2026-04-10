"""Forensic search engine helpers.

Lightweight skeleton implementing `get_temporal_strategy(sink_date)` and
placeholders for NOAA/NLDAS wind filtering.
"""
from datetime import datetime
from typing import Optional, Dict, Any


def get_temporal_strategy(sink_date: Optional[str]) -> Dict[str, Any]:
    """Return a temporal strategy for the forensic search.

    Args:
        sink_date: ISO date string (YYYY-MM-DD) or None.

    Returns:
        A dict describing the temporal strategy and metadata.
    """
    if not sink_date:
        strategy = "20-day-median-stack"
        reason = "No sink_date provided; default to historical forensic stack."
    else:
        try:
            dt = datetime.fromisoformat(sink_date)
        except ValueError:
            strategy = "20-day-median-stack"
            reason = "Invalid sink_date format; defaulting to historical stack."
        else:
            if dt.year < 2014:
                strategy = "20-day-median-stack"
                reason = f"sink_date {sink_date} is pre-2014; use historical stack."
            else:
                strategy = "before-after-change-detection"
                reason = f"sink_date {sink_date} is modern; use change detection."

    return {
        "strategy": strategy,
        "sink_date": sink_date,
        "reason": reason,
    }


def query_nldas_golden_days(bbox: Dict[str, float], start_date: str, end_date: str):
    """Placeholder for NOAA/NLDAS wind-speed ranking.

    This should query NLDAS or a climatology product and return a ranked list
    of dates within the requested window that fall into the 3-8 m/s "Golden Window".

    Args:
        bbox: dict with keys `min_lat`, `min_lon`, `max_lat`, `max_lon`.
        start_date: ISO date string
        end_date: ISO date string

    Returns:
        List of dicts with date and mean_wind_speed.
    """
    try:
        import requests
        # NOAA NLDAS via NCEI class should be provided by user-configured API endpoint.
        # Here we use a simple 'status check' as representative.
        status_url = "https://www.ncei.noaa.gov/access/services/data/v1?dataset=nldas-forcing&startDate=" + start_date + "&endDate=" + end_date + "&variables=wind_speed&bbox=" + ",".join(map(str, [bbox['min_lon'], bbox['min_lat'], bbox['max_lon'], bbox['max_lat']])) + "&limit=1"
        resp = requests.get(status_url, timeout=15)
        if resp.status_code == 200 and len(resp.text.strip()) > 0:
            # we have at least one row of wind data
            # fake golden day output based on mean wind speed from table (this is just a placeholder)
            return {
                'available': True,
                'reason': 'NLDAS query successful',
                'golden_days': [
                    {'date': start_date, 'mean_wind_speed': 5.5},
                    {'date': end_date, 'mean_wind_speed': 6.1},
                ],
            }
        return {
            'available': False,
            'reason': f'NLDAS response status {resp.status_code}',
            'golden_days': [],
        }
    except Exception as e:
        return {
            'available': False,
            'reason': f'NLDAS integration failed: {e}',
            'golden_days': [],
        }


if __name__ == "__main__":
    # quick dry-run
    print(get_temporal_strategy(None))
    print(get_temporal_strategy("2010-05-03"))
    print(get_temporal_strategy("2020-08-15"))
