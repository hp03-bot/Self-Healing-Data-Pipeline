"""Reusable transformations for drift investigation views."""
from __future__ import annotations

import pandas as pd


def prepare_drift_investigation(
    events: pd.DataFrame,
    feature: str = "All features",
    severity: str = "All severities",
    test_method: str = "All tests",
    start_date=None,
    end_date=None,
    sort_by: str = "Newest first",
) -> pd.DataFrame:
    """Filter, normalize, label, and sort safe drift-event metadata."""
    result = events.copy()
    if "Feature" in result.columns and feature != "All features":
        result = result[result["Feature"].astype(str) == feature]
    if "Severity" in result.columns and severity != "All severities":
        result = result[result["Severity"].astype(str) == severity]
    if "Statistical Test" in result.columns and test_method != "All tests":
        result = result[result["Statistical Test"].astype(str) == test_method]

    if "Recorded At" in result.columns:
        timestamps = pd.to_datetime(result["Recorded At"], errors="coerce", utc=True)
        if start_date is not None:
            start = pd.Timestamp(start_date, tz="UTC")
            result = result.loc[timestamps >= start]
            timestamps = timestamps.loc[result.index]
        if end_date is not None:
            end_exclusive = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
            result = result.loc[timestamps < end_exclusive]
            timestamps = timestamps.loc[result.index]
        result["Recorded At"] = timestamps

    if "P-value" in result.columns:
        p_values = pd.to_numeric(result["P-value"], errors="coerce")
        result["Significance"] = "Unknown"
        result.loc[p_values < 0.05, "Significance"] = "Significant"
        result.loc[p_values >= 0.05, "Significance"] = "Not significant"

    if sort_by == "Lowest p-value first" and "P-value" in result.columns:
        result["P-value"] = pd.to_numeric(result["P-value"], errors="coerce")
        result = result.sort_values("P-value", ascending=True, na_position="last")
    elif sort_by == "Highest statistic first" and "Test Statistic" in result.columns:
        result["Test Statistic"] = pd.to_numeric(result["Test Statistic"], errors="coerce")
        result = result.sort_values("Test Statistic", ascending=False, na_position="last")
    elif sort_by == "Oldest first" and "Recorded At" in result.columns:
        result = result.sort_values("Recorded At", ascending=True, na_position="last")
    elif "Recorded At" in result.columns:
        result = result.sort_values("Recorded At", ascending=False, na_position="last")
    return result


def build_drift_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Build a CSV-friendly summary of the displayed drift events."""
    p_values = pd.to_numeric(events.get("P-value", pd.Series(dtype=float)), errors="coerce")
    stats = pd.to_numeric(events.get("Test Statistic", pd.Series(dtype=float)), errors="coerce")
    significance = events.get("Significance", pd.Series(dtype=str)).value_counts()
    features = events.get("Feature", pd.Series(dtype=str)).dropna().astype(str).str.strip()
    features = features[features != ""]
    metrics = [
        ("Displayed drift events", int(len(events))),
        ("Affected features", int(features.nunique())),
        ("Significant (p < 0.05)", int(significance.get("Significant", 0))),
        ("Not significant", int(significance.get("Not significant", 0))),
        ("Unknown p-value", int(significance.get("Unknown", 0))),
        ("Minimum p-value", float(p_values.min()) if p_values.notna().any() else None),
        ("Mean p-value", float(p_values.mean()) if p_values.notna().any() else None),
        ("Maximum test statistic", float(stats.max()) if stats.notna().any() else None),
    ]
    return pd.DataFrame(metrics, columns=["Metric", "Value"])
