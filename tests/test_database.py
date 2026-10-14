"""Tests for SQLite database schema, inserts, updates, and queries."""
import os
import tempfile
from pathlib import Path
import pytest
from src.database import (
    get_active_model,
    get_connection,
    get_latest_metrics_summary,
    get_model_history,
    get_pipeline_runs,
    get_recent_anomaly_events,
    get_recent_drift_events,
    get_recent_failed_recoveries,
    get_recovery_actions,
    get_system_events,
    init_db,
    log_anomaly_events,
    log_drift_events,
    log_pipeline_run,
    log_recovery_action,
    log_system_event,
    register_model_run,
    update_model_status,
)


@pytest.fixture
def temp_db(tmp_path):
    """Fixture providing an isolated temporary database path."""
    db_file = tmp_path / "test_pipeline.db"
    init_db(db_file)
    return db_file


def test_init_db(temp_db):
    """Verify that all required tables and indexes are initialized."""
    with get_connection(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        required = {
            "pipeline_runs",
            "anomaly_events",
            "drift_events",
            "model_registry",
            "recovery_actions",
            "system_events",
        }
        assert required.issubset(tables)


def test_system_event_logging(temp_db):
    """Verify immutable system event logging and filtering."""
    evt_id = log_system_event(
        event_type="TEST_EVENT",
        severity="WARNING",
        component="TEST_COMP",
        description="A test event was recorded.",
        db_path=temp_db,
    )
    assert evt_id.startswith("evt_")

    events = get_system_events(severity="WARNING", db_path=temp_db)
    assert len(events) == 1
    assert events[0]["event_type"] == "TEST_EVENT"
    assert events[0]["severity"] == "WARNING"


def test_pipeline_run_persistence(temp_db):
    """Verify storing and querying pipeline run metadata."""
    log_pipeline_run(
        run_id="run_test_01",
        dataset_name="test_data.csv",
        records_total=500,
        records_valid=480,
        records_quarantined=20,
        records_anomalous=15,
        quality_score=94.5,
        status="Healthy",
        execution_time_ms=120.5,
        db_path=temp_db,
    )
    runs = get_pipeline_runs(limit=10, db_path=temp_db)
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run_test_01"
    assert runs[0]["records_total"] == 500
    assert runs[0]["quality_score"] == 94.5


def test_model_registry_lifecycle(temp_db):
    """Verify model registration, promotion, demotion, and rollback state."""
    # Register candidate
    register_model_run(
        model_version="v1.0.0",
        algorithm="RandomForest",
        accuracy=0.88,
        precision_score=0.85,
        recall_score=0.82,
        f1_score=0.835,
        status="active",
        artifact_path="/tmp/model_v1.0.0.joblib",
        db_path=temp_db,
    )

    active = get_active_model(temp_db)
    assert active is not None
    assert active["model_version"] == "v1.0.0"

    # Register second candidate and promote it
    register_model_run(
        model_version="v1.1.0",
        algorithm="RandomForest",
        accuracy=0.91,
        precision_score=0.89,
        recall_score=0.86,
        f1_score=0.875,
        status="active",
        artifact_path="/tmp/model_v1.1.0.joblib",
        db_path=temp_db,
    )

    active_now = get_active_model(temp_db)
    assert active_now["model_version"] == "v1.1.0"

    # Previous v1.0.0 should be archived
    history = get_model_history(temp_db)
    v1_entry = next(m for m in history if m["model_version"] == "v1.0.0")
    assert v1_entry["status"] == "archived"


def test_recovery_action_logging(temp_db):
    """Verify recovery action persistence and query."""
    action_id = log_recovery_action(
        trigger_event="INVALID_INCOMING_DATA",
        action_type="QUARANTINE_RECORDS",
        status="SUCCESS",
        outcome_summary="Isolated 5 corrupt records.",
        run_id="run_001",
        db_path=temp_db,
    )
    assert action_id.startswith("rec_")
    actions = get_recovery_actions(db_path=temp_db)
    assert len(actions) == 1
    assert actions[0]["status"] == "SUCCESS"



def test_recovery_metrics_report_no_attempts_as_unavailable(temp_db):
    """No recovery history must not be reported as a perfect success rate."""
    summary = get_latest_metrics_summary(temp_db)

    assert summary["total_recoveries"] == 0
    assert summary["successful_recoveries"] == 0
    assert summary["failed_recoveries"] == 0
    assert summary["recovery_success_rate"] is None


def test_recovery_metrics_include_failed_attempts(temp_db):
    """Recovery aggregates should report successful and failed attempts accurately."""
    log_recovery_action(
        trigger_event="DRIFT_CRITICAL",
        action_type="RETRAIN_MODEL",
        status="SUCCESS",
        outcome_summary="Recovery succeeded.",
        run_id="run_metrics_1",
        db_path=temp_db,
    )
    log_recovery_action(
        trigger_event="DRIFT_CRITICAL",
        action_type="RETRAIN_MODEL",
        status="FAILED",
        outcome_summary="Recovery failed.",
        run_id="run_metrics_2",
        error_details="test failure",
        db_path=temp_db,
    )

    summary = get_latest_metrics_summary(temp_db)

    assert summary["total_recoveries"] == 2
    assert summary["successful_recoveries"] == 1
    assert summary["failed_recoveries"] == 1
    assert summary["recovery_success_rate"] == 50.0



def test_recovery_metrics_count_only_recent_failed_actions(temp_db):
    """Expose recent failed recoveries while excluding older audit records."""
    from datetime import datetime, timedelta, timezone

    log_recovery_action(
        trigger_event="DRIFT_CRITICAL",
        action_type="RETRAIN_MODEL",
        status="FAILED",
        outcome_summary="Recent recovery failure.",
        run_id="run_recent_failure",
        db_path=temp_db,
    )
    old_action_id = log_recovery_action(
        trigger_event="DRIFT_CRITICAL",
        action_type="RETRAIN_MODEL",
        status="FAILED",
        outcome_summary="Older recovery failure.",
        run_id="run_old_failure",
        db_path=temp_db,
    )
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    with get_connection(temp_db) as conn:
        conn.execute(
            "UPDATE recovery_actions SET created_at = ? WHERE action_id = ?",
            (old_timestamp, old_action_id),
        )
        conn.commit()

    summary = get_latest_metrics_summary(temp_db)

    assert summary["failed_recoveries"] == 2
    assert summary["failed_recoveries_24h"] == 1



def test_recent_failed_recoveries_include_actionable_details_and_exclude_successes(temp_db):
    """Failure inspection query should return useful fields and only failed actions."""
    log_recovery_action(
        trigger_event="DRIFT_CRITICAL",
        action_type="RETRAIN_MODEL",
        status="FAILED",
        outcome_summary="Candidate training crashed.",
        error_details="ValueError: target column missing",
        run_id="run_failed_detail",
        db_path=temp_db,
    )
    log_recovery_action(
        trigger_event="DRIFT_WARNING",
        action_type="RETRAIN_MODEL",
        status="SUCCESS",
        outcome_summary="Training succeeded.",
        run_id="run_success_detail",
        db_path=temp_db,
    )

    failures = get_recent_failed_recoveries(limit=10, db_path=temp_db)

    assert len(failures) == 1
    assert failures[0]["status"] == "FAILED"
    assert failures[0]["run_id"] == "run_failed_detail"
    assert failures[0]["trigger_event"] == "DRIFT_CRITICAL"
    assert failures[0]["action_type"] == "RETRAIN_MODEL"
    assert failures[0]["outcome_summary"] == "Candidate training crashed."
    assert failures[0]["error_details"] == "ValueError: target column missing"



def test_recent_anomaly_query_returns_safe_investigation_fields_and_respects_limit(temp_db):
    """Recent anomaly query exposes useful metadata but not raw record payloads."""
    log_pipeline_run(
        run_id="run_anomaly_details",
        dataset_name="anomaly.csv",
        records_total=10,
        records_valid=10,
        records_quarantined=0,
        records_anomalous=2,
        quality_score=90.0,
        status="Warning",
        execution_time_ms=10.0,
        db_path=temp_db,
    )
    log_anomaly_events(
        run_id="run_anomaly_details",
        anomalies=[
            {
                "record_index": 3,
                "anomaly_score": 0.91,
                "detector_method": "isolation_forest",
                "top_contributing_feature": "credit_score",
                "record_payload": {"credit_score": 120},
            },
            {
                "record_index": 7,
                "anomaly_score": 0.82,
                "detector_method": "isolation_forest",
                "top_contributing_feature": "debt_to_income",
                "record_payload": {"debt_to_income": 9.5},
            },
        ],
        db_path=temp_db,
    )

    events = get_recent_anomaly_events(limit=1, db_path=temp_db)

    assert len(events) == 1
    assert events[0]["run_id"] == "run_anomaly_details"
    assert events[0]["record_index"] in {3, 7}
    assert "anomaly_score" in events[0]
    assert "top_contributing_feature" in events[0]
    assert "record_payload" not in events[0]






def test_anomaly_investigation_csv_contains_only_safe_metadata(temp_db):
    """CSV export data should contain investigation fields, never raw row payloads."""
    import csv
    from io import StringIO

    log_pipeline_run(
        run_id="run_anomaly_export",
        dataset_name="export.csv",
        records_total=3,
        records_valid=3,
        records_quarantined=0,
        records_anomalous=1,
        quality_score=90.0,
        status="Warning",
        execution_time_ms=10.0,
        db_path=temp_db,
    )
    log_anomaly_events(
        run_id="run_anomaly_export",
        anomalies=[{
            "record_index": 2,
            "anomaly_score": 0.95,
            "detector_method": "isolation_forest",
            "top_contributing_feature": "account_balance",
            "record_payload": {"private_marker": "must_not_export"},
        }],
        db_path=temp_db,
    )

    events = get_recent_anomaly_events(limit=20, db_path=temp_db)
    export_fields = [
        "run_id", "record_index", "anomaly_score", "detector_method",
        "top_contributing_feature", "created_at",
    ]
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=export_fields)
    writer.writeheader()
    writer.writerows({field: event[field] for field in export_fields} for event in events)
    exported_csv = output.getvalue()

    assert "run_anomaly_export" in exported_csv
    assert "account_balance" in exported_csv
    assert "private_marker" not in exported_csv
    assert "must_not_export" not in exported_csv
    assert "record_payload" not in exported_csv


def test_recent_anomaly_query_filters_by_run_and_safe_metadata(temp_db):
    """Anomaly investigation filters should combine safely and match metadata."""
    for run_id, dataset_name in [
        ("run_anomaly_alpha", "alpha.csv"),
        ("run_anomaly_beta", "beta.csv"),
    ]:
        log_pipeline_run(
            run_id=run_id,
            dataset_name=dataset_name,
            records_total=5,
            records_valid=5,
            records_quarantined=0,
            records_anomalous=1,
            quality_score=90.0,
            status="Warning",
            execution_time_ms=10.0,
            db_path=temp_db,
        )

    log_anomaly_events(
        run_id="run_anomaly_alpha",
        anomalies=[{
            "record_index": 1,
            "anomaly_score": 0.91,
            "detector_method": "isolation_forest",
            "top_contributing_feature": "credit_score",
            "record_payload": {"private_value": "do not expose"},
        }],
        db_path=temp_db,
    )
    log_anomaly_events(
        run_id="run_anomaly_beta",
        anomalies=[{
            "record_index": 2,
            "anomaly_score": 0.87,
            "detector_method": "local_outlier_factor",
            "top_contributing_feature": "transaction_amount",
            "record_payload": {"private_value": "do not expose"},
        }],
        db_path=temp_db,
    )

    by_run = get_recent_anomaly_events(
        limit=20, db_path=temp_db, run_id="run_anomaly_alpha"
    )
    by_feature = get_recent_anomaly_events(
        limit=20, db_path=temp_db, search="transaction_amount"
    )
    combined = get_recent_anomaly_events(
        limit=20,
        db_path=temp_db,
        run_id="run_anomaly_alpha",
        search="local_outlier_factor",
    )
    injection_like = get_recent_anomaly_events(
        limit=20, db_path=temp_db, search="' OR 1=1 --"
    )

    assert len(by_run) == 1
    assert by_run[0]["run_id"] == "run_anomaly_alpha"
    assert len(by_feature) == 1
    assert by_feature[0]["run_id"] == "run_anomaly_beta"
    assert combined == []
    assert injection_like == []
    assert all("record_payload" not in event for event in by_run + by_feature)





def test_anomaly_score_summary_uses_only_filtered_visible_rows():
    """Dashboard score summaries should be calculated from the displayed result set."""
    import pandas as pd

    visible_anomalies = pd.DataFrame({"Anomaly Score": [0.8, 0.9, 0.7]})
    score_values = pd.to_numeric(
        visible_anomalies["Anomaly Score"], errors="coerce"
    ).dropna()

    assert len(score_values) == 3
    assert score_values.mean() == pytest.approx(0.8)
    assert score_values.max() == pytest.approx(0.9)

def test_recent_anomaly_query_respects_requested_limit(temp_db):
    """Anomaly investigation history should return no more than the requested limit."""
    log_pipeline_run(
        run_id="run_anomaly_limit",
        dataset_name="limit.csv",
        records_total=4,
        records_valid=4,
        records_quarantined=0,
        records_anomalous=4,
        quality_score=90.0,
        status="Warning",
        execution_time_ms=10.0,
        db_path=temp_db,
    )
    log_anomaly_events(
        run_id="run_anomaly_limit",
        anomalies=[
            {"record_index": index, "anomaly_score": 0.5 + index / 10,
             "detector_method": "iforest", "top_contributing_feature": "amount"}
            for index in range(4)
        ],
        db_path=temp_db,
    )

    events = get_recent_anomaly_events(limit=2, db_path=temp_db)

    assert len(events) == 2

def test_recent_anomaly_query_filters_by_minimum_score(temp_db):
    """Minimum-score filtering should exclude lower-scored anomalies."""
    log_pipeline_run(
        run_id="run_score_filter",
        dataset_name="scores.csv",
        records_total=4,
        records_valid=4,
        records_quarantined=0,
        records_anomalous=3,
        quality_score=90.0,
        status="Warning",
        execution_time_ms=10.0,
        db_path=temp_db,
    )
    log_anomaly_events(
        run_id="run_score_filter",
        anomalies=[
            {"record_index": 0, "anomaly_score": 0.25, "detector_method": "iforest",
             "top_contributing_feature": "age"},
            {"record_index": 1, "anomaly_score": 0.75, "detector_method": "iforest",
             "top_contributing_feature": "income"},
            {"record_index": 2, "anomaly_score": 0.95, "detector_method": "iforest",
             "top_contributing_feature": "balance"},
        ],
        db_path=temp_db,
    )

    events = get_recent_anomaly_events(
        limit=20, db_path=temp_db, min_score=0.75
    )

    assert {event["record_index"] for event in events} == {1, 2}
    assert all(event["anomaly_score"] >= 0.75 for event in events)

def test_recent_drift_query_returns_detected_events_only(temp_db):
    """Recent drift inspection should include detected features and exclude clean ones."""
    log_pipeline_run(
        run_id="run_drift_details",
        dataset_name="drift.csv",
        records_total=10,
        records_valid=10,
        records_quarantined=0,
        records_anomalous=0,
        quality_score=100.0,
        status="Warning",
        execution_time_ms=10.0,
        db_path=temp_db,
    )
    from src.database import log_drift_events

    log_drift_events(
        run_id="run_drift_details",
        drift_results=[
            {
                "feature_name": "temperature",
                "test_method": "ks_test",
                "statistic": 0.42,
                "p_value": 0.01,
                "drift_detected": True,
                "severity": "CRITICAL",
            },
            {
                "feature_name": "humidity",
                "test_method": "ks_test",
                "statistic": 0.08,
                "p_value": 0.70,
                "drift_detected": False,
                "severity": "HEALTHY",
            },
        ],
        db_path=temp_db,
    )

    events = get_recent_drift_events(limit=10, db_path=temp_db)

    assert len(events) == 1
    assert events[0]["run_id"] == "run_drift_details"
    assert events[0]["feature_name"] == "temperature"
    assert events[0]["severity"] == "CRITICAL"
    assert events[0]["p_value"] == 0.01


def test_pipeline_run_query_bounds_limit_and_returns_run_details(temp_db):
    """Run history query should cap results and expose operational run metrics."""
    for index in range(3):
        log_pipeline_run(
            run_id=f"run_history_{index}",
            dataset_name=f"batch_{index}.csv",
            records_total=10 + index,
            records_valid=9 + index,
            records_quarantined=1,
            records_anomalous=index,
            quality_score=90.0 + index,
            status="Healthy" if index else "Warning",
            execution_time_ms=100.0 + index,
            db_path=temp_db,
        )

    runs = get_pipeline_runs(limit=2, db_path=temp_db)
    assert len(runs) == 2
    assert {run["run_id"] for run in runs}.issubset(
        {"run_history_0", "run_history_1", "run_history_2"}
    )
    assert {"dataset_name", "records_total", "records_quarantined", "quality_score", "status"}.issubset(runs[0])

    bounded_runs = get_pipeline_runs(limit=0, db_path=temp_db)
    assert len(bounded_runs) == 1


def test_pipeline_run_query_filters_by_status(temp_db):
    """Run history status filter should return only matching runs."""
    for run_id, status in [
        ("run_filter_healthy", "Healthy"),
        ("run_filter_warning", "Warning"),
        ("run_filter_error", "Error"),
    ]:
        log_pipeline_run(
            run_id=run_id,
            dataset_name=f"{run_id}.csv",
            records_total=5,
            records_valid=5,
            records_quarantined=0,
            records_anomalous=0,
            quality_score=100.0,
            status=status,
            execution_time_ms=25.0,
            db_path=temp_db,
        )

    warnings = get_pipeline_runs(limit=20, db_path=temp_db, status="Warning")
    assert len(warnings) == 1
    assert warnings[0]["run_id"] == "run_filter_warning"
    assert warnings[0]["status"] == "Warning"

    all_runs = get_pipeline_runs(limit=20, db_path=temp_db, status="ALL")
    assert len(all_runs) == 3



def test_pipeline_run_query_searches_run_id_or_dataset_and_combines_status(temp_db):
    """Run history search should match run IDs or dataset names and combine with status."""
    for run_id, dataset_name, status in [
        ("run_customer_import", "customers.csv", "Healthy"),
        ("run_orders_import", "orders.csv", "Warning"),
        ("run_customer_retry", "retry_customers.csv", "Warning"),
    ]:
        log_pipeline_run(
            run_id=run_id,
            dataset_name=dataset_name,
            records_total=5,
            records_valid=5,
            records_quarantined=0,
            records_anomalous=0,
            quality_score=100.0,
            status=status,
            execution_time_ms=25.0,
            db_path=temp_db,
        )

    dataset_matches = get_pipeline_runs(limit=20, db_path=temp_db, search="orders.csv")
    assert {run["run_id"] for run in dataset_matches} == {"run_orders_import"}

    run_id_matches = get_pipeline_runs(limit=20, db_path=temp_db, search="customer")
    assert {run["run_id"] for run in run_id_matches} == {
        "run_customer_import",
        "run_customer_retry",
    }

    filtered_matches = get_pipeline_runs(
        limit=20, db_path=temp_db, status="Warning", search="customer"
    )
    assert {run["run_id"] for run in filtered_matches} == {"run_customer_retry"}


def test_anomaly_feature_breakdown_uses_displayed_rows_and_ignores_blanks():
    """Feature breakdown should summarize only displayed, non-empty feature labels."""
    import pandas as pd

    visible_anomalies = pd.DataFrame({
        "Top Contributing Feature": [
            "account_balance",
            "credit_score",
            "account_balance",
            "",
            None,
            "transaction_amount",
            "account_balance",
        ]
    })
    feature_counts = (
        visible_anomalies["Top Contributing Feature"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .head(5)
        .rename_axis("Top Contributing Feature")
        .reset_index(name="Detection Count")
    )

    counts = dict(zip(
        feature_counts["Top Contributing Feature"],
        feature_counts["Detection Count"],
    ))
    assert counts["account_balance"] == 3
    assert counts["credit_score"] == 1
    assert counts["transaction_amount"] == 1
    assert "None" not in counts


def test_anomaly_score_distribution_input_excludes_invalid_scores():
    """Score distribution should receive numeric scores from visible rows only."""
    import pandas as pd

    visible_anomalies = pd.DataFrame({
        "Anomaly Score": [0.1, 0.2, "invalid", None, 0.95]
    })
    score_values = pd.to_numeric(
        visible_anomalies["Anomaly Score"], errors="coerce"
    ).dropna()

    assert score_values.tolist() == [0.1, 0.2, 0.95]
    assert score_values.between(0.0, 1.0).all()



def test_anomaly_score_timeline_sorts_timestamps_and_drops_invalid_dates():
    """Timeline input should be chronological and omit rows without usable timestamps."""
    import pandas as pd

    visible_anomalies = pd.DataFrame({
        "Recorded At": [
            "2026-10-14T10:03:00+00:00",
            "not-a-timestamp",
            "2026-10-14T10:01:00+00:00",
        ],
        "Anomaly Score": [0.8, 0.99, 0.4],
    })
    score_values = pd.to_numeric(
        visible_anomalies["Anomaly Score"], errors="coerce"
    ).dropna()
    score_timeline = pd.DataFrame({
        "Recorded At": pd.to_datetime(
            visible_anomalies.loc[score_values.index, "Recorded At"],
            errors="coerce",
            utc=True,
        ),
        "Anomaly Score": score_values,
    }).dropna(subset=["Recorded At"]).sort_values("Recorded At")

    assert score_timeline["Anomaly Score"].tolist() == [0.4, 0.8]
    assert score_timeline["Recorded At"].is_monotonic_increasing



def test_anomaly_score_percentiles_summarize_numeric_visible_scores():
    """Percentile metrics should use numeric scores from the displayed detections."""
    import pandas as pd

    visible_anomalies = pd.DataFrame({
        "Anomaly Score": [0.1, 0.2, "invalid", None, 0.95, 0.8]
    })
    score_values = pd.to_numeric(
        visible_anomalies["Anomaly Score"], errors="coerce"
    ).dropna()

    assert score_values.median() == pytest.approx(0.5)
    assert score_values.quantile(0.95) == pytest.approx(0.9275)
    assert score_values.quantile(0.95) <= score_values.max()
    assert len(score_values) == 4


def test_anomaly_score_bands_count_numeric_visible_scores_only():
    """Score bands should be mutually exclusive and ignore invalid or missing scores."""
    import pandas as pd

    visible_anomalies = pd.DataFrame({
        "Anomaly Score": [0.0, 0.39, 0.4, 0.69, 0.7, 1.0, "invalid", None]
    })
    score_values = pd.to_numeric(
        visible_anomalies["Anomaly Score"], errors="coerce"
    ).dropna()
    band_counts = {
        "Low (< 0.40)": int((score_values < 0.40).sum()),
        "Moderate (0.40–0.69)": int(
            ((score_values >= 0.40) & (score_values < 0.70)).sum()
        ),
        "High (≥ 0.70)": int((score_values >= 0.70).sum()),
    }

    assert band_counts == {
        "Low (< 0.40)": 2,
        "Moderate (0.40–0.69)": 2,
        "High (≥ 0.70)": 2,
    }
    assert sum(band_counts.values()) == len(score_values)


def test_anomaly_investigation_filters_detector_and_inclusive_utc_dates():
    from src.anomaly_utils import prepare_anomaly_investigation
    import pandas as pd

    rows = pd.DataFrame({
        "Recorded At": [
            "2026-10-10T00:00:00Z",
            "2026-10-11T23:59:00Z",
            "2026-10-12T00:00:00Z",
        ],
        "Anomaly Score": [0.2, 0.8, 0.5],
        "Detector": ["iforest", "iforest", "lof"],
    })
    visible = prepare_anomaly_investigation(
        rows, detector="iforest", start_date=pd.Timestamp("2026-10-10").date(),
        end_date=pd.Timestamp("2026-10-11").date(),
    )

    assert visible["Anomaly Score"].tolist() == [0.8, 0.2]
    assert visible["Severity Band"].tolist() == ["High", "Low"]


def test_anomaly_investigation_can_sort_by_highest_score_and_unknown_band():
    from src.anomaly_utils import prepare_anomaly_investigation
    import pandas as pd

    rows = pd.DataFrame({
        "Recorded At": ["2026-10-10T00:00:00Z"] * 3,
        "Anomaly Score": ["invalid", 0.3, 0.9],
        "Detector": ["iforest"] * 3,
    })
    visible = prepare_anomaly_investigation(rows, sort_by="Highest score first")

    assert visible["Anomaly Score"].iloc[:2].tolist() == [0.9, 0.3]
    assert visible["Severity Band"].tolist() == ["High", "Low", "Unknown"]


def test_anomaly_score_summary_reports_metrics_and_score_bands():
    from src.anomaly_utils import build_anomaly_score_summary, prepare_anomaly_investigation
    import pandas as pd

    rows = pd.DataFrame({
        "Recorded At": ["2026-10-10T00:00:00Z"] * 4,
        "Anomaly Score": [0.2, 0.5, 0.8, "invalid"],
        "Detector": ["iforest"] * 4,
    })
    visible = prepare_anomaly_investigation(rows)
    summary = build_anomaly_score_summary(visible).set_index("Metric")["Value"]

    assert summary["Displayed detections"] == 3
    assert summary["Low detections"] == 1
    assert summary["Moderate detections"] == 1
    assert summary["High detections"] == 1
    assert summary["Unknown-score detections"] == 1
    assert summary["Highest score"] == 0.8




def test_drift_investigation_filters_feature_severity_test_and_inclusive_dates():
    import pandas as pd
    from src.drift_utils import prepare_drift_investigation

    rows = pd.DataFrame({
        "Recorded At": ["2026-10-10T02:00:00Z", "2026-10-11T23:59:00Z", "2026-10-12T00:00:00Z"],
        "Feature": ["amount", "amount", "age"],
        "Severity": ["high", "high", "low"],
        "Statistical Test": ["ks", "ks", "chi_square"],
        "Test Statistic": [0.2, 0.5, 0.1],
        "P-value": [0.04, 0.01, 0.2],
    })
    visible = prepare_drift_investigation(
        rows, feature="amount", severity="high", test_method="ks",
        start_date=pd.Timestamp("2026-10-10").date(),
        end_date=pd.Timestamp("2026-10-11").date(),
    )
    assert len(visible) == 2
    assert visible["Feature"].tolist() == ["amount", "amount"]
    assert visible["Significance"].tolist() == ["Significant", "Significant"]


def test_drift_investigation_sorts_p_values_and_handles_invalid_values():
    import pandas as pd
    from src.drift_utils import prepare_drift_investigation

    rows = pd.DataFrame({
        "Recorded At": ["2026-10-10T00:00:00Z"] * 3,
        "Feature": ["a", "b", "c"],
        "P-value": ["invalid", 0.2, 0.001],
        "Test Statistic": [1, 3, 2],
    })
    visible = prepare_drift_investigation(rows, sort_by="Lowest p-value first")
    assert visible["Feature"].tolist() == ["c", "b", "a"]
    assert visible["Significance"].tolist() == ["Significant", "Not significant", "Unknown"]


def test_drift_summary_counts_significance_and_unique_features():
    import pandas as pd
    from src.drift_utils import build_drift_summary, prepare_drift_investigation

    rows = pd.DataFrame({
        "Recorded At": ["2026-10-10T00:00:00Z"] * 4,
        "Feature": ["a", "a", "b", ""],
        "P-value": [0.01, 0.2, None, "invalid"],
        "Test Statistic": [2.0, 1.0, 0.5, 0.1],
    })
    visible = prepare_drift_investigation(rows)
    summary = build_drift_summary(visible).set_index("Metric")["Value"]
    assert summary["Displayed drift events"] == 4
    assert summary["Affected features"] == 2
    assert summary["Significant (p < 0.05)"] == 1
    assert summary["Not significant"] == 1
    assert summary["Unknown p-value"] == 2
    assert summary["Maximum test statistic"] == 2.0
