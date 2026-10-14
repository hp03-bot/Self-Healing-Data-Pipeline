"""Self-Healing Data Pipeline - Overview Dashboard."""
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.anomaly_utils import build_anomaly_score_summary, prepare_anomaly_investigation
from src.drift_utils import build_drift_summary, prepare_drift_investigation

st.set_page_config(
    page_title="Self-Healing Data Pipeline",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config.settings import get_settings
from src.database import (
    get_active_model,
    get_latest_metrics_summary,
    get_model_history,
    get_pipeline_runs,
    get_recent_anomaly_events,
    get_recent_drift_events,
    get_recent_failed_recoveries,
    get_recovery_actions,
    get_system_events,
)
from src.monitoring import PipelineOrchestrator
from src.ui_components import (
    apply_custom_css,
    get_plotly_layout,
    render_header,
    render_metric_card,
)

# Initialize settings and orchestrator in session state
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = PipelineOrchestrator()

orchestrator: PipelineOrchestrator = st.session_state.orchestrator
settings = get_settings()

# Fetch latest database aggregates
summary_stats = get_latest_metrics_summary(settings.database_path)
current_status = summary_stats.get("current_status", "Healthy")
active_model = get_active_model(settings.database_path)

render_header(
    title="🛡️ Autonomous Self-Healing Data Pipeline",
    subtitle="Continuous ingestion monitoring, automated anomaly detection, drift mitigation, and governed auto-recovery.",
    current_status=current_status,
)

# Surface recent recovery failures instead of leaving them only in the audit history.
failed_recoveries_24h = summary_stats.get("failed_recoveries_24h", 0)
if failed_recoveries_24h:
    st.warning(
        f"Recovery attention required: {failed_recoveries_24h} failed recovery "
        f"action(s) recorded in the last 24 hours. Review the recovery outcomes "
        f"and audit trail below."
    )

# Show actionable context for failed recoveries, not just an aggregate count.
if summary_stats.get("failed_recoveries", 0):
    with st.expander("🔎 Inspect recent failed recovery actions", expanded=False):
        failed_actions = get_recent_failed_recoveries(
            limit=20, db_path=settings.database_path
        )
        if failed_actions:
            failed_df = pd.DataFrame(failed_actions)
            failed_df = failed_df.rename(columns={
                "action_id": "Action ID",
                "run_id": "Pipeline Run",
                "trigger_event": "Trigger",
                "action_type": "Recovery Action",
                "attempts": "Attempts",
                "outcome_summary": "Outcome",
                "error_details": "Error Details",
                "created_at": "Recorded At",
            })
            visible_columns = [
                "Recorded At", "Pipeline Run", "Trigger", "Recovery Action",
                "Attempts", "Outcome", "Error Details",
            ]
            st.dataframe(
                failed_df[visible_columns].fillna(""),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No failed recovery actions are currently recorded.")

# Surface recent anomaly detections for operator investigation.
with st.expander("🚨 Inspect recent anomaly detections", expanded=False):
    anomaly_run_filter = st.text_input(
        "Filter by exact pipeline run ID",
        key="anomaly_run_filter",
        placeholder="e.g. run_abc123",
    )
    anomaly_search = st.text_input(
        "Search run ID, detector, or contributing feature",
        key="anomaly_metadata_search",
        placeholder="e.g. isolation_forest or customer_age",
    )
    anomaly_min_score = st.slider(
        "Minimum anomaly score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        key="anomaly_min_score_filter",
        help="Show only detections at or above this score.",
    )
    anomaly_history_limit = st.selectbox(
        "Maximum recent detections",
        options=[10, 20, 50, 100],
        index=1,
        key="anomaly_history_limit",
        help="Choose how many recent detections to inspect or export.",
    )
    anomaly_events = get_recent_anomaly_events(
        limit=anomaly_history_limit,
        db_path=settings.database_path,
        run_id=anomaly_run_filter,
        search=anomaly_search,
        min_score=anomaly_min_score if anomaly_min_score > 0 else None,
    )
    if anomaly_events:
        anomaly_df = pd.DataFrame(anomaly_events).rename(columns={
            "run_id": "Pipeline Run",
            "record_index": "Record Index",
            "anomaly_score": "Anomaly Score",
            "detector_method": "Detector",
            "top_contributing_feature": "Top Contributing Feature",
            "created_at": "Recorded At",
        })
        visible_anomaly_columns = [
            "Recorded At", "Pipeline Run", "Record Index", "Anomaly Score",
            "Detector", "Top Contributing Feature",
        ]
        base_anomalies = anomaly_df[visible_anomaly_columns].fillna("")
        detector_options = ["All detectors"] + sorted(
            value for value in base_anomalies["Detector"].astype(str).unique()
            if value.strip()
        )
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            selected_detector = st.selectbox(
                "Filter by detector", detector_options, key="anomaly_detector_filter"
            )
        with filter_col2:
            date_start = st.date_input(
                "From date (UTC)", value=None, key="anomaly_date_start"
            )
        with filter_col3:
            date_end = st.date_input(
                "Through date (UTC)", value=None, key="anomaly_date_end"
            )
        sort_choice = st.selectbox(
            "Sort detections",
            options=["Newest first", "Oldest first", "Highest score first"],
            key="anomaly_sort_order",
        )
        if date_start is not None and date_end is not None and date_start > date_end:
            st.warning("The start date is after the end date. Please correct the date range.")
            visible_anomalies = base_anomalies.iloc[0:0].copy()
        else:
            visible_anomalies = prepare_anomaly_investigation(
                base_anomalies,
                detector=selected_detector,
                start_date=date_start,
                end_date=date_end,
                sort_by=sort_choice,
            )
        score_values = pd.to_numeric(
            visible_anomalies["Anomaly Score"], errors="coerce"
        ).dropna()
        if not score_values.empty:
            score_col1, score_col2, score_col3, score_col4, score_col5 = st.columns(5)
            score_col1.metric("Displayed detections", len(score_values))
            score_col2.metric("Average anomaly score", f"{score_values.mean():.3f}")
            score_col3.metric("Median anomaly score", f"{score_values.median():.3f}")
            score_col4.metric("95th percentile", f"{score_values.quantile(0.95):.3f}")
            score_col5.metric("Highest anomaly score", f"{score_values.max():.3f}")

            # Make score severity visible without changing detection or recovery decisions.
            score_band_counts = pd.Series({
                "Low (< 0.40)": int((score_values < 0.40).sum()),
                "Moderate (0.40–0.69)": int(
                    ((score_values >= 0.40) & (score_values < 0.70)).sum()
                ),
                "High (≥ 0.70)": int((score_values >= 0.70).sum()),
            }, name="Detection Count").rename_axis("Score Band").reset_index()
            st.caption("Score bands summarize the displayed numeric detections only; they do not trigger recovery actions.")
            st.dataframe(
                score_band_counts,
                use_container_width=True,
                hide_index=True,
            )
            score_distribution = px.histogram(
                pd.DataFrame({"Anomaly Score": score_values}),
                x="Anomaly Score",
                nbins=10,
                title="Anomaly Score Distribution",
            )
            score_distribution.update_layout(
                xaxis_title="Anomaly Score",
                yaxis_title="Detection Count",
                showlegend=False,
            )
            st.plotly_chart(
                score_distribution,
                use_container_width=True,
                key="anomaly_score_distribution",
            )
            score_timeline = pd.DataFrame({
                "Recorded At": pd.to_datetime(
                    visible_anomalies.loc[score_values.index, "Recorded At"],
                    errors="coerce",
                    utc=True,
                ),
                "Anomaly Score": score_values,
            }).dropna(subset=["Recorded At"]).sort_values("Recorded At")
            if not score_timeline.empty:
                score_timeline_chart = px.scatter(
                    score_timeline,
                    x="Recorded At",
                    y="Anomaly Score",
                    title="Anomaly Scores Over Time",
                    hover_data={"Recorded At": True, "Anomaly Score": ":.3f"},
                )
                score_timeline_chart.update_traces(
                    mode="lines+markers",
                    marker={"size": 8},
                )
                score_timeline_chart.update_layout(
                    xaxis_title="Detection Time (UTC)",
                    yaxis_title="Anomaly Score",
                    yaxis_range=[0, 1],
                    showlegend=False,
                )
                st.plotly_chart(
                    score_timeline_chart,
                    use_container_width=True,
                    key="anomaly_score_timeline",
                )
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
        if not feature_counts.empty:
            st.caption("Most frequent contributing features in the displayed results")
            st.dataframe(
                feature_counts,
                use_container_width=True,
                hide_index=True,
            )
        st.dataframe(
            visible_anomalies,
            use_container_width=True,
            hide_index=True,
        )
        anomaly_csv = visible_anomalies.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download filtered anomalies (CSV)",
            data=anomaly_csv,
            file_name="anomaly_investigation.csv",
            mime="text/csv",
            key="download_filtered_anomaly_events",
        )
        summary_csv = build_anomaly_score_summary(visible_anomalies).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download anomaly score summary (CSV)",
            data=summary_csv,
            file_name="anomaly_score_summary.csv",
            mime="text/csv",
            key="download_anomaly_score_summary",
        )
    else:
        st.caption("No anomaly detections match the current filters.")

# Surface recent feature-level drift detections for faster investigation.
drift_events_24h = summary_stats.get("drift_events_24h", 0)
if drift_events_24h:
    st.warning(
        f"Data drift detected: {drift_events_24h} feature-level drift event(s) "
        "recorded in the last 24 hours. Review the affected features and test results."
    )

with st.expander("📉 Inspect recent detected drift", expanded=False):
    drift_limit = st.selectbox(
        "Maximum recent drift events", options=[10, 20, 50, 100], index=1,
        key="drift_history_limit",
    )
    drift_events = get_recent_drift_events(
        limit=drift_limit, db_path=settings.database_path
    )
    if drift_events:
        drift_df = pd.DataFrame(drift_events).rename(columns={
            "run_id": "Pipeline Run",
            "feature_name": "Feature",
            "test_method": "Statistical Test",
            "statistic": "Test Statistic",
            "p_value": "P-value",
            "severity": "Severity",
            "created_at": "Recorded At",
        })
        drift_columns = [
            "Recorded At", "Pipeline Run", "Feature", "Severity",
            "Statistical Test", "Test Statistic", "P-value",
        ]
        drift_base = drift_df[drift_columns].fillna("")
        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            drift_features = ["All features"] + sorted(
                x for x in drift_base["Feature"].astype(str).unique() if x.strip()
            )
            selected_drift_feature = st.selectbox(
                "Filter by feature", drift_features, key="drift_feature_filter"
            )
        with dcol2:
            drift_severities = ["All severities"] + sorted(
                x for x in drift_base["Severity"].astype(str).unique() if x.strip()
            )
            selected_drift_severity = st.selectbox(
                "Filter by severity", drift_severities, key="drift_severity_filter"
            )
        with dcol3:
            drift_tests = ["All tests"] + sorted(
                x for x in drift_base["Statistical Test"].astype(str).unique() if x.strip()
            )
            selected_drift_test = st.selectbox(
                "Filter by statistical test", drift_tests, key="drift_test_filter"
            )
        ddate1, ddate2 = st.columns(2)
        with ddate1:
            drift_start_date = st.date_input(
                "From date (UTC)", value=None, key="drift_date_start"
            )
        with ddate2:
            drift_end_date = st.date_input(
                "Through date (UTC)", value=None, key="drift_date_end"
            )
        drift_sort = st.selectbox(
            "Sort drift events",
            options=["Newest first", "Oldest first", "Lowest p-value first", "Highest statistic first"],
            key="drift_sort_order",
        )
        if drift_start_date is not None and drift_end_date is not None and drift_start_date > drift_end_date:
            st.warning("The start date is after the end date. Please correct the date range.")
            visible_drift = drift_base.iloc[0:0].copy()
        else:
            visible_drift = prepare_drift_investigation(
                drift_base, feature=selected_drift_feature,
                severity=selected_drift_severity, test_method=selected_drift_test,
                start_date=drift_start_date, end_date=drift_end_date, sort_by=drift_sort,
            )
        drift_summary = build_drift_summary(visible_drift)
        summary_cols = st.columns(4)
        summary_cols[0].metric("Displayed drift events", len(visible_drift))
        summary_cols[1].metric("Significant (p < 0.05)", int((drift_summary["Significance"] == "Significant").sum()))
        numeric_p = pd.to_numeric(visible_drift["P-value"], errors="coerce")
        summary_cols[2].metric("Lowest p-value", f"{numeric_p.min():.4g}" if numeric_p.notna().any() else "N/A")
        summary_cols[3].metric("Affected features", visible_drift["Feature"].nunique())
        st.dataframe(visible_drift, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download filtered drift events (CSV)",
            data=visible_drift.to_csv(index=False).encode("utf-8"),
            file_name="drift_investigation.csv", mime="text/csv",
            key="download_filtered_drift_events",
        )
        st.download_button(
            "⬇️ Download drift summary (CSV)",
            data=drift_summary.to_csv(index=False).encode("utf-8"),
            file_name="drift_investigation_summary.csv", mime="text/csv",
            key="download_drift_summary",
        )
    else:
        st.caption("No detected drift events are currently recorded.")

# Provide an operator-friendly run history view with quality and processing outcomes.
with st.expander("🧾 Inspect recent pipeline runs", expanded=False):
    run_filter_col1, run_filter_col2 = st.columns([0.42, 0.58])
    with run_filter_col1:
        run_status_filter = st.selectbox(
            "Filter by run status",
            options=["ALL", "Healthy", "Warning", "Recovering", "Error"],
            index=0,
            key="pipeline_run_status_filter",
        )
    with run_filter_col2:
        run_search = st.text_input(
            "Search run ID or dataset",
            placeholder="e.g. run_123 or customers.csv",
            key="pipeline_run_search",
        )
    recent_runs = get_pipeline_runs(
        limit=20,
        db_path=settings.database_path,
        status=None if run_status_filter == "ALL" else run_status_filter,
        search=run_search,
    )
    if recent_runs:
        runs_df = pd.DataFrame(recent_runs).rename(columns={
            "run_id": "Run ID",
            "dataset_name": "Dataset",
            "records_total": "Total Records",
            "records_valid": "Valid Records",
            "records_quarantined": "Quarantined",
            "records_anomalous": "Anomalies",
            "quality_score": "Quality Score (%)",
            "status": "Status",
            "execution_time_ms": "Duration (ms)",
            "created_at": "Recorded At",
        })
        st.dataframe(
            runs_df[[
                "Recorded At", "Run ID", "Dataset", "Status", "Total Records",
                "Valid Records", "Quarantined", "Anomalies", "Quality Score (%)",
                "Duration (ms)",
            ]].fillna(""),
            use_container_width=True,
            hide_index=True,
        )
        csv_data = runs_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download filtered runs (CSV)",
            data=csv_data,
            file_name="pipeline_run_history.csv",
            mime="text/csv",
            key="download_filtered_pipeline_runs",
        )
    else:
        st.caption("No pipeline runs are currently recorded.")

# Hero Description Banner
st.markdown(
    """
    <div class="hero-banner">
        <h4 style="color:#F8FAFC; margin-bottom:8px; font-weight:600;">Autonomous MLOps & Self-Healing Telemetry</h4>
        <p style="color:#94A3B8; font-size:0.9rem; line-height:1.5; margin-bottom:12px;">
            This platform monitors incoming production datasets, validates schema constraints, quarantines corrupted records, 
            detects multivariate anomalies via <b>Isolation Forest</b>, tracks statistical distribution drift via 
            <b>Kolmogorov-Smirnov & Chi-Square tests</b>, and autonomously executes bounded recovery protocols (retries, drift retraining, and candidate model rollback).
        </p>
        <div style="display:flex; gap:16px; flex-wrap:wrap; font-size:0.8rem; color:#64748B;">
            <span>● <b>Engine Mode:</b> <span style="color:#38BDF8;">""" + settings.auto_recovery_mode.upper() + """</span></span>
            <span>● <b>Max Retries:</b> <span style="color:#CBD5E1;">""" + str(settings.max_retry_attempts) + """</span></span>
            <span>● <b>Cooldown:</b> <span style="color:#CBD5E1;">""" + str(settings.recovery_cooldown_seconds) + """s</span></span>
            <span>● <b>Storage:</b> <span style="color:#CBD5E1;">SQLite (WAL Mode)</span></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Row 1: KPI Metric Cards
col1, col2, col3, col4 = st.columns(4)

total_records = summary_stats.get("total_records", 0)
total_anomalies = summary_stats.get("total_anomalies", 0)
anom_pct = (total_anomalies / total_records * 100.0) if total_records > 0 else 0.0

with col1:
    render_metric_card(
        title="Total Records Processed",
        value=f"{total_records:,}",
        subtitle=f"Across {summary_stats.get('total_runs', 0)} recorded pipeline runs",
    )

with col2:
    render_metric_card(
        title="Anomalies Detected",
        value=f"{total_anomalies:,}",
        subtitle=f"{anom_pct:.2f}% of processed traffic",
        delta=f"{anom_pct:.1f}% rate",
        is_positive_delta=(anom_pct < 10.0),
    )

with col3:
    avg_score = summary_stats.get("avg_quality_score", 100.0)
    render_metric_card(
        title="Data Quality Score",
        value=f"{avg_score:.1f}%",
        subtitle=f"{summary_stats.get('total_quarantined', 0):,} records quarantined",
        delta="+0.8%" if avg_score >= 90 else "-3.2%",
        is_positive_delta=(avg_score >= 85.0),
    )

with col4:
    active_ver = active_model["model_version"] if active_model else "None"
    active_f1 = f"{active_model['f1_score']:.3f}" if active_model and active_model.get("f1_score") else "N/A"
    rec_rate = summary_stats.get("recovery_success_rate")
    total_recoveries = summary_stats.get("total_recoveries", 0)
    failed_recoveries = summary_stats.get("failed_recoveries", 0)
    recovery_rate_label = f"{rec_rate:.1f}%" if rec_rate is not None else "N/A"
    recovery_delta = f"{rec_rate:.0f}% success" if rec_rate is not None else "No attempts"
    render_metric_card(
        title="Active Model & Recovery",
        value=f"{active_ver} ({active_f1})",
        subtitle=f"Recovery Success Rate: {recovery_rate_label} · {failed_recoveries} failed / {total_recoveries} attempts",
        delta=recovery_delta,
        is_positive_delta=(rec_rate >= 90.0) if rec_rate is not None else False,
    )

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# Row 2: Interactive Historical Charts
chart_col1, chart_col2 = st.columns([0.60, 0.40])

runs_data = get_pipeline_runs(limit=25, db_path=settings.database_path)
runs_df = pd.DataFrame(runs_data)

with chart_col1:
    st.markdown("##### 📈 Ingestion Quality & Anomaly Volume (Recent Runs)")
    if not runs_df.empty:
        # Reverse to show chronological left-to-right
        chart_df = runs_df.iloc[::-1].copy()
        fig = go.Figure()

        # Quality score line (secondary axis)
        fig.add_trace(go.Scatter(
            x=chart_df["run_id"],
            y=chart_df["quality_score"],
            mode="lines+markers",
            name="Quality Score (%)",
            line=dict(color="#0EA5E9", width=2.5),
            marker=dict(size=6),
            yaxis="y2",
        ))

        # Anomalies detected bar
        fig.add_trace(go.Bar(
            x=chart_df["run_id"],
            y=chart_df["records_anomalous"],
            name="Anomalies Flagged",
            marker=dict(color="rgba(245, 158, 11, 0.75)", line=dict(color="#F59E0B", width=1)),
            yaxis="y",
        ))

        layout = get_plotly_layout(height=320)
        layout["yaxis2"] = dict(
            title="Quality Score (%)",
            overlaying="y",
            side="right",
            range=[0, 105],
            showgrid=False,
            font=dict(color="#0EA5E9"),
        )
        layout["yaxis"]["title"] = "Anomaly Count"
        layout["xaxis"]["title"] = "Pipeline Run ID"
        fig.update_layout(layout)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No pipeline runs recorded yet. Process an incoming dataset or run the Simulation Lab.")

with chart_col2:
    st.markdown("##### 🔄 Recovery Actions & Outcomes")
    rec_actions = get_recovery_actions(limit=50, db_path=settings.database_path)
    if rec_actions:
        rec_df = pd.DataFrame(rec_actions)
        status_counts = rec_df["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]

        color_map = {
            "SUCCESS": "#10B981",
            "FAILED": "#EF4444",
            "SKIPPED": "#64748B",
            "COOLED_DOWN": "#F59E0B",
        }
        fig_pie = px.pie(
            status_counts,
            values="Count",
            names="Status",
            color="Status",
            color_discrete_map=color_map,
            hole=0.55,
        )
        layout_pie = get_plotly_layout(height=320)
        layout_pie["showlegend"] = True
        fig_pie.update_layout(layout_pie)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No recovery actions triggered yet. The system is operating nominally.")

# Row 3: Recent Events and Quick Action Launchpad
st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
event_col, launch_col = st.columns([0.68, 0.32])

with event_col:
    st.markdown("##### 📜 Recent Pipeline & Audit Trail Events")
    recent_events = get_system_events(limit=8, db_path=settings.database_path)
    if recent_events:
        evt_table = []
        for e in recent_events:
            ts_short = e["timestamp"].split("T")[-1][:8] if "T" in e["timestamp"] else e["timestamp"][:19]
            evt_table.append({
                "Time": ts_short,
                "Severity": e["severity"],
                "Component": e["component"],
                "Event": e["event_type"],
                "Description": e["description"],
            })
        st.dataframe(pd.DataFrame(evt_table), use_container_width=True, hide_index=True)
    else:
        st.caption("No events recorded yet.")

with launch_col:
    st.markdown("##### ⚡ Quick Launchpad")
    with st.container():
        st.markdown(
            """
            <div style="background:#111827; border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:16px;">
                <p style="font-size:0.85rem; color:#94A3B8; margin-bottom:12px;">
                    Test end-to-end functionality immediately using bundled reference and challenge datasets:
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("🧪 Run Clean Production Batch (400 records)", use_container_width=True):
            with st.spinner("Processing clean batch through validation, anomaly, and drift monitors..."):
                sample_path = settings.base_dir / "data" / "sample_clean_batch.csv"
                if sample_path.exists():
                    df = pd.read_csv(sample_path)
                    res = orchestrator.process_incoming_batch(df, dataset_name="sample_clean_batch.csv")
                    st.success(f"Batch processed! Status: {res.status} | Quality: {res.quality_score:.1f}%")
                    st.rerun()

        if st.button("⚠️ Run Corrupted Ingestion Batch (Quarantine Test)", use_container_width=True):
            with st.spinner("Executing pipeline with corrupted payload..."):
                sample_path = settings.base_dir / "data" / "sample_corrupted_batch.csv"
                if sample_path.exists():
                    df = pd.read_csv(sample_path)
                    res = orchestrator.process_incoming_batch(df, dataset_name="sample_corrupted_batch.csv")
                    st.warning(f"Batch processed! Quarantined {res.records_quarantined} records into secure storage.")
                    st.rerun()

        if st.button("🔬 Open Failure Simulation Lab", type="primary", use_container_width=True):
            st.switch_page("pages/6_Simulation_Lab.py")
