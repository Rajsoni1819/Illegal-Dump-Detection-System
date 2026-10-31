"""
Streamlit Web Dashboard for AI Illegal Dumping Detection System.
Provides smart city incident management, audit workflows, evidence review,
data filtering, CSV export, analytics charts, and geospatial mapping.
"""

import os
import json
import logging
import pandas as pd
import streamlit as st
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional

from database import (
    IncidentDatabase,
    get_ist_now,
    get_ist_timestamp_str,
    VALID_STATUSES
)

# Page configuration
st.set_page_config(
    page_title="AI Smart Waste Watch | Illegal Dumping Detection",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Smart City CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .badge-review {
        background-color: #FEF3C7;
        color: #B45309;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-confirmed {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-resolved {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-dismissed {
        background-color: #F1F5F9;
        color: #64748B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .disclaimer-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 4px;
        color: #1E40AF;
        font-size: 0.9rem;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db_instance() -> IncidentDatabase:
    """Initialize or retrieve singleton IncidentDatabase."""
    config_path = "config.json"
    db_path = "data/dumping_incidents.db"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                db_path = cfg.get("database", {}).get("database_path", db_path)
        except Exception:
            pass
    return IncidentDatabase(db_path)


db = get_db_instance()

# Sidebar: System Status & Quick Controls
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1532996122724-e3c354a0b15b?auto=format&fit=crop&w=400&q=80",
        caption="Municipal Smart Surveillance"
    )
    st.title("♻️ Control Center")
    st.markdown("**Zone:** Sector 4 Central Green")
    st.markdown(f"**Current IST:** `{get_ist_timestamp_str()}`")
    st.markdown("---")

    # Refresh data button
    if st.button("🔄 Refresh Incident Feed", use_container_width=True):
        st.rerun()

    st.markdown("### 🛡️ Privacy Guarantee")
    st.caption(
        "• No facial recognition or citizen identification.\n"
        "• Automatic face blurring on all stored evidence images.\n"
        "• Rolling 30-day evidence retention policy.\n"
        "• Every alert is treated as a suspected incident requiring human officer review."
    )


# Top Header
col_header_left, col_header_right = st.columns([3, 1])
with col_header_left:
    st.markdown('<p class="main-header">Smart City AI Waste Surveillance</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Automated Video Monitoring & Illegal Dumping Incident Verification Dashboard</p>',
        unsafe_allow_html=True
    )
with col_header_right:
    st.markdown(
        f"<div style='text-align: right; padding-top: 10px;'>"
        f"<span style='background:#DCFCE7; color:#166534; padding:6px 14px; border-radius:20px; font-weight:600; font-size:0.85rem;'>"
        f"🟢 SYSTEM OPERATIONAL</span></div>",
        unsafe_allow_html=True
    )

st.markdown("""
<div class="disclaimer-box">
    <b>ℹ️ Verification Protocol:</b> All alerts displayed below are algorithmically flagged candidate events. 
    Officers must inspect the blurred evidence image and scene context before initiating field investigation.
</div>
""", unsafe_allow_html=True)


# Key Performance Indicators (Metrics)
stats = db.get_incident_stats()

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Total Incidents", stats["total"])
with m2:
    st.metric("⚠️ Needs Review", stats["needs_review"])
with m3:
    st.metric("🔴 Confirmed", stats["confirmed"])
with m4:
    st.metric("🟢 Resolved", stats["resolved"])
with m5:
    st.metric("⚪ Dismissed", stats["dismissed"])

st.markdown("---")


# Tabs for Main Content
tab_queue, tab_analytics, tab_map, tab_settings = st.tabs([
    "📋 Incident Review Queue",
    "📊 Analytics & Trends",
    "🗺️ Geospatial Site Map",
    "⚙️ Configuration & Retention"
])


# ==============================================================================
# TAB 1: Incident Review Queue
# ==============================================================================
with tab_queue:
    st.subheader("Candidate Incident Verification")

    # Filters Section
    with st.expander("🔍 Filter Incident Logs", expanded=True):
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            status_filter = st.multiselect(
                "Filter by Status",
                options=VALID_STATUSES,
                default=[]
            )
        with fcol2:
            object_filter = st.selectbox(
                "Detected Object Type",
                options=["All"] + ["bottle", "cup", "backpack", "handbag", "suitcase", "box", "bowl", "waste"]
            )
        with fcol3:
            search_id = st.text_input("Search by Incident ID", placeholder="e.g. INC-20261027-...")

    # Fetch incidents based on filter
    selected_status = status_filter[0] if len(status_filter) == 1 else None
    obj_query = None if object_filter == "All" else object_filter

    incidents_list = db.get_incidents(
        status=selected_status,
        detected_object=obj_query,
        limit=200
    )

    if len(status_filter) > 1:
        incidents_list = [inc for inc in incidents_list if inc["status"] in status_filter]

    if search_id.strip():
        incidents_list = [inc for inc in incidents_list if search_id.strip().lower() in inc["id"].lower()]

    if not incidents_list:
        st.info("No incidents found matching current filter criteria.")
    else:
        df_incidents = pd.DataFrame(incidents_list)

        # Split into Incident Selection & Evidence Review Panel
        left_col, right_col = st.columns([1.1, 0.9])

        with left_col:
            st.markdown(f"**Found {len(incidents_list)} incident records** (Click an incident to review)")

            # Format display dataframe
            display_df = df_incidents[[
                "id", "timestamp", "detected_object", "confidence", "status", "site_name"
            ]].copy()
            display_df["confidence"] = display_df["confidence"].apply(lambda c: f"{c:.2f}")

            # Incident selection
            incident_options = [
                f"{inc['id']} | {inc['detected_object']} | {inc['status']} | {inc['timestamp']}"
                for inc in incidents_list
            ]
            selected_option = st.selectbox("Select Incident to Inspect:", options=incident_options)
            selected_idx = incident_options.index(selected_option)
            active_incident = incidents_list[selected_idx]

            # Render Table
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=380
            )

            # CSV Export
            csv_data = df_incidents.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Filtered Incidents (CSV)",
                data=csv_data,
                file_name=f"dumping_incidents_export_{get_ist_now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with right_col:
            st.markdown(f"### Incident Details: `{active_incident['id']}`")

            # Evidence Image Preview
            img_path = active_incident.get("evidence_path", "")
            if img_path and os.path.exists(img_path):
                st.image(
                    img_path,
                    caption=f"Evidence Frame: {active_incident['detected_object']} (Faces blurred for privacy)",
                    use_column_width=True
                )
            else:
                st.warning("Evidence image not stored or path inaccessible.")

            # Incident Meta Details
            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                st.markdown(f"**Timestamp (IST):** {active_incident['timestamp']}")
                st.markdown(f"**Object:** `{active_incident['detected_object']}`")
                st.markdown(f"**Confidence:** `{active_incident['confidence']:.2f}`")
            with detail_col2:
                st.markdown(f"**Site:** {active_incident['site_name']}")
                st.markdown(f"**Coordinates:** `{active_incident['latitude']}, {active_incident['longitude']}`")
                curr_status = active_incident["status"]
                badge_class = {
                    "Needs Review": "badge-review",
                    "Confirmed": "badge-confirmed",
                    "Resolved": "badge-resolved",
                    "Dismissed": "badge-dismissed"
                }.get(curr_status, "badge-review")
                st.markdown(f"**Status:** <span class='{badge_class}'>{curr_status}</span>", unsafe_allow_html=True)

            if active_incident.get("notes"):
                st.info(f"**System/Audit Notes:** {active_incident['notes']}")

            # Officer Review Form
            st.markdown("#### Human Officer Audit Action")
            with st.form(key=f"audit_form_{active_incident['id']}"):
                officer_name = st.text_input("Reviewing Officer Name / ID", value="Officer Sharma")
                audit_notes = st.text_area("Audit Findings / Reason for Decision", placeholder="Observed person discarding beverage container...")

                btn_col1, btn_col2, btn_col3 = st.columns(3)
                with btn_col1:
                    confirm_btn = st.form_submit_button("🔴 Confirm Dumping", use_container_width=True)
                with btn_col2:
                    dismiss_btn = st.form_submit_button("⚪ Dismiss (False Alarm)", use_container_width=True)
                with btn_col3:
                    resolve_btn = st.form_submit_button("🟢 Mark Resolved", use_container_width=True)

                if confirm_btn:
                    db.update_incident_status(active_incident["id"], "Confirmed", audit_notes, officer_name)
                    st.success("Incident confirmed as illegal dumping.")
                    st.rerun()
                elif dismiss_btn:
                    db.update_incident_status(active_incident["id"], "Dismissed", audit_notes, officer_name)
                    st.info("Incident dismissed as false alarm.")
                    st.rerun()
                elif resolve_btn:
                    db.update_incident_status(active_incident["id"], "Resolved", audit_notes, officer_name)
                    st.success("Incident resolved and site cleared.")
                    st.rerun()


# ==============================================================================
# TAB 2: Analytics & Trends
# ==============================================================================
with tab_analytics:
    st.subheader("Smart Waste Analytics & Detection Trends")

    all_records = db.get_incidents(limit=1000)
    if not all_records:
        st.info("No incident records available to calculate analytics.")
    else:
        df_all = pd.DataFrame(all_records)
        df_all["date"] = pd.to_datetime(df_all["timestamp_iso"]).dt.date

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Daily Incident Frequency")
            daily_counts = df_all.groupby("date").size().reset_index(name="Incident Count")
            st.bar_chart(daily_counts, x="date", y="Incident Count", color="#3B82F6")

        with c2:
            st.markdown("#### Detected Waste Categories")
            cat_counts = df_all["detected_object"].value_counts().reset_index()
            cat_counts.columns = ["Waste Category", "Count"]
            st.bar_chart(cat_counts, x="Waste Category", y="Count", color="#F59E0B")

        st.markdown("#### Incident Status Distribution")
        status_dist = df_all["status"].value_counts().reset_index()
        status_dist.columns = ["Status", "Total"]
        st.dataframe(status_dist, use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 3: Geospatial Site Map
# ==============================================================================
with tab_map:
    st.subheader("Municipal Hotspot Surveillance Map")

    all_records = db.get_incidents(limit=500)
    valid_coords = [
        r for r in all_records
        if r.get("latitude") is not None and r.get("longitude") is not None
    ]

    if valid_coords:
        map_df = pd.DataFrame([
            {"latitude": float(r["latitude"]), "longitude": float(r["longitude"])}
            for r in valid_coords
        ])
        st.map(map_df, zoom=13)

        st.markdown("#### Monitored Site Navigation Links")
        for site in set(r["site_name"] for r in valid_coords):
            site_row = next(r for r in valid_coords if r["site_name"] == site)
            lat, lon = site_row["latitude"], site_row["longitude"]
            gmap_url = f"https://www.google.com/maps?q={lat},{lon}"
            st.markdown(f"• **{site}** (`{lat}, {lon}`) — [Open in Google Maps]({gmap_url})")
    else:
        # Fallback to default configured coordinates
        st.map(pd.DataFrame([{"latitude": 28.6139, "longitude": 77.2090}]), zoom=13)
        st.caption("Showing default configured site coordinates (Sector 4 Central Green).")


# ==============================================================================
# TAB 4: Configuration & Retention
# ==============================================================================
with tab_settings:
    st.subheader("System Configuration & Privacy Retention Policy")

    cfg_path = "config.json"
    current_cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            current_cfg = json.load(f)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("#### Monitored Parameters")
        st.json({
            "Site ID": current_cfg.get("site", {}).get("site_id", "SITE-DELHI-001"),
            "Site Name": current_cfg.get("site", {}).get("site_name", "Rajpath Sector 4"),
            "Event Duration Threshold": f"{current_cfg.get('event_logic', {}).get('event_duration_threshold_seconds', 3.0)}s",
            "Duplicate Cooldown": f"{current_cfg.get('event_logic', {}).get('duplicate_cooldown_seconds', 30.0)}s",
            "Face Blurring": current_cfg.get("evidence", {}).get("blur_faces", True),
            "Retention Period": f"{current_cfg.get('evidence', {}).get('retention_days', 30)} days"
        })

    with col_s2:
        st.markdown("#### Privacy & Data Retention Maintenance")
        st.write(
            "Under municipal smart-city privacy regulations, captured evidence older than "
            "the retention threshold must be purged to protect citizen privacy."
        )

        retention_days = st.slider("Select Retention Threshold (Days):", min_value=1, max_value=90, value=30)
        if st.button("🗑️ Purge Expired Records Now", type="secondary"):
            purged = db.purge_old_incidents(retention_days=retention_days)
            st.success(f"Purge complete. Deleted {purged} expired records and evidence frames.")
            st.rerun()

