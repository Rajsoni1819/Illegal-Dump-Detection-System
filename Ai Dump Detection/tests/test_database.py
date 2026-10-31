"""
Unit tests for persistent SQLite database and incident management.
"""

import os
import tempfile
import pytest
from datetime import datetime, timezone, timedelta
from database import IncidentDatabase, get_ist_now, get_ist_timestamp_str, IST, VALID_STATUSES


@pytest.fixture
def temp_db():
    """Fixture providing an isolated temporary database for each test."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_incidents.db")
    db = IncidentDatabase(db_path)
    yield db
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass


def test_db_initialization(temp_db):
    """Verify tables and indexes initialize cleanly with empty statistics."""
    assert os.path.exists(temp_db.db_path)
    stats = temp_db.get_incident_stats()
    assert stats["total"] == 0
    assert stats["needs_review"] == 0
    assert stats["confirmed"] == 0
    assert stats["resolved"] == 0
    assert stats["dismissed"] == 0


def test_incident_insertion(temp_db):
    """Verify inserting an incident returns an ID in IST format and stores attributes."""
    incident_id = temp_db.insert_incident(
        detected_object="bottle",
        confidence=0.88,
        site_id="SITE-TEST-001",
        site_name="Test Sector 1",
        latitude=28.6139,
        longitude=77.2090,
        evidence_path="evidence/test.jpg",
        tracking_id=12,
        notes="Suspected plastic bottle discarded"
    )

    assert incident_id.startswith("INC-")
    incident = temp_db.get_incident(incident_id)
    assert incident is not None
    assert incident["id"] == incident_id
    assert incident["detected_object"] == "bottle"
    assert incident["confidence"] == pytest.approx(0.88, 0.01)
    assert incident["status"] == "Needs Review"
    assert incident["site_id"] == "SITE-TEST-001"
    assert incident["site_name"] == "Test Sector 1"
    assert "IST" in incident["timestamp"]


def test_invalid_status_rejection(temp_db):
    """Verify that an invalid status raises a ValueError."""
    with pytest.raises(ValueError):
        temp_db.insert_incident(
            detected_object="bag",
            confidence=0.75,
            site_id="SITE-TEST",
            site_name="Zone A",
            latitude=0.0,
            longitude=0.0,
            evidence_path="",
            status="InvalidStatus"
        )


def test_incident_status_update(temp_db):
    """Verify human review status updates and audit trailing."""
    inc_id = temp_db.insert_incident(
        detected_object="box",
        confidence=0.91,
        site_id="SITE-TEST-002",
        site_name="Test Sector 2",
        latitude=28.5,
        longitude=77.1,
        evidence_path="evidence/box.jpg"
    )

    # 1. Update to Confirmed
    success = temp_db.update_incident_status(
        inc_id,
        new_status="Confirmed",
        notes="Officer verified carton dumped on roadside",
        reviewed_by="Officer Verma"
    )
    assert success is True
    updated = temp_db.get_incident(inc_id)
    assert updated["status"] == "Confirmed"
    assert updated["reviewed_by"] == "Officer Verma"
    assert "carton dumped" in updated["notes"]
    assert "IST" in updated["reviewed_at"]

    # 2. Update to Resolved
    temp_db.update_incident_status(inc_id, "Resolved", notes="Waste collected and cleared")
    resolved = temp_db.get_incident(inc_id)
    assert resolved["status"] == "Resolved"

    # 3. Invalid status update
    with pytest.raises(ValueError):
        temp_db.update_incident_status(inc_id, "NonExistentStatus")


def test_get_incidents_filtering(temp_db):
    """Verify query filtering by status, object type, and pagination."""
    # Insert multiple test incidents
    temp_db.insert_incident("bottle", 0.8, "SITE-1", "Zone 1", 28.0, 77.0, "", status="Needs Review")
    temp_db.insert_incident("bottle", 0.9, "SITE-1", "Zone 1", 28.0, 77.0, "", status="Confirmed")
    temp_db.insert_incident("backpack", 0.7, "SITE-2", "Zone 2", 28.1, 77.1, "", status="Needs Review")
    temp_db.insert_incident("suitcase", 0.85, "SITE-2", "Zone 2", 28.1, 77.1, "", status="Dismissed")

    # Filter by status
    needs_review = temp_db.get_incidents(status="Needs Review")
    assert len(needs_review) == 2

    confirmed = temp_db.get_incidents(status="Confirmed")
    assert len(confirmed) == 1
    assert confirmed[0]["detected_object"] == "bottle"

    # Filter by object
    bottles = temp_db.get_incidents(detected_object="bottle")
    assert len(bottles) == 2

    # Filter by site_id
    site2 = temp_db.get_incidents(site_id="SITE-2")
    assert len(site2) == 2

    # Test stats aggregation
    stats = temp_db.get_incident_stats()
    assert stats["total"] == 4
    assert stats["needs_review"] == 2
    assert stats["confirmed"] == 1
    assert stats["dismissed"] == 1
    assert stats["resolved"] == 0


def test_retention_purge(temp_db):
    """Verify that records older than retention threshold are deleted along with evidence."""
    temp_dir = tempfile.mkdtemp()
    old_evidence = os.path.join(temp_dir, "old_evidence.jpg")
    new_evidence = os.path.join(temp_dir, "new_evidence.jpg")

    with open(old_evidence, "w") as f:
        f.write("fake-image-data-old")
    with open(new_evidence, "w") as f:
        f.write("fake-image-data-new")

    # Insert old incident (45 days ago)
    old_dt = get_ist_now() - timedelta(days=45)
    temp_db.insert_incident(
        detected_object="bottle",
        confidence=0.8,
        site_id="SITE-1",
        site_name="Zone 1",
        latitude=28.0,
        longitude=77.0,
        evidence_path=old_evidence,
        custom_dt=old_dt
    )

    # Insert recent incident (2 days ago)
    recent_dt = get_ist_now() - timedelta(days=2)
    recent_id = temp_db.insert_incident(
        detected_object="cup",
        confidence=0.9,
        site_id="SITE-1",
        site_name="Zone 1",
        latitude=28.0,
        longitude=77.0,
        evidence_path=new_evidence,
        custom_dt=recent_dt
    )

    # Purge incidents older than 30 days
    purged_count = temp_db.purge_old_incidents(retention_days=30)
    assert purged_count == 1
    assert not os.path.exists(old_evidence)
    assert os.path.exists(new_evidence)

    # Verify only recent incident remains
    remaining = temp_db.get_incidents()
    assert len(remaining) == 1
    assert remaining[0]["id"] == recent_id

