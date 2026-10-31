"""
Database Management Module for AI Illegal Dumping Detection System.
Handles persistent SQLite storage, incident logging, status updates,
retrieval, filtering, and privacy-compliant evidence retention.
"""

import os
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("IllegalDumping.Database")

# Standard Indian Standard Time (IST) offset: UTC + 5:30
IST = timezone(timedelta(hours=5, minutes=30))

VALID_STATUSES = ["Needs Review", "Confirmed", "Dismissed", "Resolved"]


def get_ist_now() -> datetime:
    """Return the current datetime in Indian Standard Time (IST)."""
    return datetime.now(IST)


def get_ist_timestamp_str(dt: Optional[datetime] = None) -> str:
    """Format datetime as IST ISO 8601 string with explicit timezone offset."""
    if dt is None:
        dt = get_ist_now()
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    else:
        dt = dt.astimezone(IST)
    return dt.strftime("%Y-%m-%d %H:%M:%S IST")


class IncidentDatabase:
    """Persistent SQLite manager for waste dumping incidents."""

    def __init__(self, db_path: str = "data/dumping_incidents.db"):
        self.db_path = db_path
        # Ensure parent directory exists
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a configured SQLite connection."""
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high-concurrency read/write
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def init_db(self) -> None:
        """Initialize database schema, tables, and performance indexes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS incidents (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    timestamp_iso TEXT NOT NULL,
                    site_id TEXT NOT NULL,
                    site_name TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    detected_object TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    tracking_id INTEGER,
                    evidence_path TEXT,
                    status TEXT NOT NULL CHECK(status IN ('Needs Review', 'Confirmed', 'Dismissed', 'Resolved')),
                    notes TEXT DEFAULT '',
                    reviewed_by TEXT DEFAULT '',
                    reviewed_at TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                );
            """)

            # Indexes for high-speed filtering and analytics
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_incidents_timestamp ON incidents(timestamp_iso);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_incidents_site ON incidents(site_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_incidents_object ON incidents(detected_object);")
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

    def generate_incident_id(self, prefix: str = "INC") -> str:
        """Generate a human-readable, unique incident ID in IST format: INC-YYYYMMDD-HHMMSS-XXXX."""
        now = get_ist_now()
        import random
        rand_suffix = random.randint(1000, 9999)
        return f"{prefix}-{now.strftime('%Y%m%d-%H%M%S')}-{rand_suffix}"

    def insert_incident(
        self,
        detected_object: str,
        confidence: float,
        site_id: str,
        site_name: str,
        latitude: float,
        longitude: float,
        evidence_path: str,
        tracking_id: Optional[int] = None,
        status: str = "Needs Review",
        notes: str = "",
        custom_id: Optional[str] = None,
        custom_dt: Optional[datetime] = None
    ) -> str:
        """
        Record a suspected dumping incident into persistent storage.
        Returns the unique incident ID.
        """
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Must be one of {VALID_STATUSES}")

        incident_id = custom_id or self.generate_incident_id()
        now_ist = custom_dt or get_ist_now()
        timestamp_formatted = get_ist_timestamp_str(now_ist)
        timestamp_iso = now_ist.isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO incidents (
                    id, timestamp, timestamp_iso, site_id, site_name,
                    latitude, longitude, detected_object, confidence,
                    tracking_id, evidence_path, status, notes,
                    reviewed_by, reviewed_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', '', ?)
            """, (
                incident_id,
                timestamp_formatted,
                timestamp_iso,
                site_id,
                site_name,
                float(latitude) if latitude is not None else None,
                float(longitude) if longitude is not None else None,
                detected_object,
                float(confidence),
                int(tracking_id) if tracking_id is not None else None,
                evidence_path,
                status,
                notes,
                timestamp_iso
            ))
            conn.commit()
            logger.info(f"Logged new incident: {incident_id} ({detected_object}, {confidence:.2f})")
            return incident_id

    def update_incident_status(
        self,
        incident_id: str,
        new_status: str,
        notes: Optional[str] = None,
        reviewed_by: Optional[str] = None
    ) -> bool:
        """Update incident review status, reviewer audit signature, and notes."""
        if new_status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{new_status}'. Must be one of {VALID_STATUSES}")

        reviewed_at = get_ist_timestamp_str()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            if notes is not None:
                cursor.execute("""
                    UPDATE incidents
                    SET status = ?, notes = ?, reviewed_by = ?, reviewed_at = ?
                    WHERE id = ?
                """, (new_status, notes, reviewed_by or "System Officer", reviewed_at, incident_id))
            else:
                cursor.execute("""
                    UPDATE incidents
                    SET status = ?, reviewed_by = ?, reviewed_at = ?
                    WHERE id = ?
                """, (new_status, reviewed_by or "System Officer", reviewed_at, incident_id))
            conn.commit()
            updated = cursor.rowcount > 0
            if updated:
                logger.info(f"Incident {incident_id} status updated to '{new_status}'")
            return updated

    def get_incident(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single incident record by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_incidents(
        self,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        site_id: Optional[str] = None,
        detected_object: Optional[str] = None,
        limit: int = 500,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Query incidents with optional filters and pagination.
        Dates can be provided in ISO or 'YYYY-MM-DD' format.
        """
        query = "SELECT * FROM incidents WHERE 1=1"
        params: List[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if site_id:
            query += " AND site_id = ?"
            params.append(site_id)
        if detected_object:
            query += " AND detected_object = ?"
            params.append(detected_object)
        if start_date:
            query += " AND timestamp_iso >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp_iso <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp_iso DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_incident_stats(self) -> Dict[str, int]:
        """Compute aggregate incident statistics for dashboards and reports."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'Needs Review' THEN 1 ELSE 0 END) as needs_review,
                    SUM(CASE WHEN status = 'Confirmed' THEN 1 ELSE 0 END) as confirmed,
                    SUM(CASE WHEN status = 'Resolved' THEN 1 ELSE 0 END) as resolved,
                    SUM(CASE WHEN status = 'Dismissed' THEN 1 ELSE 0 END) as dismissed
                FROM incidents
            """)
            row = cursor.fetchone()
            if row:
                return {
                    "total": row["total"] or 0,
                    "needs_review": row["needs_review"] or 0,
                    "confirmed": row["confirmed"] or 0,
                    "resolved": row["resolved"] or 0,
                    "dismissed": row["dismissed"] or 0
                }
            return {"total": 0, "needs_review": 0, "confirmed": 0, "resolved": 0, "dismissed": 0}

    def purge_old_incidents(self, retention_days: int = 30) -> int:
        """
        Delete incidents older than retention_days to comply with privacy regulations.
        Also deletes associated local evidence image files.
        Returns the number of incidents purged.
        """
        if retention_days <= 0:
            return 0

        cutoff = (get_ist_now() - timedelta(days=retention_days)).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, evidence_path FROM incidents WHERE timestamp_iso < ?", (cutoff,))
            expired = cursor.fetchall()
            deleted_count = 0

            for item in expired:
                evidence_file = item["evidence_path"]
                if evidence_file and os.path.exists(evidence_file):
                    try:
                        os.remove(evidence_file)
                    except OSError as e:
                        logger.warning(f"Could not delete evidence file {evidence_file}: {e}")

                cursor.execute("DELETE FROM incidents WHERE id = ?", (item["id"],))
                deleted_count += 1

            conn.commit()
            if deleted_count > 0:
                logger.info(f"Purged {deleted_count} expired incidents (older than {retention_days} days).")
            return deleted_count

