"""
Event Detection and Spatial-Temporal Reasoning Module.
Implements region-of-interest (ROI) testing, stationary object tracking,
person-waste interaction analysis, duplicate-alert suppression,
and privacy-preserving face blurring.
"""

import time
import math
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

logger = logging.getLogger("IllegalDumping.EventDetection")


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    Determine if a 2D point (x, y) is inside a polygon using ray casting algorithm.
    Works for convex and concave polygons.
    """
    if len(polygon) < 3:
        return False

    x, y = point
    inside = False
    n = len(polygon)
    p1x, p1y = polygon[0]

    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def calculate_euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    """Compute Euclidean distance between two 2D points."""
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])


def get_bbox_bottom_center(bbox: List[float]) -> Tuple[float, float]:
    """Get the bottom-center point (ground contact point) of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Get the geometric center of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def blur_faces_best_effort(
    image: Any,
    person_boxes: Optional[List[List[float]]] = None,
    blur_kernel_size: int = 45
) -> Any:
    """
    Apply best-effort privacy blurring to human faces in the image.
    Uses OpenCV Haar Cascades if available and also blurs upper head regions of detected persons.
    Never raises an exception; gracefully returns original image if OpenCV unavailable.
    """
    if image is None or cv2 is None or np is None:
        return image

    output_img = image.copy()
    h, w = output_img.shape[:2]

    # 1. Attempt Haar Cascade Face Detection
    try:
        cascade_path = getattr(cv2.data, 'haarcascades', '') + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if not face_cascade.empty():
            gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30))
            for (fx, fy, fw, fh) in faces:
                # Apply heavy Gaussian blur
                k = max(15, blur_kernel_size | 1)
                sub_face = output_img[fy:fy + fh, fx:fx + fw]
                output_img[fy:fy + fh, fx:fx + fw] = cv2.GaussianBlur(sub_face, (k, k), 30)
    except Exception as e:
        logger.debug(f"Haar face cascade blur skipped: {e}")

    # 2. Reinforce privacy by blurring head/face region of all detected person bounding boxes
    if person_boxes:
        for pbox in person_boxes:
            px1, py1, px2, py2 = map(int, pbox)
            px1 = max(0, min(w - 1, px1))
            py1 = max(0, min(h - 1, py1))
            px2 = max(0, min(w, px2))
            py2 = max(0, min(h, py2))

            person_h = py2 - py1
            person_w = px2 - px1
            if person_h > 20 and person_w > 10:
                # Upper 22% is the head / face region
                head_y2 = min(h, int(py1 + 0.22 * person_h))
                k = max(21, blur_kernel_size | 1)
                head_roi = output_img[py1:head_y2, px1:px2]
                if head_roi.size > 0:
                    output_img[py1:head_y2, px1:px2] = cv2.GaussianBlur(head_roi, (k, k), 25)

    return output_img


@dataclass
class TrackedItem:
    """Tracks state and trajectory of an object across consecutive frames."""
    track_id: int
    class_name: str
    confidence: float
    bbox: List[float]
    first_seen: float
    last_seen: float
    positions: deque = field(default_factory=lambda: deque(maxlen=60))
    is_in_roi: bool = False
    roi_entry_time: Optional[float] = None
    is_stationary: bool = False
    stationary_start_time: Optional[float] = None
    stationary_duration: float = 0.0
    associated_person_id: Optional[int] = None
    min_person_distance: float = float("inf")
    alert_triggered: bool = False
    last_alert_time: float = 0.0


@dataclass
class DumpingIncidentCandidate:
    """Details of a candidate dumping event awaiting review."""
    tracking_id: int
    detected_object: str
    confidence: float
    bbox: List[float]
    location_point: Tuple[float, float]
    stationary_duration: float
    timestamp: float
    explanation: str


class DumpingEventDetector:
    """
    Spatial-Temporal Event Detection Engine.
    Analyzes tracked objects in a Region of Interest (ROI) and detects abandoned/dumped items.
    """

    def __init__(
        self,
        roi_polygon: List[Tuple[float, float]],
        event_duration_threshold: float = 3.0,
        stationary_movement_threshold: float = 25.0,
        person_separation_distance: float = 120.0,
        duplicate_cooldown_seconds: float = 30.0,
        waste_classes: Optional[List[str]] = None,
        person_classes: Optional[List[str]] = None
    ):
        self.roi_polygon = roi_polygon
        self.event_duration_threshold = float(event_duration_threshold)
        self.stationary_movement_threshold = float(stationary_movement_threshold)
        self.person_separation_distance = float(person_separation_distance)
        self.duplicate_cooldown_seconds = float(duplicate_cooldown_seconds)

        self.waste_classes = set(waste_classes or [
            "bottle", "cup", "backpack", "handbag", "suitcase", "box", "bowl", "waste"
        ])
        self.person_classes = set(person_classes or ["person"])

        # Active tracked entities: track_id -> TrackedItem
        self.tracks: Dict[int, TrackedItem] = {}
        # Spatial alert cooldown cache: list of (center_pt, alert_timestamp)
        self.recent_alerts: List[Tuple[Tuple[float, float], float]] = []

    def update_roi(self, new_roi_polygon: List[Tuple[float, float]]) -> None:
        """Dynamically update monitoring region of interest."""
        self.roi_polygon = new_roi_polygon

    def convert_normalized_roi(self, frame_width: int, frame_height: int) -> List[Tuple[float, float]]:
        """Convert normalized (0.0 - 1.0) ROI coordinates to pixel coordinates."""
        pixel_roi = []
        for pt in self.roi_polygon:
            px = pt[0] * frame_width if pt[0] <= 1.0 else pt[0]
            py = pt[1] * frame_height if pt[1] <= 1.0 else pt[1]
            pixel_roi.append((float(px), float(py)))
        return pixel_roi

    def process_frame_tracks(
        self,
        tracked_objects: List[Dict[str, Any]],
        current_time: Optional[float] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None
    ) -> List[DumpingIncidentCandidate]:
        """
        Evaluate tracked bounding boxes for a single video frame.

        Parameters:
            tracked_objects: list of dicts with keys:
                - 'track_id': int
                - 'class_name': str
                - 'confidence': float
                - 'bbox': [x1, y1, x2, y2]
            current_time: epoch seconds (defaults to time.time())
            frame_width, frame_height: dimensions to scale normalized ROI if needed

        Returns:
            List of candidate dumping events that triggered in this frame.
        """
        now = current_time if current_time is not None else time.time()
        active_ids = set()
        triggered_candidates: List[DumpingIncidentCandidate] = []

        # Convert ROI if normalized
        effective_roi = self.roi_polygon
        if frame_width and frame_height and self.roi_polygon and self.roi_polygon[0][0] <= 1.0:
            effective_roi = self.convert_normalized_roi(frame_width, frame_height)

        # 1. Separate persons from candidate waste items
        persons: List[Dict[str, Any]] = []
        waste_items: List[Dict[str, Any]] = []

        for obj in tracked_objects:
            cls = obj.get("class_name", "").lower()
            if cls in self.person_classes:
                persons.append(obj)
            elif cls in self.waste_classes:
                waste_items.append(obj)

        # 2. Update active tracks
        for obj in waste_items:
            tid = obj["track_id"]
            bbox = obj["bbox"]
            cls = obj.get("class_name", "waste")
            conf = float(obj.get("confidence", 0.5))
            active_ids.add(tid)

            ground_pt = get_bbox_bottom_center(bbox)
            in_roi = point_in_polygon(ground_pt, effective_roi)

            if tid not in self.tracks:
                item = TrackedItem(
                    track_id=tid,
                    class_name=cls,
                    confidence=conf,
                    bbox=bbox,
                    first_seen=now,
                    last_seen=now,
                    is_in_roi=in_roi,
                    roi_entry_time=now if in_roi else None
                )
                self.tracks[tid] = item
            else:
                item = self.tracks[tid]
                item.bbox = bbox
                item.confidence = conf
                item.last_seen = now
                if in_roi and not item.is_in_roi:
                    item.roi_entry_time = now
                elif not in_roi:
                    item.roi_entry_time = None
                item.is_in_roi = in_roi

            item.positions.append((ground_pt[0], ground_pt[1], now))

            # 3. Analyze movement / stationary status
            if item.is_in_roi and len(item.positions) >= 2:
                # Displacement over window of positions
                oldest_pt = item.positions[0]
                displacement = calculate_euclidean_distance(ground_pt, (oldest_pt[0], oldest_pt[1]))

                if displacement < self.stationary_movement_threshold:
                    if not item.is_stationary:
                        item.is_stationary = True
                        item.stationary_start_time = now
                    item.stationary_duration = now - (item.stationary_start_time or now)
                else:
                    # Object moved significantly
                    item.is_stationary = False
                    item.stationary_start_time = None
                    item.stationary_duration = 0.0
            else:
                item.is_stationary = False
                item.stationary_start_time = None
                item.stationary_duration = 0.0

            # 4. Check proximity to nearby persons
            min_dist = float("inf")
            nearest_person_id = None
            for p in persons:
                p_center = get_bbox_center(p["bbox"])
                dist = calculate_euclidean_distance(ground_pt, p_center)
                if dist < min_dist:
                    min_dist = dist
                    nearest_person_id = p.get("track_id")

            item.min_person_distance = min_dist
            item.associated_person_id = nearest_person_id

            # 5. Evaluate Candidate Dumping Event criteria:
            # - Inside designated ROI
            # - Stationary for >= event_duration_threshold
            # - Person is separated / not actively holding item (dist > person_separation_distance or no person)
            # - Not already alerted / suppressed by cooldown
            is_abandoned = (min_dist > self.person_separation_distance or not persons)

            if item.is_in_roi and item.is_stationary and item.stationary_duration >= self.event_duration_threshold:
                if is_abandoned and not item.alert_triggered:
                    # Check spatial duplicate alert cooldown
                    if not self._is_spatially_cooldown_active(ground_pt, now):
                        explanation = (
                            f"Object '{cls}' (ID:{tid}) left stationary in ROI for "
                            f"{item.stationary_duration:.1f}s without active handler "
                            f"(nearest person {min_dist:.1f}px away)."
                        )
                        candidate = DumpingIncidentCandidate(
                            tracking_id=tid,
                            detected_object=cls,
                            confidence=conf,
                            bbox=bbox,
                            location_point=ground_pt,
                            stationary_duration=item.stationary_duration,
                            timestamp=now,
                            explanation=explanation
                        )
                        triggered_candidates.append(candidate)
                        item.alert_triggered = True
                        item.last_alert_time = now
                        self.recent_alerts.append((ground_pt, now))
                        logger.info(f"Triggered suspected dumping candidate: {explanation}")

        # 6. Housekeeping: Remove tracks unseen for > 15 seconds
        stale_threshold = 15.0
        to_delete = [
            tid for tid, item in self.tracks.items()
            if (now - item.last_seen) > stale_threshold and tid not in active_ids
        ]
        for tid in to_delete:
            del self.tracks[tid]

        # Purge old spatial cooldown alerts older than cooldown period
        self.recent_alerts = [
            (pt, t) for (pt, t) in self.recent_alerts
            if (now - t) < self.duplicate_cooldown_seconds
        ]

        return triggered_candidates

    def _is_spatially_cooldown_active(self, point: Tuple[float, float], current_time: float) -> bool:
        """Check if an alert was triggered at or very near this point within cooldown period."""
        cooldown_radius = 50.0  # pixels
        for alert_pt, alert_time in self.recent_alerts:
            if (current_time - alert_time) < self.duplicate_cooldown_seconds:
                if calculate_euclidean_distance(point, alert_pt) < cooldown_radius:
                    return True
        return False

