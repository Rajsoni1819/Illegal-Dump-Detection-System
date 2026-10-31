"""
AI Object Detection and Tracking Engine for Illegal Dumping Detection.
Integrates Ultralytics YOLO with ByteTrack, OpenCV video processing,
spatial-temporal event detection, and privacy face blurring.
"""

import os
import time
import json
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import List, Dict, Any, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from database import IncidentDatabase, get_ist_now, get_ist_timestamp_str
from event_detection import (
    DumpingEventDetector,
    DumpingIncidentCandidate,
    blur_faces_best_effort,
    point_in_polygon
)

logger = logging.getLogger("IllegalDumping.Detector")


class DumpingDetectorEngine:
    """
    Main detection pipeline: captures frames, runs YOLO + tracker,
    evaluates dumping state machine, saves privacy-compliant evidence,
    and updates SQLite database.
    """

    def __init__(self, config_path: str = "config.json", db: Optional[IncidentDatabase] = None):
        self.config_path = config_path
        self.config = self._load_config(config_path)

        # Initialize Database
        db_path = self.config.get("database", {}).get("database_path", "data/dumping_incidents.db")
        self.db = db if db is not None else IncidentDatabase(db_path)

        # Setup Evidence Directory
        self.evidence_dir = self.config.get("evidence", {}).get("evidence_directory", "evidence")
        os.makedirs(self.evidence_dir, exist_ok=True)

        # Setup Event Detector
        event_cfg = self.config.get("event_logic", {})
        det_cfg = self.config.get("detection", {})
        self.event_detector = DumpingEventDetector(
            roi_polygon=event_cfg.get("roi_polygon_normalized", [
                [0.15, 0.35], [0.85, 0.35], [0.90, 0.88], [0.10, 0.88]
            ]),
            event_duration_threshold=event_cfg.get("event_duration_threshold_seconds", 3.0),
            stationary_movement_threshold=event_cfg.get("stationary_movement_threshold_pixels", 25.0),
            person_separation_distance=event_cfg.get("person_separation_distance_pixels", 120.0),
            duplicate_cooldown_seconds=event_cfg.get("duplicate_cooldown_seconds", 30.0),
            waste_classes=det_cfg.get("waste_classes"),
            person_classes=det_cfg.get("person_classes")
        )

        # YOLO Model
        self.model = None
        self.model_loaded = False
        self._init_yolo_model()

        # Camera & State
        self.cap = None
        self.is_running = False
        self.current_fps = 0.0
        self.last_alert_message = ""
        self.alert_display_until = 0.0

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load JSON configuration with sensible fallbacks."""
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading {path}, loading defaults: {e}")

        # Default configuration
        return {
            "camera": {"camera_index": 0, "frame_width": 1280, "frame_height": 720},
            "site": {
                "site_id": "SITE-DELHI-001",
                "site_name": "Rajpath Central Green - Sector 4",
                "latitude": 28.6139,
                "longitude": 77.2090
            },
            "detection": {
                "model_path": "models/yolov8n.pt",
                "confidence_threshold": 0.35,
                "tracker_type": "bytetrack.yaml",
                "person_classes": ["person"],
                "waste_classes": ["bottle", "cup", "backpack", "handbag", "suitcase", "box", "bowl", "waste"]
            },
            "event_logic": {
                "roi_polygon_normalized": [[0.15, 0.35], [0.85, 0.35], [0.90, 0.88], [0.10, 0.88]],
                "event_duration_threshold_seconds": 3.0,
                "stationary_movement_threshold_pixels": 25.0,
                "person_separation_distance_pixels": 120.0,
                "duplicate_cooldown_seconds": 30.0
            },
            "evidence": {
                "save_evidence": True,
                "evidence_directory": "evidence",
                "blur_faces": True,
                "face_blur_factor": 35,
                "retention_days": 30
            },
            "database": {"database_path": "data/dumping_incidents.db"},
            "notifications": {"email_enabled": False}
        }

    def _init_yolo_model(self) -> None:
        """Initialize Ultralytics YOLO model or mock for offline environments."""
        if YOLO is None:
            logger.warning("Ultralytics library not installed. Running in mock/headless mode.")
            return

        model_path = self.config.get("detection", {}).get("model_path", "models/yolov8n.pt")
        try:
            logger.info(f"Loading YOLO model from {model_path}...")
            # If models directory doesn't have it, YOLO will auto-download yolov8n.pt or use local
            self.model = YOLO(model_path if os.path.exists(model_path) else "yolov8n.pt")
            self.model_loaded = True
            logger.info("YOLO model successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}. Falling back to simulated tracker.")
            self.model_loaded = False

    def start_camera(self, source: Optional[Any] = None) -> bool:
        """
        Open video stream from camera index or video file.
        Returns True if successfully opened.
        """
        if cv2 is None:
            logger.error("OpenCV is not installed. Cannot start camera.")
            return False

        if source is None:
            source = self.config.get("camera", {}).get("camera_index", 0)

        # Allow string integers to be converted to int
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        logger.info(f"Connecting to video source: {source}")
        if self.cap is not None:
            self.cap.release()

        # Handle Windows backend (DirectShow) for faster webcam startup if int
        if isinstance(source, int) and os.name == "nt":
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            logger.warning(f"Failed to open source {source}. Attempting default fallback.")
            self.cap = cv2.VideoCapture(source)

        if self.cap.isOpened():
            self.is_running = True
            w = self.config.get("camera", {}).get("frame_width", 1280)
            h = self.config.get("camera", {}).get("frame_height", 720)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            logger.info(f"Camera opened successfully ({w}x{h}).")
            return True
        else:
            logger.error(f"Cannot access camera or video file: {source}")
            self.is_running = False
            return False

    def stop_camera(self) -> None:
        """Release camera and halt processing loop."""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info("Camera stopped.")

    def detect_and_track_frame(self, frame: Any) -> List[Dict[str, Any]]:
        """
        Run object detection and multi-object tracking on a single frame.
        Returns normalized list of tracked objects.
        """
        if frame is None:
            return []

        h, w = frame.shape[:2]
        conf_thresh = float(self.config.get("detection", {}).get("confidence_threshold", 0.35))
        tracker_type = self.config.get("detection", {}).get("tracker_type", "bytetrack.yaml")

        tracked_objects = []

        if self.model_loaded and self.model is not None:
            try:
                # Run YOLO tracking with ByteTrack
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker=tracker_type,
                    conf=conf_thresh,
                    verbose=False
                )
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        # Extract coordinates [x1, y1, x2, y2]
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_idx = int(box.cls[0].cpu().numpy())
                        cls_name = results[0].names.get(cls_idx, "unknown")

                        # Ultralytics track ID (None if not tracked yet)
                        track_id = int(box.id[0].cpu().numpy()) if box.id is not None else -1

                        tracked_objects.append({
                            "track_id": track_id,
                            "class_name": cls_name,
                            "confidence": conf,
                            "bbox": xyxy
                        })
            except Exception as e:
                logger.error(f"Inference error: {e}")

        return tracked_objects

    def process_frame(
        self,
        frame: Any,
        custom_tracks: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Any, List[DumpingIncidentCandidate]]:
        """
        Process a single frame:
        1. Run detection/tracking
        2. Evaluate dumping event state machine
        3. Save evidence & log incident if suspected dumping occurs
        4. Draw HUD overlays and bounding boxes
        """
        if frame is None:
            return None, []

        start_time = time.time()
        h, w = frame.shape[:2]

        # 1. Detect and track
        tracked_objects = custom_tracks if custom_tracks is not None else self.detect_and_track_frame(frame)

        # 2. Evaluate dumping logic
        triggered_events = self.event_detector.process_frame_tracks(
            tracked_objects=tracked_objects,
            current_time=start_time,
            frame_width=w,
            frame_height=h
        )

        # 3. Handle newly triggered dumping events
        for candidate in triggered_events:
            self._handle_incident_event(candidate, frame, tracked_objects)

        # 4. Render HUD overlays on frame
        annotated_frame = self._render_hud_overlay(frame, tracked_objects, triggered_events)

        # Calculate FPS
        elapsed = time.time() - start_time
        if elapsed > 0:
            self.current_fps = 0.9 * self.current_fps + 0.1 * (1.0 / elapsed)

        return annotated_frame, triggered_events

    def _handle_incident_event(
        self,
        candidate: DumpingIncidentCandidate,
        raw_frame: Any,
        all_tracks: List[Dict[str, Any]]
    ) -> None:
        """Create privacy-blurred evidence, record incident in DB, and send alert."""
        site_cfg = self.config.get("site", {})
        evidence_cfg = self.config.get("evidence", {})

        # Save evidence snapshot with best-effort face blurring
        evidence_rel_path = ""
        incident_id = self.db.generate_incident_id()

        if evidence_cfg.get("save_evidence", True) and raw_frame is not None and cv2 is not None:
            filename = f"{incident_id}_{candidate.detected_object}.jpg"
            evidence_full_path = os.path.join(self.evidence_dir, filename)

            # Apply privacy face blurring
            if evidence_cfg.get("blur_faces", True):
                person_boxes = [
                    t["bbox"] for t in all_tracks
                    if t.get("class_name", "").lower() in self.event_detector.person_classes
                ]
                blur_factor = evidence_cfg.get("face_blur_factor", 35)
                evidence_img = blur_faces_best_effort(raw_frame, person_boxes, blur_factor)
            else:
                evidence_img = raw_frame

            # Annotate incident bounding box on evidence
            bx1, by1, bx2, by2 = map(int, candidate.bbox)
            cv2.rectangle(evidence_img, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
            cv2.putText(
                evidence_img,
                f"SUSPECTED DUMPING: {candidate.detected_object.upper()}",
                (bx1, max(20, by1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            cv2.imwrite(evidence_full_path, evidence_img)
            evidence_rel_path = os.path.relpath(evidence_full_path).replace("\\", "/")

        # Insert incident into SQLite Database
        self.db.insert_incident(
            detected_object=candidate.detected_object,
            confidence=candidate.confidence,
            site_id=site_cfg.get("site_id", "SITE-001"),
            site_name=site_cfg.get("site_name", "Monitored Zone"),
            latitude=site_cfg.get("latitude", 28.6139),
            longitude=site_cfg.get("longitude", 77.2090),
            evidence_path=evidence_rel_path,
            tracking_id=candidate.tracking_id,
            status="Needs Review",
            notes=candidate.explanation,
            custom_id=incident_id
        )

        # Trigger visual alert banner
        self.last_alert_message = f"SUSPECTED DUMPING: {candidate.detected_object.upper()} (ID:{candidate.tracking_id})"
        self.alert_display_until = time.time() + 6.0

        # Send optional SMTP email
        if self.config.get("notifications", {}).get("email_enabled", False):
            self._send_email_notification(candidate, incident_id, evidence_rel_path)

    def _send_email_notification(
        self,
        candidate: DumpingIncidentCandidate,
        incident_id: str,
        evidence_path: str
    ) -> None:
        """Send optional email notification using configured SMTP credentials."""
        notif_cfg = self.config.get("notifications", {})
        smtp_server = os.getenv("SMTP_HOST", notif_cfg.get("smtp_server", "smtp.gmail.com"))
        smtp_port = int(os.getenv("SMTP_PORT", notif_cfg.get("smtp_port", 587)))
        smtp_user = os.getenv("SMTP_USER", notif_cfg.get("sender_email", ""))
        smtp_pass = os.getenv("SMTP_PASSWORD", "")
        recipient = os.getenv("SMTP_RECIPIENT", notif_cfg.get("recipient_email", ""))

        if not (smtp_user and smtp_pass and recipient):
            logger.info("SMTP credentials not fully configured. Email alert skipped.")
            return

        try:
            msg = MIMEMultipart()
            msg["From"] = smtp_user
            msg["To"] = recipient
            msg["Subject"] = f"[CRITICAL ALERT] Suspected Illegal Dumping - {incident_id}"

            body = (
                f"Smart City Waste Management Alert\n"
                f"-----------------------------------\n"
                f"Incident ID: {incident_id}\n"
                f"Timestamp: {get_ist_timestamp_str()}\n"
                f"Detected Object: {candidate.detected_object} (Conf: {candidate.confidence:.2f})\n"
                f"Site: {self.config.get('site', {}).get('site_name', 'Unknown')}\n"
                f"Details: {candidate.explanation}\n\n"
                f"Note: This is an unverified suspected incident. Please review on the dashboard.\n"
            )
            msg.attach(MIMEText(body, "plain"))

            # Attach evidence image if available
            if evidence_path and os.path.exists(evidence_path):
                with open(evidence_path, "rb") as f:
                    img_data = f.read()
                    image_part = MIMEImage(img_data, name=os.path.basename(evidence_path))
                    msg.attach(image_part)

            server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
            if notif_cfg.get("use_tls", True):
                server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, recipient, msg.as_string())
            server.quit()
            logger.info(f"Email alert sent successfully to {recipient}")
        except Exception as e:
            logger.warning(f"Failed to dispatch email alert: {e}")

    def _render_hud_overlay(
        self,
        frame: Any,
        tracked_objects: List[Dict[str, Any]],
        triggered_events: List[DumpingIncidentCandidate]
    ) -> Any:
        """Draw smart city HUD overlay, ROI polygon, bounding boxes, and alert banners."""
        if frame is None or cv2 is None or np is None:
            return frame

        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]

        # 1. Determine ROI polygon color and draw
        effective_roi = self.event_detector.convert_normalized_roi(w, h)
        roi_pts = np.array(effective_roi, np.int32).reshape((-1, 1, 2))

        # Check if any waste item is currently stationary inside ROI
        any_stationary = any(
            item.is_in_roi and item.is_stationary
            for item in self.event_detector.tracks.values()
        )
        is_alerting = time.time() < self.alert_display_until

        if is_alerting:
            roi_color = (0, 0, 255)       # Red: Dumping suspected
            roi_fill_alpha = 0.25
        elif any_stationary:
            roi_color = (0, 220, 255)     # Yellow: Stationary item under monitoring
            roi_fill_alpha = 0.15
        else:
            roi_color = (0, 255, 128)     # Green: Normal monitoring
            roi_fill_alpha = 0.08

        overlay = vis_frame.copy()
        cv2.fillPoly(overlay, [roi_pts], roi_color)
        cv2.addWeighted(overlay, roi_fill_alpha, vis_frame, 1 - roi_fill_alpha, 0, vis_frame)
        cv2.polylines(vis_frame, [roi_pts], isClosed=True, color=roi_color, thickness=2)

        # Label ROI
        if len(effective_roi) > 0:
            rx, ry = int(effective_roi[0][0]), int(effective_roi[0][1])
            cv2.putText(
                vis_frame,
                "ZONE: NO DUMPING DESIGNATED AREA",
                (rx + 10, ry + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                roi_color,
                2
            )

        # 2. Draw Bounding Boxes and Tracking IDs
        for obj in tracked_objects:
            tid = obj.get("track_id", -1)
            cls_name = obj.get("class_name", "obj")
            conf = float(obj.get("confidence", 0.0))
            bx1, by1, bx2, by2 = map(int, obj.get("bbox", [0, 0, 0, 0]))

            is_person = cls_name.lower() in self.event_detector.person_classes
            is_waste = cls_name.lower() in self.event_detector.waste_classes

            if is_person:
                box_color = (255, 180, 0)  # Cyan/Blue for person
            elif is_waste:
                # Check if tracked item is stationary in ROI
                item = self.event_detector.tracks.get(tid)
                if item and item.is_in_roi and item.is_stationary:
                    box_color = (0, 0, 255) if item.alert_triggered else (0, 215, 255)
                else:
                    box_color = (200, 200, 200)
            else:
                box_color = (150, 150, 150)

            cv2.rectangle(vis_frame, (bx1, by1), (bx2, by2), box_color, 2)
            label = f"#{tid} {cls_name} {conf:.2f}"
            cv2.putText(vis_frame, label, (bx1, max(18, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)

            # Draw stationary countdown bar for waste in ROI
            if is_waste:
                item = self.event_detector.tracks.get(tid)
                if item and item.is_in_roi and item.is_stationary:
                    dur = item.stationary_duration
                    thresh = self.event_detector.event_duration_threshold
                    progress = min(1.0, dur / thresh)
                    bar_w = bx2 - bx1
                    bar_y = max(5, by1 - 14)
                    # Background bar
                    cv2.rectangle(vis_frame, (bx1, bar_y), (bx2, bar_y + 5), (50, 50, 50), -1)
                    # Filled progress
                    fill_x = int(bx1 + bar_w * progress)
                    bar_col = (0, 0, 255) if progress >= 1.0 else (0, 255, 255)
                    cv2.rectangle(vis_frame, (bx1, bar_y), (fill_x, bar_y + 5), bar_col, -1)
                    cv2.putText(
                        vis_frame,
                        f"Stationary: {dur:.1f}s",
                        (bx1, bar_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        bar_col,
                        1
                    )

        # 3. Top HUD Status Bar
        hud_bg = vis_frame.copy()
        cv2.rectangle(hud_bg, (0, 0), (w, 42), (20, 25, 30), -1)
        cv2.addWeighted(hud_bg, 0.85, vis_frame, 0.15, 0, vis_frame)

        site_id = self.config.get("site", {}).get("site_id", "SITE-001")
        ist_time = get_ist_timestamp_str()
        cv2.putText(
            vis_frame,
            f"SMART WASTE MONITOR [{site_id}] | FPS: {self.current_fps:.1f} | TRACKS: {len(tracked_objects)}",
            (15, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 200),
            2
        )
        cv2.putText(
            vis_frame,
            ist_time,
            (w - 280, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1
        )

        # 4. Critical Alert Banner
        if is_alerting:
            alert_bar = vis_frame.copy()
            cv2.rectangle(alert_bar, (0, h - 50), (w, h), (0, 0, 180), -1)
            cv2.addWeighted(alert_bar, 0.9, vis_frame, 0.1, 0, vis_frame)
            cv2.putText(
                vis_frame,
                f"ALERT: {self.last_alert_message} - LOGGED TO AUDIT DATABASE",
                (20, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        return vis_frame

