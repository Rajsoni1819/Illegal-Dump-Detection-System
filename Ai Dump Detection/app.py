"""
AI Illegal Dumping Detection System - Main Application Entrypoint.
Provides command-line interface, live webcam/video monitoring loop,
real-time HUD display, and automated demo mode.
"""

import os
import sys
import time
import argparse
import logging
from typing import Optional

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from database import IncidentDatabase, get_ist_timestamp_str
from detector import DumpingDetectorEngine

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("IllegalDumping.App")


def generate_synthetic_demo_video(output_path: str = "demo_video.mp4", num_frames: int = 180) -> str:
    """
    Generate a synthetic test video simulating a person entering the ROI,
    depositing a waste item (bottle/bag), and walking away.
    Used for demo and automated verification when no physical webcam is connected.
    """
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required to generate demo video.")

    w, h = 1280, 720
    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    logger.info(f"Generating synthetic demonstration video: {output_path} ({num_frames} frames)...")

    # Simulation timeline:
    # Frame 0 - 40: Person walks in from left (x: 100 -> 500), carrying item
    # Frame 41 - 70: Person drops item at (x: 550, y: 500) inside ROI
    # Frame 71 - 140: Person walks away to the right (x: 600 -> 1200)
    # Frame 141 - 180: Item remains stationary alone in ROI -> Triggers dumping alert!

    item_x, item_y = 550, 480
    dropped = False

    for f in range(num_frames):
        # Create a clean street/pavement background
        frame = np.full((h, w, 3), 45, dtype=np.uint8)

        # Draw road / sidewalk
        cv2.rectangle(frame, (0, 300), (w, h), (75, 75, 80), -1)
        # Sidewalk curb
        cv2.line(frame, (0, 300), (w, 300), (160, 160, 160), 3)
        # Road markings
        for rx in range(0, w, 100):
            cv2.line(frame, (rx, 580), (rx + 50, 580), (220, 220, 220), 4)

        # Draw ROI Boundary (Designated No Dumping Zone)
        roi_pts = np.array([[200, 320], [1080, 320], [1150, 650], [130, 650]], np.int32)
        cv2.polylines(frame, [roi_pts], True, (0, 255, 128), 2)
        cv2.putText(frame, "MONITORED ROI: NO DUMPING ZONE", (220, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2)

        # Compute person position
        if f <= 50:
            person_x = int(100 + (f / 50.0) * 450)
            person_y = 380
        else:
            dropped = True
            person_x = int(550 + ((f - 50) / 90.0) * 650)
            person_y = 380

        # Draw Person if still in frame
        if person_x < w + 50:
            # Person head (circle)
            cv2.circle(frame, (person_x, person_y), 24, (200, 170, 150), -1)
            # Body (rectangle)
            cv2.rectangle(frame, (person_x - 22, person_y + 24), (person_x + 22, person_y + 110), (180, 90, 40), -1)
            # Legs
            cv2.line(frame, (person_x - 12, person_y + 110), (person_x - 15, person_y + 170), (40, 40, 40), 6)
            cv2.line(frame, (person_x + 12, person_y + 110), (person_x + 15, person_y + 170), (40, 40, 40), 6)
            cv2.putText(frame, "PERSON", (person_x - 30, person_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

        # Draw Waste Item (Bag / Bottle)
        if not dropped:
            # Item carried by person
            curr_item_x = person_x + 30
            curr_item_y = person_y + 60
        else:
            # Stationary dropped item on the ground inside ROI
            curr_item_x = item_x
            curr_item_y = item_y

        # Waste item representation
        cv2.rectangle(frame, (curr_item_x - 16, curr_item_y - 20), (curr_item_x + 16, curr_item_y + 20), (0, 140, 255), -1)
        cv2.putText(frame, "WASTE", (curr_item_x - 20, curr_item_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)

        out.write(frame)

    out.release()
    logger.info(f"Synthetic demo video generated successfully at {output_path}")
    return output_path


def run_monitoring_loop(
    source: Optional[Any] = None,
    config_path: str = "config.json",
    headless: bool = False,
    max_frames: Optional[int] = None
) -> None:
    """Run real-time detection and incident management pipeline."""
    logger.info("Initializing AI Illegal Dumping Detection System...")
    engine = DumpingDetectorEngine(config_path=config_path)

    # If demo video requested and not found, generate it
    if isinstance(source, str) and source.endswith(".mp4") and not os.path.exists(source):
        logger.info(f"Source video '{source}' does not exist. Creating synthetic demonstration video...")
        generate_synthetic_demo_video(source)

    success = engine.start_camera(source)
    if not success:
        logger.error("Failed to start video source. Exiting monitoring loop.")
        sys.exit(1)

    window_name = "AI Illegal Dumping Detection System (Press 'q' to Quit)"
    if not headless and cv2 is not None:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)
        except Exception as e:
            logger.warning(f"Could not open GUI window (headless environment detected): {e}")
            headless = True

    frame_count = 0
    logger.info("Monitoring started. Processing video frames...")

    try:
        while engine.is_running:
            ret, frame = engine.cap.read()
            if not ret or frame is None:
                # Video ended or camera disconnected
                if isinstance(source, str) and os.path.exists(source):
                    logger.info("Video stream completed.")
                else:
                    logger.warning("Camera frame read error or disconnected.")
                break

            # Process frame through detection and event reasoning
            annotated_frame, triggered_events = engine.process_frame(frame)
            frame_count += 1

            for event in triggered_events:
                logger.info(
                    f"⚠️ [INCIDENT LOGGED] Object: {event.detected_object} | "
                    f"TrackID: {event.tracking_id} | "
                    f"Duration: {event.stationary_duration:.1f}s | "
                    f"Time: {get_ist_timestamp_str()}"
                )

            # Display GUI window if not in headless mode
            if not headless and cv2 is not None:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or Esc to quit
                    logger.info("Exit requested by user.")
                    break

            if max_frames and frame_count >= max_frames:
                logger.info(f"Reached specified frame limit ({max_frames}). Stopping.")
                break

    except KeyboardInterrupt:
        logger.info("Monitoring interrupted via KeyboardInterrupt.")
    finally:
        engine.stop_camera()
        if not headless and cv2 is not None:
            cv2.destroyAllWindows()
        logger.info(f"Monitoring stopped. Total frames processed: {frame_count}")


def main():
    """Parse command line arguments and execute detection system."""
    parser = argparse.ArgumentParser(
        description="AI Illegal Dumping Detection System - Smart City Surveillance"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video source: camera index (e.g. '0') or video file path (e.g. 'demo_video.mp4')"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to JSON configuration file (default: config.json)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with synthetic video generation"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI OpenCV window (suitable for servers and testing)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit execution to a fixed number of frames (useful for benchmarking)"
    )
    args = parser.parse_args()

    source = args.source
    if args.demo:
        source = "demo_video.mp4"
        if not os.path.exists(source):
            generate_synthetic_demo_video(source)

    run_monitoring_loop(
        source=source,
        config_path=args.config,
        headless=args.headless,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()

