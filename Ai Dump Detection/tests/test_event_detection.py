"""
Unit tests for spatial-temporal event reasoning, ROI testing,
stationary duration logic, and duplicate-alert prevention.
"""

import time
import pytest
from event_detection import (
    point_in_polygon,
    calculate_euclidean_distance,
    get_bbox_bottom_center,
    get_bbox_center,
    blur_faces_best_effort,
    DumpingEventDetector
)

try:
    import numpy as np
except ImportError:
    np = None


# Define standard test ROI polygon
SAMPLE_ROI = [
    (100.0, 100.0),
    (500.0, 100.0),
    (500.0, 500.0),
    (100.0, 500.0)
]


def test_point_in_polygon_convex():
    """Verify standard point in polygon ray-casting behavior."""
    # Clearly inside
    assert point_in_polygon((300.0, 300.0), SAMPLE_ROI) is True
    assert point_in_polygon((150.0, 150.0), SAMPLE_ROI) is True

    # Clearly outside
    assert point_in_polygon((50.0, 50.0), SAMPLE_ROI) is False
    assert point_in_polygon((600.0, 300.0), SAMPLE_ROI) is False
    assert point_in_polygon((300.0, 600.0), SAMPLE_ROI) is False


def test_point_in_polygon_concave():
    """Verify non-convex (L-shaped) polygon inclusion test."""
    l_shaped_roi = [
        (0.0, 0.0),
        (200.0, 0.0),
        (200.0, 100.0),
        (100.0, 100.0),
        (100.0, 200.0),
        (0.0, 200.0)
    ]
    # Inside horizontal bar
    assert point_in_polygon((150.0, 50.0), l_shaped_roi) is True
    # Inside vertical bar
    assert point_in_polygon((50.0, 150.0), l_shaped_roi) is True
    # In the cut-out corner (outside)
    assert point_in_polygon((150.0, 150.0), l_shaped_roi) is False


def test_calculate_euclidean_distance():
    """Verify Euclidean distance helper."""
    assert calculate_euclidean_distance((0, 0), (3, 4)) == pytest.approx(5.0)
    assert calculate_euclidean_distance((100, 200), (100, 200)) == pytest.approx(0.0)


def test_get_bbox_anchors():
    """Verify bottom-center (ground contact) and center calculations."""
    bbox = [100.0, 200.0, 200.0, 400.0]
    center = get_bbox_center(bbox)
    bottom_center = get_bbox_bottom_center(bbox)

    assert center == (150.0, 300.0)
    assert bottom_center == (150.0, 400.0)


def test_stationary_tracking_and_event_duration():
    """
    Verify temporal state machine:
    An abandoned waste object in ROI triggers an event ONLY after
    event_duration_threshold_seconds has elapsed.
    """
    detector = DumpingEventDetector(
        roi_polygon=SAMPLE_ROI,
        event_duration_threshold=3.0,
        stationary_movement_threshold=20.0,
        person_separation_distance=100.0,
        duplicate_cooldown_seconds=30.0
    )

    t0 = 1000.0
    waste_obj = {
        "track_id": 1,
        "class_name": "bottle",
        "confidence": 0.85,
        "bbox": [280.0, 260.0, 320.0, 340.0]  # Ground contact at (300, 340) inside ROI
    }

    # Frame 1: Object first seen at t0
    events_t0 = detector.process_frame_tracks([waste_obj], current_time=t0)
    assert len(events_t0) == 0

    # Frame 2: Object stationary at t0 + 1.5s (< 3.0s threshold) -> Should NOT trigger
    events_t1 = detector.process_frame_tracks([waste_obj], current_time=t0 + 1.5)
    assert len(events_t1) == 0

    # Frame 3: Object stationary at t0 + 2.9s (< 3.0s threshold) -> Should NOT trigger
    events_t2 = detector.process_frame_tracks([waste_obj], current_time=t0 + 2.9)
    assert len(events_t2) == 0

    # Frame 4: Object stationary at t0 + 3.1s (>= 3.0s threshold) -> MUST trigger!
    events_t3 = detector.process_frame_tracks([waste_obj], current_time=t0 + 3.1)
    assert len(events_t3) == 1
    event = events_t3[0]
    assert event.tracking_id == 1
    assert event.detected_object == "bottle"
    assert event.confidence == pytest.approx(0.85)
    assert event.stationary_duration >= 3.0


def test_moving_object_does_not_trigger():
    """Verify that an object moving through the ROI never triggers stationary dumping."""
    detector = DumpingEventDetector(
        roi_polygon=SAMPLE_ROI,
        event_duration_threshold=2.0,
        stationary_movement_threshold=20.0
    )

    t0 = 1000.0
    for step in range(5):
        # Move object by 50px each second (> 20px threshold)
        x_offset = step * 50.0
        obj = {
            "track_id": 2,
            "class_name": "bottle",
            "confidence": 0.9,
            "bbox": [200.0 + x_offset, 250.0, 240.0 + x_offset, 310.0]
        }
        events = detector.process_frame_tracks([obj], current_time=t0 + step * 1.0)
        assert len(events) == 0


def test_object_outside_roi_does_not_trigger():
    """Verify that an object left stationary outside the ROI is ignored."""
    detector = DumpingEventDetector(
        roi_polygon=SAMPLE_ROI,
        event_duration_threshold=2.0
    )

    t0 = 1000.0
    # Object at (50, 50) which is outside SAMPLE_ROI (100-500)
    outside_obj = {
        "track_id": 3,
        "class_name": "backpack",
        "confidence": 0.9,
        "bbox": [30.0, 30.0, 70.0, 70.0]
    }

    detector.process_frame_tracks([outside_obj], current_time=t0)
    events = detector.process_frame_tracks([outside_obj], current_time=t0 + 5.0)
    assert len(events) == 0


def test_person_proximity_suppresses_dumping():
    """
    Verify that if a person remains close to or is holding the object
    (e.g., waiting at a bus stop with a backpack), dumping alert is suppressed.
    """
    detector = DumpingEventDetector(
        roi_polygon=SAMPLE_ROI,
        event_duration_threshold=2.0,
        person_separation_distance=80.0
    )

    t0 = 1000.0
    waste_obj = {
        "track_id": 4,
        "class_name": "backpack",
        "confidence": 0.92,
        "bbox": [290.0, 290.0, 330.0, 350.0]  # Center at (310, 320)
    }

    # Person stands 30px away from the backpack (< 80px separation threshold)
    person_obj = {
        "track_id": 101,
        "class_name": "person",
        "confidence": 0.95,
        "bbox": [320.0, 200.0, 360.0, 350.0]  # Center at (340, 275)
    }

    detector.process_frame_tracks([waste_obj, person_obj], current_time=t0)
    # 5 seconds elapse with person still holding/near backpack
    events = detector.process_frame_tracks([waste_obj, person_obj], current_time=t0 + 5.0)
    assert len(events) == 0, "Alert should be suppressed when person is holding/near item"

    # Now person walks away (distance > 80px)
    person_walked_away = {
        "track_id": 101,
        "class_name": "person",
        "confidence": 0.95,
        "bbox": [800.0, 200.0, 850.0, 350.0]
    }
    # Once person leaves and duration requirement is satisfied, alert MUST trigger!
    events_after_departure = detector.process_frame_tracks([waste_obj, person_walked_away], current_time=t0 + 6.0)
    assert len(events_after_departure) == 1


def test_duplicate_alert_prevention():
    """Verify that duplicate alerts are not fired repeatedly for the same item during cooldown."""
    detector = DumpingEventDetector(
        roi_polygon=SAMPLE_ROI,
        event_duration_threshold=2.0,
        duplicate_cooldown_seconds=30.0
    )

    t0 = 1000.0
    obj = {
        "track_id": 5,
        "class_name": "box",
        "confidence": 0.88,
        "bbox": [200.0, 200.0, 260.0, 260.0]
    }

    detector.process_frame_tracks([obj], current_time=t0)
    # First alert triggers at t0 + 2.5s
    events_1 = detector.process_frame_tracks([obj], current_time=t0 + 2.5)
    assert len(events_1) == 1

    # Subsequent frames during the next 20s must NOT fire duplicate alerts
    events_2 = detector.process_frame_tracks([obj], current_time=t0 + 5.0)
    assert len(events_2) == 0
    events_3 = detector.process_frame_tracks([obj], current_time=t0 + 15.0)
    assert len(events_3) == 0


def test_face_blurring_best_effort():
    """Verify face blurring runs gracefully and alters pixels in face region."""
    if np is None:
        pytest.skip("NumPy not available")

    # Create dummy 200x200 3-channel image
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    # Place a bright pattern in the head area
    img[20:60, 40:80] = 255

    person_box = [40.0, 20.0, 80.0, 180.0]  # Upper 22% is roughly [20:55, 40:80]
    blurred = blur_faces_best_effort(img, [person_box], blur_kernel_size=31)

    assert blurred is not None
    assert blurred.shape == img.shape
    # Head region should now be smoothed/different from original solid 255
    assert not np.array_equal(img[25:50, 45:75], blurred[25:50, 45:75])

