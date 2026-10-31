"""
Unit tests for configuration loading, validation, and fallback handling.
"""

import os
import json
import tempfile
import pytest
from detector import DumpingDetectorEngine
from event_detection import DumpingEventDetector


def test_load_valid_config():
    """Verify loading from valid config.json."""
    config_path = "config.json"
    assert os.path.exists(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert "camera" in cfg
    assert "site" in cfg
    assert "detection" in cfg
    assert "event_logic" in cfg
    assert "evidence" in cfg
    assert "database" in cfg
    assert "notifications" in cfg

    # Verify key data types
    assert isinstance(cfg["site"]["latitude"], float)
    assert isinstance(cfg["site"]["longitude"], float)
    assert isinstance(cfg["event_logic"]["roi_polygon_normalized"], list)
    assert cfg["event_logic"]["event_duration_threshold_seconds"] > 0


def test_invalid_config_fallback():
    """Verify that a corrupted or non-existent configuration gracefully falls back to defaults."""
    temp_dir = tempfile.mkdtemp()
    corrupt_cfg = os.path.join(temp_dir, "corrupt.json")

    with open(corrupt_cfg, "w") as f:
        f.write("{ INVALID JSON CONTENT :::")

    engine = DumpingDetectorEngine(config_path=corrupt_cfg)
    assert engine.config is not None
    assert "camera" in engine.config
    assert "site" in engine.config
    assert engine.config["site"]["site_id"] == "SITE-DELHI-001"
    assert engine.event_detector.event_duration_threshold == 3.0


def test_normalized_roi_scaling():
    """Verify normalized (0.0 to 1.0) coordinates scale accurately across diverse camera resolutions."""
    normalized_roi = [
        [0.1, 0.2],
        [0.9, 0.2],
        [0.9, 0.8],
        [0.1, 0.8]
    ]
    detector = DumpingEventDetector(roi_polygon=normalized_roi)

    # Scale to 1280x720 (720p HD)
    scaled_720p = detector.convert_normalized_roi(frame_width=1280, frame_height=720)
    assert scaled_720p[0] == (128.0, 144.0)
    assert scaled_720p[1] == (1152.0, 144.0)
    assert scaled_720p[2] == (1152.0, 576.0)
    assert scaled_720p[3] == (128.0, 576.0)

    # Scale to 1920x1080 (1080p Full HD)
    scaled_1080p = detector.convert_normalized_roi(frame_width=1920, frame_height=1080)
    assert scaled_1080p[0] == (192.0, 216.0)
    assert scaled_1080p[2] == (1728.0, 864.0)

