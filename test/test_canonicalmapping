from canonical_mapping import select_canonical_frame


def test_select_canonical_frame_deterministic_order():
    frame_config = {
        "frames": [
            {"frame_id": "f1", "y": 0.0},
            {"frame_id": "f2", "y": 6.0},
            {"frame_id": "f3", "y": 12.0},
        ]
    }
    reordered = {
        "frames": [
            {"frame_id": "f3", "y": 12.0},
            {"frame_id": "f1", "y": 0.0},
            {"frame_id": "f2", "y": 6.0},
        ]
    }
    assert select_canonical_frame(frame_config) == "f2"
    assert select_canonical_frame(reordered) == "f2"
