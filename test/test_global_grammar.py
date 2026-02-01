import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from analysisops import _legacy_frame_config_from_cfg
from structural_configurator import Building, configuration_package, apply_global_action, get_column_lines_x


def _girder_count(frame_config, story_index):
    return sum(
        1
        for m in frame_config["members"]
        if m.get("group") == "GIR" and int(m.get("story_index", 0)) == story_index
    )


def test_global_add_remove_column_line():
    build = Building(width_x=24.0, length_y=12.0, num_stories=2, story_h=3.0)
    cfg = configuration_package(
        build,
        y_span_target=6.0,
        sec_spacing=2.0,
        dead_kPa=2.0,
        live_kPa=2.0,
    )
    frame_config = _legacy_frame_config_from_cfg(cfg)
    span_limits = (4.0, 30.0)

    x_lines = get_column_lines_x(frame_config)
    assert x_lines == [0.0, 24.0]
    y_lines = sorted(
        {round(n.get("y", 0.0), 6) for n in frame_config["nodes"] if abs(float(n.get("z", 0.0))) < 1e-6}
    )
    initial_girders = _girder_count(frame_config, 1)
    assert initial_girders == (len(x_lines) - 1) * len(y_lines)

    parent = next(m for m in frame_config["members"] if m.get("group") == "GIR")
    parent["foo"] = 7.0
    parent["weight"] = 100.0

    ok, reason = apply_global_action(frame_config, {"type": "ADD_COLUMN_LINE", "x": 12.0}, span_limits)
    assert ok, reason
    x_lines = get_column_lines_x(frame_config)
    assert x_lines == [0.0, 12.0, 24.0]
    for s in (1, 2):
        assert _girder_count(frame_config, s) == (len(x_lines) - 1) * len(y_lines)

    girder_children = [
        m
        for m in frame_config["members"]
        if m.get("group") == "GIR"
        and abs(float(m.get("Yi", 0.0)) - float(parent.get("Yi", 0.0))) < 1e-6
        and int(m.get("story_index", 0)) == int(parent.get("story_index", 0))
    ]
    lengths = sorted(float(m.get("length", 0.0)) for m in girder_children)
    assert lengths[0] >= span_limits[0]
    assert lengths[-1] <= span_limits[1]
    for child in girder_children:
        if abs(child.get("Xi") - 0.0) < 1e-6 and abs(child.get("Xj") - 12.0) < 1e-6:
            assert child.get("foo") == 7.0
            assert child.get("weight") == 50.0

    ok, reason = apply_global_action(frame_config, {"type": "REMOVE_COLUMN_LINE", "x": 12.0}, span_limits)
    assert ok, reason
    x_lines = get_column_lines_x(frame_config)
    assert x_lines == [0.0, 24.0]
    for s in (1, 2):
        assert _girder_count(frame_config, s) == (len(x_lines) - 1) * len(y_lines)