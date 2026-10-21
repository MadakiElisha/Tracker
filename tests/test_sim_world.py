import re
import xml.dom.minidom
from pathlib import Path

SIM = Path(__file__).resolve().parents[1] / "sim"


def _world_names():
    d = xml.dom.minidom.parse(str(SIM / "worlds/tracker_world.sdf"))
    names = {m.getAttribute("name") for m in d.getElementsByTagName("model")}
    for inc in d.getElementsByTagName("include"):
        n = inc.getElementsByTagName("name")
        if n:
            names.add(n[0].firstChild.data)
    return names


def test_world_is_valid_xml_with_fuel_uris():
    d = xml.dom.minidom.parse(str(SIM / "worlds/tracker_world.sdf"))
    uris = [u.firstChild.data for u in d.getElementsByTagName("uri")]
    assert any("fuel.gazebosim.org" in u for u in uris)
    assert "model://iris_with_ardupilot" in uris


def test_every_moved_actor_exists_in_world():
    src = (SIM / "move_target.py").read_text()
    moved = set(re.findall(r'^\s+"(\w+)":\s+\w+,', src, flags=re.M))
    assert moved, "no actors found in move_target.py"
    assert moved <= _world_names()


def test_fetch_script_covers_world_models():
    world = (SIM / "worlds/tracker_world.sdf").read_text()
    fetch = (SIM / "fetch_models.sh").read_text()
    for m in set(re.findall(r"OpenRobotics/models/([\w%]+)", world)):
        assert m in fetch, m
