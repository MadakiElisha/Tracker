"""GCS tap endpoint: validation, token, config wiring."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
pytest.importorskip("flask"); pytest.importorskip("cv2")
from gcs_server import GCSServer


def client(**kw):
    g = GCSServer(frame_w=640, frame_h=480, **kw)
    return g, g.app.test_client()


def test_valid_tap_is_consumed_once():
    g, c = client()
    assert c.post('/tap', json={'x': 10, 'y': 20}).status_code == 200
    assert g.consume_tap() == (10, 20)
    assert g.consume_tap() is None


@pytest.mark.parametrize("body", [None, {}, {'x': 'a', 'y': 1}, {'x': 1}, [1, 2]])
def test_malformed_tap_is_400_not_500(body):
    g, c = client()
    r = c.post('/tap', json=body) if body is not None else c.post('/tap', data="not json")
    assert r.status_code == 400
    assert g.consume_tap() is None


@pytest.mark.parametrize("xy", [(-1, 5), (5, -1), (641, 5), (5, 481)])
def test_out_of_frame_tap_rejected(xy):
    g, c = client()
    assert c.post('/tap', json={'x': xy[0], 'y': xy[1]}).status_code == 400
    assert g.consume_tap() is None


def test_token_required_when_set():
    g, c = client(token="s3cret")
    assert c.post('/tap', json={'x': 1, 'y': 1}).status_code == 403
    assert c.post('/tap', json={'x': 1, 'y': 1}, headers={'X-Token': 'wrong'}).status_code == 403
    assert g.consume_tap() is None
    assert c.post('/tap', json={'x': 1, 'y': 1}, headers={'X-Token': 's3cret'}).status_code == 200
    assert g.consume_tap() == (1, 1)
    assert c.post('/tap?token=s3cret', json={'x': 2, 'y': 2}).status_code == 200


def test_page_sends_token_header():
    _, c = client()
    assert b"X-Token" in c.get('/').data


def test_quality_and_fps_cap_configurable():
    g, _ = client(jpeg_quality=33, fps_cap=10)
    assert g.jpeg_quality == 33 and g.frame_period == pytest.approx(0.1)
