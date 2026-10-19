"""RC switch decoding: edges, calibration window, class filter, invalid PWM."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from fc_interface5 import RCDecoder

RC = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'config.yaml')))['rc']


def feed(dec, n, **ch):
    for _ in range(n):
        dec.feed({7: ch.get('c7', 1000), 8: ch.get('c8', 1000), 9: ch.get('c9', 1000)})


def test_rising_edge_fires_once():
    d = RCDecoder(RC, cal_frames=3)
    feed(d, 5)
    feed(d, 1, c7=2000)
    assert d.consume() == (True, False, 'any')
    feed(d, 5, c7=2000)                         # held high: no repeat
    assert d.consume()[0] is False
    feed(d, 2, c7=1000)
    feed(d, 1, c7=2000)                         # new press fires again
    assert d.consume()[0] is True


def test_release_channel_is_independent():
    d = RCDecoder(RC, cal_frames=0)
    feed(d, 2)
    feed(d, 1, c8=2000)
    assert d.consume() == (False, True, 'any')


def test_switch_already_high_at_startup_does_not_fire():
    """SITL CH8 defaults to 1800: that used to raise a spurious RELEASE."""
    d = RCDecoder(RC, cal_frames=10)
    feed(d, 12, c8=1800)
    assert d.consume() == (False, False, 'any')
    feed(d, 3, c8=1000)
    feed(d, 1, c8=2000)
    assert d.consume()[1] is True               # but a real press afterwards does


def test_class_filter_three_positions():
    d = RCDecoder(RC, cal_frames=0)
    feed(d, 1, c9=1000); assert d.consume()[2] == 'any'
    feed(d, 1, c9=1500); assert d.consume()[2] == 'person'
    feed(d, 1, c9=2000); assert d.consume()[2] == 'vehicle'


def test_invalid_pwm_is_ignored_not_treated_as_low():
    d = RCDecoder(RC, cal_frames=0)
    feed(d, 2, c7=2000); d.consume()
    for bad in (0, 65535, 100):
        d.feed({7: bad, 8: bad, 9: bad})
    d.feed({7: 2000, 8: 1000, 9: 1500})        # still high: no new edge from "bad" frames
    lock, rel, flt = d.consume()
    assert lock is False
    assert flt == 'person'


def test_missing_channels_are_ignored():
    d = RCDecoder(RC, cal_frames=0)
    d.feed({})
    assert d.consume() == (False, False, 'any')
