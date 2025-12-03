from vo import geometry


def test_identity_point():
    pt = (1, 2, 3)
    assert geometry.identity_point(pt) == pt
