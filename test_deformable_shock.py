"""
Analytic-invariant tests for deformable_shock.py.
Run with  python -m pytest test_deformable_shock.py   or   python test_deformable_shock.py
"""
import numpy as np
from numpy import pi, sin, cos, exp, sqrt

import deformable_shock as ds

BASE = dict(height=13.0, kappa=0.6, epsilon=0.3, alpha=1.4, lat=15.0, lon=325.0, tilt=20.0)


def make(**kw):
    p = dict(BASE)
    p.update(kw)
    return ds.DeformableShock.from_pythea(**p)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ---------------------------------------------------------------- ellipsoid parameterisation
def test_axes_follow_pythea_definitions():
    a, b, c, rc = ds.ellipsoid_axes(13.0, 0.6, 0.3, 1.4)
    assert np.isclose(b, 0.6 * 12.0)
    assert np.isclose(c, b / 1.4)
    assert np.isclose(a, b * (1 - 0.3 ** 2)) and a < b          # eps > 0  ->  a < b
    assert np.isclose(rc, 13.0 - a)
    a2, b2, _, _ = ds.ellipsoid_axes(13.0, 0.6, -0.3, 1.4)
    assert np.isclose(a2, b2 / (1 - 0.3 ** 2)) and a2 > b2       # eps < 0  ->  a > b
    # recover epsilon from (a, b) both ways
    assert np.isclose(+sqrt(1 - a / b), 0.3)
    assert np.isclose(-sqrt(1 - b2 / a2), -0.3)


def test_sphere_special_case():
    a, b, c, _ = ds.ellipsoid_axes(10.0, 0.5, 0.0, 1.0)
    assert np.isclose(a, b) and np.isclose(b, c)


def test_apex_at_height_along_direction():
    s = make()
    lat, lon = np.radians([BASE['lat'], BASE['lon']])
    d = np.array([cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)])
    assert np.allclose(s.position(0.0, 0.0), BASE['height'] * d, atol=1e-12)
    assert np.allclose(s.nose(), BASE['height'] * d, atol=1e-12)


def test_undeformed_points_satisfy_ellipsoid_equation():
    s = make()
    X = s.surface(41, 81)['xyz'] - s.center
    e1, e2, e3 = s.frame
    a, b, c, _ = s.axes
    q = (X @ e1 / a) ** 2 + (X @ e2 / b) ** 2 + (X @ e3 / c) ** 2
    assert np.allclose(q, 1.0, atol=1e-12)


# ---------------------------------------------------------------- frame
def test_frame_orthonormal_and_tilt_zero_orientation():
    s = make(tilt=0.0)
    E = np.array(s.frame)
    assert np.allclose(E @ E.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(E), 1.0)                      # right-handed
    assert np.isclose(s.frame[1][2], 0.0)                         # b-axis in the equatorial plane
    assert s.frame[2][2] > 0                                      # c-axis toward north
    # Earth-directed apex: e1 = x, e2 = +y (toward +longitude / west limb), e3 = z
    s0 = make(lat=0.0, lon=0.0, tilt=0.0)
    assert np.allclose(s0.frame, np.eye(3), atol=1e-12)


def test_tilt_90_turns_b_axis_to_north():
    s0, s90 = make(tilt=0.0), make(tilt=90.0)
    assert np.allclose(s90.frame[1], s0.frame[2], atol=1e-12)
    assert np.allclose(s90.frame[2], -s0.frame[1], atol=1e-12)


# ---------------------------------------------------------------- normals
def test_sphere_normals_are_radial():
    s = make(epsilon=0.0, alpha=1.0)
    S = s.surface(31, 61)
    assert np.allclose(S['normals'], unit(S['xyz'] - s.center), atol=1e-7)


def test_ellipsoid_normals_analytic_including_poles():
    s = make()
    S = s.surface(31, 61)
    X = S['xyz'] - s.center
    e1, e2, e3 = s.frame
    a, b, c, _ = s.axes
    n_exp = (X @ e1 / a ** 2)[..., None] * e1 + (X @ e2 / b ** 2)[..., None] * e2 + (X @ e3 / c ** 2)[..., None] * e3
    assert np.allclose(S['normals'], unit(n_exp), atol=1e-7)


def test_dent_tilts_normal_by_predicted_angle():
    # sphere + dent: on the dent slope at angular distance sigma from the centre,
    # |grad ln R| = (D / sigma) exp(-1/2)  ->  normal tilted by arctan of that
    sigma, depth = 12.0, 0.3
    s = make(epsilon=0.0, alpha=1.0, dents=[(90.0, 0.0, sigma, depth)])
    th = pi / 2 + np.radians(sigma)                                # Delta = sigma, along theta
    n = s.normals(th, 0.0)
    r = unit(s.position(th, 0.0) - s.center)
    tilt = np.degrees(np.arccos(np.clip(n @ r, -1, 1)))
    expected = np.degrees(np.arctan(depth / np.radians(sigma) * exp(-0.5)))
    assert np.isclose(tilt, expected, atol=1e-3)
    # at the dent centre the slope is zero: normal unchanged
    n0 = s.normals(pi / 2, 0.0)
    assert np.allclose(n0, unit(s.position(pi / 2, 0.0) - s.center), atol=1e-7)


# ---------------------------------------------------------------- deformation
def test_dent_depth_and_locality():
    s0 = make()
    s = make(dents=[(90.0, 0.0, 10.0, 0.4)])
    assert np.isclose(s.radius(pi / 2, 0.0), s0.radius(pi / 2, 0.0) * exp(-0.4))
    assert np.isclose(s.radius(pi / 2, pi), s0.radius(pi / 2, pi), rtol=1e-12)      # 180 deg away
    assert np.isclose(s.radius(0.0, 0.3), s0.radius(0.0, 0.3), rtol=1e-12)          # apex, 90 deg away
    assert np.allclose(s.position(0.0, 0.0), s0.position(0.0, 0.0), atol=1e-12)


def test_negative_depth_is_a_bulge():
    s0, s = make(), make(dents=[(90.0, 0.0, 10.0, -0.4)])
    assert np.isclose(s.radius(pi / 2, 0.0), s0.radius(pi / 2, 0.0) * exp(0.4))         # pushed out
    assert np.isclose(s.radius(pi / 2, pi), s0.radius(pi / 2, pi), rtol=1e-12)          # local only
    # on a sphere the normal tilts by the same angle as for a dent, toward the other side
    sigma, D = 12.0, 0.3
    sd = make(epsilon=0.0, alpha=1.0, dents=[(90.0, 0.0, sigma, D)])
    sb = make(epsilon=0.0, alpha=1.0, dents=[(90.0, 0.0, sigma, -D)])
    th = pi / 2 + np.radians(sigma)
    ang = []
    for m in (sd, sb):
        n, r = m.normals(th, 0.0), unit(m.position(th, 0.0) - m.center)
        ang.append(np.degrees(np.arccos(np.clip(n @ r, -1, 1))))
    expected = np.degrees(np.arctan(D / np.radians(sigma) * exp(-0.5)))
    assert np.allclose(ang, expected, atol=1e-3)
    e1 = sd.frame[0]
    assert np.sign(sd.normals(th, 0.0) @ e1) != np.sign(sb.normals(th, 0.0) @ e1)   # opposite tilts


def test_semi_axes_constructor_and_pythea_equivalence():
    a, b, c, rc = ds.pythea_to_axes(13.0, 0.6, 0.3, 1.4)
    s1 = make()
    s2 = ds.DeformableShock(13.0, a, b, c, lat=BASE['lat'], lon=BASE['lon'], tilt=BASE['tilt'])
    assert np.allclose(s1.mesh(11, 21), s2.mesh(11, 21), atol=1e-12)
    assert np.allclose(s2.axes, (a, b, c, 13.0 - a))
    assert np.allclose(s2.position(0.0, 0.0), s2.nose(), atol=1e-9) and np.isclose(np.linalg.norm(s2.nose()), 13.0)


def test_legacy_pythea_json_still_loads_and_time_round_trips():
    legacy = {'model': 'shock', 'height': 13.0, 'kappa': 0.6, 'epsilon': 0.3, 'alpha': 1.4,
              'lat': BASE['lat'], 'lon': BASE['lon'], 'tilt': BASE['tilt'], 'dents': [[80, 30, 15, 0.2]]}
    m = ds.DeformableShock.from_dict(legacy)
    assert np.allclose(m.mesh(9, 17), make(dents=[(80, 30, 15, 0.2)]).mesh(9, 17), atol=1e-12)
    d = m.to_dict()
    assert 'kappa' in d and 'a' not in d and 'time' not in d                      # files are written in PyThea form
    assert np.allclose([d['kappa'], d['epsilon'], d['alpha']], [0.6, 0.3, 1.4])
    semi = {'model': 'shock', 'height': 13.0, 'a': m.a, 'b': m.b, 'c': m.c, 'lat': BASE['lat'],
            'lon': BASE['lon'], 'tilt': BASE['tilt'], 'dents': [[80, 30, 15, 0.2]]}
    assert np.allclose(ds.DeformableShock.from_dict(semi).mesh(9, 17), m.mesh(9, 17), atol=1e-12)   # a,b,c files load
    m.time = '2012-03-07T00:54:00'
    assert ds.DeformableShock.from_dict(m.to_dict()).time == '2012-03-07T00:54:00'


def test_scaled_is_self_similar_about_sun_centre():
    m = make(dents=[(70.0, 90.0, 20.0, 0.3), (60.0, 250.0, 15.0, -0.2)])
    s = m.scaled(1.7, time='2012-03-07T01:10:00')
    th, ph = ds.DeformableShock.grid(31, 61)
    assert np.allclose(s.position(th, ph), 1.7 * m.position(th, ph), atol=1e-10)    # every point scales
    assert np.allclose(s.normals(th, ph), m.normals(th, ph), atol=1e-8)             # shape (normals) unchanged
    assert s.time == '2012-03-07T01:10:00' and s.dents == m.dents


def test_pythea_round_trip_both_eccentricity_signs():
    for eps in (0.45, 0.0, -0.45):
        m = ds.DeformableShock.from_pythea(11.0, 0.8, eps, 1.3, lat=5.0, lon=20.0, tilt=-15.0)
        assert np.allclose([m.kappa, m.epsilon, m.alpha], [0.8, eps, 1.3], atol=1e-12)
        assert np.allclose(ds.axes_to_pythea(m.height, m.a, m.b, m.c), [0.8, eps, 1.3], atol=1e-12)


def test_scaled_about_surface_is_pythea_self_similar():
    m = make(dents=[(70.0, 90.0, 20.0, 0.3)])
    s = m.scaled(1.6, r0=1.0)
    assert np.isclose(s.height - 1.0, 1.6 * (m.height - 1.0))                       # h - 1 scales ...
    assert np.allclose([s.kappa, s.epsilon, s.alpha], [m.kappa, m.epsilon, m.alpha])  # ... kappa, eps, alpha do not
    p0 = m.frame[0]                                                                  # fixed point: surface below apex
    th, ph = ds.DeformableShock.grid(21, 41)
    assert np.allclose(s.position(th, ph) - p0, 1.6 * (m.position(th, ph) - p0), atol=1e-10)
    assert np.allclose(s.normals(th, ph), m.normals(th, ph), atol=1e-8)


def test_zero_depth_dent_is_identity():
    s0, s = make(), make(dents=[(40.0, 100.0, 15.0, 0.0)])
    assert np.allclose(s.mesh(11, 21), s0.mesh(11, 21), atol=1e-14)


def test_monopole_coefficient_scales_radius():
    c = np.zeros(ds.n_coeffs(2))
    c[ds.lm_index(0, 0)] = 0.2
    s0, s = make(), make(lmax=2, coeffs=c)
    S0, S = s0.surface(21, 41), s.surface(21, 41)
    assert np.allclose(S['radius'] / S0['radius'], exp(0.2 / sqrt(4 * pi)), rtol=1e-12)


def test_real_sph_harm_orthonormal():
    lmax = 4
    x, w = np.polynomial.legendre.leggauss(24)                     # exact for the polynomials involved
    nphi = 64
    phi = np.arange(nphi) * 2 * pi / nphi
    th, ph = np.meshgrid(np.arccos(x), phi, indexing='ij')
    wt = (w[:, None] * (2 * pi / nphi)) * np.ones_like(ph)
    Y = [ds.real_sph_harm(l, m, th, ph) for l in range(lmax + 1) for m in range(-l, l + 1)]
    G = np.array([[np.sum(Yi * Yj * wt) for Yj in Y] for Yi in Y])
    assert np.allclose(G, np.eye(len(Y)), atol=1e-10)


def test_lm_index_layout():
    assert [ds.lm_index(l, m) for l in range(3) for m in range(-l, l + 1)] == list(range(9))
    assert ds.n_coeffs(3) == 16


# ---------------------------------------------------------------- mesh / IO
def test_grid_order_has_no_long_jumps():
    s = make()
    pts = s.mesh(19, 37, order='grid')
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    a, b, c, _ = s.axes
    assert seg.max() < 0.25 * 2 * max(a, b, c)                     # far shorter than a line through the body
    assert pts.shape == (2 * 19 * 37, 3)


def test_dict_round_trip():
    c = np.zeros(ds.n_coeffs(3))
    c[ds.lm_index(2, -1)] = 0.05
    c[ds.lm_index(3, 2)] = -0.02
    s = make(lmax=3, coeffs=c, dents=[(80.0, 30.0, 15.0, 0.25), (120.0, 200.0, 25.0, 0.1)])
    s2 = ds.DeformableShock.from_dict(s.to_dict())
    assert np.allclose(s.mesh(13, 25), s2.mesh(13, 25), atol=1e-14)
    assert s2.to_dict() == s.to_dict()


def test_with_dent_appends():
    s = make().with_dent(90.0, 0.0, 10.0, 0.3)
    assert len(s.dents) == 1
    assert np.isclose(s.radius(pi / 2, 0.0), make().radius(pi / 2, 0.0) * exp(-0.3))


def test_mesh_sunpy_matches_mesh():
    try:
        import datetime as dt
        from sunpy.coordinates import sun
    except ImportError:
        return                                                     # sunpy not installed: nothing to check
    s = make()
    sc = s.mesh_sunpy(dt.datetime(2012, 3, 7, 0, 54), 7, 13, order='points')
    xyz = np.stack([sc.cartesian.x, sc.cartesian.y, sc.cartesian.z], axis=-1) / sun.constants.radius
    assert np.allclose(xyz.to_value(), s.mesh(7, 13), atol=1e-10)
    assert sc.frame.name == 'heliographic_stonyhurst'


if __name__ == '__main__':
    import sys
    names = [n for n in dir() if n.startswith('test_')]
    failed = 0
    for n in names:
        try:
            globals()[n]()
            print(f'PASS {n}')
        except Exception as e:                                     # noqa: BLE001
            failed += 1
            print(f'FAIL {n}: {e!r}')
    print(f'{len(names) - failed}/{len(names)} passed')
    sys.exit(1 if failed else 0)
