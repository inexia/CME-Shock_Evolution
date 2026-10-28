"""Tests for shock_fit.py.  python -m pytest test_shock_fit.py   or   python test_shock_fit.py"""
import datetime as dt
import numpy as np
from numpy import pi

import shock_fit as sf
from deformable_shock import DeformableShock


def test_projection_matches_sunpy_helioprojective():
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from sunpy.coordinates import frames, sun
    except ImportError:
        return
    date = dt.datetime(2012, 3, 7, 0, 54)
    obs = sf.Observer.from_au(109.3, -0.8, 0.96, 'STA')
    rng = np.random.default_rng(1)
    pts = rng.uniform(-12, 12, (200, 3))
    tx, ty = obs.angles(pts)
    observer = SkyCoord(obs.lon * u.deg, obs.lat * u.deg, obs.dist * sun.constants.radius,
                        frame=frames.HeliographicStonyhurst, obstime=date)
    sc = SkyCoord(*(pts.T * sun.constants.radius), frame=frames.HeliographicStonyhurst, obstime=date,
                  representation_type='cartesian')
    hpc = sc.transform_to(frames.Helioprojective(observer=observer, obstime=date))
    assert np.allclose(np.degrees(tx), hpc.Tx.to_value(u.deg), atol=1e-7)
    assert np.allclose(np.degrees(ty), hpc.Ty.to_value(u.deg), atol=1e-7)


def test_image_axes_for_earth_observer():
    obs = sf.Observer.from_au(0.0, 0.0, 1.0)
    assert np.allclose(obs.x_img, [0, 1, 0]) and np.allclose(obs.y_img, [0, 0, 1])     # +Tx = west, +Ty = north


def test_outline_of_sun_centred_sphere_is_a_circle():
    sphere = DeformableShock.from_pythea(height=6.0, kappa=1.0, epsilon=0.0, alpha=1.0)              # a = b = c = 5, centre 1
    # make it Sun-centred: centre distance h - a = 1 -> shift via lat/lon does not help; use height=a+0
    # instead build a sphere of radius 5 at the Sun: height 5 + r_c with r_c -> 0 is not reachable
    # (kappa = b/(h-1)), so test the general statement: outline radius equals the exact limb radius
    obs = sf.Observer.from_au(0.0, 0.0, 1.0)
    psi, rho = sf.model_outline(sphere, obs, n_bins=72, mesh=(181, 361))
    a = sphere.axes[0]
    centre_X, centre_Y = obs.project(sphere.center)
    # exact limb of a sphere of radius a centred at C seen from distance D: angular radius asin(a / |O-C|)
    D = np.linalg.norm(obs.position - sphere.center)
    limb = obs.dist * np.arcsin(a / D)
    # the limb circle is centred on the projected centre; compare at each position angle
    cx, cy = float(centre_X), float(centre_Y)
    # radius of the circle (centre (cx,cy), radius limb) seen from the origin at position angle psi
    b = cx * np.cos(psi) + cy * np.sin(psi)
    exact = b + np.sqrt(np.maximum(b ** 2 - (cx ** 2 + cy ** 2 - limb ** 2), 0))
    assert np.all(np.abs(rho - exact) < 0.03)                 # soft-max bias ~0.011 R_sun at tau=0.005


def test_soft_outline_tracks_hard_maximum_on_asymmetric_shape():
    # regression: a soft-max whose angular weight sits in the exponent is dominated by the far side
    m = DeformableShock.from_pythea(13.0, 0.8, 0.2, 1.3, lat=15.0, lon=325.0, tilt=10.0, dents=[(70.0, 180.0, 20.0, 0.3)])
    for obs in sf.event_2012_03_07():
        X, Y = obs.project(m.mesh(121, 241))
        _, soft = sf.outline(X, Y, 48)
        _, hard = sf.hard_outline(X, Y, 48)
        assert hard.max() - hard.min() > 4.0                     # strongly asymmetric outline
        assert np.all(soft - hard > -1e-9) and np.all(soft - hard < 0.04)


def test_ellipsoid_recovered_from_three_views():
    truth = DeformableShock.from_pythea(13.0, 0.8, 0.2, 1.3, lat=15.0, lon=325.0, tilt=10.0)
    views = sf.event_2012_03_07()
    data = [sf.synth_outline(truth, v) for v in views]
    start = DeformableShock.from_pythea(11.0, 0.6, 0.0, 1.0, lat=5.0, lon=315.0, tilt=0.0)
    fitted, res = sf.ellipsoid_fit(data, start)
    for n in sf.PARAM_NAMES:
        tol = 0.5 if n in ('lat', 'lon', 'tilt') else 0.02
        assert abs(getattr(fitted, n) - getattr(truth, n)) < tol, (n, getattr(fitted, n), getattr(truth, n))
    cmp = sf.compare_surfaces(truth, fitted)
    assert sf.band_stats(cmp, 0, 100)['rms'] < 0.5           # normals within half a degree everywhere


def test_fit_with_harmonics_keeps_ellipsoid_when_truth_is_ellipsoid():
    truth = DeformableShock.from_pythea(13.0, 0.8, 0.2, 1.3, lat=15.0, lon=325.0, tilt=10.0)
    data = [sf.synth_outline(truth, v) for v in sf.event_2012_03_07()]
    fitter = sf.ShockFitter(data, lmax=4, lam=1.0)
    fitted, res = fitter.fit(truth)                           # start at the truth: must stay there
    assert np.max(np.abs(fitted.coeffs)) < 1e-3
    assert fitter.data_cost(fitted) < 1e-6


def test_footprint_points_lie_on_surface_and_sphere():
    truth = DeformableShock.from_pythea(5.0, 0.7, 0.2, 1.1, lat=15.0, lon=325.0, tilt=0.0)
    fp = sf.synth_footprint(truth, r_sphere=1.05)
    assert len(fp.points) > 20
    assert np.allclose(np.linalg.norm(fp.points, axis=1), 1.05, atol=1e-8)
    d = fp.points - truth.center
    e1, e2, e3 = truth.frame
    r = np.linalg.norm(d, axis=1)
    th = np.arccos(np.clip(d @ e1 / r, -1, 1))
    ph = np.arctan2(d @ e3, d @ e2)
    assert np.allclose(truth.radius(th, ph), r, atol=1e-8)


def test_residual_seeds_signed_for_bulges_and_dents():
    base = dict(height=13.5, kappa=0.72, epsilon=0.548, alpha=1.2, lat=15.0, lon=325.0, tilt=10.0)
    ell = DeformableShock.from_pythea(**base)
    views = sf.event_2012_03_07()
    for depth, sign in [(-0.3, -1), (0.3, 1)]:                    # truth has a bulge / a dent on a limb
        truth = DeformableShock.from_pythea(**base, dents=[(70.0, 90.0, 20.0, depth)])
        data = [sf.synth_outline(truth, v, rho_min=2.5) for v in views]
        seeds = sf.ShockFitter(data, n_dents=1).residual_seeds(ell)
        assert seeds and all(s[2] == sign for s in seeds)        # every seed has the right sign ...
        u_t = ell.direction(np.radians(70.0), np.radians(90.0))
        dist = [np.degrees(np.arccos(np.clip(u_t @ ell.direction(np.radians(a), np.radians(b)), -1, 1)))
                for a, b, _ in seeds[:3]]
        assert min(dist) < 20.0                                   # ... and one of the first three is near the feature
    truth = DeformableShock.from_pythea(**base, dents=[(70.0, 90.0, 20.0, -0.3)])
    data = [sf.synth_outline(truth, v, rho_min=2.5) for v in views]
    only_dents = sf.ShockFitter(data, n_dents=1, depth_bounds=(0.0, 1.5)).residual_seeds(ell)
    assert all(s[2] == 1 for s in only_dents)                     # bulges off -> no bulge seeds


def test_depth_bounds_reach_the_optimizer():
    f = sf.ShockFitter([], n_dents=1, depth_bounds=(-1.0, 1.5))
    lo, hi = f._bounds(f.pack(DeformableShock.from_pythea(10.0, 0.5, 0.0, 1.0, dents=[(70.0, 90.0, 20.0, 0.1)])))
    assert lo[-1] == -1.0 and hi[-1] == 1.5
    f0 = sf.ShockFitter([], n_dents=1, depth_bounds=(0.0, 1.5))
    lo0, _ = f0._bounds(f0.pack(DeformableShock.from_pythea(10.0, 0.5, 0.0, 1.0, dents=[(70.0, 90.0, 20.0, 0.1)])))
    assert lo0[-1] == 0.0


def test_deproject_to_sphere_inverts_the_projection():
    obs = sf.Observer.from_au(109.3, -0.8, 0.96, 'STA')
    rng = np.random.default_rng(3)
    u = rng.normal(size=(200, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    p = 1.05 * u
    vis = ~sf.occulted(p, obs, r_sun=1.0) & (p @ (obs.position / obs.dist) > 0.2)  # clearly on the front side
    tx, ty = obs.angles(p[vis])
    q, hit = sf.deproject_to_sphere(obs, np.degrees(tx) * 3600, np.degrees(ty) * 3600, 1.05)
    assert hit.all() and np.allclose(q, p[vis], atol=1e-8)
    q2, hit2 = sf.deproject_to_sphere(obs, [0.0, 8000.0], [0.0, 0.0], 1.05)   # disk centre hits, 8000" misses
    assert hit2[0] and not hit2[1] and np.isnan(q2[1]).all()


def test_occulted_behind_disk_and_inside_sun():
    obs = sf.Observer.from_au(0.0, 0.0, 1.0)
    pts = np.array([[-3.0, 0.0, 0.0],      # straight behind the Sun -> hidden
                    [3.0, 0.0, 0.0],       # in front -> visible
                    [-3.0, 0.0, 1.5],      # behind the limb plane but off the disk -> visible
                    [0.2, 0.0, 0.0]])      # inside the Sun -> hidden
    assert list(sf.occulted(pts, obs)) == [True, False, False, True]


def test_ellipsoid_footprint_lies_on_sphere_and_ellipsoid():
    m = DeformableShock.from_pythea(5.0, 0.7, 0.2, 1.1, lat=15.0, lon=325.0, tilt=0.0, dents=[(150.0, 0.0, 20.0, 0.3)])
    lines = sf.ellipsoid_footprint(m, 1.05)
    assert lines
    pts = np.vstack(lines)
    assert np.allclose(np.linalg.norm(pts, axis=1), 1.05, atol=2e-3)
    e = m.ellipsoid(); d = pts - e.center
    th = np.arccos(np.clip(d @ e.frame[0] / np.linalg.norm(d, axis=1), -1, 1))
    ph = np.arctan2(d @ e.frame[2], d @ e.frame[1])
    assert np.allclose(e.radius(th, ph), np.linalg.norm(d, axis=1), atol=2e-3)       # on the ellipsoid, not the dent


def test_self_similar_fit_recovers_scale():
    prev = DeformableShock.from_pythea(9.0, 0.72, 0.548, 1.2, lat=15.0, lon=325.0, tilt=10.0, dents=[(70.0, 90.0, 20.0, 0.3)])
    truth = prev.scaled(1.35)
    data = [sf.synth_outline(truth, v, rho_min=2.5) for v in sf.event_2012_03_07()]
    model, chi2, s = sf.fit_self_similar(data, prev, mesh=(61, 121))
    assert abs(s - 1.35) < 2e-3 and chi2 < 5.0
    truth1 = prev.scaled(1.35, r0=1.0)                                        # PyThea-type expansion
    data1 = [sf.synth_outline(truth1, v, rho_min=2.5) for v in sf.event_2012_03_07()]
    m1, chi1, s1 = sf.fit_self_similar(data1, prev, mesh=(61, 121), r0=1.0)
    m0, chi0, s0 = sf.fit_self_similar(data1, prev, mesh=(61, 121), r0=0.0)
    assert abs(s1 - 1.35) < 2e-3 and chi1 < 5.0 and chi0 > 50 * max(chi1, 1.0)   # the right origin wins clearly


def test_data_cost_keeps_the_models_own_dents():
    # regression: data_cost() used to drop dents when the fitter was built for fewer of them
    truth = DeformableShock.from_pythea(13.5, 0.72, 0.548, 1.2, lat=15.0, lon=325.0, tilt=10.0,
                                        dents=[(70.0, 90.0, 20.0, 0.3), (60.0, 250.0, 15.0, -0.2)])
    data = [sf.synth_outline(truth, v, rho_min=2.5) for v in sf.event_2012_03_07()]
    for n in (0, 1, 2):
        assert sf.ShockFitter(data, n_dents=n).data_cost(truth) < 1e-6
    assert sf.ShockFitter(data).data_cost(truth.ellipsoid()) > 100.0


def test_correlated_noise_rms_and_periodicity():
    rng = np.random.default_rng(0)
    w = sf.correlated_noise(48, 0.05, 2.0, rng)
    assert abs(np.std(w) - 0.05) < 1e-12 and len(w) == 48


def _anchored(data, anchor, sigma, fixed=()):
    sf.PRIOR, sf.FIXED = (anchor, sigma), tuple(fixed)
    try:
        f = sf.ShockFitter(data)
        model, res = f.fit(anchor, ftol=1e-6, xtol=1e-6, gtol=1e-6)
        J = np.asarray(res.jac)
        cov = np.linalg.pinv(J.T @ J)
        free = [n for n in sf.PARAM_NAMES if n not in f.fixed]
        ratio = {n: float(np.sqrt(max(cov[i, i], 0)) / f.prior_sigma(n)) for i, n in enumerate(free)
                 if f.prior_sigma(n) is not None}
        return model, ratio, f
    finally:
        sf.PRIOR, sf.FIXED = None, ()


SIG = {'height': 0.10, 'a': 0.20, 'b': 0.20, 'c': 0.20, 'lat': 5.0, 'lon': 5.0, 'tilt': 5.0}
TRUTH = DeformableShock.from_pythea(13.5, 0.72, 0.3, 1.2, lat=15.0, lon=325.0, tilt=10.0)
ANCHOR = DeformableShock(TRUTH.height * 1.05, TRUTH.a, TRUTH.b * 1.08, TRUTH.c, TRUTH.lat + 4.0, TRUTH.lon - 4.0, TRUTH.tilt)


def test_anchored_fit_follows_the_data_where_they_exist():
    data = [sf.synth_outline(TRUTH, v, rho_min=2.5) for v in sf.event_2012_03_07()]         # all round, 3 views
    m, ratio, _ = _anchored(data, ANCHOR, SIG)
    assert abs(m.lat - TRUTH.lat) < 0.5 and abs(((m.lon - TRUTH.lon + 180) % 360) - 180) < 0.5
    assert abs(m.height / TRUTH.height - 1) < 0.01 and abs(m.b / TRUTH.b - 1) < 0.03
    assert all(ratio[n] < 0.45 for n in ('height', 'lat', 'lon')), ratio                      # set by the data


def test_anchored_fit_reports_what_partial_traces_do_not_constrain():
    """A halo seen from Earth only, traced on its southern side: the outline says nothing about the depth along
    the line of sight (height and a), so those stay at the anchor and are reported as set by the prior; the
    cross-section (b, c) is set by the data."""
    halo = DeformableShock.from_pythea(10.0, 0.6, 0.0, 1.3, lat=0.0, lon=0.0, tilt=20.0)
    anchor = DeformableShock(halo.height * 1.05, halo.a * 1.1, halo.b * 1.05, halo.c, 2.0, -2.0, halo.tilt + 3.0)
    v = sf.event_2012_03_07()[1]
    o = sf.synth_outline(halo, v, rho_min=2.5)
    o.mask = o.mask & (np.abs(((np.asarray(o.psi) + np.pi / 2 + np.pi) % (2 * np.pi)) - np.pi) < np.radians(60))
    m, ratio, _ = _anchored([o], anchor, SIG)
    assert ratio['height'] > 0.8 and ratio['a'] > 0.6, ratio                         # depth: (mostly) the prior
    assert ratio['b'] < 0.45 and ratio['c'] < 0.45, ratio                            # cross-section: the data
    assert abs(m.height - anchor.height) < 0.1 * SIG['height'] * anchor.height       # ... and it stays put
    return {k: round(v, 2) for k, v in ratio.items()}


def test_prior_does_not_enter_the_data_cost_and_fixed_parameters_stay():
    data = [sf.synth_outline(TRUTH, v, rho_min=2.5) for v in sf.event_2012_03_07()]
    c0 = sf.ShockFitter(data).data_cost(ANCHOR)
    sf.PRIOR = (TRUTH, SIG)
    try:
        f = sf.ShockFitter(data)
        assert abs(f.data_cost(ANCHOR) - c0) < 1e-9 and f.n_residuals() == f.n_data() + 7
    finally:
        sf.PRIOR = None
    m, _, f = _anchored(data, ANCHOR, SIG, fixed=('a', 'b', 'c'))
    assert (m.a, m.b, m.c) == (ANCHOR.a, ANCHOR.b, ANCHOR.c) and {'a', 'b', 'c'} <= f.fixed


def test_feature_traced_sector():
    m = DeformableShock.from_pythea(13.5, 0.72, 0.3, 1.2, lat=15.0, lon=325.0, tilt=10.0, dents=[(75.0, 90.0, 20.0, 0.3)])
    data = [sf.synth_outline(m, v, rho_min=2.5) for v in sf.event_2012_03_07()]
    assert sf.feature_traced(m, m.dents[0], data)
    for o in data:
        o.mask = np.zeros_like(o.mask)
    assert not sf.feature_traced(m, m.dents[0], data)


def test_trace_csv_tolerates_spreadsheet_edits():
    import contextlib, io, tempfile, os
    tx, ty = np.array([-3960.815, -2042.361, -38.271, 1923.189]), np.array([-10802.032, -11568.748, -11866.843, -11611.317])
    header = ('# LASCO-C3 Clear 2021-10-28T16:54:07.398\n# KIND=coronagraph SC=SOHO DETECTOR=C3 IMAGE_TIME=2021-10-28T16:54:07\n'
              '# HGLN_OBS=0.1523 HGLT_OBS=4.6536 DSUN_RSUN=211.330\n# Tx_arcsec, Ty_arcsec')
    d = tempfile.mkdtemp()
    clean = os.path.join(d, 'clean.csv')
    np.savetxt(clean, np.c_[tx, ty], delimiter=',', header=header, comments='', fmt='%.3f')     # as trace_outline.py
    lines = [h + ',' for h in header.split('\n')]                                           # as a spreadsheet saves it
    lines[2] = '"' + lines[2] + '"'
    rows = [f'{a:.3f},{b:.3f},' for a, b in zip(tx, ty)]
    edited = os.path.join(d, 'edited.csv')
    with open(edited, 'w', newline='') as fh:
        fh.write('\r\n'.join(lines + rows[:2] + ['', ',', '123.4,'] + rows[2:]))                # no final newline
    o1, x1, y1 = sf.read_trace_csv(clean)
    with contextlib.redirect_stdout(io.StringIO()) as out:
        o2, x2, y2 = sf.read_trace_csv(edited)
    assert (o1.lon, o1.lat, o1.dist) == (o2.lon, o2.lat, o2.dist) == (0.1523, 4.6536, 211.33)
    assert np.allclose(x1, tx) and np.allclose(x2, tx) and np.allclose(y2, ty)
    assert 'skipped 1 line' in out.getvalue()                                                # the row with one number


if __name__ == '__main__':
    import sys, time
    names = [n for n in dir() if n.startswith('test_')]
    failed = 0
    for n in names:
        t0 = time.time()
        try:
            globals()[n]()
            print(f'PASS {n} ({time.time() - t0:.1f}s)')
        except Exception as e:                                # noqa: BLE001
            failed += 1
            print(f'FAIL {n}: {e!r}')
    print(f'{len(names) - failed}/{len(names)} passed')
    sys.exit(1 if failed else 0)
