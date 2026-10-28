"""
Tests for background.py. A synthetic MAS run is written in PSI's own formats (HDF4 'Data-Set-2' with
(phi, theta, r) scales, or HDF5 'Data'), on staggered grids (every variable on its own grid), zipped with a
nested folder, the heliosphere in the corotating frame. Its fields are an exact Parker spiral, so the reader is
checked against ParkerBackground. Run:  python test_background.py
"""
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

import background as bk
from deformable_shock import DeformableShock

P = dict(v_sw=420.0, n_1au=6.0, br_1au=3.0, T=1.2e5)
TIME = '2012-03-07T00:24:00'


def _mesh(lo, hi, n, geometric=False):
    main = np.geomspace(lo, hi, n) if geometric else np.linspace(lo, hi, n)
    half = np.concatenate([[main[0] - (main[1] - main[0]) / 2], (main[1:] + main[:-1]) / 2,
                           [main[-1] + (main[-1] - main[-2]) / 2]])
    return main, half


def _code_fields(part, r, th, ph):
    """Parker spiral in MAS code units on a (r, theta, phi) grid; helio velocities in the corotating frame."""
    R, TH, PH = np.meshgrid(r, th, ph, indexing='ij')
    br = P['br_1au'] * (bk.AU_RSUN / R) ** 2
    rot = bk.OMEGA * R * bk.RSUN_KM * np.sin(TH)
    n = P['n_1au'] * (bk.AU_RSUN / R) ** 2
    return {'br': br / bk.MAS_UNITS['br'], 'bt': 0 * R, 'bp': -br * rot / P['v_sw'] / bk.MAS_UNITS['bp'],
            'vr': P['v_sw'] / bk.MAS_UNITS['vr'] + 0 * R, 'vt': 0 * R,
            'vp': (-rot if part == 'helio' else 0 * R) / bk.MAS_UNITS['vp'],
            'rho': n / bk.MAS_UNITS['rho'], 't': P['T'] / bk.MAS_UNITS['t'] + 0 * R,
            'p': 2 * n * 1e6 * bk.KB * P['T'] / bk.MAS_UNITS['p']}


def _write(path, data, r, th, ph, fmt):
    arr = np.transpose(data, (2, 1, 0)).astype(np.float32)           # PSI layout: (phi, theta, r)
    if fmt == 'hdf4':
        from pyhdf.SD import SD, SDC
        f = SD(str(path), SDC.WRITE | SDC.CREATE)
        sds = f.create('Data-Set-2', SDC.FLOAT32, arr.shape)
        for i, s in enumerate((ph, th, r)):
            sds.dim(i).setscale(SDC.FLOAT32, [float(x) for x in s])
        sds[:] = arr
        sds.endaccess()
        f.end()
    else:
        import h5py
        with h5py.File(path, 'w') as f:
            ds = f.create_dataset('Data', data=arr)
            for i, (name, s) in enumerate((('dim3', ph), ('dim2', th), ('dim1', r))):
                f[name] = np.asarray(s, np.float32)
                f[name].make_scale(name)
                ds.dims[i].attach_scale(f[name])


def make_run(folder, fmt='hdf4', parts=('corona', 'helio')):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    domains = {'corona': (1.0, 30.0, 90), 'helio': (28.0, 240.0, 110)}
    th_m, th_h = _mesh(0.0, np.pi, 73)
    ph_m, ph_h = _mesh(0.0, 2 * np.pi, 145)
    ext = 'hdf' if fmt == 'hdf4' else 'h5'
    for part in parts:
        r_m, r_h = _mesh(*domains[part], geometric=True)
        grids = {'br': (r_m, th_h, ph_h), 'bt': (r_h, th_m, ph_h), 'bp': (r_h, th_h, ph_m),
                 'vr': (r_m, th_h, ph_h), 'vt': (r_h, th_m, ph_h), 'vp': (r_h, th_h, ph_m),
                 'rho': (r_h, th_h, ph_h), 't': (r_h, th_h, ph_h), 'p': (r_h, th_h, ph_h)}
        tmp = folder / f'_build_{part}' / part
        tmp.mkdir(parents=True)
        for v, (r, th, ph) in grids.items():
            _write(tmp / f'{v}002.{ext}', _code_fields(part, r, th, ph)[v], r, th, ph, fmt)
        with zipfile.ZipFile(folder / f'{part}.zip', 'w') as z:      # nested folder inside the archive
            for f in tmp.iterdir():
                z.write(f, f'{part}/{f.name}')
        shutil.rmtree(folder / f'_build_{part}')
    return folder


def random_points(n, r_lo, r_hi, seed=0):
    rng = np.random.default_rng(seed)
    r = np.exp(rng.uniform(np.log(r_lo), np.log(r_hi), n))
    th = np.arccos(rng.uniform(-0.95, 0.95, n))
    lon = rng.uniform(-np.pi, np.pi, n)
    return np.column_stack([r * np.sin(th) * np.cos(lon), r * np.sin(th) * np.sin(lon), r * np.cos(th)])


def compare(a, b):
    rel = lambda x, y: np.nanmax(np.abs(x - y) / np.abs(y))
    ang = np.degrees(np.arccos(np.clip(np.sum(a['B'] * b['B'], 1) / np.linalg.norm(a['B'], axis=1)
                                       / np.linalg.norm(b['B'], axis=1), -1, 1)))
    du = np.nanmax(np.linalg.norm(a['u'] - b['u'], axis=1))
    return {'n': rel(a['n'], b['n']), '|B|': rel(np.linalg.norm(a['B'], axis=1), np.linalg.norm(b['B'], axis=1)),
            'B angle deg': np.nanmax(ang), 'u km/s': du, 'p': rel(a['p'], b['p']), 'VA': rel(a['VA'], b['VA']),
            'cs': rel(a['cs'], b['cs'])}


# ----------------------------------------------------------------------------- tests
def test_mas_hdf4_matches_parker(tmp):
    run = make_run(tmp / 'cr_test')
    bg = bk.MASBackground(run, b_scale=1.0, n_scale=1.0, verbose=False)
    pk = bk.ParkerBackground(**P)
    for lo, hi in ((1.05, 29.0), (31.0, 230.0)):
        xyz = random_points(400, lo, hi, seed=int(lo))
        err = compare(bg.sample(xyz, TIME), pk.sample(xyz, TIME))
        assert err['n'] < 1e-2 and err['|B|'] < 1e-2 and err['B angle deg'] < 0.3 and err['u km/s'] < 2.0, err
        assert err['p'] < 1e-2 and err['VA'] < 1e-2 and err['cs'] < 1e-2, err
    assert [p.corotating for p in bg.parts] == [False, True]                 # frames detected
    return bg


def test_cache_is_reused(tmp):
    run = make_run(tmp / 'cr_cache')
    xyz = random_points(50, 1.1, 200)
    first = bk.MASBackground(run, b_scale=1.0, n_scale=1.0, verbose=False).sample(xyz, TIME)
    for part in ('corona', 'helio'):                                       # remove zips and unpacked files
        shutil.rmtree(run / part)
        (run / f'{part}.zip').unlink()
    again = bk.MASBackground(run, b_scale=1.0, n_scale=1.0, verbose=False).sample(xyz, TIME)
    assert all(np.allclose(first[k], again[k], equal_nan=True) for k in ('n', 'B', 'u', 'p'))


def test_scale_factors(tmp):
    run = make_run(tmp / 'cr_scale', parts=('corona',))
    xyz = random_points(50, 1.5, 25)
    a = bk.MASBackground(run, b_scale=1.0, n_scale=1.0, verbose=False).sample(xyz, TIME)
    b = bk.MASBackground(run, b_scale=2.0, n_scale=0.5, verbose=False).sample(xyz, TIME)
    assert np.allclose(b['B'], 2 * a['B']) and np.allclose(b['n'], 0.5 * a['n']) and np.allclose(b['T'], a['T'])
    assert np.allclose(b['VA'], 2 * np.sqrt(2) * a['VA']) and np.allclose(b['beta'], a['beta'] / 8)


def test_hdf5_files(tmp):
    run = make_run(tmp / 'cr_h5', fmt='h5', parts=('corona',))
    bg = bk.MASBackground(run, b_scale=1.0, n_scale=1.0, verbose=False)
    xyz = random_points(200, 1.05, 29)
    err = compare(bg.sample(xyz, TIME), bk.ParkerBackground(**P).sample(xyz, TIME))
    assert err['n'] < 1e-2 and err['B angle deg'] < 0.3, err


def test_carrington_longitude_matches_sunpy():
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import frames
    for t in ('2012-03-07 00:24', '2012-03-08 12:00'):
        L0 = np.degrees(bk.carrington_L0(t))
        c = SkyCoord(37.0 * u.deg, 12.0 * u.deg, 5 * u.R_sun, frame=frames.HeliographicStonyhurst, obstime=t)
        hgc = c.transform_to(frames.HeliographicCarrington(observer='earth', obstime=t))
        d = (hgc.lon.deg - (37.0 + L0) + 180) % 360 - 180
        assert abs(d) < 0.02, (t, d)


def test_parker_theta_bn_on_sun_centred_sphere():
    pk = bk.ParkerBackground(**P)
    for h in (5.0, 21.5, 150.0):
        sphere = DeformableShock(h, h, h, h, 10.0, 30.0, 0.0)           # a = h: centre at the Sun
        up = bk.upstream_on_shock(sphere, pk, TIME, 31, 61)
        xyz = up['xyz'].reshape(-1, 3)
        th = np.arccos(xyz[:, 2] / np.linalg.norm(xyz, axis=1))
        err = np.max(np.abs(up['theta_bn'].ravel() - pk.theta_bn_radial(np.linalg.norm(xyz, axis=1), th)))
        assert err < 1e-3, (h, err)


def test_parker_field_line_trace():
    pk = bk.ParkerBackground(**P)
    line = pk.trace([bk.AU_RSUN, 0.0, 0.0], TIME, direction=-1, r_min=2.0)
    r_end = np.linalg.norm(line[-1])
    lon_end = np.degrees(np.arctan2(line[-1, 1], line[-1, 0]))
    expected = np.degrees(bk.OMEGA * (bk.AU_RSUN - r_end) * bk.RSUN_KM / P['v_sw'])
    assert abs(r_end - 2.0) < 0.05 and abs(lon_end - expected) < 0.05, (r_end, lon_end, expected)
    return expected, lon_end


def test_mas_trace_crosses_the_interface(tmp, bg):
    pk = bk.ParkerBackground(**P)
    a = bg.trace([200.0, 0.0, 0.0], TIME, direction=-1, r_min=2.0)
    b = pk.trace([200.0, 0.0, 0.0], TIME, direction=-1, r_min=2.0)
    la = np.degrees(np.arctan2(a[-1, 1], a[-1, 0]))
    lb = np.degrees(np.arctan2(b[-1, 1], b[-1, 0]))
    assert abs(la - lb) < 0.5 and abs(np.linalg.norm(a[-1]) - 2.0) < 0.1, (la, lb)
    return la, lb


def test_multi_rotation_blending(tmp):
    """Two runs (CR 2124 with a 400 km/s wind, CR 2125 with 600 km/s): data-age weights and the blended wind."""
    import datetime as dt
    from sunpy.coordinates import sun
    import event_config as cfg
    import shock_properties_dbm as sd
    runs = []
    for cr, v in ((2124, 400.0), (2125, 600.0)):
        P['v_sw'] = v
        runs.append(make_run(tmp / f'cr{cr}', parts=('corona',)))
    P['v_sw'] = 420.0
    cfg.BACKGROUND_DIRS, cfg.BACKGROUND_BLEND_TIME = runs, None
    bg = bk.load_background(b_scale=1.0, n_scale=1.0, verbose=False)
    assert isinstance(bg, bk.MultiMASBackground) and bg.crs == [2124, 2125]
    t = '2012-06-20T12:00:00'                                   # one day before the end of CR 2124
    ts = bk.posix(t)
    L0 = np.degrees(bk.carrington_L0(t))
    w_earth = bg.weights(np.radians([L0]), ts)[:, 0]
    w_west = bg.weights(np.radians([(L0 - 60) % 360]), ts)[:, 0]
    assert w_earth[0] > 0.95 and w_west[1] > 0.75, (w_earth, w_west)
    # the linear observation times against sunpy's exact ones (fractional rotation numbers)
    err = 0.0
    for cr in (2124, 2125):
        for lam in (10.0, 100.0, 250.0, 350.0):
            exact = bk.posix(sun.carrington_rotation_time(cr + (360 - lam) / 360).to_datetime())
            err = max(err, abs(bk.observed_at(bk.rotation_span(cr), lam) - exact) / 3600)
    assert err < 3.0, err                                        # hours
    # blended wind = the weighted mixture, via sample and via the DBM wind table
    pts = np.array([[10.0 * np.cos(a), 10.0 * np.sin(a), 0.0] for a in np.radians([-60.0, 0.0, 60.0])])
    s = bg.sample(pts, t)
    lam = np.mod(np.arctan2(pts[:, 1], pts[:, 0]) + bk.carrington_L0(t), 2 * np.pi)
    w = bg.weights(lam, ts)
    want = w[0] * 400.0 + w[1] * 600.0
    ur = np.sum(s['u'] * pts / 10.0, 1)
    assert np.max(np.abs(ur - want)) < 2.0, (ur, want)
    table = sd.WindTable(bg, np.arange(5.0, 25.0, 1.0), np.linspace(0, np.pi, 31), np.linspace(0, 2 * np.pi, 61, endpoint=False),
                         verbose=False)
    th = np.full(3, np.pi / 2)
    lon = np.arctan2(pts[:, 1], pts[:, 0])
    ur_t, _ = table.lookup(np.full(3, 10.0), th, lon, bk.carrington_L0(t), when=ts)
    assert np.max(np.abs(ur_t - want)) < 5.0, (ur_t, want)
    cfg.BACKGROUND_DIRS = runs[:1]
    assert isinstance(bk.load_background(verbose=False), bk.MASBackground)            # one run: as before
    return {'Earth-facing (CR 2124, 2125)': np.round(w_earth, 2).tolist(), '60 deg west': np.round(w_west, 2).tolist(),
            'blended u_r vs mixture [km/s]': round(float(np.max(np.abs(ur - want))), 2), 'obs. time vs sunpy [h]': round(err, 2)}


if __name__ == '__main__':
    tmp = Path(tempfile.mkdtemp())
    results = []
    bg = None
    tests = [('MAS HDF4 run (corona + helio) = Parker, frames detected', lambda: test_mas_hdf4_matches_parker(tmp)),
             ('converted data reused without zips or unpacked files', lambda: test_cache_is_reused(tmp)),
             ('B and density scale factors', lambda: test_scale_factors(tmp)),
             ('MAS HDF5 files', lambda: test_hdf5_files(tmp)),
             ('Carrington longitude = HGS + L0 (vs sunpy)', test_carrington_longitude_matches_sunpy),
             ('theta_Bn on a Sun-centred sphere (Parker, exact)', test_parker_theta_bn_on_sun_centred_sphere),
             ('Parker field-line trace 1 AU -> 2 R_sun', test_parker_field_line_trace),
             ('MAS field-line trace across the corona/helio boundary', lambda: test_mas_trace_crosses_the_interface(tmp, bg)),
             ('two rotations blended by data age (weights, sunpy times, mixture via sample and wind table)',
              lambda: test_multi_rotation_blending(tmp))]
    passed = 0
    for name, fn in tests:
        try:
            out = fn()
            if name.startswith('MAS HDF4'):
                bg = out
            passed += 1
            print(f'PASS  {name}' + (f'   {out}' if isinstance(out, tuple) else ''))
        except Exception as e:
            print(f'FAIL  {name}: {type(e).__name__}: {e}')
    shutil.rmtree(tmp)
    print(f'{passed}/{len(tests)} passed')
