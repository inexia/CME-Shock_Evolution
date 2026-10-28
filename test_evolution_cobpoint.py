"""
Tests for evolution_cobpoint.py, each against an exact answer. Run:  python test_evolution_cobpoint.py
"""
import contextlib
import datetime as dt
import io
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import background as bk
import event_config as cfg
import evolution_cobpoint as ec
import shock_properties_sse as sp
from deformable_shock import DeformableShock

T0 = dt.datetime(2012, 3, 7, 0, 30)
V, VSW = 1500.0, 400.0


def write_series(folder, models, times, kind='corona', method='test'):
    stem, digits = sp.SERIES_FILES[kind][:2]
    for m, t in zip(models, times):
        d = {**m.to_dict(), 'time': t.isoformat(), 'derived': {'speed_kms': V}}
        (folder / f'{stem}{int(round(m.height * 10)):0{digits}d}_{method}.json').write_text(json.dumps(d))


def setup(tmp):
    ec._LINES.clear()                                        # traced lines are cached per target and time
    cfg.FINAL_DIR = Path(tmp) / 'final'
    cfg.FIGURE_DIR = Path(tmp) / 'fig'
    for d in (cfg.FINAL_DIR, cfg.FIGURE_DIR):
        if d.exists():
            shutil.rmtree(d)
    cfg.FINAL_DIR.mkdir(parents=True)


def sphere_series():
    sp.FRONT_HALF_WIDTH_DEG = None                                   # idealised full shells
    hs = np.arange(3.0, 12.01, 1.0)
    times = [T0 + dt.timedelta(seconds=(h - 3.0) * bk.RSUN_KM / V) for h in hs]
    write_series(cfg.FINAL_DIR, [DeformableShock(h, h, h, h, 0.0, 0.0, 0.0) for h in hs], times)   # Sun-centred
    return hs


def test_parker_spiral_cobpoints_for_two_targets(tmp):
    setup(tmp)
    hs = sphere_series()
    bg = bk.ParkerBackground(v_sw=VSW, n_1au=5.0, br_1au=3.5, T=1e5)
    err = {'r': 0.0, 'lon': 0.0, 'theta_bn': 0.0, 'MA': 0.0}
    for label, lon0 in (('A', 0.0), ('B', 60.0)):
        pos = bk.AU_RSUN * np.array([np.cos(np.radians(lon0)), np.sin(np.radians(lon0)), 0.0])
        with contextlib.redirect_stdout(io.StringIO()):
            rows, _ = ec.evolution('test', bg, label, lambda t: pos, ['corona'])
        assert len(rows) == len(hs) and all(r['status'] == 'connected' for r in rows)
        for r in rows:
            h = r['apex_height_rsun']
            lon = lon0 + np.degrees(bk.OMEGA * (bk.AU_RSUN - h) * bk.RSUN_KM / VSW)     # Parker spiral, west
            tbn = np.degrees(np.arctan(bk.OMEGA * h * bk.RSUN_KM / VSW))
            VA = bg.sample(np.array([[h, 0, 0]]), r['time'])['VA'][0]
            err['r'] = max(err['r'], abs(r['cob_r_rsun'] - h))
            err['lon'] = max(err['lon'], abs(r['cob_lon_deg'] - lon))
            err['theta_bn'] = max(err['theta_bn'], abs(r['theta_bn_deg'] - tbn))
            err['MA'] = max(err['MA'], abs(r['MA'] / ((V - VSW) / VA) - 1))
    assert err['r'] < 1e-3 and err['lon'] < 0.02 and err['theta_bn'] < 0.02 and err['MA'] < 1e-3, err
    return {k: float(f'{v:.2e}') for k, v in err.items()}


def test_not_connected_and_passed(tmp):
    setup(tmp)
    sp.FRONT_HALF_WIDTH_DEG = None
    bg = bk.ParkerBackground(v_sw=VSW)
    small_far = DeformableShock.from_pythea(4.0, 0.3, 0.0, 1.0, lat=0.0, lon=180.0, tilt=0.0)       # behind the Sun
    big = DeformableShock(250.0, 250.0, 250.0, 250.0, 0.0, 0.0, 0.0)                               # past 1 AU
    write_series(cfg.FINAL_DIR, [small_far], [T0])
    write_series(cfg.FINAL_DIR, [big], [T0 + dt.timedelta(days=1)], kind='ip')
    with contextlib.redirect_stdout(io.StringIO()):
        rows, _ = ec.evolution('test', bg, 'A', lambda t: np.array([bk.AU_RSUN, 0.0, 0.0]), ['corona', 'ip'])
    got = [r['status'] for r in rows]
    assert got == ['not connected', 'spacecraft inside'], got
    return got


def test_far_side_of_a_halo_is_not_a_shock(tmp):
    """A Sun-centred shell whose apex points away from the target: with the cut, the target's line meets only
    the removed far side, so it is not connected; without the cut it would be."""
    setup(tmp)
    bg = bk.ParkerBackground(v_sw=VSW, n_1au=5.0, br_1au=3.5, T=1e5)
    hs = [6.0, 8.0]
    times = [T0 + dt.timedelta(seconds=(h - 6.0) * bk.RSUN_KM / V) for h in hs]
    write_series(cfg.FINAL_DIR, [DeformableShock(h, h, h, h, 0.0, 180.0, 0.0) for h in hs], times)   # apex behind
    pos = np.array([bk.AU_RSUN, 0.0, 0.0])
    out = {}
    for hw in (None, 90.0):
        sp.FRONT_HALF_WIDTH_DEG = hw
        ec._LINES.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            rows, _ = ec.evolution('test', bg, 'A', lambda t: pos, ['corona'])
        out[hw] = [r['status'] for r in rows]
    assert out[None] == ['connected', 'connected'] and out[90.0] == ['not connected', 'not connected'], out
    return {'no cut': out[None], 'cut at 90 deg': out[90.0]}


def test_targets_parsing_and_ephemeris():
    from sunpy.coordinates import get_earth, get_body_heliographic_stonyhurst, sun
    tg = ec.parse_targets('Earth, L1, STA, PSP, SolO, Mars')
    kinds = [k for _, k, _ in tg]
    assert kinds == ['earth', 'l1', 'horizons', 'horizons', 'horizons', 'planet'], kinds
    t = '2012-03-07T00:54:00'
    where = ec.TargetPositions([x for x in tg if x[1] != 'horizons'], [t])
    e = get_earth(t.replace('T', ' '))
    x_e = where.position('Earth', t)
    x_l1 = where.position('L1', t)
    lon, lat, r = ec.lon_lat(x_l1)
    assert abs(np.linalg.norm(x_e) - e.radius.to_value('km') / bk.RSUN_KM) < 1e-6
    assert abs(r - (e.radius.to_value('km') - 1.5e6) / bk.RSUN_KM) < 1e-6 and abs(lat - sun.B0(t).to_value('deg')) < 1e-6
    m = get_body_heliographic_stonyhurst('mars', t.replace('T', ' '))
    lon_m, lat_m, r_m = ec.lon_lat(where.position('Mars', t))
    assert abs(lon_m - m.lon.wrap_at('180d').to_value('deg')) < 1e-6 and abs(r_m - m.radius.to_value('km') / bk.RSUN_KM) < 1e-6
    return f'L1 {r / bk.AU_RSUN:.4f} AU at lat {lat:.2f}; Mars {r_m / bk.AU_RSUN:.3f} AU at lon {lon_m:.1f}'


def test_horizons_one_query_per_target_and_skipping():
    import sunpy.coordinates as sc
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    calls = []
    real = sc.get_horizons_coord

    def fake(body, time, **kw):
        calls.append((body, len(time)))
        if body in ('Parker Solar Probe', '-96'):
            raise ValueError('no ephemeris for the target at the requested time')           # not launched in 2012
        n = len(time)
        return SkyCoord(np.full(n, 110.0) * u.deg, np.zeros(n) * u.deg, np.full(n, 0.96) * u.AU,
                        frame=sc.frames.HeliographicStonyhurst, obstime=time)
    sc.get_horizons_coord = fake
    try:
        times = [(T0 + dt.timedelta(minutes=10 * i)).isoformat() for i in range(30)]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            where = ec.TargetPositions(ec.parse_targets('STA, PSP'), times)
    finally:
        sc.get_horizons_coord = real
    sta = [c for c in calls if c[0] in ('STEREO-A', '-234')]
    assert len(sta) == 1 and sta[0][1] == 30, calls                              # one query, all times
    assert 'STA' in where.pos and 'PSP' not in where.pos and 'skipped' in out.getvalue()
    lon, lat, r = ec.lon_lat(where.position('STA', times[5]))
    assert abs(lon - 110.0) < 1e-6 and abs(r / bk.AU_RSUN - 0.96) < 1e-6
    return f'{len(calls)} queries for 2 targets x 30 times; PSP skipped'


def test_evolution_figures_redrawn_with_lines(tmp):
    setup(tmp)
    sphere_series()
    bg = bk.ParkerBackground(v_sw=VSW, n_1au=5.0, br_1au=3.5, T=1e5)
    ec.EVO_QUANTITIES = ['MA', 'theta_bn']
    sp.GRID = (21, 41)
    series = sp.load_series('test', 'corona')
    plain = cfg.FIGURE_DIR / 'shock_evo' / 'test' / 'MA'                     # what shock_properties_sse.py would draw
    plain.mkdir(parents=True)
    for res_k, st in enumerate(series):
        (plain / f'shock_evo{int(round(st["model"].height * 10)):03d}_MA.png').write_bytes(b'plain')
    by_target = {}
    for label, lon0 in (('Earth', 0.0), ('STA', 110.0)):
        pos = bk.AU_RSUN * np.array([np.cos(np.radians(lon0)), np.sin(np.radians(lon0)), 0.0])
        with contextlib.redirect_stdout(io.StringIO()):
            by_target[label] = ec.evolution('test', bg, label, lambda t, p=pos: p, ['corona'])
    with contextlib.redirect_stdout(io.StringIO()):
        ec.figure('sse', 'test', by_target)
        ec.evo_figures('test', bg, by_target, ['corona'])
    figs = sorted((cfg.FIGURE_DIR / 'shock_evo' / 'test').glob('*/*.png'))
    overwritten = all(p.read_bytes()[:4] == b'\x89PNG' for p in plain.glob('*.png'))
    assert len(figs) == 20 and overwritten and (cfg.FIGURE_DIR / 'cobpoint' / 'evolution_cobpoint_sse_test.png').exists(), \
        (len(figs), overwritten)
    return f'{len(figs)} shock-evolution figures redrawn at their own paths (plain ones overwritten) + the overlay figure'


if __name__ == '__main__':
    tmp = tempfile.mkdtemp()
    tests = [('two targets, Sun-centred spheres in a Parker wind: cobpoints exact',
              lambda: test_parker_spiral_cobpoints_for_two_targets(tmp)),
             ('not connected (shock behind the Sun), then passed (target inside)', lambda: test_not_connected_and_passed(tmp)),
             ('the removed far side of a halo fit is not a shock for the cobpoint search',
              lambda: test_far_side_of_a_halo_is_not_a_shock(tmp)),
             ('targets: parsing, Earth / L1 / Mars from the ephemeris', test_targets_parsing_and_ephemeris),
             ('Horizons: one query per target for all times; a target with no ephemeris skipped',
              test_horizons_one_query_per_target_and_skipping),
             ('shock-evolution figures redrawn with the field lines, overwriting the plain ones',
              lambda: test_evolution_figures_redrawn_with_lines(tmp))]
    passed = 0
    for name, fn in tests:
        try:
            out = fn()
            passed += 1
            print(f'PASS  {name}' + (f'   {out}' if out is not None else ''))
        except Exception as e:
            import traceback
            print(f'FAIL  {name}: {type(e).__name__}: {e}')
            traceback.print_exc()
    shutil.rmtree(tmp)
    print(f'{passed}/{len(tests)} passed')
