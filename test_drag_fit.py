"""
Tests for drag_fit.py and the three versions sharing its gamma0. Run:  python test_drag_fit.py
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
import drag_fit as df
import derive_para_ip as ip
import shock_properties_sse as sp
from deformable_shock import DeformableShock

T0 = dt.datetime(2012, 3, 7, 1, 47)
RS = 695700.0
AU = bk.AU_RSUN


def setup():
    tmp = Path(tempfile.mkdtemp())
    cfg.RESULTS_DIR, cfg.FINAL_DIR, cfg.FIGURE_DIR = tmp / 'semi', tmp / 'final', tmp / 'fig'
    cfg.FINAL_DIR.mkdir(parents=True)
    sp.FRONT_HALF_WIDTH_DEG = None                                   # idealised full shells
    return tmp


def write_series(models, times, speed, method='test'):
    for m, t in zip(models, times):
        d = {**m.to_dict(), 'time': t.isoformat(), 'derived': {'speed_kms': speed}}
        (cfg.FINAL_DIR / f'shock_para_corona{int(round(m.height * 10)):03d}_{method}.json').write_text(json.dumps(d))


def sphere_series(V=1500.0, h_end=21.5):
    hs = [h_end - 2.0, h_end - 1.0, h_end]
    times = [T0 + dt.timedelta(seconds=(h - h_end) * RS / V) for h in hs]
    write_series([DeformableShock(h, h, h, h, 0.0, 0.0, 0.0) for h in hs], times, V)     # Sun-centred: v0 = V on every ray


def observe(gamma_true, v0, w, r0=21.5, D=AU):
    """A synthetic in-situ arrival from the analytic DBM solution (uniform wind), set in event_config."""
    t_arr = ip.dbm_time((D - r0) * RS, v0, w, gamma_true)
    v_arr = ip.dbm(t_arr, v0, w, gamma_true)[1]
    cfg.ARRIVAL_TIME = (T0 + dt.timedelta(seconds=float(t_arr))).strftime('%Y-%m-%d %H:%M')
    cfg.ARRIVAL_SPEED, cfg.SC_DISTANCE_AU, cfg.SC_LON, cfg.SC_LAT = float(v_arr), D / AU, 0.0, 0.0
    return float(t_arr), float(v_arr)


def test_recovers_a_known_drag():
    tmp = setup()
    sphere_series()
    bg = bk.ParkerBackground(v_sw=400.0, n_1au=5.0, br_1au=3.5, T=1e5)     # n ~ r^-2: constant N, constant gamma
    g_true = 1.0e-7
    t_arr, v_arr = observe(g_true, 1500.0, 400.0)
    df.ARRIVAL_SPEED_TYPE = 'radial'
    with contextlib.redirect_stdout(io.StringIO()):
        fit = df.fit('test', bg)
    err = fit['gamma0_per_km'] / g_true - 1
    assert abs(err) < 0.03, (fit['gamma0_per_km'], g_true)
    assert abs(fit['model']['time_residual_h']) < 0.1 and abs(fit['model']['speed_residual_kms']) < 10
    assert (cfg.FINAL_DIR / 'drag_fit_test.json').exists() and (cfg.FIGURE_DIR / 'drag_fit_test.png').exists()
    shutil.rmtree(tmp)
    return f'gamma0 {fit["gamma0_per_km"]:.3g} vs true {g_true:.3g} ({err * 100:+.1f}%); arrival {fit["model"]["time_residual_h"]:+.2f} h'


def test_sse_series_reaches_the_observed_arrival():
    tmp = setup()
    sphere_series()
    bg = bk.ParkerBackground(v_sw=400.0, n_1au=5.0, br_1au=3.5, T=1e5)
    t_arr, v_arr = observe(1.0e-7, 1500.0, 400.0)
    df.ARRIVAL_SPEED_TYPE = 'radial'
    ip.METHODS, ip.REFIT_DRAG = ['test'], True
    with contextlib.redirect_stdout(io.StringIO()):
        rep = ip.run('test', bg)
    files = sorted(cfg.FINAL_DIR.glob('shock_para_ip*_test.json'))
    steps = [json.loads(p.read_text()) for p in files]
    hs = [s['height'] for s in steps]
    assert abs(hs[0] - 21.5) < 1e-9 and abs(hs[1] - 26.5) < 1e-9 and hs[-1] >= AU - 1e-6     # 5 R_sun steps, extended
    d_sc = np.array([s['derived']['sc_ray_distance_rsun'] for s in steps])
    ts = np.array([(dt.datetime.fromisoformat(s['time']) - T0).total_seconds() for s in steps])
    t_at_sc = float(np.interp(AU, d_sc, ts))                      # when the front on the spacecraft ray reaches 1 AU
    assert abs(t_at_sc - t_arr) / 3600 < 0.15, (t_at_sc / 3600, t_arr / 3600)
    assert abs(rep['apex_height_at_arrival_rsun'] - AU) < 0.5     # a sphere: the apex is on the spacecraft ray
    shutil.rmtree(tmp)
    return f'{len(steps)} steps; the spacecraft ray reaches 1 AU {abs(t_at_sc - t_arr) / 60:.1f} min from the observed time'


def test_dbm_and_gsd_use_the_shared_gamma0():
    import shock_properties_dbm as sd
    import shock_properties_gsd as sg
    tmp = setup()
    sphere_series()
    bg = bk.ParkerBackground(v_sw=400.0, n_1au=5.0, br_1au=3.5, T=1e5)
    t_arr, v_arr = observe(1.0e-7, 1500.0, 400.0)
    df.ARRIVAL_SPEED_TYPE = 'radial'
    with contextlib.redirect_stdout(io.StringIO()):
        shared = df.fit('test', bg)
    sd.RAY_GRID, sd.H_STEP, sd.H_END, sd.REFIT_DRAG, sd.PROPERTIES, sd.SMOOTH_DEG_PER_DAY = (19, 37), 40.0, 220.0, False, False, 0.0
    table = sd.WindTable(bg, np.arange(*sd.R_TABLE), np.linspace(0, np.pi, 19), np.linspace(0, 2 * np.pi, 37, endpoint=False),
                         verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        rep = sd.run('test', bg, table)
    assert rep['gamma0_used_per_km'] == shared['gamma0_per_km']                      # no fit of its own
    chk = rep['arrival_check']
    assert abs(chk['time_residual_h']) < 0.25 and abs(chk['speed_residual_kms']) < 25, chk
    sg.RAY_GRID, sg.FIT_GRID, sg.H_STEP, sg.H_END, sg.REFIT_DRAG, sg.REFIT_GAMMA = (19, 37), (10, 19), 40.0, 220.0, False, True
    sg.PROPERTIES, sg.SMOOTH_DEG_PER_DAY, sg.FIT_DT_MIN = False, 0.0, 10.0
    ftable = sg.FieldTable(bg, np.arange(*sd.R_TABLE), np.linspace(0, np.pi, 19), np.linspace(0, 2 * np.pi, 37, endpoint=False),
                           verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        rep_g = sg.run('test', bg, ftable)
    g_ratio = rep_g['gamma0_used_per_km'] / shared['gamma0_per_km']
    assert rep_g['refit'] is not None and 0.2 < g_ratio < 5.0
    assert abs(rep_g['arrival_check']['time_residual_h']) < 0.5, rep_g['arrival_check']
    shutil.rmtree(tmp)
    return (f'DBM: shared gamma0, arrival {chk["time_residual_h"]:+.2f} h, {chk["speed_residual_kms"]:+.0f} km/s; '
            f'GSD: refit {g_ratio:.2f} x shared, arrival {rep_g["arrival_check"]["time_residual_h"]:+.2f} h')


def test_spacecraft_off_the_nose():
    """An ellipsoid pointing 40 deg from the spacecraft: the ray starts from the front's own distance and speed in
    the spacecraft direction, both below the apex values; the SSE nose then runs ahead of the spacecraft ray."""
    tmp = setup()
    hs = [19.5, 20.5, 21.5]
    V = 1500.0
    times = [T0 + dt.timedelta(seconds=(h - 21.5) * RS / V) for h in hs]
    write_series([DeformableShock.from_pythea(h, 0.7, 0.3, 1.2, lat=0.0, lon=40.0, tilt=0.0) for h in hs], times, V)
    bg = bk.ParkerBackground(v_sw=400.0, n_1au=5.0, br_1au=3.5, T=1e5)
    cfg.ARRIVAL_TIME, cfg.ARRIVAL_SPEED, cfg.SC_DISTANCE_AU, cfg.SC_LON, cfg.SC_LAT = '2012-03-08 20:00', 600.0, 1.0, 0.0, 0.0
    df.ARRIVAL_SPEED_TYPE = 'normal'
    with contextlib.redirect_stdout(io.StringIO()):
        fit = df.fit('test', bg)
    ray = fit['ray']
    assert ray['along'] == 'spacecraft' and abs(ray['apex_to_ray_angle_deg'] - 40.0) < 0.5
    assert ray['r0_rsun'] < 21.5 and ray['v0_kms'] < V and 0.5 < ray['cos_normal_ray'] < 1.0
    ip.METHODS, ip.REFIT_DRAG = ['test'], False
    with contextlib.redirect_stdout(io.StringIO()):
        rep = ip.run('test', bg)
    assert rep['apex_height_at_arrival_rsun'] > AU                 # the nose is beyond 1 AU when the flank arrives
    shutil.rmtree(tmp)
    return (f'ray at {ray["apex_to_ray_angle_deg"]:.0f} deg from the apex: r0 {ray["r0_rsun"]:.1f} R_sun, v0 {ray["v0_kms"]:.0f} km/s '
            f'(apex 21.5, {V:.0f}); nose at {rep["apex_height_at_arrival_rsun"]:.0f} R_sun at the arrival')


if __name__ == '__main__':
    tests = [('recovers a known drag from a synthetic arrival (uniform wind)', test_recovers_a_known_drag),
             ('SSE series reaches the observed arrival on the spacecraft ray', test_sse_series_reaches_the_observed_arrival),
             ('DBM uses the shared gamma0 (arrival check); GSD refits near it', test_dbm_and_gsd_use_the_shared_gamma0),
             ('spacecraft 40 deg off the nose: the ray starts from the front there', test_spacecraft_off_the_nose)]
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
    print(f'{passed}/{len(tests)} passed')
