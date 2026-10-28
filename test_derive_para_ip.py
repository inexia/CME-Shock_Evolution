"""
Tests for derive_para_ip.py, each against an exact answer. Run:  python test_derive_para_ip.py
"""
import datetime as dt
import io
import contextlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import derive_para_ip as ip
import event_config as cfg
from deformable_shock import DeformableShock

RS = ip.RSUN_KM
T0 = dt.datetime(2012, 3, 7, 1, 47, 26)


def test_dbm_matches_numerical_integration():
    worst = 0.0
    for v0, w, g in ((2150.0, 400.0, 0.4e-7), (300.0, 550.0, 1.0e-7), (900.0, 900.0, 0.5e-7)):
        f = lambda t, y: [y[1], -g * (y[1] - w) * abs(y[1] - w)]
        t = np.linspace(0, 3 * 86400, 50)
        sol = solve_ivp(f, (0, t[-1]), [0.0, v0], t_eval=t, rtol=1e-11, atol=1e-6)
        r, v = ip.dbm(t, v0, w, g)
        worst = max(worst, np.max(np.abs(r - sol.y[0])) / RS, np.max(np.abs(v - sol.y[1])))
        tt = ip.dbm_time(r[1:], v0, w, g)
        assert np.allclose(tt, t[1:], rtol=1e-8, atol=1e-3)
    assert worst < 1e-4, worst
    return float(worst)


def test_apex_ray_is_the_apex_height():
    base = DeformableShock.from_pythea(21.5, 0.6, -0.3, 1.2, lat=12, lon=-25, tilt=30)
    worst = max(abs(ip.ray_hit(ip.at_height(base, h), base.frame[0]) - h) for h in (21.5, 60.0, 215.0))
    assert worst < 1e-6, worst
    return float(worst)


def _synthetic_arrival(base, v0, gamma, w, D_au, d):
    R, slope, cosn = ip.sc_geometry(base, d)
    h_arr = ip.arrival_height(R, D_au * ip.AU_RSUN, base.height)
    t = float(ip.dbm_time((h_arr - base.height) * RS, v0, w, gamma))
    v_apex = float(ip.dbm(t, v0, w, gamma)[1])
    return T0 + dt.timedelta(seconds=t), v_apex * slope(h_arr) * cosn(h_arr), h_arr


def _setup(tmp, base, v0):
    cfg.FINAL_DIR = Path(tmp) / 'final'
    cfg.RESULTS_DIR = Path(tmp) / 'semi'
    cfg.FIGURE_DIR = Path(tmp) / 'fig'
    cfg.FINAL_DIR.mkdir(parents=True, exist_ok=True)
    d = {**base.to_dict(), 'time': T0.isoformat(), 'derived': {'speed_kms': v0}}
    (cfg.FINAL_DIR / 'shock_para_corona215_constant.json').write_text(json.dumps(d))


def test_fit_recovers_gamma_and_w(tmp):
    base = DeformableShock.from_pythea(21.5, 0.66, -0.54, 1.41, lat=13, lon=330, tilt=46)    # a halo front
    v0, g_true = 2150.0, 0.6e-7
    _setup(tmp, base, v0)
    out = {}
    for fit_w, w_true in ((False, 400.0), (True, 470.0)):
        d = np.array([1.0, 0.0, 0.0])
        t_arr, v_arr, h_arr = _synthetic_arrival(base, v0, g_true, w_true, 0.99, d)
        ip.METHODS, ip.ARRIVAL_TIME, ip.ARRIVAL_SPEED, ip.SC_DISTANCE_AU = ['constant'], t_arr.isoformat(), v_arr, 0.99
        ip.SC_DIRECTION, ip.ARRIVAL_SPEED_TYPE, ip.FIT_VSW, ip.V_SW = (0.0, 0.0), 'normal', fit_w, 400.0
        with contextlib.redirect_stdout(io.StringIO()):
            rep = ip.run('constant')
        f = rep['fit']
        assert abs(f['gamma_best'] / g_true - 1) < 1e-3 and abs(f['w_best'] - w_true) < 0.5, (fit_w, f)
        if not fit_w:
            assert abs(f['gamma_time_only'][0] / g_true - 1) < 1e-4 and abs(f['gamma_speed_only'][0] / g_true - 1) < 1e-4
        out['w fitted' if fit_w else 'w fixed'] = (round(f['gamma_best'] / 1e-7, 5), round(f['w_best'], 2))
    out['Earth reached when the apex is at'] = round(h_arr, 1)
    n = len(list(cfg.FINAL_DIR.glob('shock_para_ip*_constant.json')))
    assert n == len(np.arange(21.5, h_arr + 5.0, 5.0)), n                 # extended past the spacecraft
    out['series extended to'] = f'{n} files'
    return out


def test_series_files(tmp):
    ip.ARRIVAL_TIME = ip.ARRIVAL_SPEED = ip.SC_DISTANCE_AU = None
    ip.FIT_VSW, ip.V_SW, ip.GAMMA = False, 400.0, 0.5e-7
    for p in cfg.FINAL_DIR.glob('shock_para_ip*_constant.json'):         # files of the previous (extended) run
        p.unlink()
    with contextlib.redirect_stdout(io.StringIO()):
        ip.run('constant')
    files = sorted(cfg.FINAL_DIR.glob('shock_para_ip*_constant.json'))
    hs = [json.loads(p.read_text())['height'] for p in files]
    want = list(np.arange(21.5, 220.0, 5.0)) + [220.0]
    assert np.allclose(hs, want), hs
    first = json.loads(files[0].read_text())
    times = [dt.datetime.fromisoformat(json.loads(p.read_text())['time']) for p in files]
    assert first['time'][:19] == T0.isoformat()[:19] and all(b > a for a, b in zip(times, times[1:]))
    for p in files[::7]:
        d = json.loads(p.read_text())
        t = (dt.datetime.fromisoformat(d['time']) - T0).total_seconds()
        r, v = ip.dbm(t, 2150.0, 400.0, 0.5e-7)
        assert abs(21.5 + r / RS - d['height']) < 2e-3 and abs(v - d['derived']['speed_kms']) < 0.2
        m = DeformableShock.from_dict(d)
        assert np.allclose([m.kappa, m.epsilon, m.alpha], [0.66, -0.54, 1.41], atol=1e-9)    # self-similar
    return f'{len(files)} files, {hs[0]} ... {hs[-1]} R_sun'


if __name__ == '__main__':
    tmp = tempfile.mkdtemp()
    tests = [('DBM = numerical integration (decelerating, accelerating, w = v0)', test_dbm_matches_numerical_integration),
             ('the front on the apex ray is at the apex height', test_apex_ray_is_the_apex_height),
             ('fit recovers gamma (w fixed) and gamma + w (FIT_VSW) from a synthetic arrival',
              lambda: test_fit_recovers_gamma_and_w(tmp)),
             ('series: 5 R_sun steps to 220, self-similar, DBM times and speeds', lambda: test_series_files(tmp))]
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
