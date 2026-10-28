"""
Tests for derive_para_corona.py's 'nearest' method and the exclusion of poorly constrained fits.
Run:  python test_derive_para_corona.py
"""
import contextlib
import datetime as dt
import io
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import derive_para_corona as dc
import event_config as cfg
from deformable_shock import DeformableShock

T0 = dt.datetime(2012, 3, 7, 0, 39)
RS = 695700.0
V = 3.0 * 695700.0 / 1020.0                               # apex speed putting the fits 17 min apart [km/s]


def write_fits(folder, constraints=None):
    """Three fits with different shapes and directions, each with a dent; apex moving at V."""
    fits = [DeformableShock.from_pythea(8.0, 0.70, 0.40, 1.20, lat=14, lon=330, tilt=10, dents=[(70, 90, 20, 0.3)]),
            DeformableShock.from_pythea(11.0, 0.65, 0.35, 1.30, lat=16, lon=327, tilt=14, dents=[(65, 100, 18, 0.25)]),
            DeformableShock.from_pythea(14.0, 0.60, 0.30, 1.40, lat=18, lon=324, tilt=18, dents=[(60, 110, 16, 0.2)])]
    for i, m in enumerate(fits):
        t = T0 + dt.timedelta(seconds=(m.height - 8.0) * RS / V)
        d = {**m.to_dict(), 'time': t.isoformat()}
        if constraints and i in constraints:
            d['constraints'] = constraints[i]
        (folder / f'shock_fit_{t:%Y%m%d_%H%M}.json').write_text(json.dumps(d))
    return fits


def setup():
    tmp = Path(tempfile.mkdtemp())
    cfg.RESULTS_DIR, cfg.FINAL_DIR, cfg.FIGURE_DIR = tmp / 'semi', tmp / 'final', tmp / 'fig'
    cfg.RESULTS_DIR.mkdir(parents=True)
    return tmp


def run(methods):
    dc.METHODS = methods
    with contextlib.redirect_stdout(io.StringIO()) as out:
        dc.main()
    files = sorted(cfg.FINAL_DIR.glob('shock_para_corona*_nearest.json'))
    return {json.loads(p.read_text())['height']: json.loads(p.read_text()) for p in files}, out.getvalue()


def test_nearest_is_the_end_fit_scaled_beyond_and_interpolated_inside():
    tmp = setup()
    fits = write_fits(cfg.RESULTS_DIR)
    dc.NEAREST_KINEMATICS, dc.EXCLUDE_IF_PRIOR_SET = 'constant', None
    steps, _ = run(['nearest'])
    th, ph = DeformableShock.grid(25, 49)
    worst = 0.0
    for h, end in ((2.5, fits[0]), (5.5, fits[0]), (17.5, fits[2]), (21.5, fits[2])):
        m = DeformableShock.from_dict(steps[h])
        want = end.scaled((h - 1.0) / (end.height - 1.0), r0=1.0)                 # self-similar about the surface
        worst = max(worst, float(np.max(np.abs(m.position(th, ph) - want.position(th, ph)))))
        assert np.allclose(m.normals(th, ph), want.normals(th, ph), atol=1e-6)
        assert [list(d) for d in m.dents] == [list(d) for d in end.dents]
    assert worst < 1e-9, worst
    m = DeformableShock.from_dict(steps[9.5])                                     # between fits 0 and 1
    for k in ('kappa', 'epsilon', 'alpha', 'lat', 'lon', 'tilt'):
        lo, hi = sorted((getattr(fits[0], k), getattr(fits[1], k)))
        assert lo - 1e-9 <= getattr(m, k) <= hi + 1e-9, k
    for h in (2.5, 9.5, 21.5):                                                    # kinematics: the linear fit
        t = (dt.datetime.fromisoformat(steps[h]['time']) - T0).total_seconds()
        assert abs(t - (h - 8.0) * RS / V) < 2.0 and abs(steps[h]['derived']['speed_kms'] - V) < 0.5
    shutil.rmtree(tmp)
    return f'beyond the fits: exact self-similar copies (max {worst:.1e} R_sun), dents kept; inside: interpolated'


def test_poorly_constrained_fits_are_excluded():
    dc.NEAREST_KINEMATICS, dc.EXCLUDE_IF_PRIOR_SET = 'constant', 3
    shape_fixed = {n: {'set_by': 'fixed'} for n in ('kappa', 'epsilon', 'alpha')}
    shape_fixed.update({n: {'set_by': 'data'} for n in ('height', 'lat', 'lon', 'tilt')})
    bad = {n: {'set_by': 'prior'} for n in ('height', 'epsilon', 'lat', 'lon')}
    bad.update({n: {'set_by': 'data'} for n in ('kappa', 'alpha', 'tilt')})
    out = {}
    # 1. every fit with its shape fixed by hand (FIX): fixed parameters do not count -> all kept, nothing said
    tmp = setup()
    write_fits(cfg.RESULTS_DIR, constraints={i: shape_fixed for i in range(3)})
    with contextlib.redirect_stdout(io.StringIO()) as s:
        out['shape fixed'] = len(dc.load_fits())
    assert out['shape fixed'] == 3 and 'excluded' not in s.getvalue() and 'warning' not in s.getvalue()
    shutil.rmtree(tmp)
    # 2. one poorly constrained fit among good ones -> excluded, with the parameters named
    tmp = setup()
    write_fits(cfg.RESULTS_DIR, constraints={1: bad})
    with contextlib.redirect_stdout(io.StringIO()) as s:
        out['one bad'] = len(dc.load_fits())
    assert out['one bad'] == 2 and 'excluded shock_fit_' in s.getvalue() and 'height, epsilon, lat, lon' in s.getvalue()
    shutil.rmtree(tmp)
    # 3. all poorly constrained -> all kept, with a warning (the rule never empties the series)
    tmp = setup()
    write_fits(cfg.RESULTS_DIR, constraints={i: bad for i in range(3)})
    with contextlib.redirect_stdout(io.StringIO()) as s:
        out['all bad'] = len(dc.load_fits())
    assert out['all bad'] == 3 and 'all kept' in s.getvalue()
    shutil.rmtree(tmp)
    dc.EXCLUDE_IF_PRIOR_SET = None
    return out


def test_gui_points_join_the_kinematics_and_trends_not_the_front():
    tmp = setup()
    fits = write_fits(cfg.RESULTS_DIR)                                          # 3 fits at 8, 11, 14 R_sun
    gui = [DeformableShock.from_pythea(h, 0.6, 0.4, 1.0, lat=10, lon=332, tilt=0) for h in (4.6, 6.2)]
    for i, m in enumerate(gui):                                                 # 2 earlier GUI-only times
        tg = T0 - dt.timedelta(minutes=17 * (2 - i))
        (cfg.RESULTS_DIR / f'shock_{tg:%Y%m%d_%H%M}.json').write_text(json.dumps({**m.to_dict(), 'time': tg.isoformat()}))
    (cfg.RESULTS_DIR / f'shock_{T0:%Y%m%d_%H%M}.json').write_text(json.dumps(fits[0].to_dict()))  # superseded by the fit
    dc.NEAREST_KINEMATICS, dc.EXCLUDE_IF_PRIOR_SET, dc.USE_GUI_POINTS = 'constant', None, True
    with contextlib.redirect_stdout(io.StringIO()) as out:
        dc.METHODS = ['constant', 'nearest']
        dc.main()
    f = dc.load_fits()
    g = dc.load_gui_points(f)
    assert len(f) == 3 and len(g) == 2 and '5 measured times (3 fits, 2 GUI-only' in out.getvalue()
    ev = dc.Evolution(f, f + g)
    ts = [(p['time'] - ev.t0).total_seconds() for p in sorted(f + g, key=lambda p: p['time'])]
    hs = [p['h'] for p in sorted(f + g, key=lambda p: p['time'])]
    assert abs(ev.s - np.polyfit(ts, hs, 1)[0]) < 1e-15                         # speed from all 5 points
    assert abs(ev.h_lo - 8.0) < 1e-9 and abs(ev.k_lo - 4.6) < 1e-9              # fitted range: fits; kinematics: all
    m = DeformableShock.from_dict(json.loads((cfg.FINAL_DIR / 'shock_para_corona025_nearest.json').read_text()))
    assert abs(m.kappa - fits[0].kappa) < 1e-9 and len(m.dents) == 1            # 'nearest': the first FIT's front
    assert (cfg.FIGURE_DIR / 'derive_para_corona.png').exists()
    shutil.rmtree(tmp)
    return f'speed from 5 points {ev.s * RS:.0f} km/s; nearest front from the fits; figure written'


if __name__ == '__main__':
    tests = [('nearest: end fits scaled self-similarly beyond the range, interpolated inside, kinematics',
              test_nearest_is_the_end_fit_scaled_beyond_and_interpolated_inside),
             ('poorly constrained fits excluded', test_poorly_constrained_fits_are_excluded),
             ('GUI-only times join the kinematics and trends, not the front', test_gui_points_join_the_kinematics_and_trends_not_the_front)]
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
