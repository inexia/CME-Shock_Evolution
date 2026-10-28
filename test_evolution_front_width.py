"""
Tests for evolution_front_width.py. Run:  python test_evolution_front_width.py
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
import evolution_front_width as fw
import evolution_cobpoint as ec
import shock_properties_sse as sp
from deformable_shock import DeformableShock

T0 = dt.datetime(2021, 10, 28, 16, 0)
RS = 695700.0
V = 2500.0


def setup():
    tmp = Path(tempfile.mkdtemp())
    cfg.RESULTS_DIR, cfg.FINAL_DIR, cfg.FIGURE_DIR = tmp / 'semi', tmp / 'final', tmp / 'fig'
    cfg.FINAL_DIR.mkdir(parents=True)
    sp.FRONT_HALF_WIDTH_DEG = 90.0
    sp.GRID = (61, 121)
    hs = [5.0, 6.0, 7.0, 8.0]
    for h in hs:                                                        # self-similar, apex at V
        m = DeformableShock.from_pythea(h, 0.8, 0.2, 1.0, lat=0.0, lon=0.0, tilt=0.0)
        t = T0 + dt.timedelta(seconds=(h - hs[0]) * RS / V)
        (cfg.FINAL_DIR / f'shock_para_corona{int(round(h * 10)):03d}_test.json').write_text(
            json.dumps({**m.to_dict(), 'time': t.isoformat(), 'derived': {'speed_kms': V}}))
    return tmp


BG = bk.ParkerBackground(v_sw=400.0, n_1au=5.0, br_1au=3.5, T=1e5)


def res_at(k):
    series = sp.load_series('test', 'corona')
    return ec.load_props('corona', 'test', series, k, BG)


def test_shock_narrower_than_the_front_and_ends_at_the_threshold():
    tmp = setup()
    res = res_at(2)
    fw.CRITERION, fw.THRESHOLD = 'Mf', 1.0
    nose, a, b, eff, geo, _ = fw.widths(res)
    assert 0 < eff < geo, (eff, geo)
    # the criterion at the ends, re-read from the cut by interpolation along it, equals the threshold
    xyz, val, on = max(sp.cut_curves(res, 'Mf', 2, 0.0), key=lambda c: np.max(np.linalg.norm(c[0][:, :2], axis=1)))
    ends = []
    for e in (a, b):
        i = int(np.argmin(np.linalg.norm(xyz - e, axis=1)))
        j = i + 1 if i + 1 < len(xyz) and np.linalg.norm(xyz[i + 1] - e) < np.linalg.norm(xyz[max(i - 1, 0)] - e) else i - 1
        f = np.linalg.norm(e - xyz[i]) / max(np.linalg.norm(xyz[j] - xyz[i]), 1e-12)
        ends.append(val[i] + f * (val[j] - val[i]))
    assert np.allclose(ends, 1.0, atol=0.02), ends
    shutil.rmtree(tmp)
    return f'shock {eff:.0f} deg of a front {geo:.0f} deg; M_f at the ends {ends[0]:.3f}, {ends[1]:.3f}'


def test_monotonic_in_the_threshold():
    tmp = setup()
    res = res_at(2)
    fw.CRITERION = 'Mf'
    out = {}
    for thr in (0.0, 1.0, 1.2, 50.0):
        fw.THRESHOLD = thr
        wd = fw.widths(res)
        out[thr] = (wd[3], wd[4], fw.area_fraction(res))
    cell = 180.0 / (sp.GRID[0] - 1)                                  # grid spacing [deg]: the last cell at each edge has
    assert abs(out[0.0][0] - out[0.0][1]) <= 2 * cell                # no interpolable value -> the whole front, to 2 cells
    assert out[0.0][0] >= out[1.0][0] > out[1.2][0] > out[50.0][0] == 0.0
    assert out[0.0][2] >= out[1.0][2] > out[1.2][2] >= out[50.0][2] == 0.0
    shutil.rmtree(tmp)
    return {thr: (round(e, 1), round(a, 2)) for thr, (e, g, a) in out.items()}


def test_figures_and_csv():
    tmp = setup()
    fw.CRITERION, fw.THRESHOLD, fw.METHODS, fw.VERSIONS, fw.TARGETS = 'Mf', 1.0, ['test'], ['sse'], 'Earth'
    bk_load = bk.load_background
    bk.load_background = lambda: BG
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fw.main()
    finally:
        bk.load_background = bk_load
    steps = sorted((cfg.FIGURE_DIR / 'front_width' / 'corona_test').glob('*.png'))
    assert len(steps) == 4 and (cfg.FIGURE_DIR / 'front_width' / 'front_width_test.png').exists()
    rows = (cfg.FINAL_DIR / 'front_width_sse_test.csv').read_text().splitlines()
    assert len(rows) == 5 and rows[0].startswith('version,series,apex_height_rsun')
    shutil.copy(steps[2], '/tmp/front_width_step.png')
    shutil.copy(cfg.FIGURE_DIR / 'front_width' / 'front_width_test.png', '/tmp/front_width_summary.png')
    shutil.rmtree(tmp)
    return f'{len(steps)} step figures, the summary and the csv ({len(rows) - 1} rows)'


if __name__ == '__main__':
    tests = [('shock narrower than the front; M_f = threshold at both ends', test_shock_narrower_than_the_front_and_ends_at_the_threshold),
             ('monotonic in the threshold (0: the whole front; huge: none)', test_monotonic_in_the_threshold),
             ('figures and csv', test_figures_and_csv)]
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
