"""
Tests for shock_properties_dbm.py and ray_front.py. Run:  python test_shock_properties_dbm.py
"""
import datetime as dt
import io
import contextlib

import numpy as np

import background as bk
import derive_para_ip as dip
import shock_properties_dbm as sd
from deformable_shock import DeformableShock
from ray_front import RayFront

T0 = dt.datetime(2012, 3, 7, 1, 47)
RS = bk.RSUN_KM


def _series(models, times, speed):
    return [{'name': f's{i}', 'model': m, 'time_iso': t.isoformat(), 'time': t, 'speed_kms': speed, 'in_range': True}
            for i, (m, t) in enumerate(zip(models, times))]


def _sphere_series(V=1500.0, h_end=21.5):
    import shock_properties_sse as _sp
    _sp.FRONT_HALF_WIDTH_DEG = None                                   # idealised full shells
    hs = [h_end - 2.0, h_end - 1.0, h_end]
    times = [T0 + dt.timedelta(seconds=(h - h_end) * RS / V) for h in hs]
    return _series([DeformableShock(h, h, h, h, 10.0, 330.0, 0.0) for h in hs], times, V)


class TwoStream(bk.ParkerBackground):
    """Radial wind and r^2-normalised density that differ between the two longitude halves."""
    def __init__(self, w=(350.0, 750.0), n=(10.0, 4.0), **kw):
        super().__init__(**kw)
        self.w, self.nn = w, n

    def _sample(self, r, theta, lon, offset):
        east = (np.mod(lon, 2 * np.pi) < np.pi)
        w = np.where(east, self.w[0], self.w[1])
        n = np.where(east, self.nn[0], self.nn[1]) * (bk.AU_RSUN / r) ** 2
        e_r, _, _ = bk._spherical_basis(theta, lon)
        T = np.full_like(r, self.T)
        return bk._derived(n, T, 2 * n * 1e6 * bk.KB * T, self._field(r, theta, lon, offset), w[:, None] * e_r)


def _table(bg, grid):
    r = np.arange(21.5, 260.0 + 1e-9, 1.0)
    return sd.WindTable(bg, r, np.linspace(0, np.pi, grid[0]), np.linspace(0, 2 * np.pi, grid[1], endpoint=False),
                        verbose=False)


def test_rayfront_matches_the_ellipsoid():
    m = DeformableShock.from_pythea(21.5, 0.66, -0.54, 1.41, lat=13, lon=330, tilt=46)       # encloses the Sun
    fr = RayFront.from_surface(m, np.linspace(0, np.pi, 181), np.linspace(0, 2 * np.pi, 361, endpoint=False))
    th, ph = DeformableShock.grid(25, 49)
    x_m = m.position(th, ph)                                   # points of the ellipsoid ...
    r_m = np.linalg.norm(x_m, axis=-1)
    pos = np.nanmax(np.abs(fr.R_at(x_m / r_m[..., None]) - r_m) / r_m)   # ... must lie on the ray front
    n_m = m.normals(th, ph)
    n_f = fr.normals_at(x_m)
    ang = np.degrees(np.arccos(np.clip(np.sum(n_m * n_f, -1), -1, 1)))
    assert pos < 2e-3 and np.nanmax(ang) < 1.0 and abs(fr.height - 21.5) < 1e-3, (pos, np.nanmax(ang))
    return {'position': float(f'{pos:.1e}'), 'normal angle deg': float(f'{np.nanmax(ang):.2f}')}


def test_uniform_wind_is_classic_dbm():
    V, w, g = 1500.0, 420.0, 0.5e-7
    bg = bk.ParkerBackground(v_sw=w, n_1au=5.0)
    series = _sphere_series(V)
    sd.SMOOTH_DEG_PER_DAY, sd.GAMMA_DENSITY_SCALING = 2.0, True
    front = sd.initial_front(series, 31, 61)
    assert np.allclose(front.R, 21.5, atol=1e-6) and np.allclose(front.V, V, rtol=1e-6)
    table = _table(bg, (31, 61))
    L0 = sd.l0_function(bg, T0)
    heights = [41.5, 101.5, 201.5]
    fronts, _ = sd.propagate(front, table, L0, T0, g, 5.0 * (bk.AU_RSUN / bk.AU_RSUN) ** 2, 5.0, heights)
    worst_t, worst_r = 0.0, 0.0
    for fr, h in zip(fronts, heights):
        t = (dt.datetime.fromisoformat(fr.time) - T0).total_seconds()
        t_exact = float(dip.dbm_time((h - 21.5) * RS, V, w, g))
        worst_t = max(worst_t, abs(t - t_exact))
        worst_r = max(worst_r, float(np.nanmax(np.abs(fr.R - h))))
        v_exact = float(dip.dbm(t_exact, V, w, g)[1])
        assert abs(fr.speed - v_exact) < 0.5, (fr.speed, v_exact)
    assert worst_t < 60.0 and worst_r < 1e-3, (worst_t, worst_r)
    return {'time error s': round(worst_t, 1), 'sphere kept to R_sun': float(f'{worst_r:.1e}')}


def test_two_stream_wind_deforms_the_front_ray_by_ray():
    V, g0 = 1500.0, 0.5e-7
    bg = TwoStream(w=(350.0, 750.0), n=(10.0, 4.0))
    series = _sphere_series(V)
    sd.SMOOTH_DEG_PER_DAY, sd.GAMMA_DENSITY_SCALING = 0.0, True                # independent rays: exact per ray
    front = sd.initial_front(series, 31, 61)
    table = _table(bg, (31, 61))
    n_ref = 7.0                                                               # mean of the two halves
    fronts, _ = sd.propagate(front, table, sd.l0_function(bg, T0), T0, g0, n_ref, 5.0, [101.5])
    fr = fronts[0]
    t = (dt.datetime.fromisoformat(fr.time) - T0).total_seconds()
    TH, PH = np.meshgrid(fr.theta, fr.phi, indexing='ij')
    east = PH < np.pi
    worst = 0.0
    for mask, w, n in ((east, 350.0, 10.0), (~east, 750.0, 4.0)):
        r_exact = 21.5 + float(dip.dbm(t, V, w, g0 * n / n_ref)[0]) / RS
        v_exact = float(dip.dbm(t, V, w, g0 * n / n_ref)[1])
        inner = mask & (np.abs(PH - np.pi) > 0.2) & (np.abs(PH - 0) > 0.2) & (np.abs(PH - 2 * np.pi) > 0.2) \
            & (TH > 0.2) & (TH < np.pi - 0.2)                                  # away from the boundaries and poles
        worst = max(worst, float(np.nanmax(np.abs(fr.R[inner] - r_exact))), float(np.nanmax(np.abs(fr.V[inner] - v_exact))) / 10)
    assert worst < 0.05, worst
    fast = float(np.nanmean(fr.R[~east & (np.abs(PH - 1.5 * np.pi) < 0.5)]))
    slow = float(np.nanmean(fr.R[east & (np.abs(PH - 0.5 * np.pi) < 0.5)]))
    assert fast > slow + 5.0, (fast, slow)
    return {'per-ray error (R_sun, km/s/10)': float(f'{worst:.1e}'), 'front in fast/slow stream': (round(fast, 1), round(slow, 1))}


if __name__ == '__main__':
    tests = [('ray front sampled from an ellipsoid: positions, normals, apex', test_rayfront_matches_the_ellipsoid),
             ('uniform Parker wind: every ray = classic DBM, sphere stays a sphere', test_uniform_wind_is_classic_dbm),
             ('two-stream wind with density-scaled drag: each ray exact, front deforms',
              test_two_stream_wind_deforms_the_front_ray_by_ray)]
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
