"""
Tests for shock_properties_sse.py, each against an exact answer. Run:  python test_shock_properties.py
"""
import datetime as dt

import numpy as np

import background as bk
import shock_properties_sse as sp
from deformable_shock import DeformableShock

G = 5.0 / 3.0


def test_perpendicular_cold_limit_equals_eq_B4():
    X = np.array([1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 3.9])
    MA = np.sqrt(X * (5 + X) / (2 * (4 - X)))                      # exact for theta = 90, beta -> 0, gamma 5/3
    got = sp.rh_compression(MA, 90.0, 1e-8)
    assert np.max(np.abs(got - X)) < 1e-4, (got, X)
    return float(np.max(np.abs(got - X)))


def test_parallel_limit_is_gas_dynamic():
    beta, Ms = 2.0, 3.0                                            # c_s > V_A: the fast shock is the gas-dynamic one
    MA = Ms * np.sqrt(G * beta / 2)                                # c_s^2 = (gamma beta / 2) V_A^2
    got = float(sp.rh_compression(MA, 0.0, beta))
    want = (G + 1) * Ms ** 2 / ((G - 1) * Ms ** 2 + 2)
    assert abs(got - want) < 1e-6, (got, want)
    return got, want


def test_oblique_jump_conditions_hold():
    """At the solver's X, every Rankine-Hugoniot condition holds when evaluated independently in SI units."""
    rng = np.random.default_rng(3)
    mu0, worst = 4e-7 * np.pi, 0.0
    for _ in range(300):
        th = np.radians(rng.uniform(1, 89))
        beta = 10 ** rng.uniform(-3, 0.5)
        MA = rng.uniform(1.2, 8.0)
        rho1, B = 1e-20, 1e-5
        VA = B / np.sqrt(mu0 * rho1)
        p1 = beta * B ** 2 / (2 * mu0)
        v1 = MA * VA
        cs = np.sqrt(G * p1 / rho1)
        cf = sp.fast_speed(VA, cs, np.degrees(th))
        X = float(sp.rh_compression(MA, np.degrees(th), beta))
        if v1 <= cf:
            assert not np.isfinite(X)
            continue
        Bn, Bt1 = B * np.cos(th), B * np.sin(th)
        rho2, v2n = X * rho1, v1 / X
        Bt2 = Bt1 * X * (v1 ** 2 - Bn ** 2 / (mu0 * rho1)) / (v1 ** 2 - X * Bn ** 2 / (mu0 * rho1))
        v2t = Bn * (Bt2 - Bt1) / (mu0 * rho1 * v1)
        p2 = p1 + rho1 * v1 ** 2 - rho2 * v2n ** 2 + (Bt1 ** 2 - Bt2 ** 2) / (2 * mu0)
        h = lambda p, rho: G / (G - 1) * p / rho
        checks = [
            (rho1 * v1, rho2 * v2n),                                                       # mass
            (rho1 * v1 ** 2 + p1 + Bt1 ** 2 / (2 * mu0), rho2 * v2n ** 2 + p2 + Bt2 ** 2 / (2 * mu0)),
            (-Bn * Bt1 / mu0, rho2 * v2n * v2t - Bn * Bt2 / mu0),                         # tangential momentum
            (v1 * Bt1, v2n * Bt2 - v2t * Bn),                                              # tangential E
            (rho1 * v1 * (0.5 * v1 ** 2 + h(p1, rho1)) + v1 * Bt1 ** 2 / mu0,              # energy
             rho2 * v2n * (0.5 * (v2n ** 2 + v2t ** 2) + h(p2, rho2)) + (v2n * Bt2 ** 2 - v2t * Bt2 * Bn) / mu0)]
        for a, b in checks:
            worst = max(worst, abs(a - b) / max(abs(a), abs(b)))
        assert p2 > p1 and 1 < X <= 4 and abs(Bt2) >= abs(Bt1) * (1 - 1e-9)
    assert worst < 1e-8, worst
    return worst


def test_b3_inversion():
    X = np.array([1.3, 2.0, 3.2])
    for th in (0.0, 35.0, 90.0):
        s2, c2 = np.sin(np.radians(th)) ** 2, np.cos(np.radians(th)) ** 2
        MA = np.sqrt(s2 * X * (5 + X) / (2 * (4 - X)) + c2 * X)
        assert np.allclose(sp.b3_compression(MA, th), X, atol=1e-9)


def test_normals_of_an_ellipsoid():
    m = DeformableShock.from_pythea(12.0, 0.7, 0.4, 1.3, lat=15, lon=325, tilt=30)
    th, ph = DeformableShock.grid(41, 81)
    x = m.position(th, ph)
    e1, e2, e3 = m.frame
    a, b, c, _ = m.axes
    g = x - m.center
    grad = (g @ e1)[..., None] * e1 / a ** 2 + (g @ e2)[..., None] * e2 / b ** 2 + (g @ e3)[..., None] * e3 / c ** 2
    grad /= np.linalg.norm(grad, axis=-1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(np.sum(grad * m.normals(th, ph), -1), -1, 1)))
    assert ang.max() < 1e-3, ang.max()
    return float(ang.max())


def _series(models, times):
    return [{'name': f's{i}', 'model': m, 'time_iso': t.isoformat(), 'time': t, 'speed_kms': np.nan, 'in_range': True}
            for i, (m, t) in enumerate(zip(models, times))]


def test_normal_speed_of_self_similar_expansion():
    """PyThea expansion (kappa, eps, alpha fixed) at constant apex speed: V = v (x - P0)/(h - 1) exactly."""
    v_kms, t0 = 1500.0, dt.datetime(2012, 3, 7, 0, 30)
    hs = np.arange(5.0, 9.01, 1.0)
    times = [t0 + dt.timedelta(seconds=(h - 5.0) * bk.RSUN_KM / v_kms) for h in hs]
    feat = [(60.0, 90.0, 20.0, 0.25)]
    models = [DeformableShock.from_pythea(h, 0.7, 0.4, 1.3, lat=15, lon=325, tilt=30, dents=feat) for h in hs]
    series = _series(models, times)
    k = 2
    th, ph = DeformableShock.grid(31, 61)
    x, n = models[k].position(th, ph), models[k].normals(th, ph)
    sp.VN_METHOD = 'normal_distance'
    vn = sp.normal_speed(series, k, x, n)
    exact = np.sum(v_kms * (x - models[k].frame[0]) / (models[k].height - 1.0) * n, -1)
    rel = np.nanmax(np.abs(vn - exact)) / v_kms
    assert rel < 2e-3 and np.isfinite(vn).mean() > 0.99, rel
    return float(rel)


def test_sphere_in_parker_wind():
    """Sun-centred spheres expanding at V through a Parker wind: V_n, theta_Bn, M_A and M_f are analytic."""
    pk = bk.ParkerBackground(v_sw=400.0, n_1au=5.0, br_1au=3.5, T=1e5)
    V, t0 = 1200.0, dt.datetime(2012, 3, 7, 0, 30)
    hs = np.arange(4.0, 8.01, 1.0)
    times = [t0 + dt.timedelta(seconds=(h - 4.0) * bk.RSUN_KM / V) for h in hs]
    series = _series([DeformableShock(h, h, h, h, 10.0, 330.0, 0.0) for h in hs], times)
    sp.GRID = (21, 41)
    sp.FRONT_HALF_WIDTH_DEG = None                                 # an idealised full shell
    res = sp.properties(series, 2, pk)
    r = np.linalg.norm(res['xyz'], axis=-1)
    th = np.arccos(res['xyz'][..., 2] / r)
    up = pk.sample(res['xyz'].reshape(-1, 3), series[2]['time_iso'])
    VA, cs = up['VA'].reshape(r.shape), up['cs'].reshape(r.shape)
    tbn = pk.theta_bn_radial(r, th)
    MA = (V - 400.0) / VA
    Mf = (V - 400.0) / sp.fast_speed(VA, cs, tbn)
    errs = {'Vn': np.nanmax(np.abs(res['Vn'] - V)) / V, 'theta_bn': np.nanmax(np.abs(res['theta_bn'] - tbn)),
            'MA': np.nanmax(np.abs(res['MA'] - MA) / MA), 'Mf': np.nanmax(np.abs(res['Mf'] - Mf) / Mf)}
    assert errs['Vn'] < 1e-6 and errs['theta_bn'] < 1e-3 and errs['MA'] < 1e-6 and errs['Mf'] < 1e-6, errs
    return {k: float(f'{v:.2e}') for k, v in errs.items()}


def test_physical_front_is_a_cap():
    """Whatever the fitted ellipsoid does around the Sun, the front is the part within 90 deg of the apex and
    outside the Sun; a fit narrower than that keeps its own extent."""
    sp.FRONT_HALF_WIDTH_DEG = 90.0
    th, ph = DeformableShock.grid(31, 61)
    out = {}
    for name, m in (('halo', DeformableShock.from_pythea(21.5, 0.66, -0.54, 1.41, lat=13, lon=330, tilt=46)),
                    ('rear just inside', DeformableShock.from_pythea(21.5, 0.6, 0.0, 1.2, lat=13, lon=330, tilt=46)),
                    ('narrow', DeformableShock.from_pythea(10.0, 0.25, 0.3, 1.2, lat=13, lon=330, tilt=46))):
        x = m.position(th, ph)
        r = np.linalg.norm(x, axis=-1)
        ang = np.degrees(np.arccos(np.clip(np.sum(x * m.frame[0], -1) / r, -1, 1)))
        on = sp.on_front(m, x)
        assert np.array_equal(on, (ang <= 90.0 + 1e-9) & (r >= sp.R_MIN)), name
        out[name] = f'{on.mean() * 100:.0f}% of the surface'
    assert out['narrow'] == '100% of the surface'                 # entirely within the cap: nothing removed
    return out


def test_old_ray_fronts_get_the_cut():
    """Ray fronts written before the cut existed (no half width stored) are cut when loaded."""
    import contextlib, io, json, tempfile
    from pathlib import Path
    import event_config as cfg
    from ray_front import RayFront
    cfg.FINAL_DIR = Path(tempfile.mkdtemp())
    th, ph = np.linspace(0, np.pi, 31), np.linspace(0, 2 * np.pi, 61, endpoint=False)
    R = np.full((31, 61), 50.0)
    np.savez_compressed(cfg.FINAL_DIR / 'shock_para_dbm0500_test.npz', theta=th, phi=ph, R=R, V=R * 20, apex=[1.0, 0, 0],
                        time='2012-03-07T05:00:00', meta=json.dumps({}))
    sp.FRONT_HALF_WIDTH_DEG = 90.0
    with contextlib.redirect_stdout(io.StringIO()) as out:
        m = sp.load_series('test', 'dbm')[0]['model']
    x = m.position(*RayFront.grid(31, 61))
    ang = np.degrees(np.arccos(np.clip(x[..., 0] / np.linalg.norm(x, axis=-1), -1, 1)))
    assert m.half_width_deg == 90.0 and 'Rerun shock_properties_dbm.py' in out.getvalue()
    assert np.array_equal(sp.on_front(m, x), ang <= 90.0 + 1e-9)
    return 'old full fronts cut to 90 deg on loading, with a note to rerun'


if __name__ == '__main__':
    tests = [('perpendicular, beta -> 0: Rankine-Hugoniot = Eq. (B4)', test_perpendicular_cold_limit_equals_eq_B4),
             ('parallel: gas-dynamic compression', test_parallel_limit_is_gas_dynamic),
             ('oblique: all jump conditions hold at the solution (300 random shocks)', test_oblique_jump_conditions_hold),
             ('Eq. (B3) inversion', test_b3_inversion),
             ('normals of an ellipsoid vs its gradient [deg]', test_normals_of_an_ellipsoid),
             ('V_n of a self-similar front with a dent (relative error)', test_normal_speed_of_self_similar_expansion),
             ('expanding spheres in a Parker wind: V_n, theta_Bn, M_A, M_f', test_sphere_in_parker_wind),
             ('physical front: the part within 90 deg of the apex, outside the Sun', test_physical_front_is_a_cap),
             ('ray fronts from before the cut are cut on loading', test_old_ray_fronts_get_the_cut)]
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
