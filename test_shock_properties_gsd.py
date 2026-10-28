"""
Tests for shock_properties_gsd.py, each against an exact or independent answer. Run:  python test_shock_properties_gsd.py
"""
import datetime as dt

import numpy as np
from scipy.integrate import solve_ivp

import background as bk
import shock_properties_dbm as sd
import shock_properties_gsd as sg
import test_shock_properties_dbm as T

G = 5.0 / 3.0
RS = bk.RSUN_KM


def whitham_dM_dR(R, M, rho_of, c_of):
    """Continuous CCW relation along a radial path in a static medium: (p_M + rho c u_M) dM/dR =
    -[dp/dR|_M + rho c du/dR|_M + rho c^2 u/(u+c) (2/R)] (spherical tube, A ~ R^2)."""
    h = 1e-4 * R
    p, rho, u, c = sg.post_shock(M, rho_of(R), c_of(R))
    pp, _, up, _ = sg.post_shock(M, rho_of(R + h), c_of(R + h))
    pm, _, um, _ = sg.post_shock(M, rho_of(R - h), c_of(R - h))
    p_M = rho_of(R) * c_of(R) ** 2 / G * 4 * G * M / (G + 1)
    u_M = 2 * c_of(R) / (G + 1) * (1 + 1 / M ** 2)
    rhs = -((pp - pm) / (2 * h) + rho * c * (up - um) / (2 * h) + rho * c ** 2 * u / (u + c) * 2.0 / R)
    return rhs / (p_M + rho * c * u_M)


def test_ccw_uniform_medium_and_strong_shock_exponent():
    rho_of, c_of = (lambda R: 1e-20 + 0 * R), (lambda R: 5e4 + 0 * R)           # uniform, static
    out = {}
    for M0 in (3.0, 100.0):
        R, M = 10.0, M0
        Rs, Ms = [R], [M]
        for _ in range(4000):                                                  # the discrete update, small steps
            R_new = R * 1.002
            M = float(sg.ccw_update(np.array([M]), rho_of(R), c_of(R), rho_of(R_new), c_of(R_new),
                                    np.log((R_new / R) ** 2))[0])
            R = R_new
            Rs.append(R)
            Ms.append(M)
        sol = solve_ivp(lambda r, m: whitham_dM_dR(r, m[0], rho_of, c_of), (10.0, R), [M0], t_eval=Rs, rtol=1e-10)
        err = np.max(np.abs(np.array(Ms) - sol.y[0]) / sol.y[0])
        assert err < 2e-3, (M0, err)
        out[f'M0={M0:g}'] = f'discrete vs continuous {err:.1e}'
        if M0 == 100.0:
            slope = np.polyfit(np.log(np.array(Rs[:200]) ** 2), np.log(Ms[:200]), 1)[0]      # M ~ A^-1/lambda_inf
            lam = 1 + 2 / G + np.sqrt(2 * G / (G - 1))
            assert abs(slope + 1 / lam) < 0.01, (slope, -1 / lam)
            out['strong-shock exponent'] = f'{slope:.4f} (Whitham {-1 / lam:.4f})'
    return out


class RadialField(bk.ParkerBackground):
    """Uniform radial wind with a purely radial field: c_f = max(V_A, c_s) varies with r along every ray."""
    def _field(self, r, theta, lon, offset):
        br = self.br_1au * (bk.AU_RSUN / r) ** 2
        e_r, _, _ = bk._spherical_basis(theta, lon)
        return br[:, None] * e_r


def test_propagation_matches_1d_ccw_with_varying_fast_speed():
    V, w = 1500.0, 400.0
    bg = RadialField(v_sw=w, n_1au=5.0, br_1au=4.0, T=1e5)
    series = T._sphere_series(V)
    sg.DRIVER_HALF_ANGLE, sg.SMOOTH_DEG_PER_DAY, sd.SMOOTH_DEG_PER_DAY, sg.DRAG_ON_SHOCK = 0.0, 0.0, 0.0, False
    front = sd.initial_front(series, 31, 61)
    r = np.arange(21.5, 260.0 + 1e-9, 1.0)
    table = sg.FieldTable(bg, r, front.theta, front.phi, verbose=False)
    heights = [51.5, 101.5, 151.5]
    fronts, _ = sg.propagate(front, table, sd.l0_function(bg, T.T0), T.T0, 0.0, 1.0, 2.0, heights)
    # the same shock along one radial path, continuous: dR/dt = w + M c_f(R), dM/dR from Whitham in the wind frame
    def amb(R):
        s = bg.sample(np.array([[R, 0.0, 0.0]]), T.T0.isoformat())
        return float(s['rho'][0]), 1e3 * float(np.maximum(s['VA'][0], s['cs'][0]))            # SI
    rho_of = lambda R: amb(R)[0]
    c_of = lambda R: amb(R)[1]
    rho0, c0 = amb(21.5)
    M0 = (V - w) * 1e3 / c0
    sol = solve_ivp(lambda R, y: [whitham_dM_dR(R, y[0], rho_of, c_of)], (21.5, 160.0), [M0], dense_output=True, rtol=1e-8)
    worst = {'sphere': 0.0, 'M': 0.0, 'time': 0.0}
    t_of_R = lambda Rend: solve_ivp(lambda R, y: [RS / (w + sol.sol(R)[0] * c_of(R) / 1e3)], (21.5, Rend), [0.0], rtol=1e-8).y[0][-1]
    for fr, h in zip(fronts, heights):
        worst['sphere'] = max(worst['sphere'], float(np.nanmax(np.abs(fr.R - h))))
        worst['M'] = max(worst['M'], abs(fr.meta['M_apex'] - sol.sol(h)[0]) / sol.sol(h)[0])
        t = (dt.datetime.fromisoformat(fr.time) - T.T0).total_seconds()
        worst['time'] = max(worst['time'], abs(t - t_of_R(h)) / t_of_R(h))
    assert worst['sphere'] < 1e-2 and worst['M'] < 5e-3 and worst['time'] < 5e-3, worst
    return {k: float(f'{v:.1e}') for k, v in worst.items()}


def test_driver_floor_inside_the_cone():
    V, w = 1500.0, 400.0

    class HotUniform(RadialField):                            # constant density and sound speed, radial wind:
        def _sample(self, r, theta, lon, offset):             # Whitham's shock decays here (M ~ R^-0.45)
            e_r, _, _ = bk._spherical_basis(theta, lon)
            n = np.full_like(r, self.n_1au)
            Tm = np.full_like(r, self.T)
            return bk._derived(n, Tm, 2 * n * 1e6 * bk.KB * Tm, self._field(r, theta, lon, offset), self.v_sw * e_r)
    bg = HotUniform(v_sw=w, n_1au=5.0, br_1au=0.01, T=1e7)
    sd.GAMMA_DENSITY_SCALING = False                          # a constant-gamma driver, as in the analytic DBM
    series = T._sphere_series(V)
    sg.DRIVER_HALF_ANGLE, sg.SMOOTH_DEG_PER_DAY, sd.SMOOTH_DEG_PER_DAY, sg.DRAG_ON_SHOCK = 40.0, 0.0, 0.0, False
    front = sd.initial_front(series, 31, 61)
    r = np.arange(21.5, 260.0 + 1e-9, 1.0)
    table = sg.FieldTable(bg, r, front.theta, front.phi, verbose=False)
    gamma0 = 0.05e-7                                            # a weakly decelerating driver: faster than the free shock
    fronts, _ = sg.propagate(front, table, sd.l0_function(bg, T.T0), T.T0, gamma0, 1.0, 2.0, [101.5])
    fr = fronts[0]
    TH, PH, d = sd._dir(fr.theta, fr.phi)
    ang = np.degrees(np.arccos(np.clip(d @ fr.frame[0], -1, 1)))
    inside, outside = (ang < 35) & (TH > 0.2) & (TH < np.pi - 0.2), (ang > 60) & (TH > 0.2) & (TH < np.pi - 0.2)
    t = (dt.datetime.fromisoformat(fr.time) - T.T0).total_seconds()
    import derive_para_ip as dip
    v_drv = float(dip.dbm(t, V, w, gamma0)[1])
    assert np.nanmin(fr.V[inside]) > v_drv - 2.0 and np.nanmax(fr.V[outside]) < v_drv - 50.0, \
        (np.nanmin(fr.V[inside]), np.nanmax(fr.V[outside]), v_drv)
    return {'driver speed': round(v_drv), 'inside the cone': round(float(np.nanmin(fr.V[inside]))),
            'free flank': round(float(np.nanmax(fr.V[outside])))}


def test_capped_front_does_not_erode():
    """A front cut at 90 deg from the apex (as on real events) keeps all its rays through the GSD propagation; with
    plain bilinear lookups (0 x NaN = NaN at the edge) it lost one ring of rays per step."""
    import shock_properties_sse as _sp
    import shock_properties_dbm as sd
    from ray_front import RayFront
    import datetime as dt
    _sp.FRONT_HALF_WIDTH_DEG = 90.0
    hs = [19.5, 20.5, 21.5]
    times = [T.T0 + dt.timedelta(seconds=(h - 21.5) * RS / 1500.0) for h in hs]
    series = T._series([T.DeformableShock.from_pythea(h, 0.7, 0.2, 1.1, lat=0.0, lon=0.0, tilt=0.0) for h in hs],
                       times, 1500.0)
    bg = bk.ParkerBackground(v_sw=400.0, n_1au=5.0, br_1au=3.5, T=1e5)
    theta, phi = np.linspace(0, np.pi, 37), np.linspace(0, 2 * np.pi, 73, endpoint=False)
    table = sg.FieldTable(bg, np.arange(15.0, 300.0, 2.0), theta, phi, verbose=False)
    front0 = sd.initial_front(series, 37, 73)
    L0 = sd.l0_function(bg, times[-1])

    fronts, _ = sg.propagate(front0, table, L0, times[-1], 1e-7, 5.0, 30.0, [30.0, 40.0, 60.0, 90.0, 130.0])
    now = [int(np.isfinite(f.R).sum()) for f in fronts]
    last = fronts[-1]
    TH, PH = np.meshgrid(last.theta, last.phi, indexing='ij')
    valid = np.isfinite(last.R)
    dirs = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1)[valid]
    own = last.R_at(dirs)
    nrm = last.normals_at(own[:, None] * dirs)
    plain = RayFront._lookup                                             # the old lookup, to show the bug
    from scipy.interpolate import RegularGridInterpolator

    def old_lookup(self, name, d):
        d = np.asarray(d, float)
        th = np.arccos(np.clip(d[..., 2], -1, 1))
        ph = np.mod(np.arctan2(d[..., 1], d[..., 0]), 2 * np.pi)
        P = np.concatenate([self.phi, [self.phi[0] + 2 * np.pi]])
        A = np.concatenate([getattr(self, name), getattr(self, name)[:, :1]], axis=1)
        f = RegularGridInterpolator((self.theta, P), A, bounds_error=False, fill_value=np.nan)
        out = f(np.stack([np.clip(th, self.theta[0], self.theta[-1]), ph], -1))
        return out.reshape(d.shape[:-1]) if d.ndim > 1 else float(out.reshape(-1)[0])
    RayFront._lookup = old_lookup
    try:
        last._interp.clear()
        old_own = last.R_at(dirs)
    finally:
        RayFront._lookup = plain
        last._interp.clear()
    assert len(set(now)) == 1 and now[0] > 0, now                        # no erosion: the same rays all the way
    assert np.all(np.isfinite(own)) and np.allclose(own, last.R[valid]) and np.all(np.isfinite(nrm))
    lost = int(np.sum(~np.isfinite(old_own)))
    assert lost > 0                                                      # the old lookup lost the rim rays
    return (f'rays carrying the front per step {now}; every ray returns its own position and a normal '
            f'(the old lookup lost {lost} rim rays of {valid.sum()})')


if __name__ == '__main__':
    tests = [('CCW update = continuous Whitham rule; strong-shock exponent', test_ccw_uniform_medium_and_strong_shock_exponent),
             ('full propagation = 1-D CCW with c_f varying along the path (sphere, M, times)',
              test_propagation_matches_1d_ccw_with_varying_fast_speed),
             ('driver floor inside the cone, free decay outside', test_driver_floor_inside_the_cone),
             ('a capped front does not erode (NaN-aware lookups)', test_capped_front_does_not_erode)]
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
