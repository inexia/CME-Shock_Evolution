"""
shock_properties_gsd.py -- option B: the interplanetary front as a fast-mode shock propagating through the
structured MAS wind (geometrical shock dynamics), driven near the apex, then the shock properties on the fronts.

Run:  python shock_properties_gsd.py        (settings below; folders and background from event_config.py)

On the same Sun-centred rays as shock_properties_dbm.py, every front element moves along its normal n at
    V_n = u . n + M c_f(theta_Bn)                       (laboratory frame)
with the local wind u, the local fast-mode speed c_f at the angle theta_Bn between the MAS field and n, and the
fast Mach number M of the shock in the wind frame; on a ray, dR/dt = V_n / (n . r_hat) (Huygens: the front
spreads sideways and refracts where c_f varies). M evolves along each ray tube with Whitham's characteristic
rule (the CCW relation of geometrical shock dynamics, Whitham 1974):
    dp + rho c du + (rho c^2 u / (u + c)) dA / A = 0
applied to the post-shock state (p, rho, u, c) of the Rankine-Hugoniot relations of an effective gas whose
sound speed is c_f (gamma = 5/3), with the ambient density and c_f taken at the front's current position (so
their variation along the path enters), and the tube area A = R^2 / (n . r_hat) per unit solid angle. In a
uniform medium this is Whitham's area rule (M ~ A^-0.225 for strong shocks); a denser, slower stream or a
narrowing tube strengthens the shock, a rarefied fast stream or a widening tube weakens it. The shock cannot
fall below M = 1 (it would then be a fast-mode wave).
Mass loading (DRAG_ON_SHOCK, default): Whitham's rule alone gives a shock that does not decelerate in an r^-2
wind (the density drop along the path compensates the spherical divergence), while interplanetary shocks do;
the missing physics is the momentum given to the swept-up wind, which the drag term of model A represents. So
the drag of shock_properties_dbm.py (gamma0 N/N_ref, exact DBM step) also acts on every element's normal
speed, and the CCW rule adds the geometric and medium effects (tube area, refraction, local c_f) on top:
the model reduces to A where those vanish. gamma0 is then fitted on the whole front as in A.
Alternative (DRAG_ON_SHOCK = False): the pure Whitham shock, driven on rays within DRIVER_HALF_ANGLE of the apex
by model A (the CME pushes the nose) and free elsewhere.
The frame of the CCW relation is the local wind's (the wind's own variation along the path is neglected there;
it enters V_n).
gamma0 of the driver is fitted to an observed arrival exactly as in shock_properties_dbm.py.

Writes  fitted results/<id>/final/shock_para_gsd<hhhh>_<method>.npz, shock_para_gsd_<method>.csv,
dbm_fit_gsd_<method>.json, figure/derive_para_gsd_<method>.png, and through shock_properties_sse.py the
shock_props_gsd/ files and figure/shock_evo_gsd/.
"""
import csv
import datetime as dt
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import RegularGridInterpolator

import event_config as cfg
import background as bk
import shock_properties_sse as sp
import shock_properties_dbm as sd
import drag_fit as df
from ray_front import RayFront

# ================================ SETTINGS ================================
METHODS = ['constant', 'varied']
RAY_GRID = (91, 181)
FIT_GRID = (46, 91)
DT_MIN = 5.0
FIT_DT_MIN = 10.0
H_STEP, H_END = 5.0, 220.0
EXTEND_TO_SC = True
GAMMA = 5.0 / 3.0
DRAG_ON_SHOCK = True              # the drag of model A acts on every element's normal speed (momentum given to
                                  # the swept-up wind, which Whitham's rule lacks); gamma0 then applies to the
                                  # whole front and no driver cone is needed
DRIVER_HALF_ANGLE = 0.0           # without DRAG_ON_SHOCK: rays within this angle of the apex are driven (DBM
                                  # floor of model A) [deg]; 0: none
REFIT_DRAG = False                # True: fit the shared drag now (drag_fit.py); False: use drag_fit_<method>.json if
                                  # it exists (written by shock_properties_sse.py / drag_fit.py), else fit it
REFIT_GAMMA = True                # refit gamma0 for THIS version's kinematics (its geometric terms change the
                                  # arrival), starting from the shared gamma0, so the arrival stays right
SMOOTH_DEG_PER_DAY = 2.0
M_MIN = 1.0
COS_MIN = 0.3                     # floor of n . r_hat: a ray's crossing speed is at most V_n / COS_MIN
PROPERTIES = True
# ==========================================================================
RSUN_KM = bk.RSUN_KM
AU_RSUN = bk.AU_RSUN
DAY = 86400.0
MP, MU0 = bk.MP, bk.MU0


# ----------------------------------------------------------------------------- the background along the rays
class FieldTable(sd.WindTable):
    """WindTable plus the field (spherical components, nT), density (cm^-3) and pressure (Pa)."""

    def __init__(self, bg, r, theta, phi, verbose=True):
        super().__init__(bg, r, theta, phi, verbose)
        if self.multi is not None:                              # several runs: the sub-tables hold the fields
            return
        cache = None
        if hasattr(bg, 'folder'):
            cache = bg.folder / 'cache' / f'field_table_{len(r)}x{len(theta)}x{len(phi)}_{r[0]:g}_{r[-1]:g}.npz'
        if cache is not None and cache.exists():
            z = np.load(cache)
            fields = {k: z[k] for k in ('Br', 'Bt', 'Bp', 'n', 'p')}
        else:
            t0 = cfg.iso_time()
            offset = bg._offset(t0)
            Rg, TH, PH = np.meshgrid(self.r, self.theta, self.phi, indexing='ij')
            lon = PH - offset
            xyz = np.stack([Rg * np.sin(TH) * np.cos(lon), Rg * np.sin(TH) * np.sin(lon), Rg * np.cos(TH)], -1)
            pts = xyz.reshape(-1, 3)
            fields = {k: np.empty(len(pts), np.float32) for k in ('Br', 'Bt', 'Bp', 'n', 'p')}
            step = 400000
            if verbose:
                print(f'  tabulating the field on {len(pts):,} points ...')
            for i in range(0, len(pts), step):
                s = bg.sample(pts[i:i + step], t0)
                _, r_, th_, lon_ = bk._to_sph(pts[i:i + step])
                e_r, e_t, e_p = bk._spherical_basis(th_, lon_)
                fields['Br'][i:i + step] = np.sum(s['B'] * e_r, 1)
                fields['Bt'][i:i + step] = np.sum(s['B'] * e_t, 1)
                fields['Bp'][i:i + step] = np.sum(s['B'] * e_p, 1)
                fields['n'][i:i + step] = s['n']
                fields['p'][i:i + step] = s['p']
            fields = {k: self._fill_radial(v.reshape(Rg.shape)) for k, v in fields.items()}
            if cache is not None:
                np.savez_compressed(cache, **fields)
        ph = np.concatenate([self.phi, [self.phi[0] + 2 * np.pi]])
        pad = lambda a: np.concatenate([a, a[:, :, :1]], axis=2)
        self._f = {k: RegularGridInterpolator((self.r, self.theta, ph), pad(v), bounds_error=False, fill_value=None)
                   for k, v in fields.items()}

    def state(self, r, theta, lon_hgs, L0, when=None):
        """u_r, B (HGS Cartesian, nT), rho (kg/m^3), p (Pa) at the points; `when` weighs several runs."""
        if self.multi is not None:
            w = self._weights(lon_hgs, L0, when)
            got = [s.state(r, theta, lon_hgs, L0) for s in self.subs]
            return tuple(self.multi._blend([g[i] for g in got], w) for i in range(4))
        pts = np.stack([np.clip(r, self.r[0], self.r[-1]), theta, np.mod(lon_hgs + L0, 2 * np.pi)], -1)
        ur = self._ur(pts)
        e_r, e_t, e_p = bk._spherical_basis(theta, lon_hgs)
        B = self._f['Br'](pts)[:, None] * e_r + self._f['Bt'](pts)[:, None] * e_t + self._f['Bp'](pts)[:, None] * e_p
        return ur, B, self._f['n'](pts) * 1e6 * MP, self._f['p'](pts)


# ----------------------------------------------------------------------------- shock relations of the effective gas
def post_shock(M, rho0, c0, g=GAMMA):
    """Post-shock p, rho, u (flow speed relative to the ambient), c of a hydrodynamic shock of Mach M in a
    medium with density rho0 and sound speed c0 (p0 = rho0 c0^2 / gamma); SI."""
    p0 = rho0 * c0 ** 2 / g
    p = p0 * (2 * g * M ** 2 - (g - 1)) / (g + 1)
    rho = rho0 * (g + 1) * M ** 2 / ((g - 1) * M ** 2 + 2)
    u = 2 * c0 / (g + 1) * (M - 1.0 / M)
    c = np.sqrt(g * p / rho)
    return p, rho, u, c


def ccw_update(M, rho_old, c_old, rho_new, c_new, lnA_ratio, g=GAMMA):
    """
    Whitham's characteristic rule, discretised over one step: the post-shock state changes by the ambient
    change (rho0, c0 at fixed M) and by the tube area change ln(A_new/A_old); M adjusts so that
        dp + rho c du + (rho c^2 u / (u + c)) dA/A = 0   along the C+ characteristic behind the shock.
    """
    p0, rho_, u0, c_ = post_shock(M, rho_old, c_old, g)
    p1, _, u1, _ = post_shock(M, rho_new, c_new, g)
    p_M = rho_new * c_new ** 2 / g * 4 * g * M / (g + 1)
    u_M = 2 * c_new / (g + 1) * (1 + 1.0 / M ** 2)
    rhs = -((p1 - p0) + rho_ * c_ * (u1 - u0) + rho_ * c_ ** 2 * u0 / (u0 + c_) * lnA_ratio)
    return M + rhs / (p_M + rho_ * c_ * u_M)


def fast_speed(VA, cs, cos_bn):
    s = VA ** 2 + cs ** 2
    return np.sqrt(0.5 * (s + np.sqrt(np.maximum(s ** 2 - 4 * VA ** 2 * cs ** 2 * cos_bn ** 2, 0.0))))


# ----------------------------------------------------------------------------- propagation
def propagate(front0, table, L0, t0, gamma0, n_ref, dt_min, heights, sc=None, t_max_days=15.0):
    """
    Integrate the rays from front0 (which carries the ray speeds V of the start). Returns (fronts at the apex
    heights `heights`, arrival dict or None), as shock_properties_dbm.propagate; the fronts' V is dR/dt.
    """
    t0_s = bk.posix(t0)
    theta, phi = front0.theta, front0.phi
    TH, PH, d = sd._dir(theta, phi)
    apex = front0.frame[0]
    R = front0.R.copy()
    V_dbm = front0.V.copy()                                   # the driver (model A) on the same rays
    ok = np.isfinite(R) & np.isfinite(V_dbm)
    R[~ok], V_dbm[~ok] = np.nan, np.nan
    driven = ok & (np.degrees(np.arccos(np.clip(d @ apex, -1, 1))) <= DRIVER_HALF_ANGLE)
    dt_s = dt_min * 60.0
    sig = (SMOOTH_DEG_PER_DAY * np.sqrt(dt_s / DAY)) / np.degrees(theta[1] - theta[0]) if SMOOTH_DEG_PER_DAY else 0.0

    def ambient(R_, t):
        """wind-frame fast speed, u.n, rho, c_f and the normal at the front points of the valid rays."""
        fr = RayFront(theta, phi, R_, apex=apex)
        x = R_[ok][:, None] * d[ok]
        n = fr.normals_at(x)
        bad = ~np.all(np.isfinite(n), axis=1)                 # safety net: a valid ray never drops out of the
        n[bad] = d[ok][bad]                                   # front for want of a normal (radial instead)
        ur, B, rho, p = table.state(R_[ok], TH[ok], PH[ok], L0(t), when=t0_s + t)
        Bm = np.linalg.norm(B, axis=1)
        cos_bn = np.abs(np.sum(B * n, 1)) / np.maximum(Bm, 1e-30)
        VA = Bm * 1e-9 / np.sqrt(MU0 * rho) / 1e3
        cs = np.sqrt(GAMMA * p / rho) / 1e3
        cf = fast_speed(VA, cs, cos_bn)                       # km/s
        cosr = np.maximum(np.sum(n * d[ok], 1), COS_MIN)      # n . r_hat (floor: grazing crossings)
        return ur * cosr, rho, cf, cosr                       # u.n (radial wind), rho, c_f, n.r_hat

    # initial Mach numbers from the start speeds: M = (V_n - u.n) / c_f
    un, rho, cf, cosr = ambient(R, 0.0)
    M = np.full(R.shape, np.nan)
    M[ok] = np.maximum((V_dbm[ok] * cosr - un) / cf, M_MIN)
    A_old = (R[ok] ** 2) / cosr
    heights = list(heights)
    out, arrival = [], None
    t = 0.0
    r_apex = sd._bilinear(theta, phi, R, apex)
    r_sc = sd._bilinear(theta, phi, R, sc[0]) if sc else np.nan
    if sc and not np.isfinite(r_sc):
        sc = None
    w_prev = None
    while (heights or (sc and arrival is None)) and t < t_max_days * DAY:
        # --- the driver: model A on its rays, one DBM step
        w, N = table.lookup(R[ok], TH[ok], PH[ok], L0(t), when=t0_s + t)
        g_drag = gamma0 * (N / n_ref if sd.GAMMA_DENSITY_SCALING else 1.0)
        dv = V_dbm[ok] - w
        V_dbm[ok] = w + dv / (1.0 + g_drag * np.abs(dv) * dt_s)             # exact DBM step
        # --- the shock: V_n = u.n + M c_f, Huygens along the rays
        V_n = un + M[ok] * cf
        if DRAG_ON_SHOCK:                                      # momentum loss to the swept-up wind (exact DBM step)
            dv = V_n - un
            V_n = un + dv / (1.0 + g_drag * np.abs(dv) * dt_s)
        V_ray = V_n / cosr
        if DRIVER_HALF_ANGLE > 0:
            V_ray = np.where(driven[ok] & (V_ray < V_dbm[ok]), V_dbm[ok], V_ray)
        R_new = R.copy()
        V_ray = np.where(np.isfinite(V_ray), V_ray, V_dbm[ok])          # ... nor of a speed (plain DBM then)
        R_new[ok] = R[ok] + dt_s * V_ray / RSUN_KM
        R_new = sd._smooth(R_new, sig, theta)
        sd.pole_mean(R_new)                                    # the poles are one direction
        V_out = np.full(R.shape, np.nan)
        V_out[ok] = (R_new[ok] - R[ok]) / dt_s * RSUN_KM
        t_new = t + dt_s
        # --- the shock strength: CCW with the new ambient and tube area
        un_new, rho_new, cf_new, cosr_new = ambient(R_new, t_new)
        A_new = (R_new[ok] ** 2) / cosr_new
        M_new = ccw_update(M[ok], rho, cf * 1e3, rho_new, cf_new * 1e3, np.log(A_new / A_old))
        if DRAG_ON_SHOCK:                                      # the drag's loss of M over the step
            M_new = M_new - (M[ok] - (V_n - un) / cf)
        if DRIVER_HALF_ANGLE > 0:                              # driven rays: at least the driver's Mach number
            m_drv = (V_dbm[ok] * cosr_new - un_new) / cf_new
            M_new = np.where(driven[ok] & (m_drv > M_new), m_drv, M_new)
        M_new = np.clip(M_new, M_MIN, 100.0)
        M_grid = sd._smooth(_scatter(M_new, ok, R.shape), sig, theta)
        sd.pole_mean(M_grid)
        M_new = M_grid[ok]
        # --- outputs
        r_apex_new = sd._bilinear(theta, phi, R_new, apex)
        while heights and r_apex_new >= heights[0] - 1e-9:
            f = 0.0 if r_apex_new == r_apex else float(np.clip((heights[0] - r_apex) / (r_apex_new - r_apex), 0, 1))
            when = t0 + dt.timedelta(seconds=t + f * dt_s)
            fr = RayFront(theta, phi, R + f * (R_new - R), V_out, apex=apex, time=when.strftime('%Y-%m-%dT%H:%M:%S'),
                          meta={'M_mean': float(np.nanmean(M_new)), 'M_apex': float(sd._bilinear(theta, phi, _scatter(M_new, ok, R.shape), apex)),
                                'half_width_deg': front0.half_width_deg})
            fr.M = _scatter(M_new, ok, R.shape)
            out.append(fr)
            heights.pop(0)
        if sc and arrival is None:
            r_sc_new = sd._bilinear(theta, phi, R_new, sc[0])
            if np.isfinite(r_sc_new) and r_sc_new >= sc[1] and np.isfinite(r_sc):
                f = float(np.clip((sc[1] - r_sc) / (r_sc_new - r_sc), 0.0, 1.0)) if r_sc_new > r_sc else 0.0
                fr = RayFront(theta, phi, R + f * (R_new - R), V_out, apex=apex)
                x = sc[1] * sc[0]
                n = fr.normals_at(x[None])[0]
                v_ray = float(fr.V_at(sc[0]))
                arrival = {'t_s': t + f * dt_s, 'apex_height': sd._bilinear(theta, phi, fr.R, apex),
                           'v_ray': v_ray, 'v_normal': v_ray * float(np.dot(n, sc[0])), 'cos_normal_ray': float(np.dot(n, sc[0]))}
            r_sc = r_sc_new
        R, t, r_apex = R_new, t_new, r_apex_new
        M[ok], un, rho, cf, cosr, A_old = M_new, un_new, rho_new, cf_new, cosr_new, A_new
    return out, arrival


def _scatter(values, ok, shape):
    a = np.full(shape, np.nan)
    a[ok] = values
    return a


# ----------------------------------------------------------------------------- refit of gamma0, and run
def refit_gamma(front_fit, table, L0, t0, n_ref, sc, t_obs, v_obs, g_start):
    """gamma0 for this version's propagation, searched around the shared value: the arrival time and speed
    (df.ARRIVAL_SPEED_TYPE) weighted as in drag_fit.py."""
    from scipy.optimize import minimize_scalar
    cache = {}

    def arrive(g):
        if g not in cache:
            _, a = propagate(front_fit, table, L0, t0, g, n_ref, FIT_DT_MIN, [], sc, t_max_days=df.T_MAX_DAYS)
            cache[g] = (np.nan, np.nan) if a is None else (a['t_s'], a['v_normal'] if df.ARRIVAL_SPEED_TYPE == 'normal' else a['v_ray'])
        return cache[g]
    st, sv = df.SIGMA_TIME_H * 3600.0, df.SIGMA_SPEED
    cost = lambda g: ((arrive(g)[0] - t_obs) / st) ** 2 + ((arrive(g)[1] - v_obs) / sv) ** 2
    grid = np.geomspace(g_start / 6.0, g_start * 6.0, 19)
    J = np.array([cost(g) for g in grid])
    if not np.isfinite(J).any():
        return None
    i = int(np.nanargmin(J))
    lo, hi = np.log(grid[max(i - 1, 0)]), np.log(grid[min(i + 1, len(grid) - 1)])
    res = minimize_scalar(lambda lg: float(cost(np.exp(lg))), bounds=(lo, hi), method='bounded', options={'xatol': 1e-5})
    g_best = float(np.exp(res.x))
    t_mod, v_mod = arrive(g_best)
    return {'gamma_start': g_start, 'gamma_best': g_best, 'model_time_s': float(t_mod), 'model_speed_kms': float(v_mod),
            'time_residual_h': float((t_mod - t_obs) / 3600.0), 'speed_residual_kms': float(v_mod - v_obs)}


def run(method, bg, table):
    series = sp.load_series(method, 'corona')
    if len(series) < 2:
        print(f'[{method}] no coronal series: run derive_para_corona.py first')
        return None
    t0 = series[-1]['time']
    L0 = sd.l0_function(bg, t0)
    print(f'\n[{method}] start: {series[-1]["name"]}  h = {series[-1]["model"].height:.2f} R_sun at '
          f'{t0:%Y-%m-%d %H:%M:%S} UT, apex speed {series[-1]["speed_kms"]:.0f} km/s')
    sd.SMOOTH_DEG_PER_DAY = SMOOTH_DEG_PER_DAY
    front = sd.initial_front(series, *RAY_GRID)
    ok = np.isfinite(front.R)
    TH, PH, _ = sd._dir(front.theta, front.phi)
    _, N0 = table.lookup(front.R[ok], TH[ok], PH[ok], L0(0.0), when=bk.posix(t0))
    n_ref = float(np.nanmean(N0))
    # ---- the drag: the shared gamma0 (drag_fit.py), refitted for this version's kinematics
    drag = None if REFIT_DRAG else df.load(method)
    if drag is None:
        drag = df.fit(method, bg, series)
    gamma0, n_ref = drag['gamma0_per_km'], drag['n_ref_cm3']
    obs = df.observation()
    sc = (obs['d'], obs['dist_rsun']) if obs else None
    sc_info = {'direction_hgs_lon_lat_deg': [obs['lon'], obs['lat']], 'distance_rsun': obs['dist_rsun']} if obs else None
    have_obs, fit = obs is not None, None
    if have_obs and REFIT_GAMMA:
        t_obs = (obs['time'] - t0).total_seconds()
        front_fit = sd.initial_front(series, *FIT_GRID)
        print(f'  refitting gamma0 for this version on {FIT_GRID[0]} x {FIT_GRID[1]} rays, from the shared {gamma0:.3g} /km ...')
        fit = refit_gamma(front_fit, table, L0, t0, n_ref, sc, t_obs, obs['speed'], gamma0)
        if fit is None:
            print('  the front never reaches the spacecraft in the search range: the shared gamma0 kept')
        else:
            print(f'  gamma0 = {fit["gamma_best"]:.3g} /km ({fit["gamma_best"] / gamma0:.2f} x the shared value) -> arrival '
                  f'{t0 + dt.timedelta(seconds=fit["model_time_s"]):%Y-%m-%d %H:%M} UT ({fit["time_residual_h"]:+.2f} h), '
                  f'{fit["model_speed_kms"]:.0f} km/s ({fit["speed_residual_kms"]:+.0f})')
            gamma0 = fit['gamma_best']
    h0 = front.height
    heights = [float(h) for h in np.arange(h0, H_END + 1e-9, H_STEP)]
    if H_END - heights[-1] > 1e-6:
        heights.append(H_END)
    print(f'  propagating {RAY_GRID[0]} x {RAY_GRID[1]} rays: fast-mode shock (CCW) '
          + (f'with the drag on every element, gamma0 = {gamma0:.3g} /km ...' if DRAG_ON_SHOCK else
             f'driven within {DRIVER_HALF_ANGLE:g} deg of the apex with gamma0 = {gamma0:.3g} /km ...'))
    fronts, arrival = propagate(front, table, L0, t0, gamma0, n_ref, DT_MIN, heights[1:], sc)
    fronts = [front] + fronts
    if sc and arrival and EXTEND_TO_SC and arrival['apex_height'] > heights[-1]:
        extra = [float(h) for h in np.arange(heights[-1] + H_STEP, arrival['apex_height'] + H_STEP, H_STEP)]
        print(f'  extended to {extra[-1]:.1f} R_sun (the front passes the spacecraft when the apex is at '
              f'{arrival["apex_height"]:.1f} R_sun)')
        more, _ = propagate(front, table, L0, t0, gamma0, n_ref, DT_MIN, heights[1:] + extra, None)
        fronts = [front] + more
    rows = []
    for fr in fronts:
        if sc is not None and np.isfinite(fr.R_at(sc[0])):
            r_sc, v_sc = float(fr.R_at(sc[0])), float(fr.V_at(sc[0]))
            cosn = float(np.dot(fr.normals_at((r_sc * sc[0])[None])[0], sc[0]))
        else:
            r_sc = v_sc = cosn = np.nan
        derived = {'method': f'gsd_{method}', 'height_rsun': round(fr.height, 3), 'time': fr.time,
                   'speed_kms': round(fr.speed, 1), 'gamma0_per_km': gamma0, 'drag_on_shock': DRAG_ON_SHOCK,
                   'driver_half_angle_deg': DRIVER_HALF_ANGLE,
                   'smooth_deg_per_day': SMOOTH_DEG_PER_DAY, 'start': series[-1]['name'], 'in_fitted_range': False,
                   'M_apex': fr.meta.get('M_apex'), 'M_mean': fr.meta.get('M_mean'),
                   'sc_ray_distance_rsun': None if not np.isfinite(r_sc) else round(r_sc, 3),
                   'sc_ray_speed_radial_kms': None if not np.isfinite(v_sc) else round(v_sc, 1),
                   'sc_ray_speed_normal_kms': None if not np.isfinite(v_sc * cosn) else round(v_sc * cosn, 1)}
        fr.save(cfg.final_path(f'shock_para_gsd{int(round(fr.height * 10)):04d}_{method}.npz'),
                arrays={'M': getattr(fr, 'M', None)}, **derived)
        rows.append({'height_rsun': f'{fr.height:.2f}', 'time': fr.time, 'apex_speed_kms': f'{fr.speed:.1f}',
                     'M_apex': '' if fr.meta.get('M_apex') is None else f'{fr.meta["M_apex"]:.2f}',
                     'ray_speed_min_kms': f'{np.nanmin(fr.V):.1f}', 'ray_speed_max_kms': f'{np.nanmax(fr.V):.1f}',
                     'front_min_rsun': f'{np.nanmin(fr.R):.2f}', 'front_max_rsun': f'{np.nanmax(fr.R):.2f}',
                     'sc_ray_distance_rsun': f'{r_sc:.3f}' if np.isfinite(r_sc) else '',
                     'sc_ray_speed_normal_kms': f'{v_sc * cosn:.1f}' if np.isfinite(v_sc * cosn) else ''})
    table_path = cfg.final_path(f'shock_para_gsd_{method}.csv')
    with open(table_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    report = {'method': method, 'start': series[-1]['name'], 'start_time': t0.isoformat(),
              'driver_gamma0_used_per_km': gamma0, 'driver_half_angle_deg': DRIVER_HALF_ANGLE, 'rays': list(RAY_GRID),
              'observed': drag.get('observed'), 'spacecraft': sc_info, 'arrival_model': arrival, 'refit': fit,
              'drag_fit': {k: drag[k] for k in ('gamma0_per_km', 'n_ref_cm3', 'ray', 'fitted')}}
    if have_obs and arrival:
        t_mod = t0 + dt.timedelta(seconds=arrival['t_s'])
        v_mod = arrival['v_normal' if df.ARRIVAL_SPEED_TYPE == 'normal' else 'v_ray']
        report['arrival_check'] = {'time_residual_h': (t_mod - obs['time']).total_seconds() / 3600.0,
                                   'speed_residual_kms': v_mod - obs['speed']}
        print(f'  arrival check: {t_mod:%Y-%m-%d %H:%M} UT ({report["arrival_check"]["time_residual_h"]:+.2f} h), '
              f'{v_mod:.0f} km/s ({v_mod - obs["speed"]:+.0f})')
    cfg.final_path(f'dbm_fit_gsd_{method}.json').write_text(json.dumps(report, indent=2, default=float))
    print(f'  saved {len(fronts)} fronts shock_para_gsd<hhhh>_{method}.npz, {table_path.name}, dbm_fit_gsd_{method}.json')
    report['gamma0_used_per_km'] = gamma0
    saved = cfg.figure_path
    cfg.figure_path = lambda name: saved(name.replace('derive_para_dbm_', 'derive_para_gsd_'))
    try:
        sd.figure(method, fronts, report, t0, sc,
                  title=f'{cfg.EVENT_ID} {method}: fast-mode shock (geometrical shock dynamics) in the MAS wind, '
                        + (f'drag gamma0 = {gamma0:.2g} /km on the whole front' if DRAG_ON_SHOCK else
                           f'driver gamma0 = {gamma0:.2g} /km within {DRIVER_HALF_ANGLE:g}\u00b0 of the apex'))
    finally:
        cfg.figure_path = saved
    return report


def main():
    bg = bk.load_background()
    print(bg.summary() if hasattr(bg, 'summary') else bg.name)
    r = np.arange(sd.R_TABLE[0], sd.R_TABLE[1] + 1e-9, sd.R_TABLE[2])
    table = FieldTable(bg, r, np.linspace(0, np.pi, RAY_GRID[0]), np.linspace(0, 2 * np.pi, RAY_GRID[1], endpoint=False))
    reports = {}
    for m in METHODS:                                              # a method that fails is reported, the rest run
        try:
            reports[m] = run(m, bg, table)
            reason = None if reports[m] else 'no coronal series for this method: run derive_para_corona.py first'
        except RuntimeError as e:
            print(f'\n[{m}] not propagated: {e}')
            reports[m], reason = None, str(e)
        getattr(sp, 'record_failure', lambda *a: None)('gsd', m, reason)
    if not any(reports.values()):
        sys.exit('nothing propagated')
    if PROPERTIES:
        for m in METHODS:
            if reports[m]:
                sp.run_series('gsd', m, bg)


if __name__ == '__main__':
    main()
