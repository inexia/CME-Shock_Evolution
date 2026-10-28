"""
shock_properties_dbm.py -- interplanetary propagation of the shock front ray by ray with the drag-based model in
the structured MAS wind, then the shock properties on those fronts (the interplanetary part of step 5, with a
front that deforms in streamers, SIRs and fast streams).

Run:  python shock_properties_dbm.py        (settings below; folders and background from event_config.py)

Each Sun-centred ray (colatitude, longitude in HGS) carries the front distance R and its ray speed V:
    dV/dt = -gamma (V - w) |V - w|,     dR/dt = V,
with, at the ray's current position and time, the local radial wind speed w = u_r of the background and
    gamma = GAMMA0 * N / N_ref,  N = n (r / 1 AU)^2 :
the r^2-normalised density. In a Parker wind N is constant and gamma is the classic constant drag parameter;
in a streamer or SIR the drag is larger, in a fast rarefied stream smaller -- so the front slows down where the
wind is slow and dense and keeps its speed in fast streams, and deforms. The background turns with the Sun
during the propagation (Carrington longitude of each time). Rays are independent; a mild lateral smoothing
(SMOOTH_DEG_PER_DAY) stands in for the lateral coupling real fronts have.
Each step uses the exact DBM solution for the wind and drag at the ray's current position (stable for any gamma).
Start: the 21.5 R_sun step of each coronal series (derive_para_corona.py): R from the surface along each ray,
V from the surface's normal speed there (V_ray = V_n / (n . r_hat)). Rays that miss the surface carry no front.
Output: fronts when the apex has moved H_STEP further, up to H_END (with an observed arrival: until the front
has passed the spacecraft). gamma0 is fitted to an observed arrival time and speed (where the front crosses the
spacecraft's ray; speed along the normal there) on the coarser FIT_GRID, as in derive_para_ip.py.
Then shock_properties.run_series('dbm') computes V_n (from the ray speeds), theta_Bn, M_A, M_f and X on every
front and draws the 2 x 3 figures.

Writes  fitted results/<id>/final/shock_para_dbm<hhhh>_<method>.npz  (the fronts), shock_para_dbm_<method>.csv,
dbm_fit_dbm_<method>.json, figure/derive_para_dbm_<method>.png, and through shock_properties_sse.py the
shock_props_dbm/ files, shock_props_dbm_<method>.csv and figure/shock_evo_dbm/.
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
from scipy.ndimage import gaussian_filter

import event_config as cfg
import background as bk
import shock_properties_sse as sp
import drag_fit as df
from ray_front import RayFront

# ================================ SETTINGS ================================
METHODS = ['constant', 'varied']
RAY_GRID = (91, 181)              # rays: (colatitude, longitude) points; 91 x 181 = 2 deg
FIT_GRID = (46, 91)               # coarser rays for the gamma0 fit (4 deg)
DT_MIN = 5.0                      # integration step [min] (FIT_DT_MIN for the fit)
FIT_DT_MIN = 10.0
H_STEP, H_END = 5.0, 220.0        # output every H_STEP of apex height, up to H_END [R_sun]
EXTEND_TO_SC = True               # with an observed arrival: continue until the front has passed the spacecraft
REFIT_DRAG = False                # True: fit the drag now (drag_fit.py); False: use drag_fit_<method>.json if it
                                  # exists (written by shock_properties_sse.py / drag_fit.py), else fit it
GAMMA_DENSITY_SCALING = True      # gamma = gamma0 * N / N_ref (False: gamma0 everywhere)
SMOOTH_DEG_PER_DAY = 2.0          # lateral smoothing of R and V: Gaussian sigma reached after one day [deg]; 0: off
R_TABLE = (21.5, 300.0, 1.0)      # radial grid of the wind table [R_sun]: start, end, step
V_RAY_MAX_FACTOR = 1.5            # initial ray speeds capped at this times the apex speed (grazing rays)
PROPERTIES = True                 # run shock_properties on the fronts afterwards (properties and figures)
# ==========================================================================
RSUN_KM = bk.RSUN_KM
AU_RSUN = bk.AU_RSUN
DAY = 86400.0


# ----------------------------------------------------------------------------- the wind along the rays
class WindTable:
    """u_r and N = n (r/AU)^2 of the background on (r, colatitude, Carrington longitude), for fast lookups."""

    def __init__(self, bg, r, theta, phi, verbose=True):
        self.r, self.theta, self.phi = np.asarray(r, float), np.asarray(theta, float), np.asarray(phi, float)
        self.multi = bg if hasattr(bg, 'components') else None
        if self.multi is not None:                               # several runs: one table each, blended at lookup
            self.subs = [type(self)(c, r, theta, phi, verbose) for c in bg.components]
            return
        cache = None
        if hasattr(bg, 'folder'):
            cache = bg.folder / 'cache' / f'wind_table_{len(r)}x{len(theta)}x{len(phi)}_{r[0]:g}_{r[-1]:g}.npz'
        if cache is not None and cache.exists():
            z = np.load(cache)
            self.ur, self.N = z['ur'], z['N']
        else:
            t0 = cfg.iso_time()
            offset = bg._offset(t0)
            Rg, TH, PH = np.meshgrid(self.r, self.theta, self.phi, indexing='ij')
            lon = PH - offset                                   # HGS longitude that has Carrington longitude PH at t0
            xyz = np.stack([Rg * np.sin(TH) * np.cos(lon), Rg * np.sin(TH) * np.sin(lon), Rg * np.cos(TH)], -1)
            pts = xyz.reshape(-1, 3)
            ur, N = np.empty(len(pts), np.float32), np.empty(len(pts), np.float32)
            step = 400000
            if verbose:
                print(f'  tabulating the wind on {len(pts):,} points ...')
            for i in range(0, len(pts), step):
                s = bg.sample(pts[i:i + step], t0)
                rr = np.linalg.norm(pts[i:i + step], axis=1)
                ur[i:i + step] = np.sum(s['u'] * pts[i:i + step] / rr[:, None], 1)
                N[i:i + step] = s['n'] * (rr / AU_RSUN) ** 2
            self.ur, self.N = self._fill_radial(ur.reshape(Rg.shape)), self._fill_radial(N.reshape(Rg.shape))
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, ur=self.ur, N=self.N)
        ph = np.concatenate([self.phi, [self.phi[0] + 2 * np.pi]])
        pad = lambda a: np.concatenate([a, a[:, :, :1]], axis=2)
        grid = (self.r, self.theta, ph)
        self._ur = RegularGridInterpolator(grid, pad(self.ur), bounds_error=False, fill_value=None)
        self._N = RegularGridInterpolator(grid, pad(self.N), bounds_error=False, fill_value=None)

    @staticmethod
    def _fill_radial(a):
        """Beyond the background's outer (or inner) edge, hold the last valid shell along r."""
        a = a.copy()
        for i in range(1, a.shape[0]):
            bad = ~np.isfinite(a[i])
            a[i][bad] = a[i - 1][bad]
        for i in range(a.shape[0] - 2, -1, -1):
            bad = ~np.isfinite(a[i])
            a[i][bad] = a[i + 1][bad]
        return a

    def _weights(self, lon_hgs, L0, when):
        return self.multi.weights(np.mod(lon_hgs + L0, 2 * np.pi), when)

    def lookup(self, r, theta, lon_hgs, L0, when=None):
        """u_r [km/s] and N = n (r/AU)^2 at the points; `when` (POSIX seconds) weighs several runs."""
        if self.multi is not None:
            w = self._weights(lon_hgs, L0, when)
            got = [s.lookup(r, theta, lon_hgs, L0) for s in self.subs]
            return (self.multi._blend([g[0] for g in got], w), self.multi._blend([g[1] for g in got], w))
        pts = np.stack([np.clip(r, self.r[0], self.r[-1]), theta, np.mod(lon_hgs + L0, 2 * np.pi)], -1)
        return self._ur(pts), self._N(pts)


def l0_function(bg, t0):
    """Carrington longitude of the sub-Earth point as a linear function of time (seconds after t0)."""
    off = lambda t: (lambda o: o[0] if isinstance(o, tuple) else o)(bg._offset(t))
    a = off(t0)
    b = off(t0 + dt.timedelta(days=1))
    rate = (b - a + np.pi) % (2 * np.pi) - np.pi
    return lambda t_s: a + rate * t_s / DAY


# ----------------------------------------------------------------------------- the initial front
def initial_front(series, n_theta, n_phi):
    """The 21.5 R_sun surface on rays, with ray speeds from its normal speed."""
    k = len(series) - 1
    model = series[k]['model']
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    front = RayFront.from_surface(model, theta, phi, time=series[k]['time_iso'])
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    d = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1)
    hw = sp.half_width(model)                                    # the physical front: no far side of a halo fit
    ang = np.degrees(np.arccos(np.clip(d @ model.frame[0], -1, 1)))
    front.R = np.where(ang <= hw + 1e-9, front.R, np.nan)
    front.meta['half_width_deg'] = hw
    front.half_width_deg = hw
    ok = np.isfinite(front.R)
    xyz = front.R[..., None] * d
    n = model.normals(*local_angles(model, d))
    vn = np.full(front.R.shape, np.nan)
    vn[ok] = sp.normal_speed(series, k, xyz[ok], n[ok], 'normal_distance')
    cosn = np.sum(n * d, -1)
    v_apex = series[k]['speed_kms']
    V = vn / np.maximum(cosn, 1e-3)
    V = np.clip(V, 0.0, V_RAY_MAX_FACTOR * v_apex)
    front.V = np.where(ok, V, np.nan)
    front._interp.clear()
    return front


def local_angles(model, d):
    e1, e2, e3 = model.frame                                      # angles of the directions d about the frame
    return np.arccos(np.clip(d @ e1, -1, 1)), np.mod(np.arctan2(d @ e3, d @ e2), 2 * np.pi)


# ----------------------------------------------------------------------------- propagation
def pole_mean(a):
    """Set each pole row (one direction) to the mean of its values -- only where the front covers that pole."""
    for i in (0, -1):
        row = a[i, :]
        if np.isfinite(row).any():
            a[i, :] = np.nanmean(row)
    return a


def _smooth(a, sigma_cells, theta=None):
    """Gaussian smoothing on the (theta, phi) grid, periodic in phi; toward the poles the longitude smoothing
    widens with 1 / sin(theta) so that it stays the same angular width on the sphere."""
    if sigma_cells <= 0:
        return a
    ok = np.isfinite(a)
    if ok.all():
        out = gaussian_filter(a, sigma_cells, mode=('nearest', 'wrap'))
    else:
        num = gaussian_filter(np.where(ok, a, 0.0), sigma_cells, mode=('nearest', 'wrap'))
        den = gaussian_filter(ok.astype(float), sigma_cells, mode=('nearest', 'wrap'))
        out = np.where(ok, num / np.maximum(den, 1e-12), np.nan)
    if theta is not None:
        for i in np.flatnonzero(np.sin(theta) < 0.7):
            if np.isfinite(out[i]).all():
                out[i] = gaussian_filter(out[i], min(sigma_cells / max(np.sin(theta[i]), 0.02), a.shape[1] / 4), mode='wrap')
    return out


def _dir(theta, phi):
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    return TH, PH, np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1)


def _bilinear(theta, phi, A, d):
    """A(theta, phi) interpolated in the direction d (unit vector)."""
    ph = np.concatenate([phi, [phi[0] + 2 * np.pi]])
    f = RegularGridInterpolator((theta, ph), np.concatenate([A, A[:, :1]], axis=1), bounds_error=False,
                                fill_value=np.nan)
    th = np.arccos(np.clip(d[2], -1, 1))
    return float(f([[th, np.mod(np.arctan2(d[1], d[0]), 2 * np.pi)]])[0])


def propagate(front0, table, L0, t0, gamma0, n_ref, dt_min, heights, sc=None, t_max_days=15.0, verbose=False):
    """
    Integrate the rays from front0 (time t0). Returns (fronts at the apex heights `heights`, arrival dict or None):
    the arrival is the moment the front crosses the spacecraft (sc = (direction, distance)), with the front's
    speed there along the ray and along the normal.
    """
    t0_s = bk.posix(t0)
    theta, phi = front0.theta, front0.phi
    TH, PH, d = _dir(theta, phi)
    apex = front0.frame[0]
    R, V = front0.R.copy(), front0.V.copy()
    ok = np.isfinite(R) & np.isfinite(V)
    R[~ok], V[~ok] = np.nan, np.nan
    dt_s = dt_min * 60.0
    sig_cells = (SMOOTH_DEG_PER_DAY * np.sqrt(dt_s / DAY)) / np.degrees(theta[1] - theta[0]) if SMOOTH_DEG_PER_DAY else 0.0

    def step(R_, V_, t):
        """One step with the exact DBM solution for the wind and drag at the rays' current positions
        (unconditionally stable, exact for constant w and gamma over the step)."""
        w, N = table.lookup(R_[ok], TH[ok], PH[ok], L0(t), when=t0_s + t)
        g = gamma0 * (N / n_ref if GAMMA_DENSITY_SCALING else 1.0)
        dv = V_[ok] - w
        a = g * np.abs(dv)
        R_new, V_new = R_.copy(), V_.copy()
        with np.errstate(divide='ignore', invalid='ignore'):
            adv = w * dt_s + np.where(a > 0, np.sign(dv) * np.log1p(a * dt_s) / np.where(g > 0, g, 1.0), dv * dt_s)
        R_new[ok] = R_[ok] + adv / RSUN_KM
        V_new[ok] = w + dv / (1.0 + a * dt_s)
        return R_new, V_new

    heights = list(heights)
    out, arrival = [], None
    t = 0.0
    r_apex = _bilinear(theta, phi, R, apex)
    r_sc = _bilinear(theta, phi, R, sc[0]) if sc else np.nan
    if sc and not np.isfinite(r_sc):
        sc = None                                                 # no front on the spacecraft's ray: no arrival
    while (heights or (sc and arrival is None)) and t < t_max_days * DAY:
        R_new, V_new = step(R, V, t)
        R_new, V_new = _smooth(R_new, sig_cells, theta), _smooth(V_new, sig_cells, theta)
        for a in (R_new, V_new):                                   # the poles are one direction
            pole_mean(a)
        t_new = t + dt_s
        r_apex_new = _bilinear(theta, phi, R_new, apex)
        while heights and r_apex_new >= heights[0] - 1e-9:
            f = 0.0 if r_apex_new == r_apex else (heights[0] - r_apex) / (r_apex_new - r_apex)
            f = float(np.clip(f, 0.0, 1.0))
            when = t0 + dt.timedelta(seconds=t + f * dt_s)
            out.append(RayFront(theta, phi, R + f * (R_new - R), V + f * (V_new - V), apex=apex,
                                time=when.strftime('%Y-%m-%dT%H:%M:%S'), meta={'half_width_deg': front0.half_width_deg}))
            heights.pop(0)
        if sc and arrival is None:
            r_sc_new = _bilinear(theta, phi, R_new, sc[0])
            if np.isfinite(r_sc_new) and r_sc_new >= sc[1] and np.isfinite(r_sc):
                f = float(np.clip((sc[1] - r_sc) / (r_sc_new - r_sc), 0.0, 1.0)) if r_sc_new > r_sc else 0.0
                fr = RayFront(theta, phi, R + f * (R_new - R), V + f * (V_new - V), apex=apex)
                x = sc[1] * sc[0]
                n = fr.normals_at(x[None])[0]
                v_ray = float(fr.V_at(sc[0]))
                arrival = {'t_s': t + f * dt_s, 'apex_height': _bilinear(theta, phi, fr.R, apex),
                           'v_ray': v_ray, 'v_normal': v_ray * float(np.dot(n, sc[0])), 'cos_normal_ray': float(np.dot(n, sc[0]))}
            r_sc = r_sc_new
        R, V, t, r_apex = R_new, V_new, t_new, r_apex_new
    return out, arrival


def run(method, bg, table):
    series = sp.load_series(method, 'corona')
    if len(series) < 2:
        print(f'[{method}] no coronal series: run derive_para_corona.py first')
        return None
    t0 = series[-1]['time']
    L0 = l0_function(bg, t0)
    print(f'\n[{method}] start: {series[-1]["name"]}  h = {series[-1]["model"].height:.2f} R_sun at '
          f'{t0:%Y-%m-%d %H:%M:%S} UT, apex speed {series[-1]["speed_kms"]:.0f} km/s')
    front = initial_front(series, *RAY_GRID)
    ok = np.isfinite(front.R)
    w0, N0 = table.lookup(front.R[ok], np.meshgrid(front.theta, front.phi, indexing='ij')[0][ok],
                          np.meshgrid(front.theta, front.phi, indexing='ij')[1][ok], L0(0.0), when=bk.posix(t0))
    n_ref = float(np.nanmean(N0))
    print(f'  rays with a front: {ok.mean() * 100:.0f}%; ray speeds {np.nanmin(front.V):.0f} - {np.nanmax(front.V):.0f} km/s '
          f'(apex {front.speed:.0f}); wind there {np.nanmin(w0):.0f} - {np.nanmax(w0):.0f} km/s, '
          f'N_ref = {n_ref:.2f} cm^-3 (at 1 AU)')
    # ---- the drag: ONE gamma0 for the three versions, fitted along the spacecraft direction (drag_fit.py)
    drag = None if REFIT_DRAG else df.load(method)
    if drag is None:
        drag = df.fit(method, bg, series)
    gamma0, n_ref = drag['gamma0_per_km'], drag['n_ref_cm3']
    obs = df.observation()
    sc = (obs['d'], obs['dist_rsun']) if obs else None
    sc_info = {'direction_hgs_lon_lat_deg': [obs['lon'], obs['lat']], 'distance_rsun': obs['dist_rsun']} if obs else None
    have_obs = obs is not None
    h0 = front.height
    heights = [float(h) for h in np.arange(h0, H_END + 1e-9, H_STEP)]
    if H_END - heights[-1] > 1e-6:
        heights.append(H_END)
    print(f'  propagating {RAY_GRID[0]} x {RAY_GRID[1]} rays with gamma0 = {gamma0:.3g} /km (dt {DT_MIN:g} min) ...')
    fronts, arrival = propagate(front, table, L0, t0, gamma0, n_ref, DT_MIN, heights[1:], sc)
    fronts = [front] + fronts
    if sc and arrival and EXTEND_TO_SC and arrival['apex_height'] > heights[-1]:
        extra = [float(h) for h in np.arange(heights[-1] + H_STEP, arrival['apex_height'] + H_STEP, H_STEP)]
        print(f'  extended to {extra[-1]:.1f} R_sun: the front passes the spacecraft when the apex is at '
              f'{arrival["apex_height"]:.1f} R_sun')
        more, _ = propagate(front, table, L0, t0, gamma0, n_ref, DT_MIN, heights[1:] + extra, None)
        fronts = [front] + more
    rows = []
    for fr in fronts:
        if sc is not None:
            r_sc = float(fr.R_at(sc[0]))
            v_sc = float(fr.V_at(sc[0]))
            cosn = float(np.dot(fr.normals_at((r_sc * sc[0])[None])[0], sc[0])) if np.isfinite(r_sc) else np.nan
        else:
            r_sc = v_sc = cosn = np.nan
        derived = {'method': f'dbm_{method}', 'height_rsun': round(fr.height, 3), 'time': fr.time,
                   'speed_kms': round(fr.speed, 1), 'gamma0_per_km': gamma0, 'N_ref_cm3_at_1AU': n_ref,
                   'density_scaling': GAMMA_DENSITY_SCALING, 'smooth_deg_per_day': SMOOTH_DEG_PER_DAY,
                   'start': series[-1]['name'], 'in_fitted_range': False,
                   'sc_ray_distance_rsun': None if not np.isfinite(r_sc) else round(r_sc, 3),
                   'sc_ray_speed_radial_kms': None if not np.isfinite(v_sc) else round(v_sc, 1),
                   'sc_ray_speed_normal_kms': None if not np.isfinite(v_sc * cosn) else round(v_sc * cosn, 1)}
        fr.save(cfg.final_path(f'shock_para_dbm{int(round(fr.height * 10)):04d}_{method}.npz'), **derived)
        rows.append({'height_rsun': f'{fr.height:.2f}', 'time': fr.time, 'apex_speed_kms': f'{fr.speed:.1f}',
                     'ray_speed_min_kms': f'{np.nanmin(fr.V):.1f}', 'ray_speed_max_kms': f'{np.nanmax(fr.V):.1f}',
                     'front_min_rsun': f'{np.nanmin(fr.R):.2f}', 'front_max_rsun': f'{np.nanmax(fr.R):.2f}',
                     'sc_ray_distance_rsun': f'{r_sc:.3f}' if np.isfinite(r_sc) else '',
                     'sc_ray_speed_radial_kms': f'{v_sc:.1f}' if np.isfinite(v_sc) else '',
                     'sc_ray_speed_normal_kms': f'{v_sc * cosn:.1f}' if np.isfinite(v_sc * cosn) else ''})
    table_path = cfg.final_path(f'shock_para_dbm_{method}.csv')
    with open(table_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    report = {'method': method, 'start': series[-1]['name'], 'start_time': t0.isoformat(),
              'start_speed_kms': series[-1]['speed_kms'], 'gamma0_used_per_km': gamma0, 'N_ref_cm3_at_1AU': n_ref,
              'density_scaling': GAMMA_DENSITY_SCALING, 'smooth_deg_per_day': SMOOTH_DEG_PER_DAY,
              'rays': list(RAY_GRID), 'dt_min': DT_MIN,
              'observed': drag.get('observed'), 'spacecraft': sc_info, 'arrival_model': arrival,
              'drag_fit': {k: drag[k] for k in ('gamma0_per_km', 'n_ref_cm3', 'ray', 'fitted')}}
    if have_obs and arrival:
        t_mod = t0 + dt.timedelta(seconds=arrival['t_s'])
        v_mod = arrival['v_normal' if df.ARRIVAL_SPEED_TYPE == 'normal' else 'v_ray']
        dt_h = (t_mod - obs['time']).total_seconds() / 3600.0
        report['arrival_check'] = {'time_residual_h': dt_h, 'speed_residual_kms': v_mod - obs['speed']}
        print(f'  arrival check with the shared gamma0: {t_mod:%Y-%m-%d %H:%M} UT ({dt_h:+.2f} h), {v_mod:.0f} km/s '
              f'({v_mod - obs["speed"]:+.0f}) -- the single-ray fit and the full front differ by the lateral smoothing')
    elif have_obs:
        print('  the front never reaches the spacecraft: no arrival check')
    cfg.final_path(f'dbm_fit_dbm_{method}.json').write_text(json.dumps(report, indent=2, default=float))
    print(f'  saved {len(fronts)} fronts shock_para_dbm<hhhh>_{method}.npz, {table_path.name}, dbm_fit_dbm_{method}.json')
    figure(method, fronts, report, t0, sc)
    return report


def figure(method, fronts, report, t0, sc, title=None):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5), layout='constrained')
    times = [dt.datetime.fromisoformat(f.time) for f in fronts]
    ax[0].plot(times, [f.height for f in fronts], 'k.-', label='apex')
    ax[0].plot(times, [np.nanmax(f.R) for f in fronts], ':', color='0.5', label='farthest / nearest ray')
    ax[0].plot(times, [np.nanmin(f.R) for f in fronts], ':', color='0.5')
    ax[1].plot(times, [f.speed for f in fronts], 'k.-', label='apex')
    ax[1].plot(times, [np.nanmax(f.V) for f in fronts], ':', color='0.5', label='fastest / slowest ray')
    ax[1].plot(times, [np.nanmin(f.V) for f in fronts], ':', color='0.5')
    if sc is not None:
        ax[0].plot(times, [f.R_at(sc[0]) for f in fronts], 'o-', color='C0', ms=3, label='front on the spacecraft ray')
        if df.ARRIVAL_SPEED_TYPE == 'normal':
            v_sc = [f.V_at(sc[0]) * float(np.dot(f.normals_at((f.R_at(sc[0]) * sc[0])[None])[0], sc[0]))
                    if np.isfinite(f.R_at(sc[0])) else np.nan for f in fronts]
        else:
            v_sc = [f.V_at(sc[0]) for f in fronts]
        ax[1].plot(times, v_sc, 'o-', color='C0', ms=3, label=f'spacecraft ray ({df.ARRIVAL_SPEED_TYPE})')
        if report.get('observed'):
            t_obs = dt.datetime.fromisoformat(report['observed']['arrival_time'])
            ax[0].plot(t_obs, sc[1], '*', color='C3', ms=14, label='observed arrival')
            ax[1].plot(t_obs, report['observed']['arrival_speed_kms'], '*', color='C3', ms=14,
                       label=f'observed ({df.ARRIVAL_SPEED_TYPE})')
    ax[0].set_ylabel('distance [R$_\\odot$]')
    ax[0].set_title('front distance')
    ax[1].set_ylabel('ray speed [km/s]')
    ax[1].set_title('ray speeds')
    for a in ax[:2]:
        a.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8, interval_multiples=True))
        a.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
        a.set_xlabel('time [UT]')
        a.legend(fontsize=7)
    last = fronts[-1]
    im = ax[2].pcolormesh(np.degrees(last.phi), 90 - np.degrees(last.theta), last.V, shading='auto', cmap='viridis')
    fig.colorbar(im, ax=ax[2], shrink=0.85, label='ray speed [km/s]')
    ax[2].plot(last.lon % 360, last.lat, 'w*', ms=12, mec='k', label='apex')
    if sc is not None:
        ax[2].plot(np.degrees(np.arctan2(sc[0][1], sc[0][0])) % 360, np.degrees(np.arcsin(sc[0][2])), 'r^', ms=9,
                   mec='k', label='spacecraft direction')
    ax[2].set_xlabel('HGS longitude [deg]')
    ax[2].set_ylabel('HGS latitude [deg]')
    ax[2].set_title(f'ray speeds at the last front (apex {last.height:.0f} R$_\\odot$)')
    ax[2].legend(fontsize=7, loc='lower left')
    fig.suptitle(title or (f'{cfg.EVENT_ID} {method}: ray-by-ray DBM in the MAS wind, gamma0 = '
                           f'{report["gamma0_used_per_km"]:.2g} /km' + (' (drag fitted along the spacecraft direction)'
                                                                     if report['drag_fit']['fitted'] else '')))
    png = cfg.figure_path(f'derive_para_dbm_{method}.png')
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f'  saved {png}')


def main():
    bg = bk.load_background()
    print(bg.summary() if hasattr(bg, 'summary') else bg.name)
    r = np.arange(R_TABLE[0], R_TABLE[1] + 1e-9, R_TABLE[2])
    table = WindTable(bg, r, np.linspace(0, np.pi, RAY_GRID[0]), np.linspace(0, 2 * np.pi, RAY_GRID[1], endpoint=False))
    reports = {}
    for m in METHODS:                                              # a method that fails is reported, the rest run
        try:
            reports[m] = run(m, bg, table)
            reason = None if reports[m] else 'no coronal series for this method: run derive_para_corona.py first'
        except RuntimeError as e:
            print(f'\n[{m}] not propagated: {e}')
            reports[m], reason = None, str(e)
        getattr(sp, 'record_failure', lambda *a: None)('dbm', m, reason)
    if not any(reports.values()):
        sys.exit('nothing propagated')
    if PROPERTIES:
        for m in METHODS:
            if reports[m]:
                sp.run_series('dbm', m, bg)


if __name__ == '__main__':
    main()
