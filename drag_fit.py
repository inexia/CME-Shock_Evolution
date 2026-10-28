"""
drag_fit.py -- ONE drag parameter for the three shock-evolution versions (shock_properties_sse / _dbm / _gsd.py),
fitted along the spacecraft direction with the real background.

The drag-based model (DBM) is the base of the interplanetary evolution: without drag the front stays fast, arrives
too early and overestimates every shock parameter. The drag is set from the observation, where it was observed:
  ray        the direction of the spacecraft that saw the shock (event_config: SC_LON, SC_LAT = 0 by default),
             which need not be the nose; the point of the front on that ray, with the front's OWN speed there at
             the start (the corona series' last step), is followed, not the apex
  background the wind speed w and the density on that ray from the MAS background (background.py), through time
             (the background rotates past the fixed ray): the drag is distance-dependent,
                 gamma(r, t) = gamma0 * N(r, t) / N_ref,   N = n (r / 1 AU)^2
             (gamma ~ cross-section x wind density / mass; N_ref: mean N over the start front), and the ray's
             equation of motion is DBM's, dV/dt = -gamma (V - w) |V - w|, solved with the exact step per DT_MIN
  fit        gamma0 such that the ray reaches SC_DISTANCE_AU at ARRIVAL_TIME with ARRIVAL_SPEED (the model's
             speed along the front normal, or radial: ARRIVAL_SPEED_TYPE), weighted by SIGMA_TIME_H / SIGMA_SPEED
Used by: the SSE version keeps its shape and scales it so that its point on this ray follows the ray solution
(the nose follows from the geometry); the DBM version gives every point of the front this gamma0 with its own
local w and N; the GSD version starts from it and refits to the arrival (its geometric terms change the kinematics).

Run alone:  python drag_fit.py      (for every method in METHODS)
Writes fitted results/<id>/final/drag_fit_<method>.json and figure/<id>/drag_fit_<method>.png.
"""
import datetime as dt
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import event_config as cfg
import background as bk
import shock_properties_sse as sp

# ================================ SETTINGS ================================
METHODS = ['constant', 'varied']
DT_MIN = 5.0                           # integration step of the ray [min]
GAMMA0_DEFAULT = 0.2e-7                # [1/km] without an observed arrival
GAMMA_RANGE = (1e-9, 3e-6)             # search range [1/km]
SIGMA_TIME_H = 1.0                     # weights of the combined fit: time [h] and speed [km/s]
SIGMA_SPEED = 50.0
ARRIVAL_SPEED_TYPE = 'normal'          # compare the observed speed with the ray's 'normal' or 'radial' speed
T_MAX_DAYS = 8.0
# ==========================================================================
RSUN_KM = 695700.0
AU_RSUN = bk.AU_RSUN
DAY = 86400.0


def observation():
    """(direction unit vector, lon, lat, distance [R_sun], arrival datetime, speed) from event_config, or None."""
    need = ('ARRIVAL_TIME', 'ARRIVAL_SPEED', 'SC_DISTANCE_AU', 'SC_LON')
    if any(getattr(cfg, k, None) is None for k in need):
        return None
    lon, lat = float(cfg.SC_LON), float(getattr(cfg, 'SC_LAT', 0.0) or 0.0)
    lo, la = np.radians(lon), np.radians(lat)
    d = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    t_arr = dt.datetime.fromisoformat(str(cfg.ARRIVAL_TIME).replace(' ', 'T'))
    return {'d': d, 'lon': lon, 'lat': lat, 'dist_rsun': float(cfg.SC_DISTANCE_AU) * AU_RSUN, 'time': t_arr,
            'speed': float(cfg.ARRIVAL_SPEED)}


class RayWind:
    """w [km/s] and N = n (r/AU)^2 [cm^-3] along a fixed HGS direction d, tabulated in r and time (the background
    rotates past the ray); bilinear lookups."""

    def __init__(self, bg, d, t0, r0, r_max, t_max_s, dr=1.0, dt_h=1.0):
        self.r = np.arange(max(r0 - 2.0, 1.1), r_max + dr, dr)
        self.t = np.arange(0.0, t_max_s + dt_h * 3600.0, dt_h * 3600.0)
        self.t0 = t0
        pts = self.r[:, None] * d[None, :]
        ur, N = np.empty((len(self.t), len(self.r))), np.empty((len(self.t), len(self.r)))
        for j, ts in enumerate(self.t):
            s = bg.sample(pts, t0 + dt.timedelta(seconds=float(ts)))
            ur[j] = np.sum(s['u'] * d[None, :], 1)
            N[j] = s['n'] * (self.r / AU_RSUN) ** 2
        for a in (ur, N):                                       # outside the background's domain: hold the edge
            for j in range(len(self.t)):
                ok = np.isfinite(a[j])
                if ok.any() and not ok.all():
                    a[j] = np.interp(self.r, self.r[ok], a[j][ok])
        self.ur, self.N = ur, N

    def at(self, r, t):
        i = np.clip(np.searchsorted(self.r, r) - 1, 0, len(self.r) - 2)
        j = np.clip(np.searchsorted(self.t, t) - 1, 0, len(self.t) - 2)
        fr = np.clip((r - self.r[i]) / (self.r[i + 1] - self.r[i]), 0, 1)
        ft = np.clip((t - self.t[j]) / (self.t[j + 1] - self.t[j]), 0, 1)
        pick = lambda a: ((1 - ft) * ((1 - fr) * a[j, i] + fr * a[j, i + 1]) + ft * ((1 - fr) * a[j + 1, i] + fr * a[j + 1, i + 1]))
        return float(pick(self.ur)), float(pick(self.N))


def integrate(gamma0, n_ref, r0, v0, wind, stop_r=None, t_max_s=T_MAX_DAYS * DAY, dt_min=DT_MIN):
    """The ray's DBM solution: (t [s], R [R_sun], V [km/s]) with the exact step for the local w and gamma, until
    R >= stop_r (the last point interpolated onto stop_r) or t_max_s."""
    dt_s = dt_min * 60.0
    t, R, V = [0.0], [r0], [v0]
    while t[-1] < t_max_s:
        w, N = wind.at(R[-1], t[-1])
        g = gamma0 * N / n_ref
        dv = V[-1] - w
        a = g * abs(dv)
        adv = w * dt_s + (np.sign(dv) * np.log1p(a * dt_s) / g if a > 0 else dv * dt_s)
        R_new, V_new = R[-1] + adv / RSUN_KM, w + dv / (1.0 + a * dt_s)
        if stop_r is not None and R_new >= stop_r:
            f = (stop_r - R[-1]) / (R_new - R[-1])
            t.append(t[-1] + f * dt_s)
            R.append(stop_r)
            V.append(V[-1] + f * (V_new - V[-1]))
            break
        t.append(t[-1] + dt_s)
        R.append(R_new)
        V.append(V_new)
    return np.array(t), np.array(R), np.array(V)


def ray_start(method, bg, d, series=None):
    """The start of the ray: the front's distance and speed in direction d at the corona series' last step, the
    front normal there, N_ref over the whole start front, and the start time."""
    import shock_properties_dbm as sd
    series = series or sp.load_series(method, 'corona')
    if len(series) < 2:
        raise RuntimeError(f'[{method}] no coronal series: run derive_para_corona.py first')
    model, t0 = series[-1]['model'], series[-1]['time']
    front = sd.initial_front(series, *sd.RAY_GRID)
    r0 = float(sd._bilinear(front.theta, front.phi, front.R, d))
    v0 = float(sd._bilinear(front.theta, front.phi, front.V, d))
    if not (np.isfinite(r0) and np.isfinite(v0)):
        ang = np.degrees(np.arccos(np.clip(model.frame[0] @ d, -1, 1)))
        raise RuntimeError(f'the spacecraft direction (lon {np.degrees(np.arctan2(d[1], d[0])):.0f}, lat '
                           f'{np.degrees(np.arcsin(d[2])):.0f}) is {ang:.0f} deg from the apex of the last corona step '
                           f'(lon {model.lon % 360:.0f}, lat {model.lat:.0f}), outside the front: no drag fit. This '
                           f'method\'s direction at the corona\'s end does not face the spacecraft')
    th, ph = np.meshgrid(front.theta, front.phi, indexing='ij')
    ok = np.isfinite(front.R)
    pts = front.R[ok][:, None] * np.stack([np.sin(th[ok]) * np.cos(ph[ok]), np.sin(th[ok]) * np.sin(ph[ok]),
                                           np.cos(th[ok])], -1)
    s = bg.sample(pts, t0)
    n_ref = float(np.nanmean(s['n'] * (np.linalg.norm(pts, axis=1) / AU_RSUN) ** 2))
    n = front.normals_at((r0 * d)[None])[0]
    return {'model': model, 't0': t0, 'r0': r0, 'v0': v0, 'cos_n': float(np.dot(n, d)), 'n_ref': n_ref,
            'apex_speed': float(series[-1]['speed_kms']), 'start': series[-1]['name']}


def _roots(f, target, grid):
    vals = np.array([f(g) for g in grid]) - target
    out = []
    for i in np.flatnonzero(np.isfinite(vals[:-1]) & np.isfinite(vals[1:]) & (np.sign(vals[:-1]) != np.sign(vals[1:]))):
        lo, hi = np.log(grid[i]), np.log(grid[i + 1])
        flo = f(np.exp(lo)) - target
        for _ in range(40):
            m = 0.5 * (lo + hi)
            fm = f(np.exp(m)) - target
            lo, flo, hi = (m, fm, hi) if np.sign(fm) == np.sign(flo) else (lo, flo, m)
        out.append(float(np.exp(0.5 * (lo + hi))))
    return out


def fit(method, bg, series=None, verbose=True):
    """Fit gamma0 along the spacecraft direction (or take GAMMA0_DEFAULT without an observation); save and plot."""
    obs = observation()
    if obs is None:
        d = (series or sp.load_series(method, 'corona'))[-1]['model'].frame[0]
        start = ray_start(method, bg, d, series)
        out = {'method': method, 'gamma0_per_km': GAMMA0_DEFAULT, 'fitted': False, 'n_ref_cm3': start['n_ref'],
               'ray': {'hgs_lon_deg': float(np.degrees(np.arctan2(d[1], d[0]))), 'hgs_lat_deg': float(np.degrees(np.arcsin(d[2]))),
                       'along': 'apex', 'r0_rsun': start['r0'], 'v0_kms': start['v0'], 'cos_normal_ray': start['cos_n']},
               'start': start['start'], 'start_time': start['t0'].isoformat(), 'observed': None}
        if verbose:
            print(f'[{method}] no observed arrival in event_config.py: gamma0 = {GAMMA0_DEFAULT:.3g} /km along the apex')
        cfg.final_path(f'drag_fit_{method}.json').write_text(json.dumps(out, indent=2))
        return out
    start = ray_start(method, bg, obs['d'], series)
    t0, r0, v0 = start['t0'], start['r0'], start['v0']
    t_obs = (obs['time'] - t0).total_seconds()
    factor = start['cos_n'] if ARRIVAL_SPEED_TYPE == 'normal' else 1.0
    wind = RayWind(bg, obs['d'], t0, r0, 1.3 * obs['dist_rsun'], T_MAX_DAYS * DAY)
    w0, N0 = wind.at(r0, 0.0)
    if verbose:
        print(f'[{method}] drag fit along HGS lon {obs["lon"]:.1f}, lat {obs["lat"]:.1f} '
              f'({np.degrees(np.arccos(np.clip(start["model"].frame[0] @ obs["d"], -1, 1))):.0f} deg from the apex): '
              f'start {start["start"]} at {t0:%Y-%m-%d %H:%M} UT, r0 {r0:.1f} R_sun, ray speed {v0:.0f} km/s '
              f'(apex {start["apex_speed"]:.0f}), wind {w0:.0f} km/s, N_ref {start["n_ref"]:.2f} cm^-3')

    def arrive(g):
        t, R, V = integrate(g, start['n_ref'], r0, v0, wind, stop_r=obs['dist_rsun'])
        return (t[-1], V[-1] * factor) if R[-1] >= obs['dist_rsun'] - 1e-9 else (np.nan, np.nan)

    cache = {}

    def arr(g):
        if g not in cache:
            cache[g] = arrive(g)
        return cache[g]
    st, sv = SIGMA_TIME_H * 3600.0, SIGMA_SPEED
    cost = lambda g: ((arr(g)[0] - t_obs) / st) ** 2 + ((arr(g)[1] - obs['speed']) / sv) ** 2
    grid = np.geomspace(*GAMMA_RANGE, 120)
    J = np.array([cost(g) for g in grid])
    if not np.isfinite(J).any():
        raise RuntimeError('the ray never reaches the spacecraft distance for any gamma0 in GAMMA_RANGE')
    i = int(np.nanargmin(J))
    from scipy.optimize import minimize_scalar
    lo, hi = np.log(grid[max(i - 1, 0)]), np.log(grid[min(i + 1, len(grid) - 1)])
    res = minimize_scalar(lambda lg: float(cost(np.exp(lg))), bounds=(lo, hi), method='bounded', options={'xatol': 1e-6})
    g_best = float(np.exp(res.x))
    t_mod, v_mod = arr(g_best)
    out = {'method': method, 'gamma0_per_km': g_best, 'fitted': True, 'n_ref_cm3': start['n_ref'],
           'ray': {'hgs_lon_deg': obs['lon'], 'hgs_lat_deg': obs['lat'], 'along': 'spacecraft', 'r0_rsun': r0,
                   'v0_kms': v0, 'w0_kms': w0, 'cos_normal_ray': start['cos_n'],
                   'apex_to_ray_angle_deg': float(np.degrees(np.arccos(np.clip(start['model'].frame[0] @ obs['d'], -1, 1))))},
           'start': start['start'], 'start_time': t0.isoformat(),
           'observed': {'arrival_time': obs['time'].isoformat(), 'arrival_speed_kms': obs['speed'],
                        'sc_distance_au': obs['dist_rsun'] / AU_RSUN, 'speed_type': ARRIVAL_SPEED_TYPE},
           'model': {'arrival_time': (t0 + dt.timedelta(seconds=float(t_mod))).isoformat(), 'arrival_speed_kms': float(v_mod),
                     'time_residual_h': float((t_mod - t_obs) / 3600.0), 'speed_residual_kms': float(v_mod - obs['speed'])},
           'gamma_time_only': _roots(lambda g: arr(g)[0], t_obs, grid),
           'gamma_speed_only': _roots(lambda g: arr(g)[1], obs['speed'], grid),
           'cost_curve': {'gamma': grid.tolist(), 'cost': np.where(np.isfinite(J), J, np.nan).tolist()}}
    if verbose:
        fmt = lambda gs: ', '.join(f'{g:.3g}' for g in gs) if gs else 'none in range'
        print(f'  gamma0 matching the time alone : {fmt(out["gamma_time_only"])} /km')
        print(f'  gamma0 matching the speed alone: {fmt(out["gamma_speed_only"])} /km')
        print(f'  best: gamma0 = {g_best:.3g} /km -> arrival {out["model"]["arrival_time"][:16].replace("T", " ")} UT '
              f'({out["model"]["time_residual_h"]:+.2f} h), {v_mod:.0f} km/s ({out["model"]["speed_residual_kms"]:+.0f})')
    cfg.final_path(f'drag_fit_{method}.json').write_text(json.dumps(out, indent=2))
    figure(out, start, wind, obs)
    return out


def load(method):
    """The saved fit, or None."""
    p = cfg.FINAL_DIR / f'drag_fit_{method}.json'
    return json.loads(p.read_text()) if p.exists() else None


def ray_solution(fit_or_method, bg, series=None):
    """(t [s], R [R_sun], V [km/s], start dict, wind) of the ray with the fitted gamma0 (or the default)."""
    f = load(fit_or_method) if isinstance(fit_or_method, str) else fit_or_method
    lo, la = np.radians(f['ray']['hgs_lon_deg']), np.radians(f['ray']['hgs_lat_deg'])
    d = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    start = ray_start(f['method'], bg, d, series)
    wind = RayWind(bg, d, start['t0'], start['r0'], 1.3 * max(AU_RSUN, start['r0']) + 50.0, T_MAX_DAYS * DAY)
    t, R, V = integrate(f['gamma0_per_km'], f['n_ref_cm3'], start['r0'], start['v0'], wind)
    return t, R, V, start, wind


def figure(out, start, wind, obs):
    g = out['gamma0_per_km']
    t, R, V = integrate(g, out['n_ref_cm3'], start['r0'], start['v0'], wind, t_max_s=min(T_MAX_DAYS * DAY, 2.5 * (obs['time'] - start['t0']).total_seconds()))
    times = [start['t0'] + dt.timedelta(seconds=float(x)) for x in t]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), layout='constrained')
    ax[0].plot(times, R, 'k-', label=f'ray, gamma0 = {g:.2g} /km')
    ax[0].axhline(obs['dist_rsun'], color='0.6', ls=':', lw=1)
    ax[0].plot([obs['time']], [obs['dist_rsun']], 'o', color='C3', ms=8, label='observed arrival')
    ax[0].set_ylabel('distance on the ray [R$_\\odot$]')
    ax[0].set_title(f'ray in the spacecraft direction (lon {obs["lon"]:.0f}, lat {obs["lat"]:.0f})')
    ax[0].legend(fontsize=8)
    factor = start['cos_n'] if ARRIVAL_SPEED_TYPE == 'normal' else 1.0
    ax[1].plot(times, V * factor, 'k-', label=f'front {ARRIVAL_SPEED_TYPE} speed on the ray')
    ax[1].plot(times, [wind.at(r, x)[0] for r, x in zip(R, t)], '-', color='C0', lw=1, label='wind speed there')
    ax[1].plot([obs['time']], [obs['speed']], 'o', color='C3', ms=8, label='observed')
    ax[1].set_ylabel('[km/s]')
    ax[1].set_title('speeds along the ray')
    ax[1].legend(fontsize=8)
    cc = out['cost_curve']
    ax[2].loglog(cc['gamma'], cc['cost'], 'k-')
    ax[2].axvline(g, color='C3', lw=1, label=f'best {g:.2g}')
    for gg, lab, col in ((out['gamma_time_only'], 'time alone', 'C0'), (out['gamma_speed_only'], 'speed alone', 'C2')):
        for k, x in enumerate(gg):
            ax[2].axvline(x, color=col, ls='--', lw=1, label=lab if k == 0 else None)
    ax[2].set_xlabel('gamma0 [1/km]')
    ax[2].set_ylabel('cost')
    ax[2].set_title('fit')
    ax[2].legend(fontsize=8)
    for a in ax[:2]:
        a.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
        a.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        a.set_xlabel('time [UT]')
    fig.suptitle(f'{cfg.EVENT_ID} {out["method"]}: drag fitted along the spacecraft direction')
    png = cfg.figure_path(f'drag_fit_{out["method"]}.png')
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f'  saved {png}')


def main():
    bg = bk.load_background()
    for method in METHODS:
        try:
            fit(method, bg)
        except RuntimeError as e:
            print(e)


if __name__ == '__main__':
    main()
