"""
derive_para_ip.py -- the SSE (self-similar expansion) version of the interplanetary evolution: the shape of the
corona series' last step is kept and scaled self-similarly (PyThea: kappa, epsilon, alpha, direction and dents
fixed, lengths grown from the surface point below the apex), with the kinematics of ONE ray in the drag-based model
with the real background (drag_fit.py).

Run:  python derive_para_ip.py        (or through shock_properties_sse.py)

Kinematics: drag_fit.py follows the point of the front in the spacecraft direction (event_config: SC_LON, SC_LAT),
with the front's own speed there at the start, the local wind and a distance-dependent drag gamma = gamma0 N/N_ref
from the background, and fits gamma0 to the observed arrival (ARRIVAL_TIME, ARRIVAL_SPEED at SC_DISTANCE_AU).
Here the whole front is scaled so that its point on that ray follows the ray solution: the apex height at each
time is the height at which the self-similar surface meets the ray at the ray's distance, so the nose, which may
be far from the spacecraft direction, follows from the geometry. Without an observation the ray is the apex with
a default gamma0. The same gamma0 serves shock_properties_dbm.py (every point with its own local drag) and
shock_properties_gsd.py (which refits it for its geometric terms).

Writes fitted results/<id>/final/shock_para_ip<hhhh>_<method>.json (hhhh = apex height x 10) every H_STEP of
apex height, shock_para_ip_<method>.csv, sse_fit_<method>.json, and figure/<id>/derive_para_ip_<method>.png.
"""
import csv
import datetime as dt
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import event_config as cfg
import background as bk
import drag_fit as df
import shock_properties_sse as sp

# ================================ SETTINGS ================================
METHODS = ['constant', 'varied']   # coronal series to continue
H_STEP, H_END = 5.0, 220.0         # apex-height step and end [R_sun]
EXTEND_TO_SC = True                # with an observed arrival: continue until the front has passed the spacecraft
                                   # (on a flank the apex is far beyond the spacecraft distance by then)
REFIT_DRAG = True                  # fit the drag now (drag_fit.py); False: use fitted results/<id>/final/drag_fit_<m>.json
# ==========================================================================
RSUN_KM = 695700.0
AU_RSUN = 215.032


# ----------------------------------------------------------------------------- DBM
def dbm(t, v0, w, gamma):
    """Distance travelled [km] and speed [km/s] after t [s]; arrays broadcast."""
    t, v0, w, gamma = np.broadcast_arrays(*(np.asarray(x, float) for x in (t, v0, w, gamma)))
    dv = v0 - w
    a = gamma * np.abs(dv)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = w * t + np.where(a > 0, np.sign(dv) * np.log1p(a * t) / np.where(gamma > 0, gamma, 1.0), dv * t)
    return r, w + dv / (1.0 + a * t)


def dbm_time(dist_km, v0, w, gamma, t_max=90 * 86400.0):
    """Time [s] to travel dist_km (vectorised bisection; the motion is monotonic for v0, w > 0)."""
    dist_km, v0, w, gamma = np.broadcast_arrays(*(np.asarray(x, float) for x in (dist_km, v0, w, gamma)))
    lo, hi = np.zeros(dist_km.shape), np.full(dist_km.shape, t_max)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        far = dbm(mid, v0, w, gamma)[0] >= dist_km
        hi, lo = np.where(far, mid, hi), np.where(far, lo, mid)
    t = 0.5 * (lo + hi)
    return np.where(dbm(t_max, v0, w, gamma)[0] >= dist_km, t, np.nan)


# ----------------------------------------------------------------------------- geometry of the front
def offset(model, y):
    """|y - C| - R(direction): < 0 inside the star-shaped surface, > 0 outside."""
    g = y - model.center
    rr = np.linalg.norm(g, axis=-1)
    d = g / np.maximum(rr, 1e-12)[..., None]
    e1, e2, e3 = model.frame
    th = np.arccos(np.clip(d @ e1, -1, 1))
    ph = np.mod(np.arctan2(d @ e3, d @ e2), 2 * np.pi)
    return rr - model.radius(th, ph)


def ray_hit(model, d):
    """Distance [R_sun] from the Sun along the unit vector d to the outer crossing of the front (NaN: missed)."""
    a, b, c, _ = model.axes
    s_max = np.linalg.norm(model.center) + 2.5 * max(a, b, c) + 1.0
    S = np.linspace(0.0, s_max, 3001)
    F = offset(model, S[:, None] * d[None, :])
    idx = np.flatnonzero(np.sign(F[:-1]) != np.sign(F[1:]))
    if not idx.size:
        return np.nan
    lo, hi = S[idx[-1]], S[idx[-1] + 1]
    flo = offset(model, lo * d)
    for _ in range(60):
        m = 0.5 * (lo + hi)
        fm = offset(model, m * d)
        if np.sign(fm) == np.sign(flo):
            lo, flo = m, fm
        else:
            hi = m
    return 0.5 * (lo + hi)


def at_height(base, h, time=None):
    """The self-similar continuation of the 21.5 R_sun surface with apex height h."""
    return base.scaled((h - 1.0) / (base.height - 1.0), r0=1.0, time=time)


def sc_geometry(base, d):
    """Along the spacecraft ray d: R(h) and, at the arrival height, dR/dh and n.d. Returns a function and a dict."""
    R = lambda h: ray_hit(at_height(base, h), d)

    def slope(h):
        dh = 1e-3 * h
        return (R(h + dh) - R(h - dh)) / (2 * dh)

    def cos_normal(h):
        m = at_height(base, h)
        x = R(h) * d
        g = (x - m.center) / np.linalg.norm(x - m.center)
        e1, e2, e3 = m.frame
        n = m.normals(np.arccos(np.clip(g @ e1, -1, 1)), np.mod(np.arctan2(g @ e3, g @ e2), 2 * np.pi))
        return float(np.dot(n, d))
    return R, slope, cos_normal


def arrival_height(R, D, h0):
    """Apex height at which the front reaches distance D [R_sun] on the spacecraft ray."""
    if not np.isfinite(R(h0)):
        return np.nan
    lo, hi = h0, 2.0 * D
    while R(hi) < D and hi < 1e5:
        hi *= 2.0
    if not np.isfinite(R(hi)) or R(hi) < D:
        return np.nan
    for _ in range(60):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if R(m) < D else (lo, m)
    return 0.5 * (lo + hi)


def start_state(method):
    """The corona series' last step: (model, time, apex speed, file name, series)."""
    series = sp.load_series(method, 'corona')
    if len(series) < 2:
        return None, None, None, None, None
    last = series[-1]
    return last['model'], last['time'], float(last['speed_kms']), last['name'], series


def heights(h0, h_end):
    hs = list(np.arange(h0, h_end + 1e-9, H_STEP))
    if h_end - hs[-1] > 1e-6:
        hs.append(h_end)
    return [float(h) for h in hs]


def run(method, bg):
    base, t0, v0, src, series = start_state(method)
    if base is None:
        print(f'[{method}] no coronal series: run derive_para_corona.py first')
        return None
    h0 = base.height
    print(f'\n[{method}] start: {src}  h = {h0:.2f} R_sun at {t0:%Y-%m-%d %H:%M:%S} UT, apex speed {v0:.0f} km/s')
    fit = None if REFIT_DRAG else df.load(method)
    if fit is None:
        fit = df.fit(method, bg, series)
    t_ray, R_ray, V_ray, start, wind = df.ray_solution(fit, bg, series)
    lo, la = np.radians(fit['ray']['hgs_lon_deg']), np.radians(fit['ray']['hgs_lat_deg'])
    d = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    gamma = fit['gamma0_per_km']
    R_h, slope_h, cos_h = sc_geometry(base, d)
    # the apex height whose self-similar surface meets the ray at distance R: invert R_h(h) (monotonic)
    h_tab = np.geomspace(h0, 40.0 * h0, 600)
    R_tab = np.array([R_h(h) for h in h_tab])
    ok = np.isfinite(R_tab)
    h_tab, R_tab = h_tab[ok], R_tab[ok]
    h_of_R = lambda R: np.interp(R, R_tab, h_tab)
    h_t = h_of_R(R_ray)                                            # apex height along the ray's time axis
    v_apex_t = np.gradient(h_t, t_ray) * RSUN_KM                   # apex speed [km/s]
    h_end = H_END
    obs = fit.get('observed')
    h_arr = h_of_R(obs['sc_distance_au'] * AU_RSUN) if obs else np.nan
    if EXTEND_TO_SC and obs and h_arr > H_END:
        h_end = float(h0 + H_STEP * np.ceil((h_arr - h0) / H_STEP))
        print(f'  extended to {h_end:.1f} R_sun: the front passes the spacecraft when the apex is at {h_arr:.1f} R_sun')
    print(f'  evolution with gamma0 = {gamma:.3g} /km on the ray lon {fit["ray"]["hgs_lon_deg"]:.0f}, '
          f'lat {fit["ray"]["hgs_lat_deg"]:.0f} ({fit["ray"]["along"]}); the shape scaled self-similarly')
    rows, steps = [], []
    for h in heights(h0, h_end):
        if h > h_t[-1]:
            print(f'  the ray solution ends before the apex reaches {h:.1f} R_sun (T_MAX_DAYS in drag_fit.py)')
            break
        t = float(np.interp(h, h_t, t_ray))
        v = float(np.interp(t, t_ray, v_apex_t))
        when = t0 + dt.timedelta(seconds=t)
        m = at_height(base, h, time=when.strftime('%Y-%m-%dT%H:%M:%S'))
        r_sc = float(np.interp(t, t_ray, R_ray))
        v_rad = float(np.interp(t, t_ray, V_ray))
        v_nrm = v_rad * cos_h(h)
        derived = {'method': f'ip_sse_{method}', 'height_rsun': round(h, 3), 'time': m.time, 'speed_kms': round(v, 1),
                   'gamma0_per_km': gamma, 'ray_hgs_lon_lat_deg': [fit['ray']['hgs_lon_deg'], fit['ray']['hgs_lat_deg']],
                   'start': src, 'in_fitted_range': False,
                   'sc_ray_distance_rsun': round(r_sc, 3), 'sc_ray_speed_radial_kms': round(v_rad, 1),
                   'sc_ray_speed_normal_kms': round(v_nrm, 1),
                   'notes': [f'SSE: PyThea self-similar shape from the {h0:g} R_sun surface; kinematics of the ray in the '
                             f'{fit["ray"]["along"]} direction, DBM with the real background (drag_fit.py)']}
        path = cfg.final_path(f'shock_para_ip{int(round(h * 10)):04d}_{method}.json')
        path.write_text(json.dumps({**m.to_dict(), 'derived': derived}, indent=2))
        steps.append((when, h, v, r_sc, v_rad, v_nrm))
        rows.append({'height_rsun': f'{h:.2f}', 'time': m.time, 'apex_speed_kms': f'{v:.1f}',
                     'sc_ray_distance_rsun': f'{r_sc:.3f}', 'sc_ray_speed_radial_kms': f'{v_rad:.1f}',
                     'sc_ray_speed_normal_kms': f'{v_nrm:.1f}'})
    table = cfg.final_path(f'shock_para_ip_{method}.csv')
    with open(table, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    report = {'method': method, 'start': src, 'start_time': t0.isoformat(), 'start_height_rsun': h0,
              'start_speed_kms': v0, 'drag_fit': fit, 'apex_height_at_arrival_rsun': None if not obs else float(h_arr)}
    cfg.final_path(f'sse_fit_{method}.json').write_text(json.dumps(report, indent=2, default=float))
    print(f'  saved {len(steps)} files shock_para_ip<hhhh>_{method}.json, {table.name}, sse_fit_{method}.json')
    figure(method, steps, fit, t0, t_ray, h_t, v_apex_t, R_ray, V_ray)
    return report


def figure(method, steps, fit, t0, t_ray, h_t, v_apex_t, R_ray, V_ray):
    obs = fit.get('observed')
    fig, ax = plt.subplots(1, 3 if obs else 2, figsize=(17 if obs else 12, 4.8), layout='constrained')
    tt = np.array([t0 + dt.timedelta(seconds=float(x)) for x in t_ray])
    keep = tt <= steps[-1][0] + dt.timedelta(hours=3)
    ax[0].plot(tt[keep], h_t[keep], 'k-', label='apex (self-similar scaling)')
    ax[0].plot(tt[keep], R_ray[keep], '-', color='C0', label=f'front on the ray ({fit["ray"]["along"]} direction)')
    ax[0].plot([s[0] for s in steps], [s[1] for s in steps], 'k.', ms=4)
    if obs:
        ax[0].plot([dt.datetime.fromisoformat(obs['arrival_time'])], [obs['sc_distance_au'] * AU_RSUN], 'o', color='C3',
                   ms=8, label='observed arrival')
    ax[0].set_ylabel('[R$_\\odot$]')
    ax[0].set_title(f'{method}: gamma0 = {fit["gamma0_per_km"]:.2g} /km')
    ax[0].legend(fontsize=8)
    ax[1].plot(tt[keep], v_apex_t[keep], 'k-', label='apex')
    ax[1].plot([s[0] for s in steps], [s[5] if df.ARRIVAL_SPEED_TYPE == 'normal' else s[4] for s in steps], 'o-',
               color='C0', ms=3, label=f'front on the ray ({df.ARRIVAL_SPEED_TYPE})')
    if obs:
        ax[1].plot([dt.datetime.fromisoformat(obs['arrival_time'])], [obs['arrival_speed_kms']], 'o', color='C3', ms=8,
                   label='observed')
    ax[1].set_ylabel('[km/s]')
    ax[1].set_title('speeds')
    ax[1].legend(fontsize=8)
    for a in ax[:2]:
        a.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
        a.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        a.set_xlabel('time [UT]')
    if obs:
        cc = fit['cost_curve']
        ax[2].loglog(cc['gamma'], cc['cost'], 'k-')
        ax[2].axvline(fit['gamma0_per_km'], color='C3', lw=1, label=f'best {fit["gamma0_per_km"]:.2g} /km')
        for gg, lab, col in ((fit['gamma_time_only'], 'time alone', 'C0'), (fit['gamma_speed_only'], 'speed alone', 'C2')):
            for k, x in enumerate(gg):
                ax[2].axvline(x, color=col, ls='--', lw=1, label=lab if k == 0 else None)
        ax[2].set_xlabel('gamma0 [1/km]')
        ax[2].set_ylabel('cost')
        ax[2].set_title('drag fit (drag_fit.py)')
        ax[2].legend(fontsize=8)
    fig.suptitle(f'{cfg.EVENT_ID}: SSE interplanetary evolution ({method})')
    png = cfg.figure_path(f'derive_para_ip_{method}.png')
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f'  saved {png}')


def main():
    bg = bk.load_background()
    print(bg.summary() if hasattr(bg, 'summary') else bg.name)
    for method in METHODS:
        try:
            rep = run(method, bg)
            getattr(sp, 'record_failure', lambda *a: None)('ip', method, None if rep else 'no coronal series for this method: run derive_para_corona.py first')
        except RuntimeError as e:                                  # a method that fails is reported, the rest run
            print(f'\n[{method}] not propagated: {e}')
            getattr(sp, 'record_failure', lambda *a: None)('ip', method, str(e))


if __name__ == '__main__':
    main()
