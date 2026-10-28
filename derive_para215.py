"""
derive_para215.py -- shock parameters at the heliocentric height H_TARGET (21.5 R_sun, the usual inner boundary
of WSA-ENLIL) from all fitted times.

Run:  python derive_para215.py           (settings below; folders from event_config.py)
Reads every  fitted results/<id>/semi/shock_fit_<stamp>.json  (time taken from the file name).

1. OBSERVED   some fit reaches H_TARGET:
     'interpolate'  linear interpolation in height between the two fits that bracket H_TARGET
     'mean'         average of the N_NEAREST fits nearest to H_TARGET in height
2. PREDICTED  every fit is below H_TARGET:
     'constant'     linear height-time fit -> one constant speed and the time H_TARGET is reached; kappa,
                    epsilon, alpha, lat, lon, tilt averaged over the fits (constant kappa, eps, alpha is PyThea's
                    self-similar expansion)
     'varied'       2nd-order height-time fit (as in the CDAW catalog) -> speed and acceleration at H_TARGET;
                    direction (lat, lon, tilt) from its linear trend in height (DIRECTION_VARIED, PARAM_FIT_VS);
                    shape (kappa, eps, alpha) by default from the latest fit (SHAPE_VARIED): constant kappa, eps,
                    alpha is PyThea's self-similar assumption, and a shape trend from a few fits extrapolates
                    badly ('trend' is available, clipped to valid ranges, with a warning). If the quadratic peaks
                    below H_TARGET (strong deceleration), the shock is continued from the last fit at the
                    quadratic's speed there.
Dents/bulges: those of the fit nearest in height (FEATURES = 'nearest') or none ('none'). Features found at
different times are different features, so they are never averaged.
Angles: longitudes are averaged on the circle, tilts with their 180-deg period.

Writes  fitted results/<id>/semi/shock_215_<method>.json  (PyThea parameters, readable by the GUI's Load, plus a
'derived' block: time and speed at H_TARGET, acceleration, sources, semi-axes, cone-equivalent half widths),
fitted results/<id>/semi/shock_215_summary.csv  and  figure/derive_para215.png.
"""
import csv
import datetime as dt
import json
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import event_config as cfg
from deformable_shock import DeformableShock

# ================================ SETTINGS ================================
H_TARGET = 21.5                   # [R_sun]
PREDICT = ['constant', 'varied']  # prediction method(s) when every fit is below H_TARGET
OBS_METHOD = 'interpolate'        # 'interpolate' or 'mean' when the fits reach H_TARGET
N_NEAREST = 2                     # fits averaged by OBS_METHOD = 'mean'
PARAM_FIT_VS = 'height'           # 'varied': trends fitted against 'height' (less leverage) or 'time'
SHAPE_VARIED = 'latest'           # 'varied': kappa, eps, alpha from the 'latest' fit, their 'mean', or their 'trend'
DIRECTION_VARIED = 'trend'        # 'varied': lat, lon, tilt from their 'trend' (linear fit), 'latest' or 'mean'
FEATURES = 'nearest'              # dents/bulges of the fit nearest in height ('nearest') or 'none'
EXCLUDE = []                      # stamps to leave out, e.g. ['20120307_0039']
# ==========================================================================
RSUN_KM = 695700.0
PARS = ['kappa', 'epsilon', 'alpha', 'lat', 'lon', 'tilt']
PERIOD = {'lon': 360.0, 'tilt': 180.0}                 # angles that wrap
LIMITS = {'kappa': (0.01, 5.0), 'epsilon': (-0.95, 0.95), 'alpha': (0.1, 10.0), 'lat': (-90.0, 90.0)}


# ----------------------------------------------------------------------------- helpers
def load_fits():
    fits = []
    for p in sorted(cfg.RESULTS_DIR.glob('shock_fit_*.json')):
        m = re.match(r'^shock_fit_(\d{8}_\d{4})\.json$', p.name)
        if not m or m.group(1) in EXCLUDE:
            continue
        d = json.loads(p.read_text())
        t = cfg.stamp_date(m.group(1))
        if d.get('time'):
            t_in = dt.datetime.fromisoformat(d['time'])
            if abs((t_in - t).total_seconds()) > 60:
                print(f'warning: {p.name} says "time": "{d["time"]}" inside, but its name says '
                      f'{t:%Y-%m-%d %H:%M}; using the name')
        model = DeformableShock.from_dict(d)
        fits.append({'name': p.name, 'stamp': m.group(1), 'time': t, 'model': model,
                     'h': model.height, **{k: getattr(model, k) for k in PARS}})
    fits.sort(key=lambda f: f['time'])
    return fits


def wrap(v, period):
    return v % 360.0 if period == 360.0 else (v + 90.0) % 180.0 - 90.0


def unwrap(values, period):
    a = np.radians(np.asarray(values, float) * 360.0 / period)
    return np.degrees(np.unwrap(a)) * period / 360.0


def mean_par(name, values, weights=None):
    v = np.asarray(values, float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)
    if name in PERIOD:
        a = np.radians(v * 360.0 / PERIOD[name])
        m = np.degrees(np.arctan2(np.sum(w * np.sin(a)), np.sum(w * np.cos(a)))) * PERIOD[name] / 360.0
        return wrap(m, PERIOD[name])
    return float(np.sum(w * v) / np.sum(w))


def clip(name, v, notes):
    if name in LIMITS:
        lo, hi = LIMITS[name]
        if not lo <= v <= hi:
            notes.append(f'{name} extrapolated to {v:.3f}, clipped to [{lo}, {hi}]')
            return float(np.clip(v, lo, hi))
    return float(v)


def features_near(fits, h):
    if FEATURES == 'none':
        return []
    f = min(fits, key=lambda f: abs(f['h'] - h))
    return [list(d) for d in f['model'].dents]


def cone_half_widths(model):
    """Half widths [deg] seen from Sun centre along b and c (tangent lines); None if the Sun centre is inside."""
    a, b, c, rc = model.axes
    if rc <= a:
        return None
    s = np.sqrt(rc ** 2 - a ** 2)
    return {'b': float(np.degrees(np.arctan(b / s))), 'c': float(np.degrees(np.arctan(c / s)))}


def make_result(method, pars, t_target, speed_kms, extra, fits, notes):
    model = DeformableShock.from_pythea(H_TARGET, pars['kappa'], pars['epsilon'], pars['alpha'], pars['lat'],
                                        pars['lon'], pars['tilt'], dents=features_near(fits, H_TARGET),
                                        time=t_target.strftime('%Y-%m-%dT%H:%M:%S'))
    a, b, c, rc = model.axes
    cone = cone_half_widths(model)
    derived = {'method': method, 'target_height_rsun': H_TARGET, 'time_at_target': model.time,
               'speed_kms': round(speed_kms, 1), **extra,
               'semi_axes_rsun': {'a': round(a, 3), 'b': round(b, 3), 'c': round(c, 3), 'center': round(rc, 3)},
               'cone_half_width_deg': ({k: round(v, 1) for k, v in cone.items()} if cone else
                                       'Sun centre inside the ellipsoid (wider than a 90-deg cone)'),
               'features_from': ('none' if FEATURES == 'none' else
                                 min(fits, key=lambda f: abs(f['h'] - H_TARGET))['name']),
               'source_fits': [f['name'] for f in fits], 'notes': notes}
    return {'method': method, 'model': model, 'derived': derived}


# ----------------------------------------------------------------------------- the two cases
def observed(fits):
    notes = []
    below = [f for f in fits if f['h'] < H_TARGET]
    above = [f for f in fits if f['h'] >= H_TARGET]
    if OBS_METHOD == 'interpolate' and below and above:
        lo = max(below, key=lambda f: f['h'])
        hi = min(above, key=lambda f: f['h'])
        w = (H_TARGET - lo['h']) / (hi['h'] - lo['h'])
        pars = {}
        for k in PARS:
            y = [lo[k], hi[k]]
            if k in PERIOD:
                y = unwrap(y, PERIOD[k])
                pars[k] = wrap(y[0] + w * (y[1] - y[0]), PERIOD[k])
            else:
                pars[k] = y[0] + w * (y[1] - y[0])
        span = (hi['time'] - lo['time']).total_seconds()
        t_target = lo['time'] + dt.timedelta(seconds=w * span)
        speed = (hi['h'] - lo['h']) * RSUN_KM / span
        notes.append(f'interpolated between {lo["name"]} ({lo["h"]:.2f} R_sun) and {hi["name"]} ({hi["h"]:.2f} R_sun)')
        return [make_result('observed_interpolate', pars, t_target, speed, {}, [lo, hi], notes)]
    use = sorted(fits, key=lambda f: abs(f['h'] - H_TARGET))[:max(1, N_NEAREST)]
    use.sort(key=lambda f: f['time'])
    pars = {k: mean_par(k, [f[k] for f in use]) for k in PARS}
    if len(use) >= 2:
        ts = np.array([(f['time'] - use[0]['time']).total_seconds() for f in use])
        s, c = np.polyfit(ts, [f['h'] for f in use], 1)
        t_target = use[0]['time'] + dt.timedelta(seconds=(H_TARGET - c) / s)
        speed = s * RSUN_KM
    else:
        t_target, speed = use[0]['time'], float('nan')
        notes.append('one fit only: no speed')
    notes.append(f'average of the {len(use)} fits nearest to {H_TARGET} R_sun: ' + ', '.join(f['name'] for f in use))
    return [make_result('observed_mean', pars, t_target, speed, {}, use, notes)]


def predicted(fits):
    t0 = fits[0]['time']
    ts = np.array([(f['time'] - t0).total_seconds() for f in fits])
    hs = np.array([f['h'] for f in fits])
    out = []
    if 'constant' in PREDICT:
        notes = []
        if len(fits) > 2:
            (s, c), cov = np.polyfit(ts, hs, 1, cov=True)
            s_err = float(np.sqrt(cov[0, 0])) * RSUN_KM
        else:
            s, c = np.polyfit(ts, hs, 1)
            s_err = float('nan')
            notes.append('two fits: speed has no uncertainty estimate')
        if s <= 0:
            notes.append('the linear fit does not rise: no prediction')
        else:
            t_target = t0 + dt.timedelta(seconds=(H_TARGET - c) / s)
            pars = {k: mean_par(k, [f[k] for f in fits]) for k in PARS}
            notes.append(f'linear height-time fit over {len(fits)} fits; parameters averaged')
            out.append(make_result('constant', pars, t_target, s * RSUN_KM,
                                   {'speed_err_kms': round(s_err, 1) if np.isfinite(s_err) else None}, fits, notes))
    if 'varied' in PREDICT:
        notes = []
        if len(fits) < 3:
            notes.append('fewer than 3 fits: the 2nd-order fit reduces to a linear one')
            q = np.r_[0.0, np.polyfit(ts, hs, 1)]
        else:
            q = np.polyfit(ts, hs, 2)
            if len(fits) == 3:
                notes.append('3 fits: the 2nd-order curve passes through all of them (no uncertainty estimate)')
        roots = np.roots(q - np.r_[0.0, 0.0, H_TARGET]) if q[0] != 0 else np.array([(H_TARGET - q[2]) / q[1]])
        roots = np.sort([r.real for r in np.atleast_1d(roots) if abs(r.imag) < 1e-9 and r.real >= ts[-1] - 1e-6])
        v_of = lambda t: 2 * q[0] * t + q[1]
        if roots.size and v_of(roots[0]) > 0:
            t_s = float(roots[0])
            speed = v_of(t_s) * RSUN_KM
        else:
            v_last = v_of(ts[-1])
            h_peak = q[2] - q[1] ** 2 / (4 * q[0]) if q[0] < 0 else float('nan')
            if v_last > 0:
                t_s = ts[-1] + (H_TARGET - np.polyval(q, ts[-1])) / v_last
                speed = v_last * RSUN_KM
                notes.append(f'the 2nd-order curve peaks at {h_peak:.2f} R_sun, below {H_TARGET}: continued from the '
                             f'last fit at the curve\'s speed there ({speed:.0f} km/s, no further deceleration)')
            else:
                notes.append('the 2nd-order curve is already falling at the last fit: no prediction')
                t_s = None
        if t_s is not None:
            x_all = ts if PARAM_FIT_VS == 'time' else hs
            x_t = t_s if PARAM_FIT_VS == 'time' else H_TARGET
            deg = min(1, len(fits) - 1)
            span = x_all[-1] - x_all[0]
            pars = {}
            for k in PARS:
                how = SHAPE_VARIED if k in ('kappa', 'epsilon', 'alpha') else DIRECTION_VARIED
                if how == 'latest':
                    pars[k] = float(fits[-1][k])
                elif how == 'mean':
                    pars[k] = mean_par(k, [f[k] for f in fits])
                else:
                    y = np.array([f[k] for f in fits])
                    if k in PERIOD:
                        y = unwrap(y, PERIOD[k])
                    v = float(np.polyval(np.polyfit(x_all, y, deg), x_t)) if deg else float(y[0])
                    pars[k] = wrap(v, PERIOD[k]) if k in PERIOD else clip(k, v, notes)
            unit = 'min' if PARAM_FIT_VS == 'time' else 'R_sun'
            scale = 60.0 if PARAM_FIT_VS == 'time' else 1.0
            reach = (x_t - x_all[-1]) / scale
            notes.append(f'2nd-order height-time fit; direction: {DIRECTION_VARIED}, shape: {SHAPE_VARIED}'
                         + (f' (trends linear in {PARAM_FIT_VS}, extrapolated {reach:.1f} {unit} beyond the last fit'
                            f', {reach / max(span / scale, 1e-9):.1f}x the fitted span)'
                            if 'trend' in (DIRECTION_VARIED, SHAPE_VARIED) else ''))
            if any(n.startswith(('kappa', 'epsilon', 'alpha')) and 'clipped' in n for n in notes):
                notes.append('WARNING: the shape trend left the valid range -- the derived shape is not trustworthy; '
                             'use SHAPE_VARIED = latest or mean')
            out.append(make_result('varied', pars, t0 + dt.timedelta(seconds=t_s), speed,
                                   {'acceleration_ms2': round(2 * q[0] * RSUN_KM * 1e3, 1)}, fits, notes))
    return out, q if 'varied' in PREDICT else None


# ----------------------------------------------------------------------------- output
def save(results):
    rows = []
    tag = f'{int(round(H_TARGET * 10)):03d}'
    for r in results:
        d = {**r['model'].to_dict(), 'derived': r['derived']}
        path = cfg.results_path(f'shock_{tag}_{r["method"]}.json')
        path.write_text(json.dumps(d, indent=2))
        m = r['model']
        rows.append({'method': r['method'], 'time_at_target': m.time, 'height': H_TARGET,
                     'speed_kms': r['derived']['speed_kms'], 'kappa': f'{m.kappa:.4f}', 'epsilon': f'{m.epsilon:.4f}',
                     'alpha': f'{m.alpha:.4f}', 'lat': f'{m.lat:.2f}', 'lon': f'{m.lon:.2f}', 'tilt': f'{m.tilt:.2f}',
                     'a': f'{m.a:.3f}', 'b': f'{m.b:.3f}', 'c': f'{m.c:.3f}', 'n_features': len(m.dents),
                     'notes': ' | '.join(r['derived']['notes'])})
        print(f'saved {path}')
    csv_path = cfg.results_path(f'shock_{tag}_summary.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'saved {csv_path}')


def figure(fits, results, q):
    t0 = fits[0]['time']
    tt = [f['time'] for f in fits]
    ts = np.array([(t - t0).total_seconds() for t in tt])
    hs = np.array([f['h'] for f in fits])
    t_res = {r['method']: dt.datetime.fromisoformat(r['model'].time) for r in results}
    t_end = max(list(t_res.values()) + [tt[-1]]) + dt.timedelta(minutes=5)
    sec = lambda t: (t - t0).total_seconds()
    grid = lambda a, b: [a + (b - a) * x for x in np.linspace(0, 1, 120)]
    colors = {'constant': 'C0', 'varied': 'C3', 'observed_interpolate': 'C2', 'observed_mean': 'C2'}
    marker = {'constant': 's', 'varied': 'D', 'observed_interpolate': 'o', 'observed_mean': 'o'}
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5), layout='constrained')
    a_h, a_v = ax[0, 0], ax[0, 1]
    a_h.plot(tt, hs, 'ko', ms=7, label='fits')
    if len(fits) >= 2:
        lin = np.polyfit(ts, hs, 1)
        g = grid(t0, t_end)
        a_h.plot(g, np.polyval(lin, [sec(x) for x in g]), '--', color='C0', lw=1.2, label='linear fit')
        mid = [tt[i] + (tt[i + 1] - tt[i]) / 2 for i in range(len(tt) - 1)]
        a_v.plot(mid, np.diff(hs) / np.diff(ts) * RSUN_KM, 'ko', ms=6, label='between fits')
        a_v.plot([t0, t_end], [lin[0] * RSUN_KM] * 2, '--', color='C0', lw=1.2, label='linear fit')
    if q is not None and len(fits) >= 3 and 'varied' in t_res:
        tv = t_res['varied']
        reaches = abs(np.polyval(q, sec(tv)) - H_TARGET) < 1e-3          # the curve itself reaches the target
        t_q = tv if reaches else tt[-1]
        g = grid(t0, t_q)
        a_h.plot(g, np.polyval(q, [sec(x) for x in g]), '-', color='C3', lw=1.4, label='2nd-order fit (as used)')
        a_v.plot(g, [(2 * q[0] * sec(x) + q[1]) * RSUN_KM for x in g], '-', color='C3', lw=1.4, label='2nd-order fit')
        if not reaches:                                                   # fallback: straight on from the last fit
            v_last = 2 * q[0] * sec(tt[-1]) + q[1]
            a_h.plot([tt[-1], tv], [hs[-1], H_TARGET], ':', color='C3', lw=1.6, label='continued at the last speed')
            a_v.plot([tt[-1], tv], [v_last * RSUN_KM] * 2, ':', color='C3', lw=1.6)
    a_h.axhline(H_TARGET, color='0.5', lw=0.8, ls=':')
    for r in results:
        a_h.plot(t_res[r['method']], H_TARGET, marker[r['method']], color=colors[r['method']], ms=9, mfc='none',
                 mew=2, label=f'{r["method"]}: {r["derived"]["speed_kms"]:.0f} km/s')
        a_v.plot(t_res[r['method']], r['derived']['speed_kms'], marker[r['method']], color=colors[r['method']],
                 ms=9, mfc='none', mew=2)
    a_h.set_ylabel('apex height [R$_\\odot$]')
    a_h.set_title(f'apex height; derived at {H_TARGET} R$_\\odot$')
    a_v.set_ylabel('apex speed [km/s]')
    a_v.set_title('apex speed')
    a_h.legend(fontsize=8)
    a_v.legend(fontsize=8)
    panels = [(ax[0, 2], ['kappa', 'epsilon', 'alpha'], 'shape'), (ax[1, 0], ['lat'], 'latitude [deg]'),
              (ax[1, 1], ['lon'], 'Stonyhurst longitude [deg]'), (ax[1, 2], ['tilt'], 'tilt [deg]')]
    for a, names, title in panels:
        for j, k in enumerate(names):
            y = np.array([f[k] for f in fits])
            if k in PERIOD:
                y = unwrap(y, PERIOD[k])
            col = f'C{j}' if len(names) > 1 else 'k'
            a.plot(tt, y, 'o', color=col, ms=6, label=k if len(names) > 1 else 'fits')
            for r in results:
                v = getattr(r['model'], k)
                if k in PERIOD:                                       # put the result next to the data
                    v = v + PERIOD[k] * np.round((np.mean(y) - v) / PERIOD[k])
                a.plot(t_res[r['method']], v, marker[r['method']], color=col if len(names) > 1 else colors[r['method']],
                       ms=8, mfc='none', mew=1.8, label=None if len(names) > 1 else r['method'])
        a.set_title(title + ('  (squares: constant, diamonds: varied)' if names[0] == 'kappa' else ''))
        a.legend(fontsize=8)
    for a in ax.ravel():
        a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        a.set_xlabel('time [UT]')
    png = cfg.figure_path(f'derive_para{int(round(H_TARGET * 10)):03d}.png')
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f'saved {png}')


def main():
    fits = load_fits()
    if not fits:
        sys.exit(f'no shock_fit_<stamp>.json in "{cfg.RESULTS_DIR}"')
    print(f'{len(fits)} fits:')
    for f in fits:
        print(f'  {f["time"]:%Y-%m-%d %H:%M}  h = {f["h"]:6.2f} R_sun  kappa {f["kappa"]:.3f}  eps {f["epsilon"]:+.3f}  '
              f'alpha {f["alpha"]:.3f}  lat {f["lat"]:+6.1f}  lon {f["lon"]:6.1f}  tilt {f["tilt"]:+6.1f}  '
              f'features {len(f["model"].dents)}')
    q = None
    if any(f['h'] >= H_TARGET for f in fits):
        print(f'\nthe fits reach {H_TARGET} R_sun: OBSERVED ({OBS_METHOD})')
        results = observed(fits)
    else:
        if len(fits) < 2:
            sys.exit(f'one fit below {H_TARGET} R_sun: at least two times are needed to predict')
        print(f'\nevery fit is below {H_TARGET} R_sun: PREDICTED ({", ".join(PREDICT)})')
        results, q = predicted(fits)
    if not results:
        sys.exit('no result (see notes above)')
    for r in results:
        d, m = r['derived'], r['model']
        cone = d['cone_half_width_deg']
        print(f'\n[{r["method"]}]  at {H_TARGET} R_sun: {d["time_at_target"].replace("T", " ")} UT, '
              f'speed {d["speed_kms"]:.0f} km/s'
              + (f' +- {d["speed_err_kms"]:.0f}' if d.get('speed_err_kms') else '')
              + (f', acceleration {d["acceleration_ms2"]:.0f} m/s^2' if 'acceleration_ms2' in d else ''))
        print(f'  kappa {m.kappa:.3f}, eps {m.epsilon:+.3f}, alpha {m.alpha:.3f}, lat {m.lat:+.1f}, lon {m.lon:.1f}, '
              f'tilt {m.tilt:+.1f}  ->  a, b, c = {m.a:.2f}, {m.b:.2f}, {m.c:.2f} R_sun, centre at {m.axes[3]:.2f}')
        print('  cone-equivalent half widths: ' + (f'{cone["b"]:.1f} deg (b), {cone["c"]:.1f} deg (c)'
                                                  if isinstance(cone, dict) else cone))
        for n in d['notes']:
            print(f'  note: {n}')
    print()
    save(results)
    figure(fits, results, q)


if __name__ == '__main__':
    main()
