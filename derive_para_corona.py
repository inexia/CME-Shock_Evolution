"""
derive_para_corona.py -- shock parameters at apex heights H_START ... H_END (default 1.5 ... 21.5 R_sun,
heliocentric) in steps of H_STEP, from all fitted times: the shock's evolution through the corona.

Run:  python derive_para_corona.py        (settings below; folders from event_config.py)
Reads every  fitted results/<id>/semi/shock_fit_<stamp>.json  (time taken from the file name).

Methods (each evaluated at every height; at 21.5 R_sun 'constant' and 'varied' equal derive_para215.py):
  constant   linear height-time fit -> one speed; time at each height from the fit; kappa, epsilon, alpha,
             lat, lon, tilt averaged over the fits
  varied     2nd-order height-time fit (as in the CDAW catalog) -> time, speed and acceleration at each height
             (rising branch of the curve; where the curve cannot reach a height -- it peaks below it, or bottoms
             out above it -- the shock is continued from the last / first fit at the curve's speed there);
             direction (lat, lon, tilt) from its linear trend (DIRECTION_VARIED, PARAM_FIT_VS); shape (kappa,
             epsilon, alpha) interpolated between the fits in height and held constant beyond them
             (SHAPE_VARIED = 'interp': PyThea's self-similar shape outside the data)
  observed   interpolation between the two fits that bracket each height; only heights inside the fitted range
  nearest    the irregular front of the fits at every height: interpolated between the two bracketing fits inside
             the fitted range; beyond it, the nearest (first or last) fit scaled self-similarly to the height --
             kappa, epsilon, alpha, direction, tilt and the dents/bulges kept, every length from the surface point
             below the apex grown with h - 1. Time and speed from the height-time fit (NEAREST_KINEMATICS).
Fits the anchored fit_event.py reports as poorly constrained (EXCLUDE_IF_PRIOR_SET) are left out, so they serve
no method; EXCLUDE lists more stamps by hand.
Every step records whether it lies inside the fitted height range and, if not, how far it extrapolates.
Dents/bulges: those of the fit nearest in height (FEATURES = 'nearest'; they switch half-way between two fits)
or none ('none'). They are never averaged: features of different times are different features.

Writes  fitted results/<id>/final/shock_para_corona<hhh>_<method>.json  (hhh = height x 10: 015 ... 215; PyThea
parameters, readable by the GUI's Load, plus a 'derived' block), one table per method
fitted results/<id>/final/shock_para_corona_<method>.csv,  and  figure/derive_para_corona.png.
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
H_START, H_END, H_STEP = 1.5, 30., 1.0     # apex heights [R_sun, heliocentric]
FIT_LINE_FROM = 2.0                          # figure: the height-time fits drawn from this height to H_END
METHODS = ['constant', 'varied', 'observed', 'nearest']
NEAREST_KINEMATICS = 'varied'     # 'nearest': time and speed from the 2nd-order ('varied') or linear ('constant') fit
EXCLUDE_IF_PRIOR_SET = 3          # leave out a fit whose report (fit_event.py, anchored) shows at least this many of
                                  # its 7 ellipsoid parameters set mostly by the prior (parameters fixed with FIX do not
                                  # count); never empties the series: with fewer than 2 left, all are kept. None: keep all
PARAM_FIT_VS = 'height'           # 'varied' trends fitted against 'height' or 'time'
SHAPE_VARIED = 'interp'           # 'varied' kappa, eps, alpha: 'interp' (between fits, held beyond), 'latest', 'mean', 'trend'
DIRECTION_VARIED = 'trend'        # 'varied' lat, lon, tilt: 'trend', 'interp', 'latest' or 'mean'
FEATURES = 'nearest'              # dents/bulges of the fit nearest in height, or 'none'
EXCLUDE = []                      # stamps to leave out, e.g. ['20120307_0039']
USE_GUI_POINTS = True             # GUI saves (shock_<stamp>.json) at times without a usable fit count as measured points
                                  # for the kinematics and the 'constant' / 'varied' trends ('observed' / 'nearest' use
                                  # the full fits only: they carry the fitted irregular front)
# ==========================================================================
RSUN_KM = 695700.0
PARS = ['kappa', 'epsilon', 'alpha', 'lat', 'lon', 'tilt']
SHAPE = ('kappa', 'epsilon', 'alpha')
PERIOD = {'lon': 360.0, 'tilt': 180.0}
LIMITS = {'kappa': (0.01, 5.0), 'epsilon': (-0.95, 0.95), 'alpha': (0.1, 10.0), 'lat': (-90.0, 90.0)}


# ----------------------------------------------------------------------------- fits and angle helpers
def load_fits():
    fits, excluded = [], []
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
        fit = {'name': p.name, 'time': t, 'model': model, 'h': model.height, **{k: getattr(model, k) for k in PARS}}
        by_prior = [k for k, c in (d.get('constraints') or {}).items() if c.get('set_by') == 'prior']   # not 'fixed'
        if EXCLUDE_IF_PRIOR_SET is not None and len(by_prior) >= EXCLUDE_IF_PRIOR_SET:
            excluded.append((fit, by_prior))
            continue
        fits.append(fit)
    if excluded and len(fits) < 2:                          # the rule may not empty the series: a limitation every
        print(f'warning: {len(excluded)} fit(s) have >= {EXCLUDE_IF_PRIOR_SET} parameters set mostly by the prior, '
              f'but excluding them would leave {len(fits)}; all kept. A limitation shared by the times is likely the '
              f"event's geometry (e.g. a halo with close viewpoints), not a bad fit:")
        for f, ks in excluded:
            print(f'  {f["name"]}: {", ".join(ks)} set by the prior')
        fits += [f for f, _ in excluded]
    else:
        for f, ks in excluded:
            print(f'excluded {f["name"]}: {", ".join(ks)} set mostly by the prior -- the traces hardly constrain them '
                  f'(EXCLUDE_IF_PRIOR_SET = {EXCLUDE_IF_PRIOR_SET}; fixed parameters do not count)')
    fits.sort(key=lambda f: f['time'])
    return fits


def load_gui_points(fits):
    """GUI saves (shock_<stamp>.json) at times without a usable fit, as measured points."""
    used = {f['time'] for f in fits}
    pts = []
    for p in sorted(cfg.RESULTS_DIR.glob('shock_*.json')):
        m = re.match(r'^shock_(\d{8}_\d{4})\.json$', p.name)
        if not m or m.group(1) in EXCLUDE:
            continue
        t = cfg.stamp_date(m.group(1))
        if t in used:
            continue
        model = DeformableShock.from_dict(json.loads(p.read_text()))
        pts.append({'name': p.name, 'time': t, 'model': model, 'h': model.height, 'source': 'gui',
                    **{k: getattr(model, k) for k in PARS}})
    return pts


def wrap(v, period):
    return v % 360.0 if period == 360.0 else (v + 90.0) % 180.0 - 90.0


def unwrap(values, period):
    return np.degrees(np.unwrap(np.radians(np.asarray(values, float) * 360.0 / period))) * period / 360.0


def mean_par(k, values):
    v = np.asarray(values, float)
    if k in PERIOD:
        a = np.radians(v * 360.0 / PERIOD[k])
        return wrap(np.degrees(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))) * PERIOD[k] / 360.0, PERIOD[k])
    return float(np.mean(v))


# ----------------------------------------------------------------------------- the model
class Evolution:
    """Kinematics and parameter laws from the fits; everything evaluated per height."""

    def __init__(self, fits, points=None):
        self.fits = fits                                    # full fits: interpolation, features, the fitted range
        self.points = sorted(points or fits, key=lambda f: f['time'])   # all measured times: kinematics, trends
        self.t0 = self.points[0]['time']
        self.ts = np.array([(f['time'] - self.t0).total_seconds() for f in self.points])
        self.hs = np.array([f['h'] for f in self.points])
        self.k_by_h = sorted(self.points, key=lambda f: f['h'])
        self.k_lo, self.k_hi = self.k_by_h[0]['h'], self.k_by_h[-1]['h']
        self.by_h = sorted(fits, key=lambda f: f['h']) if fits else self.k_by_h
        self.h_lo, self.h_hi = self.by_h[0]['h'], self.by_h[-1]['h']
        n = len(self.points)
        self.notes = []
        if n > 2:
            (self.s, self.c), cov = np.polyfit(self.ts, self.hs, 1, cov=True)
            self.s_err = float(np.sqrt(cov[0, 0])) * RSUN_KM
        else:
            self.s, self.c = np.polyfit(self.ts, self.hs, 1)
            self.s_err = float('nan')
            self.notes.append('two measured times: the speed has no uncertainty estimate')
        if n >= 3:
            self.q = np.polyfit(self.ts, self.hs, 2)
            if n == 3:
                self.notes.append('3 measured times: the 2nd-order curve passes through all of them (no uncertainty)')
        else:
            self.q = np.r_[0.0, self.s, self.c]
            self.notes.append('fewer than 3 measured times: the 2nd-order fit reduces to a linear one')

    def when(self, seconds):
        return self.t0 + dt.timedelta(seconds=float(seconds))

    # --- kinematics
    def linear(self, h):
        return (h - self.c) / self.s, self.s

    def quadratic(self, h):
        """(t [s], speed [R_sun/s], note) on the rising branch of the 2nd-order curve, with fallbacks."""
        a2, b, c0 = self.q
        v = lambda t: 2 * a2 * t + b
        if abs(a2) < 1e-18:
            return (h - c0) / b, b, None
        roots = np.roots([a2, b, c0 - h])
        roots = [r.real for r in roots if abs(r.imag) < 1e-9 and v(r.real) > 0]
        t_first, t_last, eps = self.ts[0], self.ts[-1], 1e-6
        if h > self.k_hi:
            cand = sorted(r for r in roots if r >= t_last - eps)
            if cand:
                return cand[0], v(cand[0]), None
            vl = v(t_last)
            if vl <= 0:
                return None, None, 'the 2nd-order curve is already falling at the last fit'
            peak = c0 - b * b / (4 * a2) if a2 < 0 else float('nan')
            return (t_last + (h - np.polyval(self.q, t_last)) / vl, vl,
                    f'beyond the 2nd-order curve (peaks at {peak:.2f} R_sun): continued from the last fit at '
                    f'{vl * RSUN_KM:.0f} km/s')
        if h < self.k_lo:
            cand = sorted((r for r in roots if r <= t_first + eps), reverse=True)
            if cand:
                return cand[0], v(cand[0]), None
            vf = v(t_first)
            if vf <= 0:
                return None, None, 'the 2nd-order curve is not rising at the first fit'
            return (t_first + (h - np.polyval(self.q, t_first)) / vf, vf,
                    f'below the 2nd-order curve\'s lowest point: continued back from the first fit at '
                    f'{vf * RSUN_KM:.0f} km/s')
        cand = sorted(r for r in roots if t_first - eps <= r <= t_last + eps)
        if cand:
            return cand[0], v(cand[0]), None
        t = float(np.interp(h, [f['h'] for f in self.k_by_h], [(f['time'] - self.t0).total_seconds() for f in self.k_by_h]))
        return t, max(v(t), 1e-12), 'the 2nd-order curve is not monotonic here: time interpolated between fits'

    # --- parameters
    def interp(self, k, h):
        hs = [f['h'] for f in self.by_h]
        y = np.array([f[k] for f in self.by_h])
        if k in PERIOD:
            y = unwrap(y, PERIOD[k])
            return wrap(float(np.interp(h, hs, y)), PERIOD[k])
        return float(np.interp(h, hs, y))

    def trend(self, k, h, t, notes):
        x = self.ts if PARAM_FIT_VS == 'time' else self.hs
        y = np.array([f[k] for f in self.points])
        if k in PERIOD:
            y = unwrap(y, PERIOD[k])
        deg = min(1, len(self.points) - 1)
        val = float(np.polyval(np.polyfit(x, y, deg), t if PARAM_FIT_VS == 'time' else h)) if deg else float(y[0])
        if k in PERIOD:
            return wrap(val, PERIOD[k])
        lo, hi = LIMITS.get(k, (-np.inf, np.inf))
        if not lo <= val <= hi:
            notes.append(f'{k} trend gives {val:.3f}, clipped to [{lo}, {hi}]')
            val = float(np.clip(val, lo, hi))
        return val

    def par(self, k, how, h, t, notes):
        if how == 'latest':
            return float(self.points[-1][k])
        if how == 'mean':
            return mean_par(k, [f[k] for f in self.points])
        if how == 'interp':
            return self.interp(k, h)
        return self.trend(k, h, t, notes)

    def features(self, h):
        if FEATURES == 'none':
            return [], 'none'
        f = min(self.fits, key=lambda f: abs(f['h'] - h))
        return [list(d) for d in f['model'].dents], f['name']


# ----------------------------------------------------------------------------- one step
def cone_half_widths(model):
    a, b, c, rc = model.axes
    if rc <= a:
        return None
    s = np.sqrt(rc ** 2 - a ** 2)
    return float(np.degrees(np.arctan(b / s))), float(np.degrees(np.arctan(c / s)))


def step(ev, method, h):
    notes = []
    inside = ev.h_lo <= h <= ev.h_hi
    extra = {}
    if method == 'observed':
        if not inside or len(ev.fits) < 2:
            return None
        hs = [f['h'] for f in ev.by_h]
        t = float(np.interp(h, hs, [(f['time'] - ev.t0).total_seconds() for f in ev.by_h]))
        j = max(1, min(int(np.searchsorted(hs, h)), len(hs) - 1))
        lo, hi = ev.by_h[j - 1], ev.by_h[j]
        dt_s = (hi['time'] - lo['time']).total_seconds()
        speed = (hi['h'] - lo['h']) / dt_s if dt_s else float('nan')
        pars = {k: ev.interp(k, h) for k in PARS}
        notes.append(f'interpolated between {lo["name"]} and {hi["name"]}')
    elif method == 'nearest':
        if not ev.fits:
            return None
        if NEAREST_KINEMATICS == 'constant':
            if ev.s <= 0:
                return None
            t, speed = ev.linear(h)
        else:
            t, speed, note = ev.quadratic(h)
            if t is None:
                return None
            if note:
                notes.append(note)
        pars = {k: ev.interp(k, h) for k in PARS}          # between the bracketing fits; beyond them, the end fit
        if inside:
            notes.append('interpolated between the bracketing fits')
        else:
            end = ev.by_h[0] if h < ev.h_lo else ev.by_h[-1]
            notes.append(f'{end["name"]} scaled self-similarly to this height (kappa, eps, alpha, direction, '
                         f'tilt and features kept)')
    elif method == 'constant':
        if ev.s <= 0:
            return None
        t, speed = ev.linear(h)
        pars = {k: mean_par(k, [f[k] for f in ev.points]) for k in PARS}
        if np.isfinite(ev.s_err):
            extra['speed_err_kms'] = round(ev.s_err, 1)
    else:
        t, speed, note = ev.quadratic(h)
        if t is None:
            return None
        if note:
            notes.append(note)
        pars = {k: ev.par(k, SHAPE_VARIED if k in SHAPE else DIRECTION_VARIED, h, t, notes) for k in PARS}
        extra['acceleration_ms2'] = round(2 * ev.q[0] * RSUN_KM * 1e3, 1) if not note else 0.0
    feats, feats_from = ev.features(h) if ev.fits else ([], 'none')
    when = ev.when(t)
    model = DeformableShock.from_pythea(h, pars['kappa'], pars['epsilon'], pars['alpha'], pars['lat'], pars['lon'],
                                        pars['tilt'], dents=feats, time=when.strftime('%Y-%m-%dT%H:%M:%S'))
    a, b, c, rc = model.axes
    cone = cone_half_widths(model)
    beyond = 0.0 if inside else (ev.h_lo - h if h < ev.h_lo else h - ev.h_hi)
    derived = {'method': method, 'height_rsun': round(h, 3), 'time': model.time,
               'speed_kms': round(speed * RSUN_KM, 1), **extra,
               'in_fitted_range': bool(inside), 'extrapolated_rsun': round(beyond, 2),
               'semi_axes_rsun': {'a': round(a, 3), 'b': round(b, 3), 'c': round(c, 3), 'center': round(rc, 3)},
               'cone_half_width_deg': ({'b': round(cone[0], 1), 'c': round(cone[1], 1)} if cone else
                                       'Sun centre inside the ellipsoid (wider than a 90-deg cone)'),
               'features_from': feats_from, 'source_fits': [f['name'] for f in ev.fits],
               'notes': (ev.notes if method != 'observed' else []) + notes}
    return {'model': model, 'derived': derived}


# ----------------------------------------------------------------------------- output
def heights():
    n = int(round((H_END - H_START) / H_STEP))
    return [round(H_START + i * H_STEP, 6) for i in range(n + 1)]


def save(series):
    for method, steps in series.items():
        rows = []
        for s in steps:
            m, d = s['model'], s['derived']
            path = cfg.final_path(f'shock_para_corona{int(round(d["height_rsun"] * 10)):03d}_{method}.json')
            path.write_text(json.dumps({**m.to_dict(), 'derived': d}, indent=2))
            cone = d['cone_half_width_deg']
            rows.append({'height_rsun': d['height_rsun'], 'time': d['time'], 'speed_kms': d['speed_kms'],
                         'acceleration_ms2': d.get('acceleration_ms2', ''), 'kappa': f'{m.kappa:.4f}',
                         'epsilon': f'{m.epsilon:.4f}', 'alpha': f'{m.alpha:.4f}', 'lat': f'{m.lat:.2f}',
                         'lon': f'{m.lon:.2f}', 'tilt': f'{m.tilt:.2f}', 'a': f'{m.a:.3f}', 'b': f'{m.b:.3f}',
                         'c': f'{m.c:.3f}', 'center': f'{m.axes[3]:.3f}',
                         'cone_b_deg': cone['b'] if isinstance(cone, dict) else '',
                         'cone_c_deg': cone['c'] if isinstance(cone, dict) else '',
                         'in_fitted_range': d['in_fitted_range'], 'extrapolated_rsun': d['extrapolated_rsun'],
                         'n_features': len(m.dents), 'features_from': d['features_from'],
                         'notes': ' | '.join(d['notes'])})
        table = cfg.final_path(f'shock_para_corona_{method}.csv')
        with open(table, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'saved {len(steps)} files shock_para_corona<hhh>_{method}.json and {table}')


def figure(ev, series):
    """(a) height-time: the measured points and the linear and 2nd-order fits from FIT_LINE_FROM to H_END, with
    the speeds and the acceleration; (b) latitude and (c) longitude against height: the measured points and the
    methods' series."""
    colors = {'constant': 'C0', 'varied': 'C3', 'observed': 'C2', 'nearest': 'C4'}
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6), layout='constrained')
    fits = [p for p in ev.points if p.get('source') != 'gui']
    gui = [p for p in ev.points if p.get('source') == 'gui']

    def points(a, x, y_of):
        if fits:
            a.plot([x(p) for p in fits], [y_of(p) for p in fits], 'ko', ms=7, label='fits', zorder=5)
        if gui:
            a.plot([x(p) for p in gui], [y_of(p) for p in gui], 'o', mfc='w', mec='k', mew=1.4, ms=7,
                   label='GUI only', zorder=5)

    # (a) height-time
    a = ax[0]
    hh = np.linspace(FIT_LINE_FROM, H_END, 200)
    if ev.s > 0:
        a.plot([ev.when((h - ev.c) / ev.s) for h in hh], hh, '-', color='C0', lw=1.6, label='linear fit')
    span = ev.ts[-1] - ev.ts[0] if len(ev.ts) > 1 else 3600.0
    tt = np.linspace(ev.ts[0] - 3 * span - 7200, ev.ts[-1] + 3 * span + 7200, 4000)
    h_t, v_t = np.polyval(ev.q, tt), np.polyval(np.polyder(ev.q), tt)
    keep = (v_t > 0) & (h_t >= FIT_LINE_FROM) & (h_t <= H_END)              # the rising branch of the polynomial
    if keep.any():
        a.plot([ev.when(t) for t in tt[keep]], h_t[keep], '--', color='C3', lw=1.6, label='2nd-order fit')
    turn = None
    if abs(ev.q[0]) > 1e-18:
        t_v = -ev.q[1] / (2 * ev.q[0])
        h_v = float(np.polyval(ev.q, t_v))
        if ev.q[0] > 0 and FIT_LINE_FROM < h_v < ev.k_lo:                   # accelerating: lowest point below the data
            turn = h_v
    points(a, lambda p: p['time'], lambda p: p['h'])
    lines = []
    if ev.s > 0:
        err = f' $\\pm$ {ev.s_err:.0f}' if np.isfinite(ev.s_err) else ''
        lines.append(f'linear: v = {ev.s * RSUN_KM:.0f}{err} km/s')
    acc = 2 * ev.q[0] * RSUN_KM * 1e3
    t_end, v_end, note = ev.quadratic(H_END)
    lines.append(f'2nd order: a = {acc:+.1f} m/s$^2$')
    if v_end is not None:
        lines.append(f'   v({H_END:g} R$_\\odot$) = {v_end * RSUN_KM:.0f} km/s' + (' *' if note else ''))
    if note:
        lines.append(f'   * {note}')
    if turn is not None:
        vf = np.polyval(np.polyder(ev.q), ev.ts[0]) * RSUN_KM
        lines.append(f'   the curve turns at {turn:.1f} R$_\\odot$; below, "varied" continues')
        lines.append(f'   back from the first point at {vf:.0f} km/s')
    a.text(0.03, 0.97, '\n'.join(lines), transform=a.transAxes, va='top', ha='left', fontsize=9,
           bbox=dict(boxstyle='round', fc='w', ec='0.7'))
    a.set_ylim(0, H_END * 1.05)
    a.set_ylabel('apex height [R$_\\odot$]')
    a.set_xlabel('time [UT]')
    a.set_title('apex height vs time')
    a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    a.legend(fontsize=8, loc='lower right')

    # (b), (c) direction against height
    for a, key, title in ((ax[1], 'lat', 'latitude [deg]'), (ax[2], 'lon', 'Stonyhurst longitude [deg]')):
        if ev.fits:
            a.axvspan(ev.h_lo, ev.h_hi, color='0.92', zorder=0, label='fitted range')
        for method, steps in series.items():
            y = np.array([getattr(s['model'], key) for s in steps])
            if key in PERIOD and len(y):
                y = unwrap(y, PERIOD[key])
            a.plot([s['model'].height for s in steps], y, '.-', color=colors[method], ms=4, lw=1, label=method)
        ref = None
        if key in PERIOD and series:
            ref = unwrap(np.array([getattr(s['model'], key) for s in next(iter(series.values()))]), PERIOD[key])
        def y_of(p, key=key, ref=ref):
            v = p[key]
            if key in PERIOD and ref is not None and len(ref):
                v = v + PERIOD[key] * np.round((np.mean(ref) - v) / PERIOD[key])       # same branch as the curves
            return v
        points(a, lambda p: p['h'], y_of)
        a.set_title(title)
        a.set_xlabel('apex height [R$_\\odot$]')
        a.legend(fontsize=8)
    png = cfg.figure_path('derive_para_corona.png')
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f'saved {png}')


def main():
    fits = load_fits()
    gui = load_gui_points(fits) if USE_GUI_POINTS else []
    if len(fits) + len(gui) < 2:
        sys.exit(f'{len(fits)} fit(s) and {len(gui)} GUI-only time(s) in "{cfg.RESULTS_DIR}": at least two times are needed')
    ev = Evolution(fits, fits + gui)
    print(f'{len(fits) + len(gui)} measured times ({len(fits)} fits, {len(gui)} GUI-only: '
          f'{", ".join(f["name"] for f in gui) or "none"}), apex {ev.k_lo:.2f} - {ev.k_hi:.2f} R_sun; '
          f'heights {H_START} - {H_END} R_sun every {H_STEP}')
    for n in ev.notes:
        print(f'  note: {n}')
    series = {}
    for method in METHODS:
        steps = [s for s in (step(ev, method, h) for h in heights()) if s is not None]
        if steps:
            series[method] = steps
    for method, steps in series.items():
        print(f'\n[{method}]  ({len(steps)} heights)\n   h [R_sun]  time [UT]   speed   kappa    eps   alpha    lat     lon   tilt  range')
        for s in steps:
            m, d = s['model'], s['derived']
            flag = 'fitted' if d['in_fitted_range'] else f'+{d["extrapolated_rsun"]:.1f} R_sun'
            print(f'   {m.height:6.1f}   {m.time[11:19]}  {d["speed_kms"]:6.0f}  {m.kappa:6.3f} {m.epsilon:+6.3f} {m.alpha:6.3f} '
                  f'{m.lat:+6.1f} {m.lon:7.1f} {m.tilt:+6.1f}  {flag}')
        special = sorted({n for s in steps for n in s['derived']['notes'] if n not in ev.notes and 'interpolated between' not in n})
        for n in special:
            print(f'   note: {n}')
    print()
    save(series)
    figure(ev, series)


if __name__ == '__main__':
    main()
