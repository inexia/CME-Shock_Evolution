"""
evolution_cobpoint.py -- the shock at the cobpoints of one or more targets (spacecraft or planets; the
"connecting with the observer" point: where the target's magnetic field line meets the shock), through the whole
evolution: the coronal series (derive_para_corona.py, 1.5 - 21.5 R_sun) followed by the interplanetary fronts of
one of the three shock-evolution versions (shock_properties_sse.py, _dbm.py, _gsd.py).

Run:  python evolution_cobpoint.py      (settings below; folders and background from event_config.py)
Versions: VERSIONS = ['sse', 'dbm', 'gsd'] -- the corona followed by the interplanetary fronts of
shock_properties_sse.py, _dbm.py and _gsd.py; every version gets its own cobpoint files and figures (a version
whose fronts are missing is skipped). The corona part is shared: its field lines are traced once.

Targets: TARGETS = 'Earth' or 'Earth, STA, PSP, SolO' (comma-separated; a list may also hold fixed positions
(name, r [AU], HGS lon [deg], HGS lat [deg])). Earth, L1 (L1_KM sunward of Earth) and the planets come from
sunpy's built-in ephemeris (no network); spacecraft (STA, STB, PSP, SolO, SOHO, ACE, Wind, BepiColombo) from JPL
Horizons, one query per target for all times (needs the network). A target with no position then (not yet
launched, no network) is skipped with a message.

At every step of a series (apex height h, time t), for every target:
  field line   traced from the target toward the Sun through the background at t: the MAS heliospheric MHD field,
               then the coronal one, down to the photosphere (its footpoint); no Parker spiral, no source
               surface. The background is the steady pre-event state, rotated to the Carrington longitude of t.
  cobpoint     the first crossing of that line with the shock, coming from the target. No crossing: not (yet)
               connected. The target inside the shock: the shock has passed it -- its series stops there. A line that
               gets no closer to the Sun (weak field near a current sheet) stops: 'field line stalls'.
  properties   exactly as in shock_properties_sse.py, at that point: V_n, theta_Bn, u.n, V_A, c_s, beta, M_A, M_f,
               X (Rankine-Hugoniot) and X_B3.

Writes, in fitted results/<id>/final/: cobpoint_<version>_<method>_<target>.csv (one row per step) and
cobpoint_fieldlines_<version>_<method>_<target>.npz; in figure/<event>/cobpoint/:
evolution_cobpoint_<version>_<method>.png (all
targets overlaid); and, with OVERLAY_EVO, it redraws the 2 x 3 shock-evolution figures of shock_properties_*.py
(figure/<event>/shock_evo*/..., same files, same layout) with every target's field line projected onto the two
x-y panels (lower left and lower right): solid from the target to its cobpoint (a star), dashed beyond toward the
Sun. The full-front values come from shock_properties_*.py's files (recomputed if missing). Running
shock_properties_*.py alone draws the figures without lines.
"""
import csv
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
import matplotlib.patheffects as pe

import event_config as cfg
import background as bk
import shock_properties_sse as sp

# ================================ SETTINGS ================================
METHODS = ['constant', 'varied']
VERSIONS = ['sse', 'dbm', 'gsd']      # shock-evolution versions: the corona, then the interplanetary fronts of
                                      # shock_properties_sse.py / _dbm.py / _gsd.py (a missing one is skipped)
VERSION_SERIES = {'sse': 'ip', 'dbm': 'dbm', 'gsd': 'gsd'}
TARGETS = 'Earth'                     # e.g. 'Earth, STA, STB' or 'L1, STA, PSP, SolO'
L1_KM = 1.5e6                         # L1 sunward of Earth [km]
TRACE_STEP = 0.01                     # field-line step as a fraction of the heliocentric distance
STALL_STEPS = 300                     # a field line that gets no closer to the Sun (2%) over this many steps stalls
MAX_TRACE_STEPS = 6000                # (e.g. skating along a current sheet where B is weak): the trace stops there
PLOT = True
OVERLAY_EVO = True                    # redraw the shock-evolution figures with the targets' field lines
EVO_QUANTITIES = None                 # which of them (None: shock_properties.QUANTITIES, i.e. all it draws)
TARGET_COLORS = ['#FF6B6B', '#4DD0E1', '#FFD166', '#C77DFF', '#80ED99', '#FF9F1C', '#F4F1DE']
# ==========================================================================
RSUN_KM = bk.RSUN_KM
AU_RSUN = bk.AU_RSUN
HORIZONS = {'sta': ('STEREO-A', '-234'), 'stereo-a': ('STEREO-A', '-234'), 'stb': ('STEREO-B', '-235'),
            'stereo-b': ('STEREO-B', '-235'), 'psp': ('Parker Solar Probe', '-96'),
            'parker solar probe': ('Parker Solar Probe', '-96'), 'solo': ('Solar Orbiter', '-144'),
            'solar orbiter': ('Solar Orbiter', '-144'), 'soho': ('SOHO', '-21'), 'ace': ('ACE', '-92'),
            'wind': ('Wind', '-8'), 'bepicolombo': ('BepiColombo', '-121'), 'bepi': ('BepiColombo', '-121')}
PLANETS = ('mercury', 'venus', 'mars', 'jupiter', 'saturn')


# ----------------------------------------------------------------------------- targets and their positions
def parse_targets(spec):
    """[(label, kind, ref)]: kind 'earth', 'l1', 'planet', 'horizons' or 'fixed'."""
    items = [s.strip() for s in spec.split(',')] if isinstance(spec, str) else list(spec)
    out = []
    for it in items:
        if isinstance(it, (tuple, list)):
            out.append((str(it[0]), 'fixed', tuple(float(v) for v in it[1:4])))
            continue
        if not it:
            continue
        key = it.lower()
        if key == 'earth':
            out.append(('Earth', 'earth', None))
        elif key == 'l1':
            out.append(('L1', 'l1', None))
        elif key in PLANETS:
            out.append((it.capitalize(), 'planet', key))
        elif key in HORIZONS:
            out.append((it if len(it) <= 4 else HORIZONS[key][0], 'horizons', HORIZONS[key]))
        else:
            out.append((it, 'horizons', (it, it)))              # any other Horizons name, as given
    return out


def _xyz(r_rsun, lon_rad, lat_rad):
    return np.stack([r_rsun * np.cos(lat_rad) * np.cos(lon_rad), r_rsun * np.cos(lat_rad) * np.sin(lon_rad),
                     r_rsun * np.sin(lat_rad)], -1)


class TargetPositions:
    """HGS positions [R_sun] of the targets at the given times (ISO strings); Horizons queried once per target."""

    def __init__(self, targets, times):
        self.targets, self.times = targets, sorted(set(times))
        self.pos = {}
        for label, kind, ref in targets:
            try:
                xyz = self._query(kind, ref, self.times)
            except Exception as e:
                print(f'  {label}: no position ({type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ""}) -- skipped')
                continue
            ok = np.all(np.isfinite(xyz), axis=1)
            if not ok.any():
                print(f'  {label}: no position at these times -- skipped')
                continue
            self.pos[label] = {t: (x if o else None) for t, x, o in zip(self.times, xyz, ok)}

    @staticmethod
    def _query(kind, ref, times):
        ts = [str(t).replace('T', ' ') for t in times]
        if kind == 'fixed':
            r, lon, lat = ref
            return np.tile(_xyz(r * AU_RSUN, np.radians(lon), np.radians(lat)), (len(ts), 1))
        from astropy.time import Time
        if kind in ('earth', 'l1'):
            from sunpy.coordinates import get_earth
            c = get_earth(Time(ts))
            r = c.radius.to_value('km') / RSUN_KM - (L1_KM / RSUN_KM if kind == 'l1' else 0.0)
            return _xyz(np.atleast_1d(r), np.atleast_1d(c.lon.to_value('rad')), np.atleast_1d(c.lat.to_value('rad')))
        if kind == 'planet':
            from sunpy.coordinates import get_body_heliographic_stonyhurst
            c = get_body_heliographic_stonyhurst(ref, Time(ts))
        else:
            from sunpy.coordinates import get_horizons_coord, frames
            c = None
            for body in ref:                                     # the name, then the Horizons id
                try:
                    c = get_horizons_coord(body, Time(ts))
                    break
                except Exception as e:
                    err = e
            if c is None:
                raise err
            c = c.transform_to(frames.HeliographicStonyhurst(obstime=Time(ts)))
        return _xyz(np.atleast_1d(c.radius.to_value('km') / RSUN_KM), np.atleast_1d(c.lon.to_value('rad')),
                    np.atleast_1d(c.lat.to_value('rad')))

    def position(self, label, time):
        return self.pos.get(label, {}).get(time)


def lon_lat(x):
    r = np.linalg.norm(x)
    return float(np.degrees(np.arctan2(x[1], x[0]))), float(np.degrees(np.arcsin(x[2] / r))), float(r)


# ----------------------------------------------------------------------------- field line and cobpoint
def sunward_line(bg, start, time):
    """(line, stalled): the field line through `start`, followed toward the Sun (against B where B points outward),
    traced in segments of STALL_STEPS; it stops if a segment brings it no closer to the Sun."""
    B = bg.sample(start[None], time)['B'][0]
    if not np.all(np.isfinite(B)):
        return None, False
    direction = -1 if np.dot(B, start) > 0 else +1
    r_out = 1.2 * np.linalg.norm(start)
    parts, x, r_prev, total, stalled = [start[None]], np.asarray(start, float), np.linalg.norm(start), 0, False
    while total < MAX_TRACE_STEPS:
        seg = bg.trace(x, time, direction=direction, step=TRACE_STEP, r_min=1.0, r_max=r_out, max_steps=STALL_STEPS)
        if len(seg) < 2:
            break
        parts.append(seg[1:])
        total += len(seg) - 1
        x = seg[-1]
        r_now = np.linalg.norm(x)
        if r_now <= 1.0 + 1e-3 or r_now >= r_out or len(seg) < STALL_STEPS + 1:
            break
        if r_now > 0.98 * r_prev:
            stalled = True
            break
        r_prev = r_now
    return np.vstack(parts), stalled


def cobpoint(model, line):
    """('connected', point, arc length from the target, index of the line point just past it) /
    ('not connected', ...) / ('spacecraft inside', ...). Only crossings on the physical front count (outside the
    Sun, within the front's half width of the apex): the removed far side of a halo fit is ignored."""
    F = sp.surface_offset(model, line)
    entering = np.flatnonzero((F[:-1] > 0) & (F[1:] <= 0)) + 1
    for i in entering:
        a, b = line[i - 1], line[i]
        lo, hi = 0.0, 1.0
        for _ in range(60):
            m = 0.5 * (lo + hi)
            lo, hi = (m, hi) if sp.surface_offset(model, a + m * (b - a)) > 0 else (lo, m)
        point = a + 0.5 * (lo + hi) * (b - a)
        if sp.on_front(model, point[None])[0]:
            arc = np.sum(np.linalg.norm(np.diff(line[:i], axis=0), axis=1)) + np.linalg.norm(point - a)
            return 'connected', point, float(arc), int(i)
    if F[0] <= 0 and sp.on_front(model, line[0][None])[0]:
        return 'spacecraft inside', None, np.nan, -1
    return 'not connected', None, np.nan, -1


def at_point(series, k, bg, x, how):
    """The step-5 quantities at one surface point x of series step k."""
    step = series[k]
    model = step['model']
    e1, e2, e3 = model.frame
    g = (x - model.center) / np.linalg.norm(x - model.center)
    th, ph = np.arccos(np.clip(g @ e1, -1, 1)), np.mod(np.arctan2(g @ e3, g @ e2), 2 * np.pi)
    n = model.normals(th, ph)
    vn = float(sp.normal_speed(series, k, x[None], n[None], how)[0])
    s = bg.sample(x[None], step['time_iso'])
    B, u = s['B'][0], s['u'][0]
    VA, cs, beta = float(s['VA'][0]), float(s['cs'][0]), float(s['beta'][0])
    tbn = float(np.degrees(np.arccos(np.clip(abs(np.dot(B, n)) / np.linalg.norm(B), 0, 1))))
    un = float(np.dot(u, n))
    rel = vn - un
    MA = rel / VA if rel > 0 else np.nan
    Mf = rel / float(sp.fast_speed(VA, cs, tbn)) if rel > 0 else np.nan
    X = float(sp.rh_compression(MA, tbn, beta)) if np.isfinite(Mf) and Mf > 1 else np.nan
    return {'Vn_kms': vn, 'theta_bn_deg': tbn, 'u_n_kms': un, 'VA_kms': VA, 'cs_kms': cs, 'beta': beta,
            'n_cm3': float(s['n'][0]), 'B_nT': float(np.linalg.norm(B)), 'MA': MA, 'Mf': Mf, 'X': X,
            'X_B3': float(sp.b3_compression(MA, tbn)) if np.isfinite(MA) else np.nan,
            'angle_from_apex_deg': float(np.degrees(np.arccos(np.clip(np.dot(x, e1) / np.linalg.norm(x), -1, 1))))}


# ----------------------------------------------------------------------------- the shock at the targets
def front_distance(model, d):
    """The front's distance [R_sun] along the unit direction d (NaN: the physical front does not cover it)."""
    if hasattr(model, 'R_at'):                                     # a ray front (DBM, GSD)
        r = float(model.R_at(d))
    else:
        from ray_front import outer_crossing
        r = float(np.reshape(outer_crossing(model, np.asarray(d, float)[None]), -1)[0])
    if not np.isfinite(r) or not sp.on_front(model, (r * np.asarray(d))[None])[0]:
        return np.nan
    return r


def shock_at_target(version, method, bg, label, pos_of):
    """When the shock reaches a target along the target's direction, and its local properties there (the front's
    point on that line at the two bracketing steps, interpolated to the arrival). None: never reached; the reason
    is in 'status'."""
    steps, last = [], None
    for kind in ('corona', VERSION_SERIES[version]):
        series = sp.load_series(method, kind)
        for k, st in enumerate(series):
            if last is not None and st['time'] <= last:
                continue
            steps.append((kind, series, k, st))
        if series:
            last = series[-1]['time']
    if not steps:
        return {'status': 'no series'}
    prev, max_r, covered = None, 0.0, False
    for kind, series, k, st in steps:
        pos = pos_of(st['time_iso'])
        if pos is None:
            continue
        pos = np.asarray(pos, float)
        r_t = float(np.linalg.norm(pos))
        d = pos / r_t
        r_f = front_distance(st['model'], d)
        if np.isfinite(r_f):
            covered = True
            max_r = max(max_r, r_f)
        if prev is not None and np.isfinite(r_f) and np.isfinite(prev[4]) and prev[4] < prev[5] and r_f >= r_t:
            f = (prev[5] - prev[4]) / ((r_f - r_t) + (prev[5] - prev[4]))   # where (front - target) changes sign
            t0, t1 = prev[3]['time'], st['time']
            when = t0 + (t1 - t0) * f
            how = sp.vn_method(kind)
            p0 = at_point(prev[1], prev[2], bg, prev[4] * prev[6], sp.vn_method(prev[0]))
            p1 = at_point(series, k, bg, r_f * d, how)
            props = {key: (p0[key] + f * (p1[key] - p0[key]) if np.isfinite(p0[key]) and np.isfinite(p1[key])
                           else (p1[key] if f >= 0.5 else p0[key])) for key in p1}
            v_ray = (r_f - prev[4]) * RSUN_KM / max((t1 - t0).total_seconds(), 1e-9)
            return {'status': 'reached', 'time': when, 'r_rsun': (1 - f) * prev[5] + f * r_t, 'v_ray_kms': v_ray,
                    'series': kind, 'lon': float(np.degrees(np.arctan2(d[1], d[0]))),
                    'lat': float(np.degrees(np.arcsin(d[2]))), **props}
        prev = (kind, series, k, st, r_f, r_t, d)
    if not covered:
        return {'status': 'the front never covers this direction'}
    return {'status': f'not reached (front at most {max_r:.0f} R_sun along this direction)'}


def print_shock_at_targets(results):
    """results: [(label, version, method, dict)]"""
    obs = None
    if getattr(cfg, 'ARRIVAL_TIME', None):
        import datetime as _dt
        obs = _dt.datetime.fromisoformat(str(cfg.ARRIVAL_TIME).replace(' ', 'T'))
    print('\n================ the shock at the targets (along each target\'s direction) ================')
    print(f'{"target":8s} {"version/method":16s} {"arrival (UT)":16s} ' + (f'{"vs obs.":>8s} ' if obs else '')
          + f'{"r":>6s} {"V_n":>6s} {"V_n-u.n":>7s} {"th_Bn":>6s} {"M_A":>5s} {"M_f":>5s} {"X":>5s} '
          f'{"n":>6s} {"B":>6s}')
    f = lambda v, fmt: (fmt.format(v) if isinstance(v, float) and np.isfinite(v) else '-').rjust(len(fmt.format(0.0)))
    for label, version, method, r in results:
        head = f'{label:8s} {version + "/" + method:16s} '
        if r.get('status') != 'reached':
            print(head + r.get('status', ''))
            continue
        dt_h = (r['time'] - obs).total_seconds() / 3600.0 if obs else None
        print(head + f'{r["time"]:%Y-%m-%d %H:%M} ' + (f'{dt_h:+7.1f}h ' if obs else '')
              + f'{f(r["r_rsun"], "{:6.1f}")} {f(r["Vn_kms"], "{:6.0f}")} {f(r["Vn_kms"] - r["u_n_kms"], "{:7.0f}")} '
              f'{f(r["theta_bn_deg"], "{:6.1f}")} {f(r["MA"], "{:5.2f}")} {f(r["Mf"], "{:5.2f}")} {f(r["X"], "{:5.2f}")} '
              f'{f(r["n_cm3"], "{:6.1f}")} {f(r["B_nT"], "{:6.1f}")}')
    print('(r [R_sun]; V_n: lab-frame normal speed, V_n - u.n: shock frame [km/s]; theta_Bn [deg]; X: Rankine-Hugoniot '
          'compression; n [cm^-3], B [nT] upstream' + (f'; vs obs.: minus the observed {obs:%Y-%m-%d %H:%M} UT)' if obs else ')'))


def shock_at_targets(bg, where, versions, methods):
    results = []
    if where is None:
        return results
    for label in where.pos:
        for version in versions:
            for method in methods:
                if not sp.load_series(method, VERSION_SERIES[version]):
                    continue
                results.append((label, version, method,
                                shock_at_target(version, method, bg, label, lambda t, L=label: where.position(L, t))))
    return results


# ----------------------------------------------------------------------------- one target, one method
_LINES = {}                                                   # (target, time) -> (line, stalled): traced once


def evolution(method, bg, label, pos_of, kinds=('corona', 'ip')):
    """Rows (one per step) and field lines of one target along the series `kinds` (followed in this order);
    pos_of(time_iso) -> its HGS position or None."""
    rows, lines = [], {}
    arrived = False
    for kind in kinds:
        series = sp.load_series(method, kind)
        if not series:
            print(f'  [{kind}] no {sp.SERIES_FILES[kind][0]}<h>_{method} files')
            continue
        how = sp.vn_method(kind)
        for k, step in enumerate(series):
            if arrived:
                break
            if rows and step['time'] <= rows[-1]['_time']:
                continue                                   # the interplanetary series starts where the corona ends
            t = step['time_iso']
            pos = pos_of(t)
            row = {'target': label, 'series': kind, 'step': k, 'apex_height_rsun': round(step['model'].height, 3),
                   'time': t, '_time': step['time']}
            if pos is None:
                row['status'] = 'no target position'
                rows.append(row)
                continue
            pos = np.asarray(pos, float)
            sc_lon, sc_lat, sc_r = lon_lat(pos)
            row.update({'sc_r_rsun': round(sc_r, 3), 'sc_lon_deg': round(sc_lon, 3), 'sc_lat_deg': round(sc_lat, 3)})
            if (label, t, tuple(np.round(pos, 6))) not in _LINES:
                _LINES[(label, t, tuple(np.round(pos, 6)))] = sunward_line(bg, pos, t)
            line, stalled = _LINES[(label, t, tuple(np.round(pos, 6)))]
            if line is None or len(line) < 2:
                row['status'] = 'no field line (target outside the background)'
                rows.append(row)
                continue
            status, x, arc, idx = cobpoint(step['model'], line)
            if status == 'not connected' and stalled:
                status = 'field line stalls'                     # it never reached the shock or the Sun
            row['status'] = status
            foot = line[-1]
            if np.linalg.norm(foot) <= 1.0 + 1e-3:
                row['foot_lon_deg'], row['foot_lat_deg'], _ = (round(v, 3) for v in lon_lat(foot))
            lines[f'{kind}_{k:03d}'] = line
            if status == 'connected':
                lon, lat, r = lon_lat(x)
                row.update({'cob_r_rsun': round(r, 4), 'cob_lon_deg': round(lon, 3), 'cob_lat_deg': round(lat, 3),
                            'fieldline_length_rsun': round(arc, 3), '_cob': x, '_cob_index': idx})
                row.update({k_: (round(v, 4) if np.isfinite(v) else '') for k_, v in at_point(series, k, bg, x, how).items()})
            elif status == 'spacecraft inside':
                arrived = True
            rows.append(row)
            q = lambda key, f='{:.3g}': (f.format(row[key]) if isinstance(row.get(key), float) else '-')
            print(f'   {label:8s} {kind:6s} h {step["model"].height:6.1f}  {t[5:16]}  {status:17s} r_cob '
                  f'{q("cob_r_rsun", "{:.2f}")}  {q("angle_from_apex_deg", "{:.1f}")} deg  V_n {q("Vn_kms", "{:.0f}")}  '
                  f'theta_Bn {q("theta_bn_deg", "{:.1f}")}  M_A {q("MA")}  M_f {q("Mf")}  X {q("X")}')
    return rows, lines


def tag_of(label):
    return ''.join(ch for ch in label if ch.isalnum() or ch in '-_') or 'target'


def save(version, method, label, rows, lines):
    cols = ['target', 'series', 'apex_height_rsun', 'time', 'status', 'cob_r_rsun', 'cob_lon_deg', 'cob_lat_deg',
            'angle_from_apex_deg', 'fieldline_length_rsun', 'Vn_kms', 'theta_bn_deg', 'u_n_kms', 'VA_kms', 'cs_kms',
            'beta', 'n_cm3', 'B_nT', 'MA', 'Mf', 'X', 'X_B3', 'foot_lon_deg', 'foot_lat_deg', 'sc_r_rsun',
            'sc_lon_deg', 'sc_lat_deg']
    table = cfg.final_path(f'cobpoint_{version}_{method}_{tag_of(label)}.csv')
    with open(table, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(cfg.final_path(f'cobpoint_fieldlines_{version}_{method}_{tag_of(label)}.npz'), **lines)
    print(f'  saved {table.name} and cobpoint_fieldlines_{version}_{method}_{tag_of(label)}.npz')


# ----------------------------------------------------------------------------- figures
def figure(version, method, by_target):
    """All targets overlaid: field lines and cobpoints (x-y), and the shock at their cobpoints vs apex height."""
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5), layout='constrained')
    a0 = ax[0, 0]
    all_h = []
    for i, (label, (rows, lines)) in enumerate(by_target.items()):
        col = TARGET_COLORS[i % len(TARGET_COLORS)]
        con = [r for r in rows if r['status'] == 'connected']
        h = np.array([r['apex_height_rsun'] for r in con])
        all_h += list(h)
        get = lambda key: np.array([r[key] if isinstance(r.get(key), float) else np.nan for r in con])
        keys = list(lines.keys())
        for j, key in enumerate(keys):
            ln = lines[key]
            a0.plot(ln[:, 0], ln[:, 1], '-', color=col, lw=0.7, alpha=0.15 + 0.6 * j / max(len(keys) - 1, 1))
        if con:
            cob = np.array([r['_cob'] for r in con])
            a0.plot(cob[:, 0], cob[:, 1], '*', color=col, ms=6, mec='k', mew=0.3)
        last = next((r for r in reversed(rows) if 'sc_r_rsun' in r), None)
        if last is not None:
            p = _xyz(last['sc_r_rsun'], np.radians(last['sc_lon_deg']), np.radians(last['sc_lat_deg']))
            a0.plot(p[0], p[1], 'o', color=col, mec='k', ms=8, label=label)
        ax[0, 1].plot(h, get('cob_r_rsun'), 'o-', color=col, ms=3, label=label)
        ax[0, 2].plot(h, get('Vn_kms') - get('u_n_kms'), 'o-', color=col, ms=3, label=label)
        ax[1, 0].plot(h, get('theta_bn_deg'), 'o-', color=col, ms=3, label=label)
        ax[1, 1].plot(h, get('MA'), 'o-', color=col, ms=3, label=f'{label} M$_A$')
        ax[1, 1].plot(h, get('Mf'), ':', color=col, lw=1.2, label=f'{label} M$_f$')
        ax[1, 2].plot(h, get('X'), 'o-', color=col, ms=3, label=label)
    a0.add_patch(plt.Circle((0, 0), 1, color='orange'))
    a0.set_aspect('equal', adjustable='datalim')
    a0.set_xlabel('x [R$_\\odot$] (to Earth)')
    a0.set_ylabel('y [R$_\\odot$]')
    a0.set_title('field lines (older: fainter) and cobpoints (stars), x-y projection', fontsize=10)
    a0.legend(fontsize=8, loc='best')
    titles = [(ax[0, 1], 'cobpoint distance [R$_\\odot$]'), (ax[0, 2], 'V$_n$ - u$\\cdot$n (shock frame) [km/s]'),
              (ax[1, 0], '$\\theta_{Bn}$ [deg] (45: quasi-perp. above)'), (ax[1, 1], 'M$_A$ (solid), M$_f$ (dotted)'),
              (ax[1, 2], 'X = $\\rho_2/\\rho_1$ (Rankine-Hugoniot)')]
    for a, title in titles:
        a.set_title(title, fontsize=10)
        a.set_xscale('log')
        if all_h:
            ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500) if min(all_h) / 1.2 <= t <= max(all_h) * 1.2]
            if ticks:
                a.set_xticks(ticks)
        a.xaxis.set_major_formatter(ScalarFormatter())
        a.xaxis.set_minor_formatter(NullFormatter())
        a.set_xlabel('apex height [R$_\\odot$]')
        a.legend(fontsize=7)
    ax[1, 0].axhline(45, color='0.6', lw=0.8, ls=':')
    ax[1, 0].set_ylim(0, 90)
    ax[1, 1].axhline(1, color='0.6', lw=0.8, ls=':')
    ax[1, 2].set_ylim(1, 4)
    fig.suptitle(f'{cfg.EVENT_ID} {version.upper()} {method}: the shock at the cobpoints of ' + ', '.join(by_target))
    out = cfg.FIGURE_DIR / 'cobpoint'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'evolution_cobpoint_{version}_{method}.png', dpi=120)
    plt.close(fig)
    print(f'  saved {out / f"evolution_cobpoint_{version}_{method}.png"}')


def load_props(kind, method, series, k, bg):
    """The full-front properties of step k: shock_properties_*.py's file, or computed now."""
    stem, digits, props_name, _ = sp.SERIES_FILES[kind]
    tag = f'{int(round(series[k]["model"].height * 10)):0{digits}d}'
    path = cfg.FINAL_DIR / props_name / method / f'{props_name}{tag}.npz'
    if path.exists():
        z = dict(np.load(path, allow_pickle=False))
        res = {k_: (v.item() if v.ndim == 0 else v) for k_, v in z.items()}
        res['time'] = str(res['time'])
        on = sp.on_front(series[k]['model'], res['xyz'])           # the physical front, also for files written
        res['on_front'] = on                                       # before it existed
        for q in ('Vn', 'theta_bn', 'MA', 'Mf', 'X', 'X_B3', 'u_n_rel'):
            if q in res:
                res[q] = np.where(on, res[q], np.nan)
        if 'shock' in res:
            res['shock'] = res['shock'] & on
        return res
    res = sp.properties(series, k, bg, sp.vn_method(kind))
    res['center'] = np.asarray(series[k]['model'].center, float)
    return res


def evo_figures(method, bg, by_target, kinds=('corona', 'ip')):
    """Redraw the 2 x 3 shock-evolution figures of every series step with the targets' field lines and cobpoints in
    the two x-y panels, over the plain ones that shock_properties_*.py drew (same files, same layout)."""
    quantities = EVO_QUANTITIES or sp.QUANTITIES
    for kind in kinds:
        series = sp.load_series(method, kind)
        if not series:
            continue
        stem, digits, props_name, fig_name = sp.SERIES_FILES[kind]
        results = [load_props(kind, method, series, k, bg) for k in range(len(series))]
        span = sp.axis_span(results)
        if kind != 'corona':                                    # interplanetary: keep the targets in view
            r_t = [r['sc_r_rsun'] for rows, _ in by_target.values() for r in rows if 'sc_r_rsun' in r and r['series'] == kind]
            if r_t:
                span = max(span, 1.08 * max(r_t))
        lim = sp.colour_limits(results, sp.COLOR_LIMITS_IP if kind != 'corona' else None)
        for q in quantities:
            (cfg.FIGURE_DIR / fig_name / method / q).mkdir(parents=True, exist_ok=True)
        bg_lim = None
        for k, res in enumerate(results):
            maps = sp.background_maps(bg, res['time'], span)
            bg_lim = bg_lim or sp.background_limits(maps)
            overlay = []
            for i, (label, (rows, lines)) in enumerate(by_target.items()):
                row = next((r for r in rows if r['series'] == kind and r['step'] == k), None)
                if row is not None and lines.get(f'{kind}_{k:03d}') is not None:
                    overlay.append((label, TARGET_COLORS[i % len(TARGET_COLORS)], row, lines[f'{kind}_{k:03d}']))
            tag = f'{int(round(res["height"] * 10)):0{digits}d}'
            for q in quantities:
                sp.figure(res, q, method, lim, cfg.FIGURE_DIR / fig_name / method / q / f'{fig_name}{tag}_{q}.png',
                          span, maps, bg_lim, overlay=overlay)
        print(f'  redrew {len(results) * len(quantities)} figures in {cfg.FIGURE_DIR / fig_name / method} '
              f'with the field lines of {", ".join(by_target)}')


# ----------------------------------------------------------------------------- main
def main():
    targets = parse_targets(TARGETS)
    if not targets:
        sys.exit('TARGETS is empty')
    bg = bk.load_background()
    print(bg.summary() if hasattr(bg, 'summary') else bg.name)
    kinds_of = {v: ['corona', VERSION_SERIES[v]] for v in VERSIONS}
    times = [s['time_iso'] for m in METHODS for v in VERSIONS for kind in kinds_of[v] for s in sp.load_series(m, kind)]
    if not times:
        sys.exit('no series: run derive_para_corona.py and shock_properties_sse/_dbm/_gsd.py first')
    print(f'targets: {", ".join(t[0] for t in targets)}')
    where = TargetPositions(targets, times)
    for method in METHODS:
        redrawn = set()                                       # the corona figures are shared: redrawn once
        for version in VERSIONS:
            ip_kind = VERSION_SERIES[version]
            if not sp.load_series(method, ip_kind):
                note = sp.failure_note(ip_kind, method)
                why = (f'shock_properties_{version}.py could not make them -- {note}' if note else
                       f'shock_properties_{version}.py has not been run for this method (METHODS there?)')
                print(f'\n[{version} / {method}] no interplanetary fronts ({sp.SERIES_FILES[ip_kind][0]}<h>_{method}): '
                      f'{why} -- skipped')
                continue
            by_target = {}
            for label, _, _ in targets:
                if label not in where.pos:
                    continue
                print(f'\n[{version} / {method}] cobpoint of {label}')
                rows, lines = evolution(method, bg, label, lambda t, L=label: where.position(L, t), kinds_of[version])
                if rows:
                    save(version, method, label, rows, lines)
                    by_target[label] = (rows, lines)
            if not by_target:
                continue
            if PLOT:
                figure(version, method, by_target)
            if OVERLAY_EVO:
                todo = [k for k in kinds_of[version] if k not in redrawn]
                evo_figures(method, bg, by_target, todo)
                redrawn.update(todo)
    print_shock_at_targets(shock_at_targets(bg, where, VERSIONS, METHODS))


if __name__ == '__main__':
    main()
