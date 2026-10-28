"""
check_fits.py -- do the shock fits of an event make physical sense? Fit residuals do not tell (a fit can match the
traces with a shell that swallows the Sun); these checks do. Reads every GUI save (shock_<stamp>.json, and
gcs_<stamp>.json for the flux rope) and fit (shock_fit_<stamp>.json) in fitted results/<id>/semi.

Run:  python check_fits.py      (folders from event_config.py)

Per time (the fit where there is one, else the GUI save; both are drawn in the figure):
  base on the Sun   the rear point (centre - a along the apex) must be inside the Sun: behind it (< -1 R_sun) the
                    shell swallows the Sun; beyond the surface (> FOOTPRINT_R) the shell has no base, no EUV wave
  base ring         where the shell meets r = FOOTPRINT_R: its angular radius from the apex direction (and Mm)
  dome              width / height above the surface, 2 b / (h - 1)
  shape             kappa, epsilon, alpha within RANGES (broad sanity bounds, adjustable, not literature statistics)
  flux rope         (with gcs_<stamp>.json) the upper half of the GCS flux rope must lie inside the shock; the
                    stand-off at the apex, shock apex - flux-rope front, must be positive
Across times (chronological):
  apex speed        the height must increase; speed below SPEED_MAX
  EUV-wave speed    the base ring's expansion speed (its arc radius on r = FOOTPRINT_R) within EUV_SPEED
  direction         the apex direction may not jump more than DIR_JUMP_DEG between consecutive times
  shape             kappa, epsilon, alpha may not jump more than SHAPE_JUMP between consecutive times

Writes fitted results/<id>/semi/check_fits.csv (one row per time, with its flags) and figure/<id>/check_fits.png.
"""
import csv
import json
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import event_config as cfg
import shock_fit as sf
from deformable_shock import DeformableShock

# ================================ SETTINGS ================================
RANGES = {'kappa': (0.2, 1.5), 'epsilon': (-0.7, 0.7), 'alpha': (0.6, 1.8)}
DOME_WIDTH = (0.5, 3.0)              # 2 b / (h - 1)
SPEED_MAX = 4000.0                   # apex speed [km/s]
EUV_SPEED = (50.0, 1500.0)           # base-ring expansion speed [km/s]
DIR_JUMP_DEG = 15.0                  # apex direction change between consecutive times [deg]
SHAPE_JUMP = {'kappa': 0.3, 'epsilon': 0.3, 'alpha': 0.4}
OUTSIDE_TOL = 0.01                   # flux rope: tolerated fraction of its upper half outside the shock
# ==========================================================================
RSUN_KM = 695700.0
STAMP = re.compile(r'^(shock|gcs|shock_fit)_(\d{8}_\d{4})\.json$')


def load():
    """{stamp: {'gui': model, 'fit': model, 'gcs': params}}"""
    out = {}
    for p in sorted(cfg.RESULTS_DIR.glob('*.json')):
        m = STAMP.match(p.name)
        if not m:
            continue
        kind, stamp = m.groups()
        d = json.loads(p.read_text())
        slot = out.setdefault(stamp, {})
        if kind == 'gcs':
            slot['gcs'] = d
        else:
            slot['fit' if kind == 'shock_fit' else 'gui'] = DeformableShock.from_dict(d)
    return dict(sorted(out.items()))


def base_ring(model):
    """Angular radius of the base ring (mean, min, max) [deg] from the apex direction, or None."""
    lines = sf.ellipsoid_footprint(model, cfg.FOOTPRINT_R)
    if not lines:
        return None
    ring = np.vstack(lines)
    ang = np.degrees(np.arccos(np.clip(ring @ model.frame[0] / np.linalg.norm(ring, axis=1), -1, 1)))
    return float(ang.mean()), float(ang.min()), float(ang.max())


def flux_rope(params, when, shock):
    """(flux-rope front height, fraction of its upper half outside the shock) or None (no GCS mesh code)."""
    try:
        import astropy.units as u
        from geometry import gcs_mesh_sunpy
        import shock_properties_sse as sp
    except Exception:
        return None
    mesh = gcs_mesh_sunpy(when, np.radians(params['half_angle']), params['height'], 12, 24, 21, params['kappa'],
                          np.radians(params['lat']), np.radians(params['lon']), np.radians(params['tilt']))
    c = mesh.cartesian
    x = np.stack([c.x.to_value(u.m), c.y.to_value(u.m), c.z.to_value(u.m)], -1).reshape(-1, 3) / (RSUN_KM * 1e3)
    r = np.linalg.norm(x, axis=1)
    front = float(r.max())
    upper = r > 1.0 + 0.5 * (front - 1.0)
    outside = float(np.mean(sp.surface_offset(shock, x[upper]) > 0)) if upper.any() else 0.0
    return front, outside


def per_time(stamp, slot):
    model = slot.get('fit') or slot.get('gui')
    a, b, c, rc = model.axes
    when = cfg.stamp_date(stamp)
    row = {'time': when, 'stamp': stamp, 'source': 'fit' if 'fit' in slot else 'GUI', 'model': model,
           'height': model.height, 'kappa': model.kappa, 'epsilon': model.epsilon, 'alpha': model.alpha,
           'lat': model.lat, 'lon': model.lon % 360.0, 'tilt': model.tilt, 'rear': rc - a,
           'dome': 2 * b / max(model.height - 1.0, 1e-9), 'flags': []}
    ring = base_ring(model)
    row['ring_deg'] = ring[0] if ring else np.nan
    row['ring_Mm'] = np.radians(ring[0]) * cfg.FOOTPRINT_R * RSUN_KM / 1e3 if ring else np.nan
    if row['rear'] < -1.0:
        row['flags'].append(f'rear point {row["rear"]:+.2f} R_sun: the shell swallows the Sun')
    elif row['rear'] > cfg.FOOTPRINT_R or ring is None:
        row['flags'].append(f'rear point {row["rear"]:+.2f} R_sun: no base on the Sun (no EUV-wave footprint)')
    for k, (lo, hi) in RANGES.items():
        if not lo <= row[k] <= hi:
            row['flags'].append(f'{k} {row[k]:.2f} outside [{lo:g}, {hi:g}]')
    if not DOME_WIDTH[0] <= row['dome'] <= DOME_WIDTH[1]:
        row['flags'].append(f'dome width/height {row["dome"]:.2f} outside [{DOME_WIDTH[0]:g}, {DOME_WIDTH[1]:g}]')
    row['cme_front'] = row['standoff'] = row['outside'] = np.nan
    if 'gcs' in slot:
        fr = flux_rope(slot['gcs'], when, model)
        if fr is not None:
            row['cme_front'], row['outside'] = fr
            row['standoff'] = model.height - fr[0]
            if row['standoff'] <= 0:
                row['flags'].append(f'stand-off {row["standoff"]:+.2f} R_sun: the flux rope is ahead of the shock')
            if row['outside'] > OUTSIDE_TOL:
                row['flags'].append(f'{100 * row["outside"]:.0f}% of the flux rope\'s upper half sticks out of the shock')
    return row


def across(rows):
    for prev, row in zip(rows[:-1], rows[1:]):
        dt_s = (row['time'] - prev['time']).total_seconds()
        row['speed'] = (row['height'] - prev['height']) * RSUN_KM / dt_s
        row['euv_speed'] = (row['ring_Mm'] - prev['ring_Mm']) * 1e3 / dt_s
        if row['speed'] <= 0:
            row['flags'].append(f'apex height decreases since {prev["time"]:%H:%M}')
        elif row['speed'] > SPEED_MAX:
            row['flags'].append(f'apex speed {row["speed"]:.0f} km/s since {prev["time"]:%H:%M}')
        if np.isfinite(row['euv_speed']):
            if row['euv_speed'] < 0:
                row['flags'].append(f'base ring shrinks since {prev["time"]:%H:%M} ({row["euv_speed"]:.0f} km/s)')
            elif not EUV_SPEED[0] <= row['euv_speed'] <= EUV_SPEED[1]:
                row['flags'].append(f'EUV-wave speed {row["euv_speed"]:.0f} km/s since {prev["time"]:%H:%M}')
        jump = np.degrees(np.arccos(np.clip(prev['model'].frame[0] @ row['model'].frame[0], -1, 1)))
        row['dir_jump'] = jump
        if jump > DIR_JUMP_DEG:
            row['flags'].append(f'apex direction jumps {jump:.0f} deg since {prev["time"]:%H:%M}')
        for k, lim in SHAPE_JUMP.items():
            if abs(row[k] - prev[k]) > lim:
                row['flags'].append(f'{k} jumps {row[k] - prev[k]:+.2f} since {prev["time"]:%H:%M}')
    if rows:
        rows[0].update({'speed': np.nan, 'euv_speed': np.nan, 'dir_jump': np.nan})


def report(rows):
    print(f'{"time":>5s} {"src":3s} {"h":>6s} {"kappa":>5s} {"eps":>5s} {"alpha":>5s} {"lat":>6s} {"lon":>6s} {"tilt":>6s} '
          f'{"rear":>6s} {"ring":>5s} {"dome":>5s} {"v_apex":>6s} {"v_EUV":>6s} {"stand":>6s}')
    f = lambda v, fmt: (fmt.format(v) if np.isfinite(v) else '-').rjust(len(fmt.format(0.0)))
    for r in rows:
        print(f'{r["time"]:%H:%M} {r["source"]:3s} {r["height"]:6.2f} {r["kappa"]:5.2f} {r["epsilon"]:+5.2f} {r["alpha"]:5.2f} '
              f'{r["lat"]:+6.1f} {r["lon"]:6.1f} {r["tilt"]:+6.1f} {r["rear"]:+6.2f} {f(r["ring_deg"], "{:5.1f}")} '
              f'{r["dome"]:5.2f} {f(r["speed"], "{:6.0f}")} {f(r["euv_speed"], "{:6.0f}")} {f(r["standoff"], "{:+6.2f}")}')
        for flag in r['flags']:
            print(f'        ! {flag}')
    n = sum(1 for r in rows if r['flags'])
    print(f'\n{len(rows)} time(s); {n} with flags' + ('' if n else ': nothing unphysical found'))
    print('(rear [R_sun]: inside the Sun -1 ... 1; ring: base-ring angular radius [deg]; dome: width / height; '
          'v [km/s]; stand: shock apex - flux-rope front [R_sun])')


def save_csv(rows):
    cols = ['time', 'source', 'height', 'kappa', 'epsilon', 'alpha', 'lat', 'lon', 'tilt', 'rear', 'ring_deg', 'ring_Mm',
            'dome', 'speed', 'euv_speed', 'dir_jump', 'cme_front', 'standoff', 'outside', 'flags']
    path = cfg.results_path('check_fits.csv')
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r['time'].isoformat()] + [r['source']] +
                       [f'{r[k]:.4g}' if np.isfinite(r[k]) else '' for k in cols[2:-1]] + ['; '.join(r['flags'])])
    print(f'saved {path}')


def figure(rows, data):
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5), layout='constrained')
    t = [r['time'] for r in rows]
    flagged = [r['time'] for r in rows if r['flags']]

    def mark(a):
        for tf in flagged:
            a.axvline(tf, color='C3', alpha=0.25, lw=6, zorder=0)
    # (a) heights
    a = ax[0, 0]
    for key, label in (('gui', 'shock: GUI'), ('fit', 'shock: fit')):          # the GUI ring under the fit's dot
        pts = [(cfg.stamp_date(s), d[key].height) for s, d in data.items() if key in d]
        if pts:
            kw = dict(mfc='w', mec='k', mew=1.3, ms=10, zorder=3) if key == 'gui' else dict(color='k', ms=6, zorder=4)
            a.plot(*zip(*pts), 'o', ls='none', label=label, **kw)
    cme = [(r['time'], r['cme_front']) for r in rows if np.isfinite(r['cme_front'])]
    if cme:
        a.plot(*zip(*cme), '^', color='C1', ms=7, label='flux-rope front')
    a.set_ylabel('apex height [R$_\\odot$]')
    a.set_title('heights (red bands: times with flags)')
    a.legend(fontsize=8)
    # (b) base ring
    a = ax[0, 1]
    a.plot(t, [r['ring_deg'] for r in rows], 'o-', color='C2')
    for r in rows[1:]:
        if np.isfinite(r['euv_speed']):
            a.annotate(f'{r["euv_speed"]:.0f}', (r['time'], r['ring_deg']), textcoords='offset points', xytext=(0, 7),
                       ha='center', fontsize=8, color='C2')
    a.set_ylabel('base-ring radius [deg from the apex]')
    a.set_title('EUV-wave footprint (labels: expansion speed, km/s)')
    # (c) shape
    a = ax[0, 2]
    for k, col in (('kappa', 'C0'), ('epsilon', 'C1'), ('alpha', 'C4')):
        a.plot(t, [r[k] for r in rows], 'o-', color=col, label=k)
        for v in RANGES[k]:
            a.axhline(v, color=col, ls=':', lw=0.8)
    a.set_title('shape (dotted: sanity bounds)')
    a.legend(fontsize=8)
    # (d) direction, (e) tilt
    a = ax[1, 0]
    a.plot(t, [r['lat'] for r in rows], 'o-', label='latitude')
    lon = np.degrees(np.unwrap(np.radians([r['lon'] for r in rows])))
    a.plot(t, lon - 360.0 * (np.mean(lon) > 180), 'o-', label='Stonyhurst longitude')
    a.set_ylabel('[deg]')
    a.set_title('apex direction')
    a.legend(fontsize=8)
    a = ax[1, 1]
    a.plot(t, [r['tilt'] for r in rows], 'o-', color='C5')
    a.set_ylabel('tilt [deg]')
    a.set_title('tilt')
    # (f) the base on the Sun
    a = ax[1, 2]
    a.axhspan(-1.0, cfg.FOOTPRINT_R, color='C2', alpha=0.12, label='base on the Sun')
    a.plot(t, [r['rear'] for r in rows], 'o-', color='k', label='rear point')
    a.plot(t, [r['dome'] for r in rows], 's--', color='C7', label='dome width / height')
    a.set_title('rear point [R$_\\odot$] and dome width / height')
    a.legend(fontsize=8)
    for a in ax.flat:
        mark(a)
        a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        a.set_xlabel('time [UT]')
    fig.suptitle(f'{cfg.EVENT_ID}: physical checks of the shock fits')
    png = cfg.figure_path('check_fits.png')
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f'saved {png}')


def main():
    data = load()
    if not data:
        raise SystemExit(f'no shock_<stamp>.json or shock_fit_<stamp>.json in "{cfg.RESULTS_DIR}"')
    rows = [per_time(s, d) for s, d in data.items() if 'fit' in d or 'gui' in d]
    across(rows)
    report(rows)
    save_csv(rows)
    figure(rows, data)


if __name__ == '__main__':
    main()
