"""
evolution_front_width.py -- how wide is the SHOCK, as opposed to the front? Coronagraphs show a wide front, but
in situ a wide shock is not always seen: the shock strength (M_f, M_A or the compression ratio X) falls toward the
flanks, so the outer parts of the front may no longer be a shock, or only a weak one. This script measures, through
the whole evolution, the part of the front that is still a shock, in the ecliptic (the plane z = 0 of the x-y panels):

  nose       the front's point in the plane farthest from the Sun
  ends       from the nose, along the front's cut through the plane, in both directions, the points where the
             criterion (CRITERION: 'Mf', 'MA' or 'X') falls to THRESHOLD (interpolated between grid points), or the
             front itself ends -- the two lines drawn from the Sun go to these points
  widths     effective width = the angle at the Sun between the two ends; geometric width = the same for the whole
             front cut (what a coronagraph shows in that plane)
  area       the fraction of the front's area (3-D) where the criterion is above the threshold

Run:  python evolution_front_width.py      (after shock_properties_sse/_dbm/_gsd.py; folders from event_config.py)
For each version (VERSIONS: the corona followed by the SSE, DBM or GSD fronts) and method:
  figure/<id>/front_width/<version>_<method>/front_width<h>.png   one per step: x-y over the wind density and speed,
                         the front's cut coloured by the criterion, the two lines from the Sun to the shock's ends
                         (dashed beyond, to the edge), the widths, and the targets (positions only)
  figure/<id>/front_width/front_width_<method>.png   the widths, the ends' longitudes (with the targets') and the
                         shock's share of the front's area against apex height, for all versions
  fitted results/<id>/final/front_width_<version>_<method>.csv
The corona part is shared by the versions: its figures are drawn once.
"""
import csv

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
import evolution_cobpoint as ec

# ================================ SETTINGS ================================
VERSIONS = ['sse', 'dbm', 'gsd']
METHODS = ['constant', 'varied']
CRITERION = 'Mf'                       # 'Mf', 'MA' or 'X': the shock ends where this falls to THRESHOLD
THRESHOLD = 1.0                        # e.g. 1 for M_f or M_A; for X (compression ratio) 1 means no compression --
                                       # a value like 1.5 asks for a shock strong enough to be seen in situ
PLOT_STEPS = True                      # one x-y figure per step
TARGETS = 'Earth'                      # shown on the figures (positions only), as in evolution_cobpoint.py
EXTEND_LINES = True                    # the two lines continued (dashed) beyond the ends, to the edge of the box
# ==========================================================================
LINE_COLOR = '#FF4136'


def _lon(p):
    return float(np.degrees(np.arctan2(p[1], p[0])))


def _angle(a, b):
    """Angle at the Sun between two points of the plane [deg], 0..180."""
    return float(abs((_lon(a) - _lon(b) + 180.0) % 360.0 - 180.0))


def _end(xyz, val, i, j):
    """The point between i (criterion >= threshold) and its neighbour j where the criterion reaches the threshold
    (linear in the criterion); at j itself if the value there is missing (the front or the shock region ends)."""
    if not np.isfinite(val[j]):
        return xyz[i]
    f = (val[i] - THRESHOLD) / (val[i] - val[j]) if val[i] != val[j] else 0.0
    return xyz[i] + float(np.clip(f, 0.0, 1.0)) * (xyz[j] - xyz[i])


def widths(res):
    """In the plane z = 0: (nose, end_a, end_b, effective width, geometric width, geometric ends) or None."""
    best = None
    for xyz, val, on in sp.cut_curves(res, CRITERION, 2, 0.0):
        idx = np.flatnonzero(on)
        if idx.size < 2:
            continue
        r = np.linalg.norm(xyz[:, :2], axis=1)
        k = idx[np.argmax(r[idx])]
        if best is None or r[k] > best[0]:
            best = (r[k], xyz, val, on, k)
    if best is None:
        return None
    _, xyz, val, on, k = best
    n = len(xyz)
    closed = np.linalg.norm(xyz[0] - xyz[-1]) < 1e-6
    good = on & np.isfinite(val) & (val >= THRESHOLD)

    def walk(step, mask):
        """From the nose k along the curve while mask holds: (last index inside, first index outside or None)."""
        i = k
        for _ in range(n):
            j = i + step
            if closed:
                j %= n
            elif not 0 <= j < n:
                return i, None
            if j == k or not mask[j]:
                return i, j
            i = j
        return i, None

    (ga, gja), (gb, gjb) = walk(-1, on), walk(+1, on)
    geom = (xyz[ga], xyz[gb])
    if not good[k]:                                   # no shock even at the nose
        return xyz[k], None, None, 0.0, _angle(*geom), geom
    (ia, ja), (ib, jb) = walk(-1, good), walk(+1, good)
    end_a = _end(xyz, val, ia, ja) if ja is not None else xyz[ia]
    end_b = _end(xyz, val, ib, jb) if jb is not None else xyz[ib]
    eff = _angle(end_a, xyz[k]) + _angle(xyz[k], end_b)
    return xyz[k], end_a, end_b, eff, _angle(geom[0], xyz[k]) + _angle(xyz[k], geom[1]), geom


def area_fraction(res):
    """The share of the front's area where the criterion is above the threshold."""
    w = sp.area_weights(res['xyz'])
    on = res.get('on_front', np.isfinite(res['xyz'][..., 0]))
    v = res[CRITERION]
    good = on & np.isfinite(v) & (v >= THRESHOLD)
    tot = float(np.nansum(np.where(on, w, 0.0)))
    return float(np.nansum(np.where(good, w, 0.0)) / tot) if tot > 0 else np.nan


def step_figure(kind, method, res, lim, span, maps, bg_lim, wd, targets, path):
    q = CRITERION
    cmap = plt.get_cmap(sp.CMAPS.get(q, 'plasma'))
    norm = Normalize(*lim[q])
    fig, axs = plt.subplots(1, 2, figsize=(15, 7.6), layout='constrained')
    stroke = [pe.withStroke(linewidth=2.5, foreground='k')]
    for ax, field in zip(axs, ('nr2', 'ur')):
        m = maps['xy']
        if field == 'ur':
            im = ax.pcolormesh(m['a'], m['b'], m['ur'], shading='auto', cmap=sp.BG_CMAPS['ur'],
                               vmin=bg_lim['ur'][0], vmax=bg_lim['ur'][1])
            blabel = 'solar wind: u$_r$ [km/s]'
        else:
            im = ax.pcolormesh(m['a'], m['b'], m['nr2'], shading='auto', cmap=sp.BG_CMAPS['nr2'],
                               norm=LogNorm(*bg_lim['nr2']))
            blabel = 'solar wind: n (r / 1 AU)$^2$ [cm$^{-3}$]'
        fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, pad=0.02, label=blabel)
        for xyz, val, on in sp.cut_curves(res, q, 2, 0.0):
            keep = on[:-1] & on[1:]
            seg = np.stack([xyz[:-1, :2], xyz[1:, :2]], axis=1)[keep]
            sv = (0.5 * (val[:-1] + val[1:]))[keep]
            cols = cmap(norm(np.nan_to_num(sv, nan=lim[q][0])))
            cols[~np.isfinite(sv) | (sv < THRESHOLD)] = (0.7, 0.7, 0.7, 1.0)      # not a shock by the criterion: grey
            ax.add_collection(LineCollection(seg, colors='k', linewidths=5.0, zorder=3))
            ax.add_collection(LineCollection(seg, colors=cols, linewidths=3.0, zorder=4))
        ax.add_patch(plt.Circle((0, 0), 1.0, color='orange', zorder=5))
        if wd is not None and wd[1] is not None:
            for e in (wd[1], wd[2]):
                ax.plot([0, e[0]], [0, e[1]], '-', color=LINE_COLOR, lw=2.2, zorder=6)
                if EXTEND_LINES:
                    u = e[:2] / np.linalg.norm(e[:2])
                    far = u * 2.0 * span
                    ax.plot([e[0], far[0]], [e[1], far[1]], '--', color=LINE_COLOR, lw=1.1, alpha=0.8, zorder=6)
                ax.plot(e[0], e[1], 'o', color=LINE_COLOR, mec='k', ms=7, zorder=7)
        for i, (label, pos) in enumerate(targets):
            if pos is None:
                continue
            p = pos if np.max(np.abs(pos[:2])) <= span else pos[:2] / np.max(np.abs(pos[:2])) * 0.97 * span
            col = ec.TARGET_COLORS[i % len(ec.TARGET_COLORS)]
            ax.plot(p[0], p[1], 'o', color=col, mec='k', ms=9, zorder=8)
            ax.text(p[0], p[1] + 0.03 * span, f' {label}', color=col, fontsize=10, path_effects=stroke, zorder=9,
                    ha='left' if p[0] < 0.6 * span else 'right')
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_aspect('equal')
        ax.set_xlabel('x [R$_\\odot$] (to Earth)')
        ax.set_ylabel('y [R$_\\odot$]')
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axs, shrink=0.6, label=f'shock: {sp.LABELS.get(q, q)} (grey: below {THRESHOLD:g})')
    eff = 'no shock in the plane' if wd is None else (f'shock {wd[3]:.0f}$\\degree$ wide' if wd[1] is not None
                                                       else 'no shock at the nose')
    geo = '' if wd is None else f' of a front {wd[4]:.0f}$\\degree$ wide'
    fig.suptitle(f'{cfg.EVENT_ID} {kind}/{method}: apex {res["height"]:.1f} R$_\\odot$ at '
                 f'{str(res["time"]).replace("T", " ")[:19]} UT -- plane z = 0: {eff}{geo} '
                 f'({CRITERION} $\\geq$ {THRESHOLD:g})', fontsize=12)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def run_version(version, method, bg, where, targets, drawn):
    """Rows (one per step) of one version: the corona, then its interplanetary fronts."""
    kinds = ['corona', ec.VERSION_SERIES[version]]
    rows, last_time = [], None
    for kind in kinds:
        series = sp.load_series(method, kind)
        if not series:
            note = sp.failure_note(kind, method) if kind != 'corona' else None
            print(f'  [{version}/{method}] no {sp.SERIES_FILES[kind][0]} series' + (f': {note}' if note else '') + ' -- skipped')
            continue
        results = [ec.load_props(kind, method, series, k, bg) for k in range(len(series))]
        span = sp.axis_span(results)
        lim = sp.colour_limits(results, sp.COLOR_LIMITS_IP if kind != 'corona' else None)
        out = cfg.FIGURE_DIR / 'front_width' / (f'corona_{method}' if kind == 'corona' else f'{version}_{method}')
        draw = PLOT_STEPS and (kind != 'corona' or (method not in drawn))
        if draw:
            out.mkdir(parents=True, exist_ok=True)
        bg_lim = None
        for k, (step, res) in enumerate(zip(series, results)):
            if last_time is not None and step['time'] <= last_time:
                continue                                   # the interplanetary fronts start where the corona ends
            wd = widths(res)
            frac = area_fraction(res)
            row = {'version': version, 'series': kind, 'apex_height_rsun': round(res['height'], 3),
                   'time': str(res['time']), 'width_shock_deg': np.nan, 'width_front_deg': np.nan,
                   'end_a_lon_deg': np.nan, 'end_b_lon_deg': np.nan, 'nose_lon_deg': np.nan,
                   'shock_area_fraction': round(frac, 4) if np.isfinite(frac) else np.nan}
            if wd is not None:
                row.update({'width_front_deg': round(wd[4], 2), 'nose_lon_deg': round(_lon(wd[0]), 2),
                            'width_shock_deg': round(wd[3], 2)})
                if wd[1] is not None:
                    row.update({'end_a_lon_deg': round(_lon(wd[1]), 2), 'end_b_lon_deg': round(_lon(wd[2]), 2)})
            rows.append(row)
            if draw:
                maps = sp.background_maps(bg, res['time'], span)
                bg_lim = bg_lim or sp.background_limits(maps)
                tpos = [(label, where.position(label, step['time_iso'])) for label in where.pos] if where else []
                step_figure(kind, method, res, lim, span, maps, bg_lim, wd, tpos,
                            out / f'front_width{int(round(res["height"] * 10)):0{sp.SERIES_FILES[kind][1]}d}.png')
        if series:
            last_time = series[-1]['time']
        if draw:
            print(f'  saved {len(series)} figures in {out}')
    if PLOT_STEPS:
        drawn.add(method)
    return rows


def summary_figure(method, by_version, target_lons):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), layout='constrained')
    for i, (version, rows) in enumerate(by_version.items()):
        col = f'C{i}'
        h = np.array([r['apex_height_rsun'] for r in rows])
        get = lambda k: np.array([r[k] for r in rows], float)
        ax[0].plot(h, get('width_shock_deg'), 'o-', color=col, ms=3, label=f'{version}: shock')
        ax[0].plot(h, get('width_front_deg'), ':', color=col, lw=1.2, label=f'{version}: front')
        for k, key in enumerate(('end_a_lon_deg', 'end_b_lon_deg')):
            ax[1].plot(h, get(key), '-', color=col, lw=1.4, label=f'{version}: shock ends' if k == 0 else None)
        ax[1].plot(h, get('nose_lon_deg'), ':', color=col, lw=1)
        ax[2].plot(h, get('shock_area_fraction'), 'o-', color=col, ms=3, label=version)
    for j, (label, lon) in enumerate(target_lons.items()):
        ax[1].axhline(lon, color=ec.TARGET_COLORS[j % len(ec.TARGET_COLORS)], ls='--', lw=1, label=label)
    ax[0].set_title(f'width in the ecliptic: shock ({CRITERION} $\\geq$ {THRESHOLD:g}, solid) and front (dotted)')
    ax[0].set_ylabel('[deg]')
    ax[1].set_title('Stonyhurst longitude of the shock\'s ends (nose: dotted)')
    ax[1].set_ylabel('[deg]')
    ax[2].set_title('share of the front\'s area that is a shock (3-D)')
    ax[2].set_ylim(0, 1.02)
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    for a in ax:
        a.set_xscale('log')
        a.xaxis.set_major_formatter(ScalarFormatter())
        a.xaxis.set_minor_formatter(NullFormatter())
        a.set_xlabel('apex height [R$_\\odot$]')
        a.legend(fontsize=7)
    fig.suptitle(f'{cfg.EVENT_ID} {method}: the shock\'s width against the front\'s')
    out = cfg.FIGURE_DIR / 'front_width'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'front_width_{method}.png', dpi=120)
    plt.close(fig)
    print(f'saved {out / f"front_width_{method}.png"}')


def save_csv(version, method, rows):
    path = cfg.final_path(f'front_width_{version}_{method}.csv')
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'  saved {path}')


def main():
    bg = bk.load_background()
    print(bg.summary() if hasattr(bg, 'summary') else bg.name)
    targets = ec.parse_targets(TARGETS) if TARGETS else []
    times = [s['time_iso'] for m in METHODS for v in VERSIONS for kind in ('corona', ec.VERSION_SERIES[v])
             for s in sp.load_series(m, kind)]
    where = ec.TargetPositions(targets, times) if (targets and times) else None
    drawn = set()
    for method in METHODS:
        by_version = {}
        for version in VERSIONS:
            print(f'\n[{version} / {method}] {CRITERION} >= {THRESHOLD:g}')
            rows = run_version(version, method, bg, where, targets, drawn)
            if rows:
                save_csv(version, method, rows)
                by_version[version] = rows
        if by_version:
            t_lons = {}
            if where:
                for label in where.pos:
                    p = next((where.position(label, t) for t in times if where.position(label, t) is not None), None)
                    if p is not None:
                        t_lons[label] = _lon(p)
            summary_figure(method, by_version, t_lons)
    ec.print_shock_at_targets(ec.shock_at_targets(bg, where, VERSIONS, METHODS))


if __name__ == '__main__':
    main()
