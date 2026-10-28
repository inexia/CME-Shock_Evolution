"""
trace_outline.py -- click the shock in each view and save the points for fit_event.py.

Run:  python trace_outline.py        (event settings: event_config.py)
One window per coronagraph view: click along the shock boundary (the outer edge of the diffuse front).
Left click = add point, right click = remove last, Enter (or middle click) = finish this view.
Coronagraph views show radial guide lines from Sun centre every GUIDE_STEP deg of position angle (every 20 deg
highlighted and labelled, PA from north through east): click where each line crosses the shock edge. 4 deg is
about half a fit bin (7.5 deg) -- finer gains nothing, the edge itself is the limit. Where the edge runs nearly
along a line (flanks reaching down to the occulter), click the edge itself; extra clicks are fine.
Leave untraceable parts (occulter, pylon, too faint) untraced.
Output: fitted results/<id>/semi/trace_<stamp>_<SC>_<DETECTOR>.csv (Tx, Ty in arcsec; view kind, image time and
observer in the header).
A view that already has a trace file (EXISTING): 'ask' in the terminal, 'skip' it, 'add' points to it (its points
are shown on the image while you click; the new ones are added), or 'replace' it. Before a file is changed, the old
one is copied to fitted results/<id>/semi/trace_backup/ with the time of the change, so nothing is lost.
"""
import datetime as dt
import os
import shutil
import signal
from pathlib import Path

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

import event_config as cfg
from gcs_shock_gui import load_view, image_norm, panel_title     # Helioviewer loading as in the GUI
import shock_fit as sf

# ================================ SETTINGS ================================
EXISTING = 'ask'                      # a view with a trace file: 'ask' (in the terminal), 'skip', 'add', 'replace'
OVERWRITE = False                     # (older setting) True = 'replace'
GUIDE_STEP = 10.                      # radial guide lines every this many deg of position angle; 0 = none
GUIDE_LABEL_EVERY = 20                # highlight and label every this many deg (rounded to a multiple of GUIDE_STEP)
GUIDE_ALPHA = 0.6                     # opacity of the 4-deg lines; lower it if they hide a faint edge
FOV_RSUN = {'C2': 6.0, 'C3': 30.0, 'COR1': 4.0, 'COR2': 15.0}   # outer field edge, for line length / labels
# ==========================================================================


def clear_signal_wakeup_fd():
    """
    sunpy's downloads (parfive / asyncio) leave the process-wide signal wakeup fd pointing at their
    event loop's socket; once that socket is closed, matplotlib's Qt event loop (used by ginput) fails
    when restoring it: OSError [Errno 9] Bad file descriptor. Nothing else needs it here, so clear it.
    """
    try:
        signal.set_wakeup_fd(-1)
    except (ValueError, OSError):
        pass


def draw_guides(ax, smap, detector):
    """Radial lines from Sun centre every GUIDE_STEP deg of position angle (PA from solar north through east)."""
    if not GUIDE_STEP:
        return
    obs = sf.observer_from_map(smap)
    arcsec = lambda rho: np.degrees(np.asarray(rho) / obs.dist) * 3600.0          # projected R_sun -> arcsec
    fov = FOV_RSUN.get(detector, 30.0)
    r = np.array([sf.OCCULTER_RSUN.get(detector, 1.0), 1.6 * fov])
    n = int(round(360.0 / GUIDE_STEP))
    every = max(1, int(round(GUIDE_LABEL_EVERY / GUIDE_STEP)))              # labelled lines lie on the grid
    for k in range(n):
        pa = k * GUIDE_STEP
        major = k % every == 0
        p = np.radians(pa)
        line = SkyCoord(-arcsec(r) * np.sin(p) * u.arcsec, arcsec(r) * np.cos(p) * u.arcsec, frame=smap.coordinate_frame)
        ax.plot_coord(line, '-', color='yellow' if major else 'cyan', lw=1.0 if major else 0.7,
                      alpha=min(1.0, GUIDE_ALPHA + 0.2) if major else GUIDE_ALPHA, scalex=False, scaley=False)
        if major:
            lab = SkyCoord(-arcsec(0.9 * fov) * np.sin(p) * u.arcsec, arcsec(0.9 * fov) * np.cos(p) * u.arcsec,
                           frame=smap.coordinate_frame)
            x, y = smap.world_to_pixel(lab)
            ax.text(float(x.value), float(y.value), f'{pa:.0f}\u00b0', color='yellow', fontsize=8,
                    ha='center', va='center', alpha=0.9)


def ask(path, n):
    """What to do with an existing trace file, asked in the terminal."""
    choices = {'s': 'skip', 'a': 'add', 'r': 'replace'}
    while True:
        a = input(f'{Path(path).name} exists ({n} points): [s]kip, [a]dd points, [r]eplace? [s] ').strip().lower()
        if not a or a[0] in choices:
            return choices[a[0] if a else 's']


def backup(path):
    """Copy a trace file to trace_backup/ (named with the time of the change) before it is changed."""
    folder = Path(path).parent / 'trace_backup'
    folder.mkdir(exist_ok=True)
    stem = f'{Path(path).stem}_{dt.datetime.now():%Y%m%d_%H%M%S}'
    dest, k = folder / f'{stem}.csv', 1
    while dest.exists():                                          # two changes within a second: keep both
        k += 1
        dest = folder / f'{stem}_{k}.csv'
    shutil.copy(path, dest)
    return dest


def trace_one(kind, sc, detector, date):
    out = cfg.trace_path(sc, detector)
    mode, old = 'new', None
    if os.path.exists(out):
        _, tx0, ty0 = sf.read_trace_csv(out)
        mode = 'replace' if OVERWRITE else EXISTING
        if mode == 'ask':
            mode = ask(out, len(tx0))
        if mode == 'skip':
            print(f'{out} exists, skipped')
            return
        old = (tx0, ty0)
    smap = load_view(kind, sc, detector, date)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection=smap)
    smap.plot(axes=ax, cmap='Greys_r', norm=image_norm(smap, kind))
    draw_guides(ax, smap, detector)
    if old is not None:                                           # the existing points, for reference
        c0 = SkyCoord(old[0] * u.arcsec, old[1] * u.arcsec, frame=smap.coordinate_frame)
        ax.plot_coord(c0, 'x', color='orange', ms=7, mew=1.5, scalex=False, scaley=False,
                      label=f'existing trace ({len(old[0])} points)' + (' -- kept' if mode == 'add' else ' -- replaced'))
        ax.legend(loc='lower left', fontsize=8)
    hint = {'add': 'click points to ADD to the existing trace', 'replace': 'retrace: click the shock edge anew'}.get(
        mode, 'click where the guide lines cross the shock edge')
    ax.set_title(f'{panel_title(smap, kind, sc, detector)}\n{hint}; Enter to finish', fontsize=10)
    clear_signal_wakeup_fd()           # after the downloads, right before matplotlib's event loop
    pts = plt.ginput(n=-1, timeout=0)
    plt.close(fig)
    if mode == 'add' and not pts:
        print(f'{sc} {detector}: no points added, {out} unchanged')
        return
    if mode != 'add' and len(pts) < 3:
        print(f'{sc} {detector}: fewer than 3 points clicked, nothing saved' + (f' ({out} unchanged)' if old else ''))
        return
    c = smap.pixel_to_world(np.array([p[0] for p in pts]) * u.pix, np.array([p[1] for p in pts]) * u.pix)
    tx, ty = c.Tx.to_value(u.arcsec), c.Ty.to_value(u.arcsec)
    if mode == 'add':
        tx, ty = np.r_[old[0], tx], np.r_[old[1], ty]
    if old is not None:
        print(f'old trace kept as {backup(out)}')
    obs = sf.observer_from_map(smap)
    header = (f'# {smap.nickname} {smap.date.isot}\n'
              f'# KIND={kind} SC={sc} DETECTOR={detector} IMAGE_TIME={smap.date.isot[:19]}\n'
              f'# HGLN_OBS={obs.lon:.4f} HGLT_OBS={obs.lat:.4f} DSUN_RSUN={obs.dist:.3f}\n'
              '# Tx_arcsec, Ty_arcsec')
    np.savetxt(out, np.c_[tx, ty], delimiter=',', header=header, comments='', fmt='%.3f')
    print(f'saved {len(tx)} points to {out}' + (f' ({len(pts)} added)' if mode == 'add' else ''))


def main():
    for kind, sc, detector in cfg.views():
        if kind == 'coronagraph':                                 # the coronagraph views only
            trace_one(kind, sc, detector, cfg.date())


if __name__ == '__main__':
    main()
