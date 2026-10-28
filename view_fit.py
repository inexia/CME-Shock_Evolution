"""
view_fit.py -- look at the fit for TIME (event_config.py): on every image with your traces, and in 3D.

Run:  python view_fit.py
EUV row (if EUV_ON = True): your EUV clicks (cyan), the footprint of the ELLIPSOID fit on the r = FOOTPRINT_R
    sphere (red) and its faint grid -- the EUV wave is compared with the ellipsoid only.
Coronagraph row: your clicks (cyan), final fit outline (orange), ellipsoid-only outline (dashed blue), faint
    grid of the final fit, and the RMS distance between the traces and each outline.
3D panel: final surface coloured by heliocentric distance, the Sun, observer directions, and a dot at the centre
    of each dent (magenta) and bulge (orange-red) with its depth D.
Saved to figure/fit_view_<stamp>.png.
"""
import json
import os

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.gridspec import GridSpec

import event_config as cfg
from gcs_shock_gui import load_view, image_norm, panel_title, clean_axes, plot_xyz
from trace_outline import clear_signal_wakeup_fd
import shock_fit as sf
from deformable_shock import DeformableShock

# ================================ SETTINGS ================================
SHOW_GRID = True                              # faint wire grid of the fitted surfaces on the images
SHOW = True                                   # open the interactive window (3D panel can be rotated)
# ==========================================================================


def load_model(path):
    return DeformableShock.from_dict(json.loads(path.read_text())) if path.exists() else None


def outline_skycoord(model, observer, smap, rho_min, n_bins=360):
    X, Y = observer.project(model.mesh(181, 361))
    psi, rho = sf.hard_outline(X, Y, n_bins)
    rho = np.where(rho > rho_min, rho, np.nan)
    psi, rho = np.r_[psi, psi[:1]], np.r_[rho, rho[:1]]
    tx = np.degrees(rho * np.cos(psi) / observer.dist) * 3600 * u.arcsec
    ty = np.degrees(rho * np.sin(psi) / observer.dist) * 3600 * u.arcsec
    return SkyCoord(tx, ty, frame=smap.coordinate_frame)


def trace_rms(model, od):
    _, rho = sf.model_outline(model, od.observer, n_bins=od.n_bins)
    d = (rho - od.rho)[od.mask]
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d ** 2))) if d.size else np.nan


def image_panel(fig, spec, kind, sc, det, fit, ell, first):
    smap = load_view(kind, sc, det, cfg.date())
    ax = fig.add_subplot(spec, projection=smap)
    smap.plot(axes=ax, cmap='Greys_r', norm=image_norm(smap, kind))
    clean_axes(ax)
    obs = sf.observer_from_map(smap, sc)
    kw = dict(scalex=False, scaley=False)
    date = smap.date
    path = cfg.trace_path(sc, det)
    tobs = tx = ty = None
    if os.path.exists(path):
        tobs, tx, ty = sf.read_trace_csv(path, sc)
        ax.plot_coord(SkyCoord(tx * u.arcsec, ty * u.arcsec, frame=smap.coordinate_frame), 'o', mfc='none',
                      mec='cyan', mew=0.8, ms=4, label='traced', **kw)
    line3 = 'no trace file'
    if kind == 'euv':
        base = ell or fit.ellipsoid()
        if SHOW_GRID:
            plot_xyz(ax, base.mesh(19, 37, order='grid'), obs, date, '-', color='orange', lw=0.3, alpha=0.4)
        for k, line in enumerate(sf.ellipsoid_footprint(base, cfg.FOOTPRINT_R)):
            plot_xyz(ax, line, obs, date, '-', color='red', lw=1.8, label='ellipsoid footprint' if k == 0 else None)
        if tx is not None:
            pts, hit = sf.deproject_to_sphere(tobs, tx, ty, cfg.FOOTPRINT_R)
            if hit.any():
                fp = sf.FootprintData(pts[hit])
                r = np.sqrt(sf.ShockFitter([], [fp]).data_cost(base) / hit.sum()) * fp.sigma
                line3 = f'footprint rms {r:.2f} R$_\\odot$ ({hit.sum()} on-disk points)'
            else:
                line3 = 'no on-disk points'
    else:
        if SHOW_GRID:
            plot_xyz(ax, fit.mesh(19, 37, order='grid'), obs, date, '-', color='orange', lw=0.3, alpha=0.35)
        rho_min = sf.OCCULTER_RSUN.get(det, 2.5)
        if ell is not None:
            ax.plot_coord(outline_skycoord(ell, obs, smap, rho_min), '--', color='deepskyblue', lw=1.5,
                          label='ellipsoid fit', **kw)
        ax.plot_coord(outline_skycoord(fit, obs, smap, rho_min), '-', color='orange', lw=2.0, label='final fit', **kw)
        if tx is not None:
            psi, rho = sf.trace_to_polar(tobs, tx, ty)
            od = sf.outline_from_trace(tobs, psi, rho, rho_min=rho_min)
            line3 = f'trace rms: fit {trace_rms(fit, od):.2f}' + (f', ellipsoid {trace_rms(ell, od):.2f}' if ell else '')
            print(f'{sc:5s} {det:8s}: {line3}')
            line3 += r' R$_\odot$'
    if first:
        ax.legend(loc='lower left', fontsize=7, framealpha=0.7)
    ax.set_title(f'{panel_title(smap, kind, sc, det)}\n{line3}', fontsize=9)
    return obs


def panel_3d(fig, spec, fit, ell, observers):
    ax = fig.add_subplot(spec, projection='3d')
    ax.computed_zorder = False
    S = fit.surface(61, 121)
    xyz = S['xyz']
    r = np.linalg.norm(xyz, axis=-1)
    norm = colors.Normalize(vmin=1.0, vmax=float(r.max()))
    cmap = plt.get_cmap('viridis')
    ax.plot_surface(xyz[..., 0], xyz[..., 1], xyz[..., 2], facecolors=cmap(norm(r)), rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False, alpha=0.45, zorder=1)
    u_, v_ = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
    ax.plot_surface(np.cos(u_) * np.sin(v_), np.sin(u_) * np.sin(v_), np.cos(v_), color='red', linewidth=0, zorder=0)
    L = float(np.abs(xyz).max()) * 1.1
    seen = set()
    for obs in observers:
        if obs.name in seen:
            continue
        seen.add(obs.name)
        d = obs.position / np.linalg.norm(obs.position)
        ax.plot([0, L * d[0]], [0, L * d[1]], [0, L * d[2]], color='k', lw=1, zorder=3)
        ax.text(*(L * 1.05 * d), obs.name, fontsize=9)
    for th, ph, sg, dp in fit.dents:                      # each feature: a dot at its centre with its depth
        col, label = ('magenta', f' dent, D = {dp:+.2f}') if dp >= 0 else ('orangered', f' bulge, D = {dp:+.2f}')
        p = fit.position(np.radians(th), np.radians(ph))
        ax.scatter(*p, color=col, s=40, depthshade=False, zorder=6)
        ax.text(*(p * 1.03), label, color=col, fontsize=8, zorder=7)
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L, L)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=30, azim=-60)
    ax.set_xlabel(r'X (Earth) [R$_\odot$]')
    ax.set_ylabel(r'Y [R$_\odot$]')
    ax.set_zlabel(r'Z (north) [R$_\odot$]')
    ax.set_title(f'fitted surface, HGS, {cfg.TIME} UT\ncolour = heliocentric distance', fontsize=10)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.5, pad=0.1, label=r'r [R$_\odot$]')


def main():
    fit = load_model(cfg.results_path(f'shock_fit_{cfg.stamp()}.json'))
    if fit is None:
        raise SystemExit(f'no fit for {cfg.TIME}: run  python fit_event.py  first')
    ell = load_model(cfg.results_path(f'shock_fit_ellipsoid_{cfg.stamp()}.json'))
    print(f'final fit ({fit.time}): {fit}')
    for th, ph, sg, dp in fit.dents:
        print(f'  {"dent " if dp >= 0 else "bulge"}: {th:.1f} deg from apex, azimuth {ph % 360:.1f}, width {sg:.1f} deg, '
              f'D = {dp:+.2f} (radius x {np.exp(-dp):.2f})')
    vs = cfg.views()
    rows = ['euv', 'coronagraph'] if any(k == 'euv' for k, _, _ in vs) else ['coronagraph']
    scs = cfg.parse_spacecraft()
    fig = plt.figure(figsize=(4.3 * (len(scs) + 1.4), 4.8 * len(rows)), layout='constrained')
    gs = GridSpec(len(rows), len(scs) + 1, figure=fig, width_ratios=[1] * len(scs) + [1.4])
    observers, first = [], True
    for r, kind in enumerate(rows):
        for i, sc in enumerate(scs):
            det = next(d for k, s, d in vs if k == kind and s == sc)
            observers.append(image_panel(fig, gs[r, i], kind, sc, det, fit, ell, first))
            first = False
    panel_3d(fig, gs[:, len(scs)], fit, ell, observers)
    png = cfg.figure_path(f'fit_view_{cfg.stamp()}.png')
    fig.savefig(png, dpi=130, bbox_inches='tight')
    print(f'saved {png}')
    if SHOW:
        clear_signal_wakeup_fd()
        plt.show()


if __name__ == '__main__':
    main()
