"""
experiment_identifiability.py
=============================
Go/no-go experiment for the irregular-shock reconstruction:
can 2-3 image outlines (optionally plus the EUV-wave footprint) recover a streamer-scale dent
in the shock surface well enough to give cobpoint normals to ~10 degrees?

Truth: a PyThea ellipsoid with one Gaussian dent (sigma 20 deg, depth 0.3 in ln R) on the flank,
at four azimuths, so that the dent sits on a limb for some viewpoints and between limbs for others.
Viewing geometries: three views at ~120 deg (2012-03-07), two views at 60 deg (STEREO-A / Earth
today), and two views plus the footprint of the surface on the r = 1.05 R_sun sphere (EUV wave).
Methods: ellipsoid only (PyThea baseline), ellipsoid + one compact dent (multi-start), and
optionally ellipsoid + spherical harmonics to L=4 (shown to leak into unobserved regions).

Outputs (in OUT): identifiability_results.csv, identifiability_<case>.png (normal-error maps of
the dent fit, noise-free), identifiability_summary.png (band normal-error RMS vs dent azimuth).
Run:  python experiment_identifiability.py      (settings below; QUICK=True ~ 1 h)
"""
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import shock_fit as sf
from deformable_shock import DeformableShock

# ================================ SETTINGS ================================
QUICK = True                     # reduced run; set False for the full Monte Carlo
COMPARE_HARMONIC = False         # also fit spherical harmonics to L=4 (slow; documents the leakage failure)
NOISE = [0.0, 0.05]              # outline noise RMS [R_sun] (Zhou et al. tracing scatter ~0.03-0.08)
N_MC = 2 if QUICK else 30        # Monte Carlo draws per noisy configuration
RHO_MIN = 2.5                    # occulter: outline bins below this projected radius are unobserved
SIGMA_OUT = 0.05                 # assumed outline uncertainty used to weight the fit [R_sun]
DENT_THETA, DENT_SIGMA, DENT_DEPTH = 70.0, 20.0, 0.3   # dent: angle from apex, width, depth (ln R)
DENT_PHIS = [0.0, 90.0, 180.0, 270.0]                 # dent azimuths swept (position relative to the limbs)
BAND = (1.5, 5.0)                # heliocentric band where cobpoints form [R_sun]
OUT = Path('.')
# ==========================================================================

# ellipsoid baselines (PyThea form, converted by from_pythea): the late one reaches down below the EUV sphere on the back, as the 00:54 UT
# shock did (Zhou et al. cobpoints at 1.4-2.7 R_sun); the early one is the ~00:25 UT stage
BASES = {
    'late_apex13': dict(height=13.5, kappa=0.72, epsilon=0.548, alpha=1.2, lat=15.0, lon=325.0, tilt=10.0),
    'early_apex5': dict(height=5.0, kappa=0.7, epsilon=0.2, alpha=1.1, lat=15.0, lon=325.0, tilt=10.0),
}
GEOMETRIES = {
    '3 views 120deg': dict(views=sf.event_2012_03_07(), footprint=False),
    '2 views 60deg': dict(views=sf.two_views(60.0), footprint=False),
    '2 views 60deg + EUV': dict(views=sf.two_views(60.0), footprint=True),
}
METHODS = ['ellipsoid', 'dent'] + (['harmonic_L4'] if COMPARE_HARMONIC else [])


def truth_model(case, dent_phi):
    return DeformableShock.from_pythea(**BASES[case], dents=[(DENT_THETA, dent_phi, DENT_SIGMA, DENT_DEPTH)])


def perturbed_start(truth):
    """Where a user would start: wrong by ~10-20 % in size and 10 deg in direction, round."""
    r = 0.8 * 0.5 * (truth.b + truth.c)
    return DeformableShock(truth.height * 0.9, r, r, r, lat=truth.lat - 10.0, lon=truth.lon - 10.0, tilt=0.0)


def limb_distance(truth, views):
    """
    Smallest angular distance (in units of the dent width) between the dent centre and the
    silhouette curve of any view: ~0 means the dent is on a limb, >2 means no view sees it edge-on.
    """
    S = truth.surface(121, 241)
    P, n = S['xyz'].reshape(-1, 3), S['normals'].reshape(-1, 3)
    u = truth.direction(S['theta'], S['phi']).reshape(-1, 3)
    u_d = truth.direction(np.radians(DENT_THETA), np.radians(truth.dents[0][1]))
    best = np.inf
    for v in views:
        d = P - v.position
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        limb = np.abs(np.sum(n * d, axis=1)) < 0.05
        if limb.any():
            best = min(best, np.degrees(np.arccos(np.clip(u[limb] @ u_d, -1, 1))).min())
    return best / DENT_SIGMA


def dent_direction(truth, fitted):
    """(theta, phi, r) of the truth dent centre as seen from the fitted model's centre/frame."""
    p = truth.position(np.radians(DENT_THETA), np.radians(truth.dents[0][1]))
    d = p - fitted.center
    e1, e2, e3 = fitted.frame
    r = np.linalg.norm(d)
    return np.arccos(np.clip(d @ e1 / r, -1, 1)), np.arctan2(d @ e3, d @ e2), r


def evaluate(truth, fitted):
    cmp = sf.compare_surfaces(truth, fitted, 61, 121)
    all_ = sf.band_stats(cmp, 0.0, 1e3)
    band = sf.band_stats(cmp, *BAND)
    flank = sf.band_stats(cmp, 0.0, 1e3, apex_angle_min=60.0)
    th, ph, r = dent_direction(truth, fitted)
    m = {'rms_all': all_['rms'], 'p95_all': all_['p95'], 'rms_band': band['rms'], 'p95_band': band['p95'],
         'n_band': band['n'], 'rms_flank': flank['rms'], 'rad_rms_band': band['rad_rms'],
         'dent_depth_rec': -float(fitted.log_deformation(th, ph)),
         'dent_radial_err': float(fitted.radius(th, ph) - r), 'dent_sep': np.nan}
    if fitted.dents:
        d = fitted.dents[0]
        u_t = truth.direction(np.radians(DENT_THETA), np.radians(truth.dents[0][1]))
        u_f = fitted.direction(np.radians(d[0]), np.radians(d[1]))
        m['dent_sep'] = float(np.degrees(np.arccos(np.clip(u_t @ u_f, -1, 1))))
    return cmp, m


def make_data(truth, geom, noise, rng):
    outlines = [sf.synth_outline(truth, v, sigma=SIGMA_OUT, noise=noise, rng=rng, rho_min=RHO_MIN)
                for v in geom['views']]
    footprints = []
    if geom['footprint']:
        fp = sf.synth_footprint(truth, 1.05, sigma=SIGMA_OUT, noise=noise, rng=rng)
        if len(fp.points):
            footprints.append(fp)
    return outlines, footprints


def fit_method(method, outlines, footprints, ellipsoid):
    tol = dict(ftol=1e-6, xtol=1e-6, gtol=1e-6)
    if method == 'ellipsoid':
        return ellipsoid
    if method == 'dent':
        return sf.ShockFitter(outlines, footprints, n_dents=1).fit_dents_multistart(ellipsoid, **tol)[0]
    if method == 'harmonic_L4':
        return sf.ShockFitter(outlines, footprints, lmax=4, lam=1.0).fit(ellipsoid, **tol)[0]
    raise ValueError(method)


def run():
    rows, maps = [], {}
    fields = ['case', 'geometry', 'dent_phi', 'limb_dist', 'noise', 'draw', 'method', 'n_footprint', 'fit_s',
              'outline_rms', 'rms_all', 'p95_all', 'rms_band', 'p95_band', 'n_band', 'rms_flank',
              'rad_rms_band', 'dent_depth_rec', 'dent_radial_err', 'dent_sep']
    csv_path = OUT / 'identifiability_results.csv'
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for cname in BASES:
            for dent_phi in DENT_PHIS:
                truth = truth_model(cname, dent_phi)
                for gname, geom in GEOMETRIES.items():
                    ld = limb_distance(truth, geom['views'])
                    for noise in NOISE:
                        for draw in range(1 if noise == 0 else N_MC):
                            rng = np.random.default_rng(1000 * draw + 7)
                            outlines, footprints = make_data(truth, geom, noise, rng)
                            if geom['footprint'] and not footprints:
                                print(f'{cname} / {gname}: surface does not reach r=1.05 -> no footprint, skipped')
                                break
                            t0 = time.time()
                            ell, _ = sf.ShockFitter(outlines, footprints, lmax=0).fit(
                                perturbed_start(truth), ftol=1e-6, xtol=1e-6, gtol=1e-6)
                            t_ell = time.time() - t0
                            for method in METHODS:
                                t0 = time.time()
                                fitted = fit_method(method, outlines, footprints, ell)
                                fit_s = t_ell if method == 'ellipsoid' else time.time() - t0
                                fitter = sf.ShockFitter(outlines, footprints, lmax=fitted.lmax,
                                                        n_dents=len(fitted.dents))
                                n_data = fitter.n_residuals() - len(fitter.coef_lm)
                                out_rms = float(np.sqrt(fitter.data_cost(fitted) / n_data)) * SIGMA_OUT
                                cmp, m = evaluate(truth, fitted)
                                row = {'case': cname, 'geometry': gname, 'dent_phi': dent_phi,
                                       'limb_dist': round(ld, 2), 'noise': noise, 'draw': draw, 'method': method,
                                       'n_footprint': sum(len(f.points) for f in footprints),
                                       'fit_s': round(fit_s, 1), 'outline_rms': round(out_rms, 4), **m}
                                rows.append(row)
                                writer.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                                                 for k, v in row.items()})
                                fh.flush()
                                if draw == 0:
                                    maps[(cname, gname, dent_phi, noise, method)] = cmp
                                print(f'{cname:12s} {gname:20s} dent@{dent_phi:<3.0f} limb {ld:4.1f}s noise={noise:<5} '
                                      f'd{draw} {method:9s} | outline rms {out_rms:.3f} | n-err rms all '
                                      f'{m["rms_all"]:5.1f} band {m["rms_band"]:5.1f} flank {m["rms_flank"]:5.1f} | '
                                      f'dent {m["dent_depth_rec"]:+.2f}/{DENT_DEPTH} sep {m["dent_sep"]:5.1f} | '
                                      f'{fit_s:.0f}s', flush=True)
    plot_maps(maps)
    plot_summary(rows)
    print(f'wrote {csv_path}')


def plot_maps(maps, method='dent'):
    for cname in BASES:
        keys = [k for k in maps if k[0] == cname and k[3] == 0.0 and k[4] == method]
        if not keys:
            continue
        geoms = [g for g in GEOMETRIES if any(k[1] == g for k in keys)]
        fig, axes = plt.subplots(len(geoms), len(DENT_PHIS), figsize=(3.6 * len(DENT_PHIS), 2.9 * len(geoms)),
                                 squeeze=False)
        for i, g in enumerate(geoms):
            for j, dphi in enumerate(DENT_PHIS):
                ax = axes[i, j]
                cands = [k for k in keys if k[1] == g and k[2] == dphi]
                if not cands:
                    ax.axis('off')
                    continue
                cmp = maps[cands[0]]
                err = cmp['normal_err'].reshape(61, 121)
                im = ax.imshow(err, origin='lower', extent=[0, 360, 0, 180], vmin=0, vmax=15, cmap='magma_r',
                               aspect='auto')
                band = ((cmp['r_helio'] >= BAND[0]) & (cmp['r_helio'] <= BAND[1])).reshape(61, 121)
                if band.any() and not band.all():
                    ax.contour(np.linspace(0, 360, 121), np.linspace(0, 180, 61), band.astype(float),
                               levels=[0.5], colors='cyan', linewidths=0.8)
                ax.plot(dphi, DENT_THETA, 'w+', ms=10, mew=1.5)
                ell = maps.get((cname, g, dphi, 0.0, 'ellipsoid'))
                e0 = np.sqrt(np.mean(ell['normal_err'] ** 2)) if ell is not None else np.nan
                ax.set_title(f'{g}\ndent@{dphi:.0f}: rms {np.sqrt(np.mean(err ** 2)):.1f} deg (ellipsoid {e0:.1f})',
                             fontsize=8)
                if i == len(geoms) - 1:
                    ax.set_xlabel('phi [deg]')
                if j == 0:
                    ax.set_ylabel('theta from apex [deg]')
        fig.colorbar(im, ax=axes, label=f'normal error [deg], {method} fit', shrink=0.8)
        fig.suptitle(f'{cname}: noise-free; cyan = {BAND[0]}-{BAND[1]} R_sun band, + = dent centre')
        fig.savefig(OUT / f'identifiability_{cname}.png', dpi=110, bbox_inches='tight')
        plt.close(fig)


def plot_summary(rows):
    """Flank and band normal-error RMS vs dent azimuth per geometry/method (noise-free solid, noisy dashed)."""
    fig, axes = plt.subplots(2, len(BASES), figsize=(6 * len(BASES), 7.5), squeeze=False)
    colors = {g: c for g, c in zip(GEOMETRIES, ['C0', 'C1', 'C2'])}
    markers = {'ellipsoid': 's', 'dent': 'o', 'harmonic_L4': '^'}
    for col, cname in enumerate(BASES):
        for row_i, (key, label) in enumerate([('rms_flank', 'flank (>60 deg from apex)'),
                                              ('rms_band', f'{BAND[0]}-{BAND[1]} R_sun band')]):
            ax = axes[row_i, col]
            for g in GEOMETRIES:
                for method in METHODS:
                    for noise, ls in zip(NOISE, ['-', '--']):
                        pts = []
                        for dphi in DENT_PHIS:
                            sel = [r for r in rows if r['case'] == cname and r['geometry'] == g
                                   and r['noise'] == noise and r['method'] == method and r['dent_phi'] == dphi
                                   and np.isfinite(r[key])]
                            if sel:
                                pts.append((dphi, np.mean([r[key] for r in sel])))
                        if pts:
                            ax.plot(*zip(*pts), ls, marker=markers[method], color=colors[g], ms=5,
                                    label=f'{g}, {method}, noise {noise}')
            ax.axhline(10, color='k', lw=0.8, ls=':')
            ax.set_xlabel('dent azimuth phi [deg]')
            ax.set_ylabel(f'normal-error RMS, {label} [deg]')
            ax.set_title(cname)
            if row_i == 0:
                ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / 'identifiability_summary.png', dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    run()
