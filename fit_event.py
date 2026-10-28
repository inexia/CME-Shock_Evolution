"""
fit_event.py -- fit the shock surface to the traced views, at every traced time in one run (or at one time).

Run:  python fit_event.py
Times: TIMES = 'all' fits every time that has traces in fitted results/<id>/semi (recognised by the time stamp
in the trace file names, trace_<YYYYMMDD_HHMM>_<SC>_<DET>.csv), in chronological order, so each fit can start
from and be compared with the one before; 'config' fits only TIME of event_config.py; or give a list of times.
SKIP_FITTED skips times that already have a fit. A time that cannot be fitted (too few traced views, no start)
is reported and skipped; a table at the end lists every time.
At each time:
1. data       coronagraph traces -> outlines (EUV traces, from earlier sessions, only with USE_EUV_TRACES:
              on-disk points on the r = FOOTPRINT_R sphere as the EUV-wave footprint, off-limb points as an outline)
2. start      the GUI's shock_<stamp>.json for this time and/or a self-similar scaling of the fit nearest
              in time (earlier or later), whichever matches the coronagraph outlines better
3. ellipsoid  least-squares fit to all the data
4. limb map   which parts of the surface the coronagraph outlines constrain
5. features   dents/bulges added one at a time to the CORONAGRAPH outlines only (EUV not used), kept only
              if the data need them
6. evolution  against the fit nearest in time (earlier or later): is this time consistent with self-similar
              expansion of it, about the surface point below the apex (PyThea: kappa, eps, alpha fixed) or
              about Sun centre? (Going back in time, the scale is simply below 1.)
Writes  fitted results/<id>/semi/shock_fit_<stamp>.json, shock_fit_ellipsoid_<stamp>.json (time inside) and a
row of fitted results/<id>/semi/shock_fit_summary.csv.
"""
import csv
import datetime as dt
import json
import os
import re
import sys

import numpy as np

import event_config as cfg
import shock_fit as sf
from deformable_shock import DeformableShock

# ================================ SETTINGS ================================
TIMES = 'all'                          # 'all': every traced time (chronological); 'config': TIME of event_config.py;
                                       # or a list, e.g. ['2012-03-07 00:39', '2012-03-07 00:54']
SKIP_FITTED = False                    # skip times that already have shock_fit_<stamp>.json
ANCHOR = 'gui'                         # fit anchored on this time's GUI save (shock_<stamp>.json): each ellipsoid
                                       # parameter moves from it only as far as the traced data justify (a Gaussian
                                       # prior of width PRIOR_SIGMA), so a partly traced front does not drift. No GUI
                                       # save: the best start is the anchor. 'start': always the best start. None: free
PRIOR_SIGMA = {'height': 0.10, 'kappa': 0.20, 'alpha': 0.20,     # fraction of the anchor value
               'epsilon': 0.10,                                  # absolute
               'lat': 5.0, 'lon': 5.0, 'tilt': 5.0}              # deg; None for a parameter: free
FIX = ['height', 'kappa', 'alpha', 'epsilon']                               # the GUI's parameters held exactly at the anchor, e.g. ['kappa', 'epsilon',
                                       # 'alpha'] (the shape; it then grows self-similarly with the height)
SHAPE_FROM_NEAREST = False             # the anchor takes its shape (kappa, epsilon, alpha) from the nearest fit: for
                                       # a partly seen time, often with FIX = ['kappa', 'epsilon', 'alpha']
FEATURE_MAX = 0.20                     # a dent or bulge changes the local radius by at most this fraction
FEATURES_IN_TRACED = True              # a dent/bulge counts only if its centre projects into a traced sector
USE_EUV_TRACES = False                 # EUV traces are no longer made (trace_outline.py traces coronagraphs only);
                                       # True: use any EUV trace files from earlier sessions (footprint, dome)
OCCULTER = {}                          # override coronagraph inner edge [R_sun], e.g. {'COR2': 3.0}
SIGMA_OUT = 0.05                       # outline / footprint uncertainty [R_sun]
LIMB_MARGIN = 0.05                     # EUV clicks within this of the footprint sphere's limb are ignored [R_sun]
N_BINS = 48
MAX_DENTS = 1                          # dents/bulges added one at a time up to this number; 0 = ellipsoid only
ALLOW_BULGE = True                     # features may push the surface out (D < 0) as well as in (D > 0)
MIN_DEPTH = 0.1                        # a feature must have |D| >= this (ln R) ...
MAX_SIGMA = 45.0                       # ... and be at most this wide [deg] to count
REAR_INSIDE_SUN = True                 # keep the ellipsoid's rear point inside the Sun (a <= (h + 1)/2): a shock is
                                       # driven from one side, and an ellipsoid reaching behind the Sun is a fit
                                       # artefact (its far side is unobservable); False: as before
SELF_SIMILAR_ORIGINS = {'surface': 1.0,         # PyThea's self-similar mode: kappa, eps, alpha fixed
                        'Sun centre': 0.0}      # all lengths from Sun centre scaled; tested at each new time
# ==========================================================================
RSUN_KM = 695700.0
TOL = dict(ftol=1e-6, xtol=1e-6, gtol=1e-6)


def load_traces():
    coro, euv_out, foot, missing = [], [], [], []
    for kind, sc, det in cfg.views():
        if kind == 'euv' and not USE_EUV_TRACES:
            continue
        path = cfg.trace_path(sc, det)
        if not os.path.exists(path):
            missing.append((kind, sc, det, path))
            continue
        obs, tx, ty = sf.read_trace_csv(path, name=f'{sc} {det}')
        if kind == 'coronagraph':
            psi, rho = sf.trace_to_polar(obs, tx, ty)
            rmin = OCCULTER.get(det, sf.OCCULTER_RSUN.get(det, 2.5))
            od = sf.outline_from_trace(obs, psi, rho, n_bins=N_BINS, sigma=SIGMA_OUT, rho_min=rmin)
            used = od.rho[od.mask]
            print(f'{sc:5s} {det:8s}: {len(tx):3d} points -> {od.mask.sum()}/{N_BINS} bins'
                  + (f', outline {used.min():.1f}-{used.max():.1f} R_sun' if used.size else ''))
            if od.mask.any():
                coro.append(od)
        else:
            pts, hit = sf.deproject_to_sphere(obs, tx, ty, cfg.FOOTPRINT_R)
            _, rho_all = sf.trace_to_polar(obs, tx, ty)
            # off-disk clicks count as the edge of the EUV dome only if clearly above the footprint sphere's
            # limb; near-limb misses (click scatter) are ambiguous and dropped. A dome edge only exists while
            # the shock is small enough to fit in the EUV field of view.
            dome = ~hit & (rho_all > cfg.FOOTPRINT_R + LIMB_MARGIN)
            n_on, n_dome, n_drop = int(hit.sum()), int(dome.sum()), int((~hit & ~dome).sum())
            if n_on:
                foot.append(sf.FootprintData(pts[hit], sigma=SIGMA_OUT, name=f'{sc} {det} footprint'))
            if n_dome >= 3:
                psi, rho = sf.trace_to_polar(obs, tx[dome], ty[dome])
                od = sf.outline_from_trace(obs, psi, rho, n_bins=N_BINS, sigma=SIGMA_OUT, rho_min=1.0)
                if od.mask.any():
                    euv_out.append(od)
            print(f'{sc:5s} {det:8s}: {len(tx):3d} points -> {n_on} on the disk (footprint on r = '
                  f'{cfg.FOOTPRINT_R}), {n_dome} above the limb (dome edge)'
                  + (f', {n_drop} at the limb ignored' if n_drop else ''))
    for kind, sc, det, path in missing:
        print(f'{sc:5s} {det:8s}: {path} not found -- trace it with trace_outline.py')
    return coro, euv_out, foot


def nearest_fit():
    """(model, stamp) of the final fit nearest in time to TIME, earlier or later, or (None, None)."""
    p, st, _ = cfg.nearest_result(['shock_fit'])
    if p is None:
        return None, None
    return DeformableShock.from_dict(json.loads(p.read_text())), st


def rms(fitter, model):
    n = fitter.n_data()
    return float(np.sqrt(fitter.data_cost(model) / n)) * SIGMA_OUT if n else float('nan')


def write_summary(row):
    path = cfg.results_path('shock_fit_summary.csv')
    rows = []
    if path.exists():
        with open(path, newline='') as fh:
            rows = [r for r in csv.DictReader(fh) if r.get('time') != row['time']]
    rows.append(row)
    rows.sort(key=lambda r: r['time'])
    fields = list(row.keys())
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})
    return path


CONSTRAINTS = {}                       # the last fit's report (saved with it)


class SkipTime(Exception):
    """This time cannot be fitted (the reason is the message)."""


def traced_times():
    """Every time with at least one trace file, from the stamps in the names (chronological, 'YYYY-MM-DD HH:MM')."""
    stamps = set()
    for p in cfg.RESULTS_DIR.glob('trace_*.csv'):
        m = re.match(r'^trace_(\d{8}_\d{4})_', p.name)
        if m:
            stamps.add(m.group(1))
    return [cfg.stamp_date(s).strftime('%Y-%m-%d %H:%M') for s in sorted(stamps)]


def main():
    if TIMES == 'config':
        times = [cfg.TIME]
    elif TIMES == 'all':
        times = traced_times()
        if not times:
            sys.exit(f'no trace files in "{cfg.RESULTS_DIR}": run trace_outline.py first')
    else:
        times = sorted(TIMES, key=lambda t: cfg.stamp(t))
    print(f'{len(times)} time(s) to fit: {", ".join(t[-5:] for t in times)}')
    keep = cfg.TIME
    report = []
    try:
        for t in times:
            cfg.TIME = t
            if SKIP_FITTED and cfg.results_path(f'shock_fit_{cfg.stamp()}.json').exists():
                report.append((t, 'skipped (already fitted)', None))
                continue
            try:
                report.append((t, 'fitted', fit_one()))
            except SkipTime as e:
                print(f'\n  skipped: {e}')
                report.append((t, f'not fitted: {e}', None))
    finally:
        cfg.TIME = keep
    print('\n================ summary ================')
    print('   time (UT)          result')
    for t, status, row in report:
        extra = '' if not row else (f'  apex {row["height"]} R_sun, {row["n_features"]} feature(s)'
                                    + (f', {row["evolution"]}' if row.get('evolution') else '')
                                    + (f', {row["apex_speed_kms"]} km/s' if row.get('apex_speed_kms') else ''))
        print(f'   {t}   {status}{extra}')
    print(f'\nfits and summary in {cfg.RESULTS_DIR}  ->  python view_fit.py')


def fit_one():
    sf.REAR_INSIDE_SUN = REAR_INSIDE_SUN
    print(f'\n=== {cfg.TIME} UT')
    coro, euv_out, foot = load_traces()
    if len(coro) < 1 or len(coro) + len(euv_out) + (1 if foot else 0) < 2:
        raise SkipTime('not enough traced views: need coronagraph outlines from at least two views (with '
                       'USE_EUV_TRACES: one coronagraph view plus EUV). Run  python trace_outline.py  first.')
    f_all = sf.ShockFitter(coro + euv_out, foot)
    f_coro = sf.ShockFitter(coro, mesh=(121, 241))

    # ---- start: GUI file for this time and/or self-similar continuation of the previous fit
    starts = []
    gui_file = cfg.results_path(f'shock_{cfg.stamp()}.json')
    if gui_file.exists():
        starts.append(('GUI ' + gui_file.name, DeformableShock.from_dict(json.loads(gui_file.read_text()))))
    prev, prev_stamp = nearest_fit()
    ss = {}                                       # origin label -> (model, chi2, scale)
    if prev is not None:
        for label, r0 in SELF_SIMILAR_ORIGINS.items():
            m, chi2, scale = sf.fit_self_similar(coro, prev, time=cfg.iso_time(), r0=r0)
            ss[label] = (m, chi2, scale)
            starts.append((f'{prev_stamp} fit expanded self-similarly about the {label} (x{scale:.3f})', m))
    if not starts and os.path.exists('shock_params.json'):
        starts.append(('shock_params.json', DeformableShock.from_dict(json.load(open('shock_params.json')))))
    if not starts:
        raise SkipTime('no start: save a rough shock in the GUI (Save writes shock_<stamp>.json) or fit an earlier time')
    name, start = min(starts, key=lambda s: f_coro.data_cost(s[1].ellipsoid()))
    print(f'\nstart: {name}')
    start = start.ellipsoid()
    start.time = cfg.iso_time()

    # ---- the anchor: the fit may leave it only as far as the traced data justify
    anchor = None
    if ANCHOR:
        gui = next((m for n, m in starts if n.startswith('GUI ')), None)
        if ANCHOR == 'gui' and gui is not None:
            anchor, a_name = gui.ellipsoid(), 'the GUI save'
        else:
            anchor, a_name = start, f'the best start ({name})'
        if SHAPE_FROM_NEAREST and ss:
            m = ss[min(ss, key=lambda k: ss[k][1])][0]
            anchor = DeformableShock.from_pythea(anchor.height, m.kappa, m.epsilon, m.alpha, anchor.lat, anchor.lon,
                                                 anchor.tilt)
            a_name += f', with the shape of the {prev_stamp} fit'
        anchor.time = cfg.iso_time()
        sf.PRIOR = (anchor, {k: v for k, v in PRIOR_SIGMA.items() if v is not None})
        sf.FIXED = tuple(FIX)
        start = anchor
        f_all = sf.ShockFitter(coro + euv_out, foot)             # with the prior and the fixed parameters
        print(f'anchored on {a_name}: ' + ', '.join(
            f'{k} {"fixed" if k in FIX else (f"+-{v * 100:.0f}%" if k in ("height", "kappa", "alpha") else (f"+-{v:g}" if k == "epsilon" else f"+-{v:g} deg"))}'
            for k, v in PRIOR_SIGMA.items() if v is not None or k in FIX))
    try:
        fitted, ell, foot_rms, k_sel = fit_surface(f_all, f_coro, start, coro, euv_out, foot, anchor)
    finally:
        sf.PRIOR, sf.FIXED = None, ()
    return finish(fitted, ell, foot_rms, f_coro, coro, euv_out, ss, prev, prev_stamp)


def constraint_report(fitter, res, anchor, ell):
    """Each ellipsoid parameter: anchor -> fit +- uncertainty, and whether the data or the prior set it."""
    J = np.asarray(res.jac)
    cov = np.linalg.pinv(J.T @ J)
    free = [n for n in sf.PARAM_NAMES if n not in fitter.fixed]
    out = {}
    print('  parameter    anchor ->    fit   +- 1 sigma   set by')
    for n in sf.PARAM_NAMES:
        a0, v = getattr(anchor, n), getattr(ell, n)
        if n not in free:
            print(f'  {n:8s} {a0:9.3f}          (fixed)')
            out[n] = {'anchor': round(a0, 4), 'fit': round(a0, 4), 'set_by': 'fixed'}
            continue
        i = free.index(n)
        s_post = float(np.sqrt(max(cov[i, i], 0.0)))
        s_pri = fitter.prior_sigma(n)
        if s_pri is None or s_post < 0.45 * s_pri:
            who = 'the data'
        elif s_post > 0.8 * s_pri:
            who = 'the prior (the traces hardly constrain it)'
        else:
            who = 'data and prior'
        print(f'  {n:8s} {a0:9.3f} -> {v:9.3f}  +- {s_post:7.3f}   {who}')
        out[n] = {'anchor': round(a0, 4), 'fit': round(v, 4), 'sigma': round(s_post, 4),
                  'set_by': {'the data': 'data', 'data and prior': 'both'}.get(who, 'prior')}
    return out


def fit_surface(f_all, f_coro, start, coro, euv_out, foot, anchor):
    # ---- ellipsoid on all data
    ell, res = f_all.fit(start, **TOL)
    ell.time = cfg.iso_time()
    global CONSTRAINTS
    CONSTRAINTS = constraint_report(f_all, res, anchor, ell) if anchor is not None else {}
    foot_rms = rms(sf.ShockFitter([], foot), ell) if foot else float('nan')
    print(f'ellipsoid (coronagraph + EUV): {ell}\n  coronagraph outline rms {rms(f_coro, ell):.3f} R_sun'
          + (f', EUV dome outline rms {rms(sf.ShockFitter(euv_out), ell):.3f}' if euv_out else '')
          + (f', footprint rms {foot_rms:.3f} R_sun' if foot else ''))
    th, ph, dist = sf.limb_map(ell, [o.observer for o in coro])
    r = np.linalg.norm(ell.surface(61, 121)['xyz'], axis=-1)
    for lo, hi in [(0, 1e3), (1.5, 5.0), (5.0, 10.0)]:
        sel = (r >= lo) & (r <= hi)
        if sel.any():
            where = 'whole surface' if hi > 100 else f'{lo}-{hi} R_sun'
            print(f'  limb map: {100 * (dist[sel] < 20).mean():.0f}% of the {where} lies within 20 deg of a '
                  f'coronagraph limb')

    # ---- dents/bulges, coronagraph only
    fitted = ell
    if MAX_DENTS > 0:
        depth = (-np.log(1.0 + FEATURE_MAX) if ALLOW_BULGE else 0.0, -np.log(1.0 - FEATURE_MAX))   # radius x e^-D
        fitter = sf.ShockFitter(coro, n_dents=MAX_DENTS, depth_bounds=depth)
        print(f'\nadding up to {MAX_DENTS} {"dents/bulges" if ALLOW_BULGE else "dents"} (coronagraph outlines only) ...')
        models, costs = fitter.fit_dents_sequential(ell, **TOL)
        k_sel, info = fitter.select_dents(models, costs, min_depth=MIN_DEPTH, max_sigma=MAX_SIGMA)
        print(f'  error correlation along the outline: f = {info["f"]:.1f} bins (N = {info["N"]}, '
              f'~{info["N"] / info["f"]:.0f} independent)')
        for k, c in enumerate(costs):
            st = next((s for s in info['steps'] if s['k'] == k), None)
            line = f'  K={k}: outline rms {np.sqrt(c / fitter.n_data()) * SIGMA_OUT:.3f} R_sun'
            if st:
                line += (f' | chi2 drop / f = {st["dchi2_eff"]:.1f} (needs > {st["threshold"]:.1f}), feature-like: '
                         f'{"yes" if st["shape_ok"] else "no"} -> {"accepted" if st["accepted"] else "rejected"}')
            print(line)
        if FEATURES_IN_TRACED and k_sel:
            k = k_sel
            while k > 0 and not all(sf.feature_traced(models[k], d, coro) for d in models[k].dents):
                k -= 1
            if k < k_sel:
                print(f'  {k_sel - k} feature(s) dropped: centre outside every traced sector (FEATURES_IN_TRACED)')
                k_sel = k
        fitted = models[k_sel]
        fitted.time = cfg.iso_time()
        if k_sel:
            th, ph, dist = sf.limb_map(fitted, [o.observer for o in coro])
        for k, d in enumerate(fitted.dents):
            p = fitted.position(np.radians(d[0]), np.radians(d[1]))
            i, j = int(round(d[0] / 3.0)), int(round((d[1] % 360) / 3.0)) % 121
            print(f'  {"dent " if d[3] >= 0 else "bulge"} {k}: {d[0]:.1f} deg from apex, azimuth {d[1] % 360:.1f}, '
                  f'width {d[2]:.1f} deg, D = {d[3]:+.2f} (radius x {np.exp(-d[3]):.2f}); centre at '
                  f'{np.linalg.norm(p):.2f} R_sun, HGS lon {np.degrees(np.arctan2(p[1], p[0])) % 360:.1f} lat '
                  f'{np.degrees(np.arcsin(p[2] / np.linalg.norm(p))):.1f}; nearest limb {dist[min(i, 60), j]:.0f} deg away')
    print(f'\nselected: {fitted}')
    return fitted, ell, foot_rms, (len(fitted.dents) if fitted is not ell else 0)


def finish(fitted, ell, foot_rms, f_coro, coro, euv_out, ss, prev, prev_stamp):
    # ---- evolution test against the previous time
    verdict, speed, best = '', float('nan'), None
    if ss:
        best = min(ss, key=lambda k: ss[k][1])
        ss_model, ss_chi2, ss_scale = ss[best]
        chi2_final = f_coro.data_cost(fitted)
        fac = f_coro.residual_correlation_factor(fitted)
        n = f_coro.n_data()
        extra = len(sf.PARAM_NAMES) + 4 * len(fitted.dents) - 1
        drop = (ss_chi2 - chi2_final) / fac
        thr = extra * np.log(max(n / fac, 2.0))
        dt_s = (cfg.date() - cfg.stamp_date(prev_stamp)).total_seconds()              # < 0: reference is later
        speed = (fitted.height - prev.height) * RSUN_KM / dt_s
        verdict = 'shape changed' if drop > thr else f'self-similar about the {best}'
        print(f'\nevolution relative to the {prev_stamp} fit ({dt_s / 60:+.0f} min to this time):')
        for label, (m, chi2, scale) in ss.items():
            print(f'  self-similar about the {label:16s}: x{scale:.3f}, outline rms {np.sqrt(chi2 / n) * SIGMA_OUT:.3f} R_sun'
                  + ('   <- better' if label == best and len(ss) > 1 else ''))
        print(f'  free fit (shape allowed to change)  : outline rms {np.sqrt(chi2_final / n) * SIGMA_OUT:.3f} R_sun\n'
              f'  chi2 drop / f = {drop:.1f} for {extra} extra parameters (needs > {thr:.1f}) -> {verdict.upper()}\n'
              f'  apex {prev.height:.2f} -> {fitted.height:.2f} R_sun, mean speed {speed:.0f} km/s; kappa, eps, alpha '
              f'{prev.kappa:.3f}, {prev.epsilon:.3f}, {prev.alpha:.3f} -> {fitted.kappa:.3f}, {fitted.epsilon:.3f}, '
              f'{fitted.alpha:.3f}')

    # ---- save
    out = cfg.results_path(f'shock_fit_{cfg.stamp()}.json')
    out_ell = cfg.results_path(f'shock_fit_ellipsoid_{cfg.stamp()}.json')
    for path, model in ((out, fitted), (out_ell, ell)):
        with open(path, 'w') as fh:                              # with the fit report: what set each parameter
            json.dump({**model.to_dict(), 'constraints': CONSTRAINTS}, fh, indent=2)
    row = {
        'time': cfg.iso_time(), 'height': f'{fitted.height:.3f}', 'kappa': f'{fitted.kappa:.4f}',
        'epsilon': f'{fitted.epsilon:.4f}', 'alpha': f'{fitted.alpha:.4f}',
        'lat': f'{fitted.lat:.2f}', 'lon': f'{fitted.lon:.2f}', 'tilt': f'{fitted.tilt:.2f}',
        'a': f'{fitted.a:.3f}', 'b': f'{fitted.b:.3f}', 'c': f'{fitted.c:.3f}',
        'n_features': len(fitted.dents),
        'features': '; '.join(f'{d[0]:.1f}/{d[1] % 360:.1f}/{d[2]:.1f}/{d[3]:+.2f}' for d in fitted.dents),
        'coronagraph_rms': f'{rms(f_coro, fitted):.4f}', 'footprint_rms': f'{foot_rms:.4f}',
        'views': ' '.join(o.observer.name.replace(' ', '_') for o in coro + euv_out),
        'reference_fit': prev_stamp or '', 'selfsim_origin': best or '',
        'selfsim_scale': f'{ss[best][2]:.4f}' if best else '',
        'evolution': verdict, 'apex_speed_kms': f'{speed:.0f}' if np.isfinite(speed) else '',
        'n_set_by_prior': sum(1 for c in CONSTRAINTS.values() if c['set_by'] == 'prior') if CONSTRAINTS else ''}
    summary = write_summary(row)
    print(f'\nsaved {out}, {out_ell.name}; summary row in {summary}')
    return row


if __name__ == '__main__':
    main()
