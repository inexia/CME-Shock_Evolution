"""
experiment_slow_shock.py -- can the DBM and GSD versions show a fading shock? The front of the corona series' last
step is scaled to several apex speeds at the start of the interplanetary propagation, propagated with a fixed
drag by both versions, and at several heights the share of the front that is still a shock is reported: the area
fraction with M_f > 1 (3-D), and in the ecliptic the shock's width by two criteria -- M_f >= 1 (the physical onset)
and X >= X_DETECT (what an in-situ observer would call a shock). Nothing in fitted results/ is touched.

Run:  python experiment_slow_shock.py      (the background and corona series of event_config.py)
"""
import contextlib
import io

import numpy as np

import event_config as cfg
import background as bk
import shock_properties_sse as sp
import shock_properties_dbm as sd
import shock_properties_gsd as sg
import evolution_front_width as fw
import drag_fit as df

# ================================ SETTINGS ================================
METHOD = 'constant'
SPEEDS = [1500.0, 900.0, 600.0]        # apex speeds at the start [km/s]; the whole front is scaled by the same factor
HEIGHTS = [60.0, 120.0, 214.0]         # report at these apex heights [R_sun]
X_DETECT = 1.3                         # compression ratio an in-situ observer needs to identify a shock
GAMMA0 = None                          # [1/km]; None: the drag fit of METHOD if it exists, else drag_fit.GAMMA0_DEFAULT
RAY_GRID = (46, 91)
DT_MIN = 5.0
# ==========================================================================


def shares(front, bg, kind):
    """(area fraction with M_f > 1, shock width [deg] with M_f >= 1, with X >= X_DETECT, front width [deg])."""
    series = [{'name': 'x', 'model': front, 'time_iso': front.time, 'time': None, 'speed_kms': front.speed, 'in_range': False}]
    res = sp.properties(series, 0, bg, sp.vn_method(kind))
    res['center'] = np.zeros(3)
    w = sp.area_weights(res['xyz'])
    on = res['on_front']
    tot = float(np.nansum(np.where(on, w, 0.0)))
    frac = float(np.nansum(np.where(on & (res['Mf'] > 1), w, 0.0)) / tot) if tot > 0 else np.nan
    fw.CRITERION, fw.THRESHOLD = 'Mf', 1.0
    a = fw.widths(res)
    fw.CRITERION, fw.THRESHOLD = 'X', X_DETECT
    b = fw.widths(res)
    return frac, (a[3] if a else np.nan), (b[3] if b else np.nan), (a[4] if a else np.nan)


def main():
    bg = bk.load_background()
    series = sp.load_series(METHOD, 'corona')
    if len(series) < 2:
        raise SystemExit(f'no coronal series for {METHOD}: run derive_para_corona.py first')
    fit = df.load(METHOD)
    gamma0 = GAMMA0 or (fit['gamma0_per_km'] if fit else df.GAMMA0_DEFAULT)
    n_ref = fit['n_ref_cm3'] if fit else None
    t0 = series[-1]['time']
    theta, phi = np.linspace(0, np.pi, RAY_GRID[0]), np.linspace(0, 2 * np.pi, RAY_GRID[1], endpoint=False)
    print(f'start: {series[-1]["name"]} at {t0:%Y-%m-%d %H:%M} UT, apex {series[-1]["model"].height:.1f} R_sun, '
          f'apex speed {series[-1]["speed_kms"]:.0f} km/s; gamma0 {gamma0:.3g} /km')
    with contextlib.redirect_stdout(io.StringIO()):
        table = sd.WindTable(bg, np.arange(*sd.R_TABLE), theta, phi, verbose=False)
        ftable = sg.FieldTable(bg, np.arange(*sd.R_TABLE), theta, phi, verbose=False)
    L0 = sd.l0_function(bg, t0)
    front_ref = sd.initial_front(series, *RAY_GRID)
    if n_ref is None:
        th, ph = np.meshgrid(theta, phi, indexing='ij')
        ok = np.isfinite(front_ref.R)
        _, N0 = table.lookup(front_ref.R[ok], th[ok], ph[ok], L0(0.0), when=bk.posix(t0))
        n_ref = float(np.nanmean(N0))
    v_apex = front_ref.speed
    print(f'\n{"version":8s} {"V_apex0":>8s} {"height":>7s} {"shock area":>11s} {"width Mf>=1":>12s} {"width X>={:g}".format(X_DETECT):>13s} {"front":>7s}')
    for v in SPEEDS:
        f = v / v_apex
        front0 = sd.initial_front(series, *RAY_GRID)
        front0.V = front0.V * f                                    # the whole front slower or faster by the same factor
        front0._interp.clear()
        for version, prop, tab in (('dbm', sd.propagate, table), ('gsd', sg.propagate, ftable)):
            with contextlib.redirect_stdout(io.StringIO()):
                fronts, _ = prop(front0, tab, L0, t0, gamma0, n_ref, DT_MIN, HEIGHTS)
            for fr in fronts:
                frac, w_mf, w_x, w_front = shares(fr, bg, version)
                g = lambda x, fmt: fmt.format(x) if np.isfinite(x) else '-'
                print(f'{version:8s} {v:8.0f} {fr.height:7.1f} {g(100 * frac, "{:10.0f}%")} {g(w_mf, "{:11.0f}d")} '
                      f'{g(w_x, "{:12.0f}d")} {g(w_front, "{:6.0f}d")}')
    print('\n(shock area: share of the front with M_f > 1; widths in the ecliptic: of the shock by each criterion, and '
          'of the whole front)')


if __name__ == '__main__':
    main()
