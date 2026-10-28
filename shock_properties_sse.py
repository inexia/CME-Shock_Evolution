"""
shock_properties_sse.py -- step 5, self-similar-expansion version (SSE): shock properties over the whole front at
every height in the corona (derive_para_corona.py) and in interplanetary space, where the front expands
self-similarly with its apex on the drag-based model (derive_para_ip.py), using the background solar wind.
The three versions of the shock evolution: shock_properties_sse.py (this), shock_properties_dbm.py (ray-by-ray
drag in the structured wind) and shock_properties_gsd.py (that plus geometrical shock dynamics). This module also
holds the shared functions the other two use (properties, Rankine-Hugoniot, figures).
Running it: PROPAGATE runs derive_para_ip.py first (the drag from drag_fit.py, fitted along the spacecraft direction
with the observed arrival in event_config.py); then the properties
and figures of the corona and the SSE interplanetary series.

Run:  python shock_properties.py      (settings below; folders from event_config.py)
Reads  fitted results/<id>/final/shock_para_corona<hhh>_<method>.json  (derive_para_corona.py) and the background
(background.py, event_config.BACKGROUND_DIR).

At every point x of a (theta, phi) grid on each surface:
  n         outward unit normal of the irregular front, n ~ dx/dtheta x dx/dphi of the full parameterisation
            x = C + R_ell exp(S) u (dents and bulges included; DeformableShock.normals)
  V_n       speed of the front along n in the laboratory frame (HGS, inertial) [km/s]. Only this component of
            a front's motion is physical. VN_METHOD = 'normal_distance': the distance from x along n to the
            surfaces a fraction VN_FRACTION of the way to the neighbouring heights (apex height, semi-axes,
            direction and time interpolated linearly), over the time difference
            (central difference, one-sided at the ends). Those surfaces carry this step's dents/bulges, so a
            feature moves with the front instead of jumping between fits. The error falls as the square of the
            distance used: ~0.03% of the apex speed at 0.1 of a 1 R_sun step (up to 3% at the rim of a dent
            if the full step were used). 'self_similar': V = (dh/dt) (x - P0) / (h - 1) about the
            surface point P0 below the apex (PyThea's expansion). The shock velocity is V_sh = V_n n.
  theta_Bn  angle between the upstream B and n, 0-90 deg
  M_A       (V_n - u.n) / V_A, the normal Alfven Mach number in the shock frame
  M_f       (V_n - u.n) / c_f, with the fast speed at theta_Bn:
            c_f^2 = [V_A^2 + c_s^2 + sqrt((V_A^2 + c_s^2)^2 - 4 V_A^2 c_s^2 cos^2 theta_Bn)] / 2
  X         density compression rho2/rho1 from the Rankine-Hugoniot conditions of an oblique MHD shock (the
            fast-mode root) for the local M_A, theta_Bn, upstream beta and GAMMA -- no downstream data needed.
  X_B3      the beta -> 0, gamma = 5/3 interpolation M_A^2 = (M_Aperp sin)^2 + (M_Apar cos)^2 with
            M_Aperp = sqrt(X (5 + X) / (2 (4 - X))), M_Apar = sqrt(X), for comparison.
Masks: points with r < R_MIN are dropped; where the front does not move into the wind (V_n <= u.n) M_A, M_f and
X are NaN; where M_f <= 1 there is no fast shock and X is NaN.

Writes, in the event folders: fitted results/<id>/final/shock_props/<method>/shock_props<hhh>.npz (every array on
the grid), shock_props_<method>.csv (per height: values at the apex direction, area-weighted statistics over the
front) and figure/shock_evo/<method>/<quantity>/shock_evo<hhh>_<quantity>.png for the corona; the SSE
interplanetary series goes to shock_props_sse/, shock_props_sse_<method>.csv and figure/shock_evo_sse/ (heights
with 4 digits). The figures are 2 x 3: the front seen from the VIEWS, then the x-y and x-z planes over the wind
density and the x-y plane over the wind speed, with the front's cut coloured by the quantity; colour ranges
(COLOR_LIMITS) and axes (AXIS_LIM) are the same at every height of a series, so the evolution shows.
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
from scipy.interpolate import RegularGridInterpolator

import event_config as cfg
import background as bk
from deformable_shock import DeformableShock

# ================================ SETTINGS ================================
PROPAGATE = True                        # run derive_para_ip.py first (the SSE interplanetary series)
SERIES = ['corona', 'ip']               # 'corona': derive_para_corona.py series; 'ip': derive_para_ip.py series
METHODS = ['constant', 'varied']        # which derived series
GRID = (61, 121)                        # (theta, phi) points on each surface
R_MIN = 1.05                            # drop surface points below this heliocentric distance [R_sun]
FRONT_HALF_WIDTH_DEG = 90.0             # the physical front: within this angle of the apex (seen from the Sun) and
                                        # outside the Sun. A shock is driven from one side, so its lateral extent is
                                        # limited; the parts of a fitted ellipsoid that wrap around the Sun are
                                        # unconstrained by the data. A narrower fit keeps its own extent. None: no cut
VN_METHOD = {'corona': 'normal_distance',   # 'normal_distance' or 'self_similar' per series; the ip series is
             'ip': 'self_similar',          # self-similar by construction, so its V_n is exact that way;
             'dbm': 'ray', 'gsd': 'ray'}    # ray fronts carry their ray speeds: V_n = V_ray (n . r_hat)
VN_FRACTION = 0.1                       # normal distance to surfaces this fraction of the way to the neighbours
GAMMA = 5.0 / 3.0
PLOT = True
QUANTITIES = ['Vn', 'theta_bn', 'MA', 'Mf', 'X']
VIEWS = [('from Earth', 0, 0),          # (title, HGS latitude, HGS longitude) of each viewpoint [deg];
         ('from 90\u00b0 W', 0, 90),       # e.g. ('from STEREO-A', 0, 110) for March 2012
         ('from solar north', 90, 0)]
PLOT_GRID = (31, 61)                    # coarser grid for the drawn surface (speed)
BG_GRID = 201                           # points across the background maps (planes through the Sun)
BG_CMAPS = {'nr2': 'cividis', 'ur': 'Blues_r'}   # wind density n (r/1 AU)^2 and radial speed u_r
BG_LIMITS = {'nr2': None, 'ur': None}  # fixed ranges, e.g. {'nr2': (1, 50), 'ur': (250, 750)}; None: from the
                                        # first height of the series (then fixed for the series)
COLOR_LIMITS = {'Vn': None, 'theta_bn': (0, 90), 'MA': (1, 10), 'Mf': (1, 10), 'X': (1, 4), 'X_B3': (1, 4)}
                                        # fixed colour ranges; None: from all heights of the method
COLOR_LIMITS_IP = {}                    # overrides for the ip series, e.g. {'MA': (1, 30), 'Mf': (1, 30)}
AXIS_LIM = None                         # half-width of the plotted box [R_sun], the same at every height so
                                        # the evolution shows; None: fitted to the largest front of the series
# ==========================================================================
RSUN_KM = bk.RSUN_KM
LABELS = {'Vn': 'V$_n$ [km/s] (lab frame)', 'theta_bn': '$\\theta_{Bn}$ [deg]', 'MA': 'M$_A$', 'Mf': 'M$_f$',
          'X': 'X = $\\rho_2/\\rho_1$ (Rankine-Hugoniot)', 'X_B3': 'X (Eq. B3, $\\beta \\to 0$)'}
CMAPS = {'Vn': 'viridis', 'theta_bn': 'coolwarm', 'MA': 'plasma', 'Mf': 'plasma', 'X': 'cividis', 'X_B3': 'cividis'}
SHOCK_ONLY = ('MA', 'Mf', 'X', 'X_B3')   # drawn only where M_f > 1 (grey elsewhere); the saved data keep all values


# ----------------------------------------------------------------------------- MHD shock relations
def fast_speed(VA, cs, theta_bn_deg):
    s = VA ** 2 + cs ** 2
    c2 = np.cos(np.radians(theta_bn_deg)) ** 2
    return np.sqrt(0.5 * (s + np.sqrt(np.maximum(s ** 2 - 4 * VA ** 2 * cs ** 2 * c2, 0.0))))


def _rh_residual(X, a, bt, p1, gamma):
    """Energy-flux jump (normalised by rho1 v1^3) of an oblique MHD shock in the normal-incidence frame, with
    mass, normal/tangential momentum and induction already imposed. a = V_An^2/v1^2, bt = B_t1^2/(mu0 rho1 v1^2),
    p1 = p1/(rho1 v1^2). Zero at the jump's compression ratio X."""
    b = X * (1.0 - a) / (1.0 - X * a)                     # B_t2 / B_t1
    p2 = p1 + (1.0 - 1.0 / X) + 0.5 * bt * (1.0 - b ** 2)
    v2n, v2t = 1.0 / X, np.sqrt(a * bt) * (b - 1.0)
    g = gamma / (gamma - 1.0)
    f1 = 0.5 + g * p1 + bt
    f2 = 0.5 * (v2n ** 2 + v2t ** 2) + g * p2 / X + v2n * bt * b ** 2 - v2t * b * np.sqrt(bt * a)
    return f2 - f1


def rh_compression(MA, theta_bn_deg, beta, gamma=GAMMA, n_grid=600):
    """
    Compression ratio X of the fast-mode MHD shock with normal Alfven Mach number MA (shock frame), angle
    theta_Bn and upstream beta = 2 mu0 p / B^2. Fast-shock root: 1 < X < min((g+1)/(g-1), M_An^2), B_t2 >= B_t1.
    NaN where no fast shock exists.
    """
    MA, th, beta = np.broadcast_arrays(np.asarray(MA, float), np.asarray(theta_bn_deg, float),
                                       np.asarray(beta, float))
    shape = MA.shape
    MA, th, beta = MA.ravel(), th.ravel(), np.nan_to_num(beta.ravel(), nan=0.0)
    out = np.full(MA.shape, np.nan)
    ok = np.isfinite(MA) & np.isfinite(th) & (MA > 0)
    if not ok.any():
        return out.reshape(shape)
    c2 = np.cos(np.radians(th[ok])) ** 2
    a = c2 / MA[ok] ** 2
    bt = (1.0 - c2) / MA[ok] ** 2
    p1 = beta[ok] / (2.0 * MA[ok] ** 2)
    x_max = (gamma + 1.0) / (gamma - 1.0)
    upper = np.where(a > 0, np.minimum(x_max, 1.0 / np.maximum(a, 1e-300)), x_max) * (1.0 - 1e-9)
    q = np.linspace(1e-6, 1.0, n_grid)
    Xs = 1.0 + (upper[:, None] - 1.0) * q[None, :]
    R = _rh_residual(Xs, a[:, None], bt[:, None], p1[:, None], gamma)
    b = Xs * (1 - a[:, None]) / (1 - Xs * a[:, None])
    change = (np.sign(R[:, :-1]) != np.sign(R[:, 1:])) & np.isfinite(R[:, :-1]) & np.isfinite(R[:, 1:]) \
        & (b[:, :-1] >= 1.0 - 1e-9)
    has = change.any(1)
    j = np.where(has, n_grid - 2 - np.argmax(change[:, ::-1], axis=1), 0)       # the largest root
    rows = np.arange(len(j))
    lo, hi = Xs[rows, j], Xs[rows, j + 1]
    flo = _rh_residual(lo, a, bt, p1, gamma)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = _rh_residual(mid, a, bt, p1, gamma)
        left = np.sign(fm) == np.sign(flo)
        lo, flo = np.where(left, mid, lo), np.where(left, fm, flo)
        hi = np.where(left, hi, mid)
    res = np.where(has, 0.5 * (lo + hi), np.nan)
    out[np.flatnonzero(ok)] = res
    return out.reshape(shape)


def b3_compression(MA, theta_bn_deg):
    """Invert M_A^2 = sin^2 X(5+X)/(2(4-X)) + cos^2 X for X in (1, 4); NaN for M_A <= 1."""
    MA, th = np.broadcast_arrays(np.asarray(MA, float), np.asarray(theta_bn_deg, float))
    s2, c2 = np.sin(np.radians(th)) ** 2, np.cos(np.radians(th)) ** 2
    f = lambda X: s2 * X * (5 + X) / (2 * (4 - X)) + c2 * X
    lo, hi = np.ones_like(MA), np.full_like(MA, 4.0 - 1e-12)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        big = f(mid) > MA ** 2
        hi, lo = np.where(big, mid, hi), np.where(big, lo, mid)
    return np.where(np.isfinite(MA) & np.isfinite(th) & (MA > 1), 0.5 * (lo + hi), np.nan)


# ----------------------------------------------------------------------------- the derived series
def vn_method(kind='corona'):
    return VN_METHOD if isinstance(VN_METHOD, str) else VN_METHOD.get(kind, 'normal_distance')


_NOTED = set()                          # series whose 'old fronts' note was printed


def record_failure(kind, method, reason):
    """Remember why a method's interplanetary fronts were not made (None: they were; the note is removed), so the
    scripts that use the fronts can say why they are missing."""
    p = FINAL_FAILED(kind, method)
    if reason:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f'{dt.datetime.now():%Y-%m-%d %H:%M}  {reason}\n')
    elif p.exists():
        p.unlink()


def failure_note(kind, method):
    """The recorded reason a method's fronts are missing, or None."""
    p = FINAL_FAILED(kind, method)
    return p.read_text().strip() if p.exists() else None


FINAL_FAILED = lambda kind, method: cfg.FINAL_DIR / f'{kind}_failed_{method}.txt'


def load_series(method, kind='corona'):
    stem, digits = SERIES_FILES[kind][:2]
    steps = []
    if kind in ('dbm', 'gsd'):
        from ray_front import RayFront
        old = 0
        for p in cfg.FINAL_DIR.glob(f'{stem}*_{method}.npz'):
            if not re.match(rf'^{stem}\d{{{digits}}}_{method}\.npz$', p.name):
                continue
            m = RayFront.load(p)
            if m.meta.get('half_width_deg') is None:            # propagated before the front was limited to a cap
                m.half_width_deg = FRONT_HALF_WIDTH_DEG
                old += 1
            steps.append({'name': p.name, 'model': m, 'time_iso': m.time, 'time': dt.datetime.fromisoformat(m.time),
                          'speed_kms': m.speed, 'in_range': False})
        if old and FRONT_HALF_WIDTH_DEG is not None and (kind, method) not in _NOTED:
            _NOTED.add((kind, method))
            print(f'  note: {old} {kind} fronts were propagated before the front was limited to {FRONT_HALF_WIDTH_DEG:g} deg '
                  f'about the apex; the cut is applied now. Rerun shock_properties_{kind}.py to propagate the cap itself.')
        return sorted(steps, key=lambda s: s['model'].height)
    for p in cfg.FINAL_DIR.glob(f'{stem}*_{method}.json'):
        if not re.match(rf'^{stem}\d{{{digits}}}_{method}\.json$', p.name):
            continue
        d = json.loads(p.read_text())
        steps.append({'name': p.name, 'model': DeformableShock.from_dict(d), 'time_iso': d['time'],
                      'time': dt.datetime.fromisoformat(d['time']), 'speed_kms': d['derived']['speed_kms'],
                      'in_range': d['derived'].get('in_fitted_range')})
    return sorted(steps, key=lambda s: s['model'].height)


def local_angles(model, d):
    e1, e2, e3 = model.frame
    return np.arccos(np.clip(d @ e1, -1, 1)), np.mod(np.arctan2(d @ e3, d @ e2), 2 * np.pi)


def surface_offset(model, y):
    """|y - C| - R(direction): < 0 inside the (star-shaped) surface, > 0 outside."""
    g = y - model.center
    rr = np.linalg.norm(g, axis=-1)
    th, ph = local_angles(model, g / np.maximum(rr, 1e-12)[..., None])
    return rr - model.radius(th, ph)


def ray_distance(model, x, n, span, n_coarse=81):
    """Signed distance s along n from points x to the surface of `model` (the crossing nearest s = 0)."""
    S = np.linspace(-span, span, n_coarse)
    F = surface_offset(model, x[:, None, :] + S[None, :, None] * n[:, None, :])
    change = np.sign(F[:, :-1]) != np.sign(F[:, 1:])
    mid = 0.5 * (S[:-1] + S[1:])
    score = np.where(change, np.abs(mid)[None, :], np.inf)
    j = np.argmin(score, axis=1)
    has = np.isfinite(score[np.arange(len(j)), j])
    lo, hi = S[j], S[j + 1]
    flo = surface_offset(model, x + lo[:, None] * n)
    for _ in range(50):
        m = 0.5 * (lo + hi)
        fm = surface_offset(model, x + m[:, None] * n)
        left = np.sign(fm) == np.sign(flo)
        lo, flo = np.where(left, m, lo), np.where(left, fm, flo)
        hi = np.where(left, hi, m)
    return np.where(has, 0.5 * (lo + hi), np.nan)


def toward(model, neighbour, f):
    """The surface a fraction f of the way from `model` to `neighbour`: apex height, semi-axes and latitude
    interpolated linearly, longitude and tilt across their wrap; it carries `model`'s dents/bulges, so a feature
    moves with the front. (Lengths, not kappa/eps/alpha: exact both for PyThea's expansion, where a, b, c grow
    with h - 1, and for fronts centred on the Sun.)"""
    lerp = lambda p, q: p + f * (q - p)
    lon = model.lon + f * ((neighbour.lon - model.lon + 180.0) % 360.0 - 180.0)
    tilt = model.tilt + f * ((neighbour.tilt - model.tilt + 90.0) % 180.0 - 90.0)
    m = model.to_dict()
    return DeformableShock(lerp(model.height, neighbour.height), lerp(model.a, neighbour.a), lerp(model.b, neighbour.b),
                           lerp(model.c, neighbour.c), lerp(model.lat, neighbour.lat), lon, tilt,
                           lmax=m.get('lmax', 0), coeffs=m.get('coeffs') or None, dents=m['dents'])


def normal_speed(series, k, xyz, nrm, how=None):
    """V_n [km/s] at the points xyz (normals nrm) of step k."""
    step = series[k]
    model = step['model']
    shape = xyz.shape[:-1]
    x, n = xyz.reshape(-1, 3), nrm.reshape(-1, 3)
    how = how or vn_method()
    if how == 'ray':                                              # ray front: the rays carry their speeds
        r_hat = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        return (model.ray_speed(x) * np.sum(n * r_hat, 1)).reshape(shape)
    if how == 'self_similar' or len(series) < 2:
        p0 = model.frame[0]
        v = step['speed_kms'] * (x - p0) / (model.height - 1.0)
        return np.sum(v * n, 1).reshape(shape)
    reach = np.max(np.linalg.norm(x - model.frame[0], axis=1)) / (model.height - 1.0)
    dist = {}
    for j in (k - 1, k + 1):
        if 0 <= j < len(series):
            other = toward(model, series[j]['model'], VN_FRACTION)
            span = 2.0 * abs(other.height - model.height) * max(reach, 1.0) + 0.2
            dist[j] = (ray_distance(other, x, n, span),
                       VN_FRACTION * (series[j]['time'] - step['time']).total_seconds())
    if len(dist) == 2:
        (s_m, t_m), (s_p, t_p) = dist[k - 1], dist[k + 1]
        v = (s_p - s_m) / (t_p - t_m)
    else:
        (s, t), = dist.values()
        v = s / t
    return (v * RSUN_KM).reshape(shape)


# ----------------------------------------------------------------------------- one height
def area_weights(xyz):
    dth = np.gradient(xyz, axis=0)
    dph = np.gradient(xyz, axis=1)
    w = np.linalg.norm(np.cross(dth, dph), axis=-1)
    w[:, -1] = 0.0                                   # phi = 2 pi duplicates phi = 0
    return w


def half_width(model):
    """Angular extent of the physical front about the apex direction [deg]."""
    if hasattr(model, 'half_width_deg'):                        # a ray front remembers its cut
        return float(model.half_width_deg or 180.0)
    return 180.0 if FRONT_HALF_WIDTH_DEG is None else float(FRONT_HALF_WIDTH_DEG)


def on_front(model, xyz):
    """Which points of a surface belong to the physical front: outside the Sun and within half_width of the apex."""
    xyz = np.asarray(xyz, float)
    r = np.linalg.norm(xyz, axis=-1)
    with np.errstate(invalid='ignore'):
        ang = np.degrees(np.arccos(np.clip(np.sum(xyz * model.frame[0], -1) / np.maximum(r, 1e-12), -1, 1)))
    return (r >= R_MIN) & (ang <= half_width(model) + 1e-9)


def properties(series, k, bg, how=None):
    step = series[k]
    model = step['model']
    up = bk.upstream_on_shock(model, bg, step['time_iso'], *GRID)
    xyz, nrm = up['xyz'], up['normal']
    r = np.linalg.norm(xyz, axis=-1)
    keep = on_front(model, xyz)
    Vn = np.where(keep, normal_speed(series, k, xyz, nrm, how), np.nan)
    rel = Vn - up['u_n']
    moving = keep & (rel > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        MA = np.where(moving, rel / up['VA'], np.nan)
        cf = fast_speed(up['VA'], up['cs'], up['theta_bn'])
        Mf = np.where(moving, rel / cf, np.nan)
    shock = moving & (Mf > 1)
    X = np.where(shock, rh_compression(MA, up['theta_bn'], up['beta']), np.nan)
    X_B3 = np.where(moving, b3_compression(MA, up['theta_bn']), np.nan)
    out = {k_: v for k_, v in up.items()}
    out.update({'Vn': Vn, 'u_n_rel': rel, 'MA': MA, 'Mf': Mf, 'X': X, 'X_B3': X_B3, 'cf': cf, 'on_front': keep,
                'theta_bn': np.where(keep, up['theta_bn'], np.nan), 'area': area_weights(xyz) * keep,
                'shock': shock, 'height': model.height, 'time': step['time_iso']})
    return out


def summary_row(res, method):
    w = res['area']
    s = res['shock']
    a_front, a_shock = w.sum(), (w * s).sum()
    def wmed(q, m):
        v, ww = q[m], w[m]
        if not v.size or ww.sum() <= 0:
            return np.nan
        o = np.argsort(v)
        c = np.cumsum(ww[o])
        return float(v[o][np.searchsorted(c, 0.5 * c[-1])])
    apex = lambda q: float(q[0, 0])
    qperp = (w * s * (res['theta_bn'] > 45)).sum() / a_shock if a_shock > 0 else np.nan
    f = lambda v: '' if not np.isfinite(v) else f'{v:.4g}'
    return {'method': method, 'height_rsun': f'{res["height"]:.2f}', 'time': res['time'],
            'apex_Vn_kms': f(apex(res['Vn'])), 'apex_theta_bn': f(apex(res['theta_bn'])), 'apex_MA': f(apex(res['MA'])),
            'apex_Mf': f(apex(res['Mf'])), 'apex_X': f(apex(res['X'])), 'apex_X_B3': f(apex(res['X_B3'])),
            'front_area_rsun2': f(a_front), 'shock_area_fraction': f(a_shock / a_front if a_front else np.nan),
            'quasi_perp_fraction_of_shock': f(qperp),
            'median_Vn_kms': f(wmed(res['Vn'], s)), 'median_MA': f(wmed(res['MA'], s)),
            'max_MA': f(np.nanmax(np.where(s, res['MA'], np.nan)) if s.any() else np.nan),
            'median_Mf': f(wmed(res['Mf'], s)), 'median_X': f(wmed(res['X'], s)),
            'max_X': f(np.nanmax(np.where(s, res['X'], np.nan)) if s.any() else np.nan)}


# ----------------------------------------------------------------------------- figures
def colour_limits(results, overrides=None):
    """COLOR_LIMITS (with overrides) where set; otherwise one range from all heights of the method."""
    fixed = {**COLOR_LIMITS, **(overrides or {})}
    lim = {}
    for q in ('Vn', 'theta_bn', 'MA', 'Mf', 'X', 'X_B3'):
        if fixed.get(q) is not None:
            lim[q] = tuple(float(x) for x in fixed[q])
            continue
        v = np.concatenate([r[q][np.isfinite(r[q])].ravel() for r in results] + [np.array([np.nan])])
        v = v[np.isfinite(v)]
        hi = np.percentile(v, 99) if v.size else 1.0
        lim[q] = (0.0 if q in ('Vn', 'theta_bn') else 1.0, max(hi, 1.5))
    return lim


def axis_span(results):
    """Half-width of the plotted box: AXIS_LIM, or the largest front of the series (points outside the Sun)."""
    if AXIS_LIM is not None:
        return float(AXIS_LIM)
    big = 1.15
    for r in results:
        xyz = r['xyz'].reshape(-1, 3)
        out = np.linalg.norm(xyz, axis=1) >= 1.0
        if out.any():
            big = max(big, float(np.nanmax(np.abs(xyz[out]))))
    return 1.08 * big


def _quads(xyz):
    """Grid (n, m, 3) -> its (n-1)(m-1) quadrilateral faces (k, 4, 3)."""
    q = np.stack([xyz[:-1, :-1], xyz[1:, :-1], xyz[1:, 1:], xyz[:-1, 1:]], axis=2)
    return q.reshape(-1, 4, 3)


def _face_values(v):
    f = np.stack([v[:-1, :-1], v[1:, :-1], v[1:, 1:], v[:-1, 1:]], axis=-1)
    return np.where(np.all(np.isfinite(f), -1), np.mean(f, -1), np.nan).ravel()


def background_maps(bg, time, span, n=None):
    """The background on the planes through the Sun: x-y (z = 0) and x-z (y = 0), over the plotted box:
    n (r / 1 AU)^2 [cm^-3] and the radial wind speed u_r [km/s]."""
    n = n or BG_GRID
    c = np.linspace(-span, span, n)
    A, B = np.meshgrid(c, c)
    out = {}
    for plane, (ia, ib) in (('xy', (0, 1)), ('xz', (0, 2))):
        xyz = np.zeros(A.shape + (3,))
        xyz[..., ia], xyz[..., ib] = A, B
        r = np.linalg.norm(xyz, axis=-1)
        s = bg.sample(xyz.reshape(-1, 3), time)
        with np.errstate(invalid='ignore', divide='ignore'):
            nr2 = s['n'].reshape(A.shape) * (r / bk.AU_RSUN) ** 2
            ur = np.sum(s['u'] * (xyz.reshape(-1, 3) / np.maximum(r.reshape(-1, 1), 1e-9)), 1).reshape(A.shape)
        inside = r < 1.0
        out[plane] = {'a': c, 'b': c, 'nr2': np.where(inside, np.nan, nr2), 'ur': np.where(inside, np.nan, ur)}
    return out


def background_limits(maps):
    """Fixed colour ranges for the background panels of a series (BG_LIMITS, else from the first height)."""
    lim = {}
    for key, log in (('nr2', True), ('ur', False)):
        if BG_LIMITS.get(key) is not None:
            lim[key] = tuple(BG_LIMITS[key])
            continue
        v = np.concatenate([m[key][np.isfinite(m[key])].ravel() for m in maps.values()])
        v = np.log10(v[v > 0]) if log else v
        lo, hi = (np.percentile(v, [2, 98]) if v.size else (0.0, 1.0))
        lim[key] = (10 ** lo, 10 ** hi) if log else (lo, hi)
    return lim


def cut_curves(res, q, axis, value=0.0):
    """Curves where the front crosses the plane {coordinate `axis` = value}: list of ((M, 3) xyz, (M,) q)."""
    import contourpy
    th = res['theta'][:, 0]
    ph = res['phi'][0, :]
    gen = contourpy.contour_generator(x=ph, y=th, z=res['xyz'][..., axis] - value)
    fx = [RegularGridInterpolator((th, ph), res['xyz'][..., i]) for i in range(3)]
    fq = RegularGridInterpolator((th, ph), res[q])
    curves = []
    for line in gen.lines(0.0):
        if len(line) < 2:
            continue
        tp = np.column_stack([np.clip(line[:, 1], th[0], th[-1]), np.clip(line[:, 0], ph[0], ph[-1])])
        on = RegularGridInterpolator((th, ph), res.get('on_front', np.isfinite(res['xyz'][..., 0])).astype(float))(tp)
        curves.append((np.column_stack([f(tp) for f in fx]), fq(tp), on >= 0.5))
    return curves


def _draw_3d(ax, faces, colours, span, name, lat, lon):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.ticker import MaxNLocator
    ax.add_collection3d(Poly3DCollection(faces, facecolors=colours, edgecolors=colours, linewidths=0.25))
    ax.quiver(0, 0, 0, 0.85 * span, 0, 0, color='k', lw=1, arrow_length_ratio=0.07)
    ax.text(0.7 * span, 0, 0.06 * span, 'Earth', fontsize=8)
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-span, span)
    ax.set_box_aspect((1, 1, 1), zoom=1.2)
    ax.view_init(elev=lat, azim=lon)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel('x [R$_\\odot$]')
    ax.set_ylabel('y [R$_\\odot$]')
    ax.set_zlabel('z [R$_\\odot$]')
    sight = np.abs([np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
                    np.cos(np.radians(lat)) * np.sin(np.radians(lon)), np.sin(np.radians(lat))])
    for axis, along in zip((ax.xaxis, ax.yaxis, ax.zaxis), sight):
        if along > 0.95:                                       # this axis points at the viewer: hide it
            axis.set_ticks([])
            axis.set_label_text('')
    ax.set_title(f'{name}  (HGS lon {lon:g}\u00b0, lat {lat:g}\u00b0)', fontsize=10)


def draw_field_lines(ax, overlay, span):
    """Targets' field lines projected onto the x-y plane: solid from the target to its cobpoint (star), dashed
    beyond toward the Sun; a line that never reached the shock dashed throughout. overlay: [(label, colour, row,
    line)] with row from evolution_cobpoint.py (status, '_cob', '_cob_index')."""
    import matplotlib.patheffects as pe
    stroke = [pe.withStroke(linewidth=2.5, foreground='k')]
    for label, col, row, line in overlay:
        if line is None:
            continue
        status, i = row.get('status', ''), row.get('_cob_index', -1)
        if status == 'connected' and i > 0:
            to = np.vstack([line[:i], row['_cob'][None]])
            ax.plot(to[:, 0], to[:, 1], '-', color=col, lw=2.4, zorder=6)
            ax.plot(np.r_[row['_cob'][0], line[i:, 0]], np.r_[row['_cob'][1], line[i:, 1]], '--', color=col,
                    lw=1.1, alpha=0.7, zorder=6)
            ax.plot(row['_cob'][0], row['_cob'][1], '*', color=col, mec='k', mew=0.8, ms=15, zorder=8)
        else:
            ax.plot(line[:, 0], line[:, 1], '--', color=col, lw=1.4, alpha=0.8, zorder=6)
        inside = np.flatnonzero((np.abs(line[:, 0]) <= span) & (np.abs(line[:, 1]) <= span))
        if inside.size:
            p = line[inside[0]]
            if inside[0] == 0:
                ax.plot(p[0], p[1], 'o', color=col, mec='k', ms=9, zorder=8)
            note = {'not connected': ' (not connected)', 'spacecraft inside': ' (shock passed)',
                    'field line stalls': ' (field line stalls)'}.get(status, '')
            ax.text(p[0], p[1] + 0.025 * span, f' {label}{note}', color=col, fontsize=10, zorder=9, path_effects=stroke,
                    ha='left' if p[0] < 0.6 * span else 'right', va='bottom')


def _draw_cut(fig, ax, res, q, cmap, norm, lo_q, span, plane, field, bg_map, bg_lim, overlay=None):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm
    import matplotlib.patheffects as pe
    ia, ib, normal_axis = (0, 1, 2) if plane == 'xy' else (0, 2, 1)
    names = 'xyz'
    if bg_map is not None:
        if field == 'nr2':
            im = ax.pcolormesh(bg_map['a'], bg_map['b'], bg_map['nr2'], shading='auto', cmap=BG_CMAPS['nr2'],
                               norm=LogNorm(*bg_lim['nr2']))
            label = 'n (r / 1 AU)$^2$ [cm$^{-3}$]'
        else:
            im = ax.pcolormesh(bg_map['a'], bg_map['b'], bg_map['ur'], shading='auto', cmap=BG_CMAPS['ur'],
                               vmin=bg_lim['ur'][0], vmax=bg_lim['ur'][1])
            label = 'u$_r$ [km/s]'
        fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, pad=0.02, label=f'solar wind: {label}')
    pts = res['xyz'][res.get('on_front', np.isfinite(res['xyz'][..., 0]))][::4]
    ax.scatter(pts[:, ia], pts[:, ib], s=0.3, color='w', alpha=0.15, linewidths=0, zorder=2)
    curves = cut_curves(res, q, normal_axis, 0.0)
    for xyz, val, on in curves:
        keep_seg = on[:-1] & on[1:]                                # segments off the physical front are not drawn
        seg = np.stack([xyz[:-1][:, [ia, ib]], xyz[1:][:, [ia, ib]]], axis=1)[keep_seg]
        sv = (0.5 * (val[:-1] + val[1:]))[keep_seg]
        cols = cmap(norm(np.nan_to_num(sv, nan=lo_q)))
        cols[~np.isfinite(sv)] = (0.75, 0.75, 0.75, 1.0)
        ax.add_collection(LineCollection(seg, colors='k', linewidths=5.0, zorder=3))
        ax.add_collection(LineCollection(seg, colors=cols, linewidths=3.0, zorder=4))
    if not curves:
        ax.text(0.5, 0.95, 'the front does not cross this plane (white: its projection)', transform=ax.transAxes,
                ha='center', va='top', fontsize=8, color='w',
                path_effects=[pe.withStroke(linewidth=2, foreground='k')])
    ax.add_patch(plt.Circle((0, 0), 1.0, color='orange', zorder=5))
    if overlay and plane == 'xy':                              # the targets replace the generic Earth mark
        draw_field_lines(ax, overlay, span)
    elif span > bk.AU_RSUN:
        ax.plot(bk.AU_RSUN, 0, 'o', color='deepskyblue', mec='k', ms=7, zorder=6)
        ax.text(bk.AU_RSUN, 0.04 * span, 'Earth', fontsize=8, ha='center', color='w',
                path_effects=[pe.withStroke(linewidth=2, foreground='k')])
    else:
        ax.annotate('', xy=(0.98 * span, -0.9 * span), xytext=(0.72 * span, -0.9 * span),
                    arrowprops=dict(arrowstyle='->', color='w'))
        ax.text(0.85 * span, -0.86 * span, 'Earth', fontsize=8, ha='center', color='w',
                path_effects=[pe.withStroke(linewidth=2, foreground='k')])
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect('equal')
    ax.set_xlabel(f'{names[ia]} [R$_\\odot$]')
    ax.set_ylabel(f'{names[ib]} [R$_\\odot$]')
    ax.set_title(f'{names[ia]}-{names[ib]} plane ({names[normal_axis]} = 0)', fontsize=10)


def figure(res, q, method, lim, path, span=None, maps=None, bg_lim=None, overlay=None):
    """2 x 3. Top: the front seen from the VIEWS, coloured by q (the Sun and the front are one depth-sorted set of
    faces). Bottom: the planes through the Sun -- x-y and x-z over the wind density n (r/1 AU)^2, and x-y over the
    radial wind speed -- with the front's cut coloured by q (black-edged) and its projection in faint white."""
    if q in SHOCK_ONLY:
        res = {**res, q: np.where(res['shock'], res[q], np.nan)}
    cmap = plt.get_cmap(CMAPS.get(q, 'viridis'))
    norm = plt.Normalize(*lim[q])
    sub = (slice(None, None, max(1, GRID[0] // PLOT_GRID[0])), slice(None, None, max(1, GRID[1] // PLOT_GRID[1])))
    faces = _quads(res['xyz'][sub])
    vals = _face_values(res[q][sub])
    outside = np.linalg.norm(faces.mean(1), axis=1) >= 1.0               # faces inside the Sun are never seen
    on = res.get('on_front', np.isfinite(res['xyz'][..., 0]))[sub]
    on_face = np.all(np.stack([on[:-1, :-1], on[1:, :-1], on[1:, 1:], on[:-1, 1:]], -1), -1).ravel()
    faces, vals = faces[outside & on_face], vals[outside & on_face]         # the removed far side is not drawn
    colours = cmap(norm(np.nan_to_num(vals, nan=lim[q][0])))
    colours[~np.isfinite(vals)] = (0.8, 0.8, 0.8, 0.9)
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 37), np.linspace(0, np.pi, 19))
    sun = _quads(np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)], -1))
    all_faces = np.concatenate([faces, sun])
    all_colours = np.concatenate([colours, np.tile((1.0, 0.62, 0.1, 0.8), (len(sun), 1))])
    if span is None:
        span = max(np.nanmax(np.abs(faces)) if len(faces) else 1.0, 1.15) * 1.08
    fig = plt.figure(figsize=(17, 12.5), layout='constrained')
    panels = []
    for i, (name, lat, lon) in enumerate(VIEWS[:3]):
        panels.append(fig.add_subplot(2, 3, i + 1, projection='3d', proj_type='ortho'))
        _draw_3d(panels[-1], all_faces, all_colours, span, name, lat, lon)
    for i, (plane, field) in enumerate((('xy', 'nr2'), ('xz', 'nr2'), ('xy', 'ur'))):
        panels.append(fig.add_subplot(2, 3, 4 + i))
        _draw_cut(fig, panels[-1], res, q, cmap, norm, lim[q][0], span, plane, field,
                  None if maps is None else maps[plane], bg_lim, overlay)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    vv = res[q][np.isfinite(res[q])]
    over, under = bool(vv.size and vv.max() > lim[q][1]), bool(vv.size and vv.min() < lim[q][0])
    fig.colorbar(sm, ax=panels, shrink=0.5, label=f'shock: {LABELS.get(q, q)}',
                 extend={(True, True): 'both', (True, False): 'max', (False, True): 'min'}.get((over, under), 'neither'))
    grey = 'no fast shock (M$_f \\leq$ 1) or masked' if q in SHOCK_ONLY else 'masked'
    fig.suptitle(f'{cfg.EVENT_ID} {method}: apex height {res["height"]:.1f} R$_\\odot$ at '
                 f'{res["time"].replace("T", " ")} UT   (shock grey: {grey})')
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
SERIES_FILES = {'corona': ('shock_para_corona', 3, 'shock_props', 'shock_evo'),   # file stem, height digits,
                'ip': ('shock_para_ip', 4, 'shock_props_sse', 'shock_evo_sse'),     # output folder, figure folder
                'dbm': ('shock_para_dbm', 4, 'shock_props_dbm', 'shock_evo_dbm'),   # ray fronts (shock_properties_dbm.py)
                'gsd': ('shock_para_gsd', 4, 'shock_props_gsd', 'shock_evo_gsd')}   # ray fronts (shock_properties_gsd.py)


def run_series(kind, method, bg):
    stem, digits, props_name, fig_name = SERIES_FILES[kind]
    series = load_series(method, kind)
    if not series:
        print(f'\n[{kind} / {method}] no {stem}<h>_{method}.json in "{cfg.FINAL_DIR}"')
        return
    how = vn_method(kind)
    print(f'\n[{kind} / {method}] {len(series)} heights, V_n by {how}')
    out_dir = cfg.FINAL_DIR / props_name / method
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = lambda h: f'{int(round(h * 10)):0{digits}d}'
    results, rows = [], []
    print('   h [R_sun]  time      apex: V_n  theta_Bn   M_A   M_f    X  (B3)  | shock area  q-perp  median M_A  X')
    for k in range(len(series)):
        res = properties(series, k, bg, how)
        res['center'] = series[k]['model'].center
        results.append(res)
        np.savez_compressed(out_dir / f'{props_name}{tag(res["height"])}.npz',
                            **{k_: v for k_, v in res.items() if isinstance(v, np.ndarray)},
                            height=res['height'], time=res['time'])
        row = summary_row(res, method)
        rows.append(row)
        g = lambda key: row[key] or '-'
        print(f'   {res["height"]:6.1f}  {res["time"][5:16]}  {g("apex_Vn_kms"):>6} {g("apex_theta_bn"):>8} '
              f'{g("apex_MA"):>6} {g("apex_Mf"):>5} {g("apex_X"):>5} ({g("apex_X_B3"):>5}) | '
              f'{g("shock_area_fraction"):>9} {g("quasi_perp_fraction_of_shock"):>7} {g("median_MA"):>9} '
              f'{g("median_X"):>5}')
    table = cfg.final_path(f'{props_name}_{method}.csv')
    with open(table, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'saved {len(results)} files in {out_dir} and {table}')
    if PLOT:
        lim = colour_limits(results, COLOR_LIMITS_IP if kind in ('ip', 'dbm', 'gsd') else None)
        span = axis_span(results)
        for q in QUANTITIES:
            (cfg.FIGURE_DIR / fig_name / method / q).mkdir(parents=True, exist_ok=True)
        bg_lim = None
        for res in results:                       # the background maps once per height, for every quantity
            maps = background_maps(bg, res['time'], span)
            bg_lim = bg_lim or background_limits(maps)
            for q in QUANTITIES:
                figure(res, q, method, lim, cfg.FIGURE_DIR / fig_name / method / q /
                       f'{fig_name}{tag(res["height"])}_{q}.png', span, maps, bg_lim)
        print(f'saved {len(results) * len(QUANTITIES)} figures in {cfg.FIGURE_DIR / fig_name / method}')


def main():
    bg = bk.load_background()
    print(bg.summary() if hasattr(bg, 'summary') else bg.name)
    if PROPAGATE and 'ip' in SERIES:
        import derive_para_ip as dip                           # the SSE propagation: drag_fit.py + self-similar scaling
        for method in METHODS:
            try:
                rep = dip.run(method, bg)
                record_failure('ip', method, None if rep else 'no coronal series for this method: run derive_para_corona.py first')
            except RuntimeError as e:                              # a method that fails is reported, the rest run
                print(f'\n[{method}] not propagated: {e}')
                record_failure('ip', method, str(e))
    for kind in SERIES:
        for method in METHODS:
            run_series(kind, method, bg)


if __name__ == '__main__':
    main()
