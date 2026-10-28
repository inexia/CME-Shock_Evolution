"""
shock_fit.py
============
Synthetic outlines, constraints and least-squares fitting for DeformableShock.

Pieces
------
Observer          a viewpoint (HGS lon, lat, distance); projects HGS points to the image plane
                  with the exact helioprojective relation (checked against sunpy in the tests)
outline()         the outer boundary of a projected point cloud as a function of position angle,
                  as a *smooth* soft-maximum so the fit has usable derivatives
OutlineData       one observed/synthetic outline (view, position angles, radii, mask, sigma)
FootprintData     3D points the surface must pass through (e.g. the EUV-wave front at 1.05 R_sun)
ShockFitter       least-squares fit of the ellipsoid parameters + harmonic deformation (l >= 2)
                  to any number of outlines and footprints, with a biharmonic smoothness penalty
compare_surfaces  normal-direction and radial errors of a fitted surface against a truth surface

Units: R_sun for lengths, degrees for the ellipsoid angles, radians inside the maths.
"""
import re

import numpy as np
from numpy import pi, sin, cos, sqrt
from scipy.optimize import least_squares, brentq

from deformable_shock import DeformableShock, harmonic_basis, lm_index, n_coeffs

AU_RSUN = 215.03                       # 1 au in solar radii
PARAM_NAMES = ('height', 'kappa', 'epsilon', 'alpha', 'lat', 'lon', 'tilt')   # PyThea's parameters, as in the GUI


# ----------------------------------------------------------------------------- observers
class Observer:
    """Viewpoint at Stonyhurst (lon, lat) [deg] and heliocentric distance [R_sun]."""

    def __init__(self, lon, lat, dist, name=''):
        self.lon, self.lat, self.dist, self.name = float(lon), float(lat), float(dist), name
        la, lo = np.radians([self.lat, self.lon])
        self.position = self.dist * np.array([cos(la) * cos(lo), cos(la) * sin(lo), sin(la)])
        self.z_img = self.position / self.dist                    # Sun -> observer
        north = np.array([0.0, 0.0, 1.0])
        y = north - (north @ self.z_img) * self.z_img
        self.y_img = y / np.linalg.norm(y)                        # solar north on the image
        self.x_img = np.cross(self.y_img, self.z_img)             # solar west on the image (+Tx)

    @classmethod
    def from_au(cls, lon, lat, dist_au, name=''):
        return cls(lon, lat, dist_au * AU_RSUN, name)

    def angles(self, points):
        """Helioprojective (Tx, Ty) in radians for HGS points (..., 3) [R_sun]."""
        d = np.asarray(points, float) - self.position
        dx, dy, dz = d @ self.x_img, d @ self.y_img, -(d @ self.z_img)
        return np.arctan2(dx, dz), np.arcsin(dy / np.linalg.norm(d, axis=-1))

    def project(self, points):
        """Image-plane coordinates (X, Y) in projected R_sun (angle x distance)."""
        tx, ty = self.angles(points)
        return self.dist * tx, self.dist * ty

    def __repr__(self):
        return f'Observer({self.name or "?"}: lon={self.lon:.1f}, lat={self.lat:.1f}, {self.dist / AU_RSUN:.2f} au)'


def event_2012_03_07():
    """STB, Earth/SOHO and STA on 2012-03-07 (Zhou et al. 2026: STA 109.3 W, STB 117.6 E of Earth)."""
    return [Observer.from_au(-117.6, 3.0, 1.07, 'STB'),
            Observer.from_au(0.0, -7.2, 0.99, 'Earth'),
            Observer.from_au(109.3, -0.8, 0.96, 'STA')]


def two_views(separation=60.0):
    """Earth plus a second viewpoint `separation` degrees ahead (STEREO-A today)."""
    return [Observer.from_au(0.0, 0.0, 1.0, 'Earth'), Observer.from_au(separation, 0.0, 0.96, 'STA')]


# ----------------------------------------------------------------------------- outlines
def outline(X, Y, n_bins=48, tau=0.005, taper=0.5):
    """
    Outer boundary of the projected points as a function of position angle about the Sun.

    Bin j (width w, centre psi_j) takes every point with full weight inside |dpsi| <= w/2 and with a
    cosine taper from 1 to 0 over w/2 < |dpsi| <= w/2 + taper*w, so a point entering or leaving a bin
    changes the result continuously. Within the bin the boundary radius is the soft maximum
        rho_j = tau * log sum_i exp( weight_ij * rho_i / tau ),
    i.e. the largest weighted radius up to a bias of tau * ln(N_eff) ~ 0.01 R_sun at tau = 0.005.
    Returns (psi_centres [rad, from +X toward +Y], rho [projected R_sun]).
    """
    X, Y = np.ravel(X), np.ravel(Y)
    ok = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[ok], Y[ok]
    rho = np.hypot(X, Y)
    psi = np.arctan2(Y, X)
    width = 2 * pi / n_bins
    centres = -pi + (np.arange(n_bins) + 0.5) * width
    i0 = np.floor((psi + pi) / width).astype(int) % n_bins
    j = (i0[:, None] + np.array([-1, 0, 1])[None, :]) % n_bins                   # own bin and neighbours
    dpsi = np.abs((psi[:, None] - centres[j] + pi) % (2 * pi) - pi)
    edge = 0.5 * width
    t = np.clip((dpsi - edge) / (taper * width), 0.0, 1.0)
    weight = 0.5 * (1.0 + np.cos(pi * t))                                         # 1 inside, 0 beyond the taper
    z = (weight * rho[:, None] / tau).ravel()
    jf = j.ravel()
    zmax = np.full(n_bins, -np.inf)
    np.maximum.at(zmax, jf, z)
    with np.errstate(invalid='ignore', divide='ignore'):
        total = np.bincount(jf, weights=np.exp(z - zmax[jf]), minlength=n_bins)
        rho_out = tau * (zmax + np.log(total))
    return centres, np.where(np.isfinite(rho_out), rho_out, np.nan)          # NaN = no points at this angle


def hard_outline(X, Y, n_bins=48):
    """Plain maximum radius per position-angle bin (reference / plotting only, not smooth)."""
    X, Y = np.ravel(X), np.ravel(Y)
    ok = np.isfinite(X) & np.isfinite(Y)
    rho, psi = np.hypot(X[ok], Y[ok]), np.arctan2(Y[ok], X[ok])
    width = 2 * pi / n_bins
    centres = -pi + (np.arange(n_bins) + 0.5) * width
    i0 = np.floor((psi + pi) / width).astype(int) % n_bins
    out = np.full(n_bins, -np.inf)
    np.maximum.at(out, i0, rho)
    return centres, np.where(np.isfinite(out), out, np.nan)


class OutlineData:
    """One outline: observer, bin centres, radii, validity mask and 1-sigma uncertainty [R_sun]."""

    def __init__(self, observer, psi, rho, sigma=0.05, mask=None, rho_min=None):
        self.observer, self.psi, self.rho, self.sigma = observer, np.asarray(psi), np.asarray(rho), float(sigma)
        self.mask = np.ones(len(self.rho), bool) if mask is None else np.asarray(mask, bool)
        if rho_min is not None:                     # occulted / unobserved part of the outline
            self.mask &= self.rho > rho_min
        self.n_bins = len(self.rho)


def feature_traced(model, dent, outlines):
    """Whether a dent/bulge centre projects into a traced position-angle bin of at least one view."""
    x = np.asarray(model.position(np.radians(dent[0]), np.radians(dent[1])), float).reshape(1, 3)
    for o in outlines:
        X, Y = o.observer.project(x)
        psi = np.arctan2(np.ravel(Y)[0], np.ravel(X)[0])
        j = int(np.argmin(np.abs((np.asarray(o.psi) - psi + np.pi) % (2 * np.pi) - np.pi)))
        if o.mask[j]:
            return True
    return False


def outline_from_trace(observer, psi_traced, rho_traced, n_bins=48, sigma=0.05, rho_min=None, max_gap=None,
                       tau=0.005):
    """
    OutlineData from a hand-traced boundary: psi_traced [rad, from +Tx (solar west) toward +Ty (north)],
    rho_traced [projected R_sun = angle_rad * observer.dist, i.e. arcsec / rsun_obs].
    The trace is resampled densely and passed through the same outline() operator the fit uses (so a
    traced boundary and a model boundary are compared like with like). A bin is used only where the
    trace covers its whole window (centre +- max_gap, default one bin width), so a partial trace is
    fine: the bins at the ends of the arc are masked.
    """
    psi_traced, rho_traced = np.ravel(psi_traced), np.ravel(rho_traced)
    ok = np.isfinite(psi_traced) & np.isfinite(rho_traced)
    psi_traced, rho_traced = psi_traced[ok], rho_traced[ok]
    width = 2 * pi / n_bins
    centres = -pi + (np.arange(n_bins) + 0.5) * width
    order = np.argsort(psi_traced)
    p, r = psi_traced[order], rho_traced[order]
    # dense resampling along the traced arc(s) only (no interpolation across gaps)
    dense_psi, dense_rho = [], []
    for k in range(len(p)):
        k2 = (k + 1) % len(p)
        dpsi = (p[k2] - p[k] + pi) % (2 * pi) - pi
        if abs(dpsi) > 2 * width:                                  # a gap in the trace: do not bridge it
            continue
        t = np.linspace(0, 1, max(2, int(abs(dpsi) / np.radians(0.25)) + 1), endpoint=False)
        dense_psi.append(p[k] + t * dpsi)
        dense_rho.append(r[k] + t * (r[k2] - r[k]))
    dense_psi, dense_rho = np.concatenate(dense_psi), np.concatenate(dense_rho)
    _, rho = outline(dense_rho * np.cos(dense_psi), dense_rho * np.sin(dense_psi), n_bins=n_bins, tau=tau)
    # a bin is observed only if the trace covers its whole window (centre +- one bin width)
    off = (dense_psi[None, :] - centres[:, None] + pi) % (2 * pi) - pi
    reach = width if max_gap is None else max_gap
    mask = ((np.abs(off) <= 0.5 * width).any(axis=1) & ((off <= -0.5 * width) & (off >= -reach)).any(axis=1)
            & ((off >= 0.5 * width) & (off <= reach)).any(axis=1))
    return OutlineData(observer, centres, rho, sigma=sigma, mask=mask & np.isfinite(rho), rho_min=rho_min)


# occulter / inner field-of-view edge used to mask outline bins [R_sun, projected]
OCCULTER_RSUN = {'COR1': 1.5, 'COR2': 2.5, 'C2': 2.5, 'C3': 3.7}


def trace_filename(sc, detector, time_str):
    """Shared naming convention of trace_outline.py and fit_event.py, e.g. trace_STB_COR2_0054.csv."""
    hhmm = time_str.strip()[-5:].replace(':', '')
    return f'trace_{sc.upper()}_{detector.upper()}_{hhmm}.csv'


_NUMBER = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'


def read_trace_csv(path, name=''):
    """
    Read a trace_outline.py CSV -> (Observer, Tx [arcsec], Ty [arcsec]). Tolerant of hand edits and of files
    re-saved by a spreadsheet: the observer values are found by name anywhere in a comment line (trailing commas,
    quotes, Windows line endings are fine), and each data row gives its first two numbers; rows without two numbers
    are skipped and reported.
    """
    kw, tx, ty, bad = {}, [], [], []
    with open(path, newline=None) as fh:
        for i, raw in enumerate(fh, 1):
            line = raw.strip().strip('"\'').strip()
            if not line or not line.strip(',;'):
                continue
            if line.startswith('#'):
                for key in ('HGLN_OBS', 'HGLT_OBS', 'DSUN_RSUN'):
                    m = re.search(key + r'\s*=\s*"?(' + _NUMBER + ')', line)
                    if m:
                        kw[key] = float(m.group(1))
                continue
            nums = re.findall(_NUMBER, line)
            if len(nums) >= 2:
                tx.append(float(nums[0]))
                ty.append(float(nums[1]))
            else:
                bad.append(i)
    if not {'HGLN_OBS', 'HGLT_OBS', 'DSUN_RSUN'} <= set(kw):
        raise ValueError(f'{path}: header lacks the observer position (HGLN_OBS, HGLT_OBS, DSUN_RSUN)')
    if bad:
        print(f'  {path}: skipped {len(bad)} line(s) without two numbers (line {", ".join(map(str, bad[:5]))}'
              f'{" ..." if len(bad) > 5 else ""})')
    return Observer(kw['HGLN_OBS'], kw['HGLT_OBS'], kw['DSUN_RSUN'], name or str(path)), np.array(tx), np.array(ty)


def observer_from_map(smap, name=''):
    """Observer from a sunpy map (works with any of the observer keywords sunpy understands)."""
    from sunpy.coordinates import sun
    o = smap.observer_coordinate.heliographic_stonyhurst
    return Observer(o.lon.to_value('deg'), o.lat.to_value('deg'),
                    (o.radius / sun.constants.radius).decompose().value, name)


def trace_to_polar(observer, tx_arcsec, ty_arcsec):
    """Helioprojective (Tx, Ty) in arcsec -> (psi [rad], rho [projected R_sun]) for outline_from_trace."""
    tx, ty = np.radians(np.asarray(tx_arcsec) / 3600.0), np.radians(np.asarray(ty_arcsec) / 3600.0)
    return np.arctan2(ty, tx), observer.dist * np.hypot(tx, ty)


def limb_map(model, views, n_theta=61, n_phi=121, tol=0.05):
    """
    For each (theta, phi) grid point of `model`, the angular distance [deg] to the nearest silhouette
    curve of any view (where the line of sight is tangent: |n . d| < tol). Small values mark the parts
    of the surface whose shape the outlines constrain; large values are reconstructible only with
    extra data (EUV footprint, pB depth, another view). Returns (theta, phi, dist_deg).
    """
    S = model.surface(n_theta, n_phi)
    P, n = S['xyz'].reshape(-1, 3), S['normals'].reshape(-1, 3)
    u = model.direction(S['theta'], S['phi']).reshape(-1, 3)
    dist = np.full(len(P), np.inf)
    for v in views:
        d = P - v.position
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        limb = np.abs(np.sum(n * d, axis=1)) < tol
        if limb.any():
            ang = np.degrees(np.arccos(np.clip(u @ u[limb].T, -1, 1))).min(axis=1)
            dist = np.minimum(dist, ang)
    return S['theta'], S['phi'], dist.reshape(n_theta, n_phi)


class FootprintData:
    """3D points (M, 3) in HGS the surface must pass through, with 1-sigma uncertainty [R_sun]."""

    def __init__(self, points, sigma=0.05, name='footprint'):
        self.points, self.sigma, self.name = np.asarray(points, float).reshape(-1, 3), float(sigma), name


def model_outline(model, observer, n_bins=48, mesh=(61, 121), tau=0.005, **kw):
    X, Y = observer.project(model.mesh(*mesh))
    return outline(X, Y, n_bins=n_bins, tau=tau, **kw)


def correlated_noise(n, sigma, corr_bins, rng):
    """Zero-mean noise with RMS `sigma`, smoothed over ~corr_bins bins (tracing errors are not white)."""
    w = rng.standard_normal(n)
    if corr_bins > 0:
        k = np.arange(-3 * int(np.ceil(corr_bins)), 3 * int(np.ceil(corr_bins)) + 1)
        g = np.exp(-0.5 * (k / corr_bins) ** 2)
        w = np.convolve(np.r_[w, w, w], g / g.sum(), mode='same')[n:2 * n]   # periodic in position angle
        w /= w.std() if w.std() > 0 else 1.0
    return sigma * w


def synth_outline(truth, observer, sigma=0.05, noise=0.0, corr_bins=2.0, rho_min=None, rng=None,
                  n_bins=48, mesh=(61, 121), tau=0.005):
    """Synthetic 'observed' outline of the truth surface (same forward operator as the fit, plus noise)."""
    psi, rho = model_outline(truth, observer, n_bins=n_bins, mesh=mesh, tau=tau)
    if noise > 0:
        rng = np.random.default_rng() if rng is None else rng
        rho = rho + correlated_noise(len(rho), noise, corr_bins, rng)
    return OutlineData(observer, psi, rho, sigma=sigma, rho_min=rho_min)


def surface_sphere_intersection(model, r_sphere=1.05, n_phi=72, n_theta=361):
    """
    Points where the surface crosses the heliocentric sphere |x| = r_sphere (the footprint of a
    shock on the low corona). For each azimuth all theta roots are returned; empty if none.
    """
    pts = []
    thetas = np.linspace(0.0, pi, n_theta)
    for phi in np.linspace(0.0, 2 * pi, n_phi, endpoint=False):
        f = np.linalg.norm(model.position(thetas, phi), axis=-1) - r_sphere
        for k in np.flatnonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0):
            th = brentq(lambda t: np.linalg.norm(model.position(t, phi)) - r_sphere, thetas[k], thetas[k + 1])
            pts.append(model.position(th, phi))
    return np.array(pts).reshape(-1, 3)


def synth_footprint(truth, r_sphere=1.05, sigma=0.05, noise=0.0, rng=None, n_phi=72):
    pts = surface_sphere_intersection(truth, r_sphere, n_phi=n_phi)
    if noise > 0 and len(pts):
        rng = np.random.default_rng() if rng is None else rng
        # jitter along the sphere surface (tangential), the footprint height itself is fixed
        tang = rng.standard_normal(pts.shape)
        radial = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        tang -= (tang * radial).sum(1, keepdims=True) * radial
        pts = pts + noise * tang
    return FootprintData(pts, sigma=sigma, name=f'footprint r={r_sphere}')


# ----------------------------------------------------------------------------- fitting
REAR_INSIDE_SUN = False       # fit_event.py sets it: penalise ellipsoids whose rear point lies behind the Sun
PRIOR = None                  # fit_event.py sets it: (anchor DeformableShock, {param: sigma}) -- every fitter then adds
                              # (p - p_anchor) / sigma for those ellipsoid parameters (height, kappa, alpha: sigma as
                              # a fraction of the anchor value; epsilon: absolute; lat, lon, tilt: degrees)
FIXED = ()                    # fit_event.py sets it: ellipsoid parameters every fitter holds at the start value
REAR_WEIGHT = 20.0            # residual per R_sun the rear point sits beyond the far-side surface


class ShockFitter:
    """
    Least-squares fit of a DeformableShock to outlines and footprints.

    Parameters: the 7 ellipsoid parameters plus the harmonic coefficients a_lm for 2 <= l <= lmax
    (l = 0, 1 are omitted: scale and position are already carried by the ellipsoid).
    Residuals: (rho_model - rho_obs)/sigma per outline bin, (R_model - R_obs)/sigma per footprint
    point (radial, about the model centre), and sqrt(lam) * l (l+1) * a_lm as the smoothness penalty.
    """

    def __init__(self, outlines, footprints=(), lmax=0, lam=1.0, mesh=(61, 121), tau=0.005,
                 fixed=(), bounds=None, n_dents=0, depth_bounds=(-1.0, 1.5)):
        self.outlines, self.footprints = list(outlines), list(footprints)
        # dent depth D: radius x exp(-D) at the centre. D > 0 dent (depression), D < 0 bulge (protrusion).
        # depth_bounds = (0, ...) restricts the fit to depressions only.
        self.depth_bounds = (float(depth_bounds[0]), float(depth_bounds[1]))
        self.lmax, self.lam, self.mesh, self.tau = int(lmax), float(lam), tuple(mesh), float(tau)
        self.fixed = set(fixed) | set(FIXED)
        self.n_dents = int(n_dents)                 # compact Gaussian dents (theta, phi, sigma, depth) fitted directly
        self.coef_lm = [(l, m) for l in range(2, self.lmax + 1) for m in range(-l, l + 1)]
        self.user_bounds = bounds or {}
        # the (theta, phi) grid is the same for every trial model: cache it and the harmonic basis
        self._th, self._ph = DeformableShock.grid(*self.mesh)
        self._basis = harmonic_basis(self._th, self._ph, self.lmax) if self.lmax >= 2 else None

    # --- parameter vector <-> model
    def names(self):
        return ([n for n in PARAM_NAMES if n not in self.fixed] + [f'a_{l}_{m}' for l, m in self.coef_lm]
                + [f'dent{k}_{q}' for k in range(self.n_dents) for q in ('theta', 'phi', 'sigma', 'depth')])

    def pack(self, model):
        p = [getattr(model, n) for n in PARAM_NAMES if n not in self.fixed]
        full = np.zeros(n_coeffs(max(self.lmax, model.lmax)))
        full[:n_coeffs(model.lmax)] = model.coeffs
        p += [full[lm_index(l, m)] for l, m in self.coef_lm]
        dents = list(model.dents)[:self.n_dents]
        dents += [(90.0, 0.0, 20.0, 0.0)] * (self.n_dents - len(dents))      # missing dents start flat
        for d in dents:
            p += list(d)
        return np.array(p)

    def unpack(self, p, template):
        kw = {n: getattr(template, n) for n in PARAM_NAMES}
        i = 0
        for n in PARAM_NAMES:
            if n not in self.fixed:
                kw[n] = p[i]
                i += 1
        coeffs = np.zeros(n_coeffs(self.lmax))
        for (l, m), val in zip(self.coef_lm, p[i:i + len(self.coef_lm)]):
            coeffs[lm_index(l, m)] = val
        i += len(self.coef_lm)
        dents = [tuple(p[i + 4 * k: i + 4 * k + 4]) for k in range(self.n_dents)]
        return DeformableShock.from_pythea(kw['height'], kw['kappa'], kw['epsilon'], kw['alpha'], kw['lat'], kw['lon'],
                                           kw['tilt'], lmax=self.lmax, coeffs=coeffs, dents=dents, time=template.time)

    def _bounds(self, p0):
        lo = {'height': 1.05, 'kappa': 0.05, 'epsilon': -0.95, 'alpha': 0.2, 'lat': -90.0, 'tilt': -90.0}
        hi = {'height': 60.0, 'kappa': 3.0, 'epsilon': 0.95, 'alpha': 5.0, 'lat': 90.0, 'tilt': 90.0}
        lo.update({k: v[0] for k, v in self.user_bounds.items()})
        hi.update({k: v[1] for k, v in self.user_bounds.items()})
        lower, upper = [], []
        i = 0
        for n in PARAM_NAMES:
            if n in self.fixed:
                continue
            if n == 'lon':
                lower.append(p0[i] - 180.0)
                upper.append(p0[i] + 180.0)
            else:
                lower.append(lo[n])
                upper.append(hi[n])
            i += 1
        lower += [-2.0] * len(self.coef_lm)
        upper += [2.0] * len(self.coef_lm)
        i = len(lower)
        for k in range(self.n_dents):
            phi0 = p0[i + 4 * k + 1]
            lower += [0.0, phi0 - 180.0, 5.0, self.depth_bounds[0]]
            upper += [180.0, phi0 + 180.0, 60.0, self.depth_bounds[1]]
        return np.array(lower), np.array(upper)

    def _scale(self):
        s = {'height': 1.0, 'kappa': 0.05, 'epsilon': 0.05, 'alpha': 0.1, 'lat': 5.0, 'lon': 5.0, 'tilt': 5.0}
        return np.array([s[n] for n in PARAM_NAMES if n not in self.fixed] + [0.05] * len(self.coef_lm)
                        + [10.0, 10.0, 5.0, 0.1] * self.n_dents)

    # --- residuals
    def residuals(self, p, template):
        try:
            model = self.unpack(p, template)
        except ValueError:                                   # outside the physical domain
            return np.full(self.n_residuals(), 1e3)
        pts = model.position(self._th, self._ph, basis=self._basis).reshape(-1, 3)
        res = []
        for o in self.outlines:
            X, Y = o.observer.project(pts)
            _, rho = outline(X, Y, n_bins=o.n_bins, tau=self.tau)
            rho = np.where(np.isfinite(rho), rho, 0.0)          # model does not reach this angle
            res.append(((rho - o.rho) / o.sigma)[o.mask])
        for f in self.footprints:
            d = f.points - model.center
            r_obs = np.linalg.norm(d, axis=1)
            e1, e2, e3 = model.frame
            th = np.arccos(np.clip(d @ e1 / r_obs, -1, 1))
            ph = np.arctan2(d @ e3, d @ e2)
            res.append((model.radius(th, ph) - r_obs) / f.sigma)
        if self.coef_lm:
            i0 = len(PARAM_NAMES) - len(self.fixed)
            res.append(np.array([sqrt(self.lam) * l * (l + 1) * a for (l, m), a in zip(self.coef_lm, p[i0:])]))
        if REAR_INSIDE_SUN:                                   # the rear point (r_c - a along the apex) stays
            a, b, c, rc = model.axes                           # inside the Sun: a shock is driven from one side
            res.append(np.array([REAR_WEIGHT * max(0.0, -1.0 - (rc - a))]))
        if PRIOR is not None:
            res.append(np.array(self.prior_terms(model)))
        return np.concatenate(res) if res else np.zeros(0)

    def prior_names(self):
        return [n for n in PARAM_NAMES if PRIOR is not None and n not in self.fixed and PRIOR[1].get(n) is not None]

    def prior_sigma(self, n):
        """The prior width of parameter n in its own units (R_sun or deg), or None."""
        if PRIOR is None or PRIOR[1].get(n) is None:
            return None
        anchor, sig = PRIOR
        return sig[n] * abs(getattr(anchor, n)) if n in ('height', 'kappa', 'alpha') else sig[n]

    def prior_terms(self, model):
        """(p - p_anchor) / sigma for the parameters with a prior (longitude across its wrap, tilt mod 180)."""
        anchor = PRIOR[0]
        out = []
        for n in self.prior_names():
            v, v0 = getattr(model, n), getattr(anchor, n)
            d = ((v - v0 + 180.0) % 360.0 - 180.0) if n == 'lon' else (((v - v0 + 90.0) % 180.0 - 90.0) if n == 'tilt'
                                                                       else v - v0)
            out.append(d / max(self.prior_sigma(n), 1e-9))
        return out

    def n_residuals(self):
        n = sum(int(o.mask.sum()) for o in self.outlines) + sum(len(f.points) for f in self.footprints)
        return n + len(self.coef_lm) + (1 if REAR_INSIDE_SUN else 0) + len(self.prior_names())

    def residual_seeds(self, model, n_seeds=6, min_sep=25.0, avoid=()):
        """
        Data-driven starting positions for a new dent or bulge. In every usable bin where the model outline
        lies OUTSIDE the observed one (rho_model - rho_obs > 2 sigma: the surface must be pushed in -> dent,
        sign +1) or, if bulges are allowed, INSIDE it (< -2 sigma: pushed out -> bulge, sign -1), find the
        surface grid point that defines the model outline there (a limb point) and take its direction.
        Returns [(theta_deg, phi_deg, sign)], ordered by |residual|, kept >= min_sep degrees apart and away
        from the directions in `avoid` (existing dents/bulges).
        """
        pts = model.position(self._th, self._ph, basis=self._basis).reshape(-1, 3)
        th, ph = self._th.ravel(), self._ph.ravel()
        cands = []
        for o in self.outlines:
            X, Y = o.observer.project(pts)
            rho_pt, psi_pt = np.hypot(X, Y), np.arctan2(Y, X)
            width = 2 * pi / o.n_bins
            ibin = np.floor((psi_pt + pi) / width).astype(int) % o.n_bins
            _, rho_mod = outline(X, Y, n_bins=o.n_bins, tau=self.tau)
            resid = np.where(o.mask & np.isfinite(rho_mod), rho_mod - o.rho, 0.0)
            allow_bulge = self.depth_bounds[0] < 0
            for j in np.flatnonzero((resid > 2 * o.sigma) | ((resid < -2 * o.sigma) & allow_bulge)):
                inb = np.flatnonzero(ibin == j)
                if inb.size:
                    k = inb[np.argmax(rho_pt[inb])]                  # the limb point that sets this bin's outline
                    cands.append((abs(resid[j]) / o.sigma, th[k], ph[k], 1 if resid[j] > 0 else -1))
        cands.sort(key=lambda c: -c[0])
        e = np.array(model.frame)
        def unit(t, p):
            return np.array([np.cos(t), np.sin(t) * np.cos(p), np.sin(t) * np.sin(p)])
        taken = [unit(np.radians(a), np.radians(b)) for a, b in avoid]
        seeds = []
        for _, t, p, sgn in cands:
            u = unit(t, p)
            if all(np.degrees(np.arccos(np.clip(u @ v, -1, 1))) >= min_sep for v in taken):
                seeds.append((float(np.degrees(t)), float(np.degrees(p)), sgn))
                taken.append(u)
            if len(seeds) >= n_seeds:
                break
        return seeds

    def fit_dents_sequential(self, ellipsoid, n_seeds=6, min_sep=25.0, sigma0=20.0, depth0=0.1,
                             fallback=((70.0, 0.0), (70.0, 90.0), (70.0, 180.0), (70.0, 270.0)),
                             polish_mesh=(121, 241), **kw):
        """
        Add dents one at a time up to self.n_dents. For dent k, try starts seeded from the outline residuals
        of the current (k-1)-dent model (plus the fixed azimuth fallbacks), each with ALL parameters free,
        and keep the lowest cost. The search runs on this fitter's (coarse) surface grid; each best model
        is then polished on the denser polish_mesh grid, because a coarse grid makes the cost rugged at the
        noise level (the outline jumps between grid points) and can trap the fit in a spurious minimum.
        Returns (models, costs): models[k] is the best k-dent model (k = 0..K) and costs[k] its data chi^2
        on the polish grid, so the number of dents can be chosen afterwards (select_by_bic).
        """
        def polish(model, k):
            if not polish_mesh or tuple(polish_mesh) == tuple(self.mesh):
                f = ShockFitter(self.outlines, self.footprints, mesh=self.mesh, tau=self.tau, n_dents=k,
                                depth_bounds=self.depth_bounds)
                return model, f.data_cost(model)
            f = ShockFitter(self.outlines, self.footprints, mesh=polish_mesh, tau=self.tau, n_dents=k,
                            depth_bounds=self.depth_bounds)
            refined, _ = f.fit(model, **kw)
            return refined, f.data_cost(refined)

        m0, c0 = polish(DeformableShock.from_dict({**ellipsoid.to_dict(), 'dents': [], 'lmax': 0, 'coeffs': []}), 0)
        models, costs = [m0], [c0]
        for k in range(1, self.n_dents + 1):
            current = models[-1]
            existing = [list(d) for d in current.dents]
            sub = ShockFitter(self.outlines, self.footprints, mesh=self.mesh, tau=self.tau, n_dents=k,
                              depth_bounds=self.depth_bounds)
            starts = sub.residual_seeds(current, n_seeds, min_sep, avoid=[d[:2] for d in existing])
            starts += [(a, b, 1) for a, b in fallback if all((a, b) != s[:2] for s in starts)]
            best = None
            for th0, ph0, sgn in starts:
                d = current.to_dict()
                d['dents'] = existing + [[th0, ph0, sigma0, sgn * depth0]]
                model, res = sub.fit(DeformableShock.from_dict(d), **kw)
                if best is None or res.cost < best[1].cost:
                    best = (model, res)
            refined, cost = polish(best[0], k)
            models.append(refined)
            costs.append(cost)
        return models, costs

    def n_data(self):
        return sum(int(o.mask.sum()) for o in self.outlines) + sum(len(f.points) for f in self.footprints)

    def residual_correlation_factor(self, model, max_lag=8):
        """
        Effective-sample factor f = 1 + 2 sum_k rho_k of the outline residuals (rho_k: autocorrelation at
        lag k bins, pooled over views, summed until it first drops to <= 0). Tracing errors are smooth
        along the outline, so neighbouring bins are not independent; chi^2 differences shrink by f and
        the number of independent data points is ~N / f. Clipped to [1, 10].
        """
        n_d = len(model.dents)
        f = ShockFitter(self.outlines, self.footprints, mesh=self.mesh, tau=self.tau, lmax=model.lmax, n_dents=n_d)
        pts = model.position(f._th, f._ph, basis=f._basis).reshape(-1, 3)
        series = []
        for o in self.outlines:
            X, Y = o.observer.project(pts)
            _, rho = outline(X, Y, n_bins=o.n_bins, tau=self.tau)
            r = np.where(o.mask & np.isfinite(rho), (rho - o.rho) / o.sigma, np.nan)
            series.append(r)
        num = np.zeros(max_lag + 1)
        den = 0.0
        for r in series:
            ok = np.isfinite(r)
            if ok.sum() < 4:
                continue
            x = np.where(ok, r - np.nanmean(r), np.nan)
            den += np.nansum(x * x)
            for k in range(1, max_lag + 1):
                num[k] += np.nansum(x * np.roll(x, -k))     # periodic in position angle; NaN pairs drop out
        if den <= 0:
            return 1.0
        rho_k = num[1:] / den
        total = 0.0
        for v in rho_k:
            if v <= 0:
                break
            total += v
        return float(np.clip(1.0 + 2.0 * total, 1.0, 10.0))

    def select_dents(self, models, costs, min_depth=0.1, max_sigma=45.0):
        """
        Choose the number of dents from the sequence returned by fit_dents_sequential. Dent k is accepted
        only if (a) it looks like a feature: |depth| >= min_depth and width <= max_sigma (a shallow, very
        broad 'dent' is a global shape correction, not a depression), and (b) it passes a BIC test with
        correlated errors: (chi2_{k-1} - chi2_k) / f > 4 ln(N / f), f from residual_correlation_factor of
        the largest model. Stops at the first rejected dent. Returns (k, info dict).
        """
        N = max(self.n_data(), 2)
        f = self.residual_correlation_factor(models[-1])
        threshold = 4.0 * np.log(max(N / f, 2.0))
        k_sel, steps = 0, []
        for k in range(1, len(models)):
            new = [d for d in models[k].dents if not any(np.allclose(d, e) for e in models[k - 1].dents)]
            d = max(models[k].dents, key=lambda x: 0) if not new else new[0]
            dchi = (costs[k - 1] - costs[k]) / f
            shape_ok = all(abs(dd[3]) >= min_depth and dd[2] <= max_sigma for dd in models[k].dents)
            ok = shape_ok and dchi > threshold
            steps.append({'k': k, 'dchi2_eff': dchi, 'threshold': threshold, 'shape_ok': shape_ok, 'accepted': ok})
            if not ok:
                break
            k_sel = k
        return k_sel, {'f': f, 'N': N, 'steps': steps}

    def select_by_bic(self, models, costs):
        """Plain BIC (independent errors), kept for comparison; prefer select_dents."""
        N = max(self.n_data(), 2)
        bic = [c + (7 + 4 * k) * np.log(N) for k, c in enumerate(costs)]
        return int(np.argmin(bic)), bic

    def fit_dents_multistart(self, ellipsoid, starts=((70.0, 0.0), (70.0, 90.0), (70.0, 180.0), (70.0, 270.0)),
                             sigma0=20.0, depth0=0.1, **kw):
        """
        Fit ellipsoid + n_dents compact dents, trying several starting dent positions (the problem is
        non-convex in the dent position) and keeping the lowest-cost solution. Returns (model, result).
        """
        best = None
        for th0, ph0 in starts:
            d = ellipsoid.to_dict()
            d['dents'] = [[th0, ph0 + 37.0 * k, sigma0, depth0] for k in range(self.n_dents)]
            model, res = self.fit(DeformableShock.from_dict(d), **kw)
            if best is None or res.cost < best[1].cost:
                best = (model, res)
        return best

    def data_cost(self, model):
        """
        Sum of squared, sigma-normalised data residuals (no penalty) of `model` exactly as given, including
        all its dents/bulges and harmonics, whatever this fitter's own n_dents / lmax are.
        """
        f = self
        if len(model.dents) != self.n_dents or model.lmax != self.lmax:
            f = ShockFitter(self.outlines, self.footprints, lmax=model.lmax, lam=self.lam, mesh=self.mesh,
                            tau=self.tau, n_dents=len(model.dents), depth_bounds=(-1e3, 1e3))
        r = f.residuals(f.pack(model), model)
        return float(np.sum(r[:f.n_data()] ** 2))                 # the data rows only (no penalty or prior)

    def fit(self, start, max_nfev=200, verbose=0, **kw):
        """Fit from a starting DeformableShock; returns (model, scipy result)."""
        p0 = self.pack(start)
        lower, upper = self._bounds(p0)
        p0 = np.clip(p0, lower + 1e-9, upper - 1e-9)
        res = least_squares(self.residuals, p0, args=(start,), bounds=(lower, upper), x_scale=self._scale(),
                            max_nfev=max_nfev, verbose=verbose, **kw)
        return self.unpack(res.x, start), res


def ellipsoid_fit(outlines, start, footprints=(), **kw):
    """Convenience: ellipsoid-only (PyThea-equivalent) fit."""
    return ShockFitter(outlines, footprints, lmax=0, **kw).fit(start)


# ----------------------------------------------------------------------------- EUV / low-corona geometry
def deproject_to_sphere(observer, tx_arcsec, ty_arcsec, r_sphere=1.05):
    """
    Line of sight through each helioprojective (Tx, Ty) [arcsec] -> its first (observer-side) crossing of
    the sphere |x| = r_sphere, in HGS [R_sun]. Returns (points (N, 3), hit (N,) bool); rows without a
    crossing (the line of sight misses the sphere: off-limb points) are NaN.
    """
    tx = np.radians(np.ravel(tx_arcsec) / 3600.0)
    ty = np.radians(np.ravel(ty_arcsec) / 3600.0)
    # unit vector from the observer, inverse of Observer.angles(): Tx = atan2(dx, dz), Ty = asin(dy)
    dy = np.sin(ty)
    dx = np.cos(ty) * np.sin(tx)
    dz = np.cos(ty) * np.cos(tx)
    d = dx[:, None] * observer.x_img + dy[:, None] * observer.y_img - dz[:, None] * observer.z_img
    o = observer.position
    b = d @ o                                         # |o + t d|^2 = r^2  ->  t^2 + 2 b t + |o|^2 - r^2 = 0
    disc = b * b - (o @ o - r_sphere ** 2)
    hit = disc >= 0
    t = -b - np.sqrt(np.where(hit, disc, 0.0))
    pts = o + t[:, None] * d
    pts[~hit] = np.nan
    return pts, hit


def occulted(points, observer, r_sun=1.0):
    """True where an HGS point [R_sun] is inside the Sun or hidden behind the solar disk for this observer."""
    p = np.asarray(points, float).reshape(-1, 3)
    o = observer.position
    d = p - o
    t = np.clip(-(d @ o) / np.einsum('ij,ij->i', d, d), 0.0, 1.0)     # closest approach of the segment o -> p
    closest = o + t[:, None] * d
    behind = (np.einsum('ij,ij->i', closest, closest) < r_sun ** 2) & (t < 1.0 - 1e-9)
    inside = np.einsum('ij,ij->i', p, p) < r_sun ** 2
    return behind | inside


def ellipsoid_footprint(model, r_sphere=1.05, n_theta=361, n_phi=181):
    """
    Curve(s) where the model's ELLIPSOID (dents/bulges ignored) crosses the sphere |x| = r_sphere: the
    footprint of the shock on the low corona, i.e. the EUV-wave front. Returns a list of (M, 3) HGS
    polylines (crossings ordered by azimuth phi; one polyline per crossing index).
    """
    ell = model.ellipsoid()
    th = np.linspace(0.0, pi, n_theta)
    ph = np.linspace(0.0, 2 * pi, n_phi)
    TH, PH = np.meshgrid(th, ph, indexing='ij')
    r = np.linalg.norm(ell.position(TH, PH), axis=-1) - r_sphere            # (n_theta, n_phi)
    lines = {}
    for j in range(n_phi):
        idx = np.flatnonzero(np.sign(r[:-1, j]) * np.sign(r[1:, j]) < 0)
        for n, i in enumerate(idx):
            f = r[i, j] / (r[i, j] - r[i + 1, j])
            t = th[i] + f * (th[i + 1] - th[i])
            lines.setdefault(n, []).append(ell.position(t, ph[j]))
    return [np.array(v) for v in lines.values() if len(v) > 1]


def fit_self_similar(outlines, previous, footprints=(), mesh=(121, 241), time=None, r0=0.0):
    """
    Best self-similar continuation of an earlier surface: one parameter s, model = previous.scaled(s, r0=r0)
    (shape, direction, tilt and dents fixed; lengths measured from r0 * e1 scaled: r0 = 1 is PyThea's
    self-similar mode -- kappa, epsilon, alpha fixed -- and r0 = 0 is about Sun centre). Returns (model, chi2, s).
    """
    f = ShockFitter(outlines, footprints, mesh=mesh, n_dents=len(previous.dents))
    s0 = 1.0
    ratios = []
    for o in outlines:
        _, rho = model_outline(previous, o.observer, n_bins=o.n_bins)
        ok = o.mask & np.isfinite(rho)
        if ok.any():
            ratios.append((np.max(o.rho[ok]) - r0) / max(np.max(rho[ok]) - r0, 1e-3))
    if ratios:
        s0 = float(np.clip(np.median(ratios), 0.3, 5.0))

    def res(x):
        m = previous.scaled(x[0], r0=r0)
        return f.residuals(f.pack(m), m)
    r = least_squares(res, [s0], bounds=([0.2], [20.0]), x_scale=[0.05], ftol=1e-8, xtol=1e-8)
    model = previous.scaled(r.x[0], time=time, r0=r0)
    return model, f.data_cost(model), float(r.x[0])


# ----------------------------------------------------------------------------- evaluation
def compare_surfaces(truth, fitted, n_theta=61, n_phi=121):
    """
    Sample the truth surface on a (theta, phi) grid; at the same *directions from the fitted centre*
    evaluate the fitted surface. Returns per-point: heliocentric distance of the truth point,
    angle from the truth apex axis [deg], normal-direction error [deg] and radial error [R_sun].
    """
    S = truth.surface(n_theta, n_phi)
    P, n_true = S['xyz'].reshape(-1, 3), S['normals'].reshape(-1, 3)
    d = P - fitted.center
    r_obs = np.linalg.norm(d, axis=1)
    e1, e2, e3 = fitted.frame
    th = np.arccos(np.clip(d @ e1 / r_obs, -1, 1))
    ph = np.arctan2(d @ e3, d @ e2)
    n_fit = fitted.normals(th, ph)
    ang = np.degrees(np.arccos(np.clip(np.sum(n_true * n_fit, axis=1), -1, 1)))
    return {'r_helio': np.linalg.norm(P, axis=1), 'apex_angle': np.degrees(S['theta'].ravel()),
            'normal_err': ang, 'radial_err': fitted.radius(th, ph) - r_obs,
            'theta': S['theta'].ravel(), 'phi': S['phi'].ravel()}


def band_stats(cmp, r_lo=1.5, r_hi=5.0, apex_angle_min=0.0):
    """RMS and 95th-percentile normal error inside a heliocentric band (optionally flanks only)."""
    sel = (cmp['r_helio'] >= r_lo) & (cmp['r_helio'] <= r_hi) & (cmp['apex_angle'] >= apex_angle_min)
    if not sel.any():
        return {'n': 0, 'rms': np.nan, 'p95': np.nan, 'rad_rms': np.nan}
    e = cmp['normal_err'][sel]
    return {'n': int(sel.sum()), 'rms': float(sqrt(np.mean(e ** 2))), 'p95': float(np.percentile(e, 95)),
            'rad_rms': float(sqrt(np.mean(cmp['radial_err'][sel] ** 2)))}
