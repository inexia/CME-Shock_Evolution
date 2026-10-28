"""
deformable_shock.py
===================
Irregular (non-ellipsoidal) CME-shock surface for 3D reconstruction.

Model
-----
The surface is star-shaped about the centre C of a PyThea-style ellipsoid:

    x(th, ph) = C + R(th, ph) * u(th, ph)
    R(th, ph) = R_ell(th, ph) * exp( S(th, ph) )

    R_ell = [ (cos th / a)^2 + (sin th cos ph / b)^2 + (sin th sin ph / c)^2 ]^(-1/2)
    S     = sum_{l<=lmax} sum_{|m|<=l}  a_lm Y_lm(th, ph)           (real spherical harmonics)
          + sum_k  -D_k exp( -Delta_k^2 / (2 sigma_k^2) )            (Gaussian dents)

(th, ph) are polar / azimuthal angles in the ellipsoid's principal frame:
    e1 : apex axis, from the Sun through the ellipsoid centre to the apex (semi-axis a)
    e2 : semi-axis b; for tilt = 0 it lies in the solar equatorial plane, toward +longitude
    e3 : semi-axis c = e1 x e2; for tilt = 0 it points toward solar north
    u  = cos th e1 + sin th cos ph e2 + sin th sin ph e3
Tilt rotates e2 and e3 about e1 (positive tilt turns the b-axis toward e3).
Delta_k is the angular distance on the unit sphere of directions between u and the
dent centre direction (th_k, ph_k); D_k is the dent depth in ln(R) units
(D = 0.3 -> the radius is reduced by a factor exp(-0.3) = 0.74 at the dent centre).
A negative D_k is a bulge: D = -0.3 -> the radius grows by exp(0.3) = 1.35 at its centre.

Ellipsoid parameters follow PyThea (Kouloumvakos et al. 2022, Sect. 2.4.2); JSON files, the GUI and all
printouts use them:
    height  apex heliocentric height h [R_sun]
    kappa   self-similar constant,  kappa = b / (h - 1)
    alpha   aspect ratio,           alpha = b / c
    epsilon "eccentricity",  eps = +sqrt(1 - a/b) for a <= b,  eps = -sqrt(1 - b/a) for a > b
with the semi-axes a (along the apex direction e1), b (e2), c (e3) and the centre at r_c = h - a.
Internally the class stores a, b, c (the fitter steps in them: epsilon has a flat spot at the sphere,
da/d(eps) = 0 at eps = 0); pythea_to_axes() / axes_to_pythea() convert exactly, both ways.
Self-similar expansion: kappa, epsilon, alpha fixed while h changes = expansion about the surface point
below the apex (scaled(s, r0=1)); scaled(s, r0=0) is self-similar about Sun centre instead.

Coordinates: Heliographic Stonyhurst (HGS) Cartesian in units of R_sun, as in geometry.py
(x toward Earth, z along the solar rotation axis, y = z x x, i.e. toward +longitude).

Normals are computed from central differences of the (analytic) position function in
(th, ph) with step 1e-5 rad, i.e. exact to ~1e-10 -- no noise from a point cloud.
"""
import numpy as np
from numpy import pi, sin, cos, sqrt, exp
from scipy.special import lpmv, gammaln

try:
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import frames, sun
    _HAVE_SUNPY = True
except ImportError:          # the geometry works without sunpy; only mesh_sunpy needs it
    _HAVE_SUNPY = False


# ----------------------------------------------------------------------------- harmonics
def lm_index(l, m):
    """Position of coefficient a_lm in the flat coefficient array (m runs from -l to l)."""
    return l * l + l + m


def n_coeffs(lmax):
    return (lmax + 1) ** 2


def real_sph_harm(l, m, theta, phi):
    """
    Real, orthonormal spherical harmonic Y_lm(theta, phi).
    theta: polar angle [rad], phi: azimuth [rad]. m > 0: cos(m phi) type, m < 0: sin(|m| phi) type.
    Built on scipy.special.lpmv (stable across SciPy versions).
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    am = abs(m)
    lognorm = 0.5 * (np.log((2 * l + 1) / (4 * pi)) + gammaln(l - am + 1) - gammaln(l + am + 1))
    p = lpmv(am, l, cos(theta)) * exp(lognorm)
    if m > 0:
        return sqrt(2.0) * p * cos(am * phi)
    if m < 0:
        return sqrt(2.0) * p * sin(am * phi)
    return p * np.ones_like(phi)


def harmonic_basis(theta, phi, lmax):
    """
    Matrix B (N, (lmax+1)^2) with B[:, lm_index(l, m)] = Y_lm(theta, phi) for the flattened grid,
    so that the harmonic deformation is  S = B @ coeffs.  Precompute it once for a fixed grid.
    """
    theta, phi = np.ravel(theta), np.ravel(phi)
    B = np.empty((theta.size, n_coeffs(lmax)))
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            B[:, lm_index(l, m)] = real_sph_harm(l, m, theta, phi)
    return B


# ----------------------------------------------------------------------------- ellipsoid
def pythea_to_axes(height, kappa, epsilon, alpha):
    """
    PyThea ellipsoid parameters -> (a, b, c, r_c):  b = kappa (h - 1), c = b / alpha,
    a = b (1 - eps^2) for eps >= 0, b / (1 - eps^2) for eps < 0, r_c = h - a.
    """
    height, kappa, epsilon, alpha = (float(v) for v in (height, kappa, epsilon, alpha))
    if height <= 1.0:
        raise ValueError('height must exceed 1 R_sun (kappa is defined as b / (height - 1))')
    if kappa <= 0.0 or alpha <= 0.0:
        raise ValueError('kappa and alpha must be positive')
    if not -1.0 < epsilon < 1.0:
        raise ValueError('epsilon must lie in (-1, 1)')
    b = kappa * (height - 1.0)
    c = b / alpha
    a = b * (1.0 - epsilon ** 2) if epsilon >= 0.0 else b / (1.0 - epsilon ** 2)
    return a, b, c, height - a


ellipsoid_axes = pythea_to_axes            # old name, kept for scripts written against it


def axes_to_pythea(height, a, b, c):
    """(height, a, b, c) -> PyThea (kappa, epsilon, alpha); exact inverse of pythea_to_axes."""
    kappa = b / (height - 1.0)
    alpha = b / c
    epsilon = np.sqrt(max(0.0, 1.0 - a / b)) if a <= b else -np.sqrt(max(0.0, 1.0 - b / a))
    return float(kappa), float(epsilon), float(alpha)


def principal_frame(lat_deg, lon_deg, tilt_deg):
    """Orthonormal frame (e1, e2, e3) in HGS Cartesian coordinates (see module doc)."""
    lat, lon, tilt = np.radians([lat_deg, lon_deg, tilt_deg])
    e1 = np.array([cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)])
    e2_0 = np.array([-sin(lon), cos(lon), 0.0])          # unit(z x e1); equatorial, +longitude
    e3_0 = np.cross(e1, e2_0)                              # local north
    e2 = cos(tilt) * e2_0 + sin(tilt) * e3_0
    e3 = -sin(tilt) * e2_0 + cos(tilt) * e3_0
    return e1, e2, e3


def grid_polyline_order(n_theta, n_phi):
    """
    Index order that draws a (n_theta x n_phi) grid as one polyline: rings first, then
    meridians, each alternating direction so consecutive segments stay short (no line
    through the body between pole and pole).
    """
    idx = np.arange(n_theta * n_phi).reshape(n_theta, n_phi)
    rings = [idx[i] if i % 2 == 0 else idx[i][::-1] for i in range(n_theta)]
    # rings end on the last row (south pole) -> start the meridians from the south pole
    merid = [idx[:, j][::-1] if j % 2 == 0 else idx[:, j] for j in range(n_phi)]
    return np.concatenate(rings + merid)


# ----------------------------------------------------------------------------- the model
class DeformableShock:
    """
    Parameters
    ----------
    height : apex heliocentric height [R_sun]
    a, b, c : semi-axes along e1 (apex direction), e2, e3 [R_sun]
    lat, lon, tilt : apex latitude, Stonyhurst longitude and tilt [deg]
    lmax : maximum spherical-harmonic degree of the deformation (0 = none)
    coeffs : flat array of (lmax+1)^2 coefficients a_lm, ordered by lm_index; default zeros
    dents : iterable of (theta_deg, phi_deg, sigma_deg, depth) Gaussian dents in the
            principal frame (theta from the apex axis, phi from e2); depth < 0 is a bulge
    time : optional ISO time string the surface refers to (carried through to_dict / from_dict)
    """

    def __init__(self, height, a, b, c, lat=0.0, lon=0.0, tilt=0.0,
                 lmax=0, coeffs=None, dents=(), time=None):
        self.height, self.a, self.b, self.c = (float(v) for v in (height, a, b, c))
        if self.height <= 1.0:
            raise ValueError('apex height must exceed 1 R_sun')
        if min(self.a, self.b, self.c) <= 0.0:
            raise ValueError('semi-axes a, b, c must be positive')
        self.time = None if time is None else str(time)
        self.lat, self.lon, self.tilt = float(lat), float(lon), float(tilt)
        self.lmax = int(lmax)
        n = n_coeffs(self.lmax)
        self.coeffs = np.zeros(n) if coeffs is None else np.asarray(coeffs, dtype=float).reshape(-1)
        if self.coeffs.shape != (n,):
            raise ValueError(f'coeffs must have (lmax+1)^2 = {n} entries, got {self.coeffs.shape}')
        self.dents = []
        for d in dents:
            d = tuple(float(v) for v in d)
            if len(d) != 4:
                raise ValueError('each dent is (theta_deg, phi_deg, sigma_deg, depth)')
            if d[2] <= 0.0:
                raise ValueError('dent sigma must be positive')
            self.dents.append(d)
        self._axes = (self.a, self.b, self.c, self.height - self.a)
        self._frame = principal_frame(self.lat, self.lon, self.tilt)
        self._center = self._axes[3] * self._frame[0]

    # ----- geometry helpers
    @property
    def axes(self):
        """(a, b, c, r_center) in R_sun."""
        return self._axes

    @property
    def frame(self):
        """(e1, e2, e3) unit vectors in HGS."""
        return self._frame

    @property
    def center(self):
        """Ellipsoid centre in HGS [R_sun]."""
        return self._center

    def direction(self, theta, phi):
        """Unit direction u(theta, phi) in HGS; output shape = broadcast(theta, phi).shape + (3,)."""
        theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
        e1, e2, e3 = self._frame
        st = sin(theta)
        return (cos(theta)[..., None] * e1 + (st * cos(phi))[..., None] * e2 + (st * sin(phi))[..., None] * e3)

    def log_deformation(self, theta, phi, basis=None):
        """
        S(theta, phi): harmonic part plus dents.
        basis: optional precomputed harmonic_basis(theta, phi, self.lmax) for this exact grid.
        """
        theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
        s = np.zeros(theta.shape)
        if basis is not None and self.coeffs.any():
            s = s + (basis @ self.coeffs).reshape(theta.shape)
        elif self.coeffs.any():
            for l in range(self.lmax + 1):
                for m in range(-l, l + 1):
                    a = self.coeffs[lm_index(l, m)]
                    if a != 0.0:
                        s = s + a * real_sph_harm(l, m, theta, phi)
        for th_d, ph_d, sg_d, depth in self.dents:
            if depth == 0.0:
                continue
            th_d, ph_d, sg_d = np.radians([th_d, ph_d, sg_d])
            cosd = cos(theta) * cos(th_d) + sin(theta) * sin(th_d) * cos(phi - ph_d)
            delta = np.arccos(np.clip(cosd, -1.0, 1.0))
            s = s - depth * exp(-0.5 * (delta / sg_d) ** 2)
        return s

    def log_radius(self, theta, phi, basis=None):
        theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
        a, b, c, _ = self._axes
        st, ct = sin(theta), cos(theta)
        inv2 = (ct / a) ** 2 + (st * cos(phi) / b) ** 2 + (st * sin(phi) / c) ** 2
        return -0.5 * np.log(inv2) + self.log_deformation(theta, phi, basis)

    def radius(self, theta, phi, basis=None):
        """R(theta, phi): distance from the ellipsoid centre [R_sun]."""
        return exp(self.log_radius(theta, phi, basis))

    def position(self, theta, phi, basis=None):
        """Surface points in HGS [R_sun]; shape = broadcast(theta, phi).shape + (3,)."""
        return self._center + self.radius(theta, phi, basis)[..., None] * self.direction(theta, phi)

    def normals(self, theta, phi, delta=1e-5):
        """Outward unit normals at (theta, phi), same shape as position()."""
        theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
        x_th = (self.position(theta + delta, phi) - self.position(theta - delta, phi)) / (2 * delta)
        x_ph = (self.position(theta, phi + delta) - self.position(theta, phi - delta)) / (2 * delta)
        n = np.cross(x_th, x_ph)
        pole = (theta < 1e-9) | (theta > pi - 1e-9)
        if np.any(pole):
            # x_ph vanishes at the poles: use two orthogonal chords through the pole instead
            th = np.where(theta < pi / 2, delta, pi - delta)
            x_s = (self.position(th, 0.0) - self.position(th, pi)) / (2 * delta)
            x_t = (self.position(th, pi / 2) - self.position(th, 1.5 * pi)) / (2 * delta)
            n = np.where(pole[..., None], np.cross(x_s, x_t), n)
        # star-shaped surface: the outward normal always points away from the centre
        r = self.position(theta, phi) - self._center
        sign = np.sign(np.sum(n * r, axis=-1))
        sign = np.where(sign == 0, 1.0, sign)
        n = n * sign[..., None]
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    # ----- sampled surface
    @staticmethod
    def grid(n_theta=25, n_phi=49):
        """(theta, phi) grids, theta in [0, pi], phi in [0, 2 pi] (both endpoints included)."""
        return np.meshgrid(np.linspace(0.0, pi, n_theta), np.linspace(0.0, 2 * pi, n_phi), indexing='ij')

    def surface(self, n_theta=25, n_phi=49):
        """
        Regular (theta, phi) sampling of the surface.
        Returns dict with 'theta', 'phi' (2D grids), 'radius', 'xyz' (n_theta, n_phi, 3) in HGS [R_sun]
        and 'normals' (same shape). phi includes both 0 and 2 pi so rings close.
        """
        th, ph = self.grid(n_theta, n_phi)
        return {'theta': th, 'phi': ph, 'radius': self.radius(th, ph),
                'xyz': self.position(th, ph), 'normals': self.normals(th, ph)}

    def mesh(self, n_theta=25, n_phi=49, order='points'):
        """
        (N, 3) array of surface points in HGS [R_sun].
        order='points': row-major grid; order='grid': polyline order for drawing a wire grid.
        """
        th, ph = self.grid(n_theta, n_phi)
        xyz = self.position(th, ph).reshape(-1, 3)
        if order == 'grid':
            return xyz[grid_polyline_order(n_theta, n_phi)]
        if order == 'points':
            return xyz
        raise ValueError("order must be 'points' or 'grid'")

    def mesh_sunpy(self, date, n_theta=25, n_phi=49, order='points'):
        """Surface points as a SunPy SkyCoord (HGS, Cartesian), directly plottable with plot_coord."""
        if not _HAVE_SUNPY:
            raise ImportError('mesh_sunpy needs astropy and sunpy')
        m = self.mesh(n_theta, n_phi, order).T * sun.constants.radius
        return SkyCoord(*m, frame=frames.HeliographicStonyhurst, obstime=date,
                        representation_type='cartesian')

    def nose(self, n_theta=91, n_phi=181):
        """Surface point farthest from the Sun (on the sampling grid) in HGS [R_sun]."""
        xyz = self.surface(n_theta, n_phi)['xyz'].reshape(-1, 3)
        return xyz[np.argmax(np.linalg.norm(xyz, axis=1))]

    # ----- (de)serialisation; the dict is what gcs_shock_gui.py stores in shock_params.json
    @property
    def kappa(self):
        return axes_to_pythea(self.height, self.a, self.b, self.c)[0]

    @property
    def epsilon(self):
        return axes_to_pythea(self.height, self.a, self.b, self.c)[1]

    @property
    def alpha(self):
        return axes_to_pythea(self.height, self.a, self.b, self.c)[2]

    def to_dict(self):
        kappa, epsilon, alpha = axes_to_pythea(self.height, self.a, self.b, self.c)
        d = {'model': 'shock', 'time': self.time, 'height': self.height, 'kappa': kappa, 'epsilon': epsilon,
             'alpha': alpha, 'lat': self.lat, 'lon': self.lon, 'tilt': self.tilt,
             'lmax': self.lmax, 'coeffs': self.coeffs.tolist(), 'dents': [list(d) for d in self.dents]}
        if d['time'] is None:
            del d['time']
        return d

    @classmethod
    def from_dict(cls, d):
        """Reads the PyThea form (kappa, epsilon, alpha); files with semi-axes (a, b, c) also load."""
        lmax = int(d.get('lmax', 0))
        coeffs = d.get('coeffs') or None
        if 'kappa' in d:
            a, b, c, _ = pythea_to_axes(d['height'], d['kappa'], d['epsilon'], d['alpha'])
        else:
            a, b, c = d['a'], d['b'], d['c']
        return cls(d['height'], a, b, c, d.get('lat', 0.0), d.get('lon', 0.0), d.get('tilt', 0.0),
                   lmax=lmax, coeffs=coeffs, dents=d.get('dents', ()), time=d.get('time'))

    @classmethod
    def from_pythea(cls, height, kappa, epsilon, alpha, lat=0.0, lon=0.0, tilt=0.0, **kw):
        """Construct from PyThea's (height, kappa, epsilon, alpha)."""
        a, b, c, _ = pythea_to_axes(height, kappa, epsilon, alpha)
        return cls(height, a, b, c, lat, lon, tilt, **kw)

    def scaled(self, s, time=None, r0=0.0):
        """
        Self-similar copy: every length measured from the point r0 * e1 (r0 = 0: Sun centre; r0 = 1: the
        surface point below the apex) times s, i.e. h -> r0 + s (h - r0) and a, b, c -> s a, s b, s c.
        Direction, tilt and the dents/bulges (angles and log-depths) are unchanged, so the shape is
        identical. r0 = 1 keeps PyThea's kappa, epsilon, alpha fixed (its self-similar mode).
        """
        s, r0 = float(s), float(r0)
        return DeformableShock(r0 + s * (self.height - r0), s * self.a, s * self.b, s * self.c, self.lat,
                               self.lon, self.tilt, lmax=self.lmax, coeffs=self.coeffs, dents=self.dents,
                               time=self.time if time is None else time)

    def ellipsoid(self):
        """The same surface without dents/bulges and harmonics."""
        return DeformableShock.from_dict({**self.to_dict(), 'dents': [], 'lmax': 0, 'coeffs': []})

    def with_dent(self, theta_deg, phi_deg, sigma_deg, depth):
        """Copy of the model with one more Gaussian dent."""
        d = self.to_dict()
        d['dents'].append([theta_deg, phi_deg, sigma_deg, depth])
        return DeformableShock.from_dict(d)

    def __repr__(self):
        a, b, c, rc = self._axes
        kappa, epsilon, alpha = axes_to_pythea(self.height, a, b, c)
        return (f'DeformableShock(h={self.height:.2f} Rs, kappa={kappa:.3f}, eps={epsilon:.3f}, alpha={alpha:.3f} | '
                f'a={a:.2f}, b={b:.2f}, c={c:.2f}, r_c={rc:.2f} Rs, '
                f'lat={self.lat:.1f}, lon={self.lon:.1f}, tilt={self.tilt:.1f} deg, '
                f'lmax={self.lmax}, dents={len(self.dents)})')
