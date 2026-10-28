"""
ray_front.py -- a shock front stored on Sun-centred rays: R(Theta, Phi) [R_sun] and the ray speed V(Theta, Phi)
[km/s] on an HGS grid of colatitude Theta and longitude Phi. Used for the interplanetary propagation ray by ray
(shock_properties_dbm.py). Where the front is missing R is NaN.

It offers the DeformableShock surface interface, so shock_properties_sse.py and evolution_cobpoint.py work with it:
grid(), position(theta, phi), normals(theta, phi), radius(theta, phi) and frame use local angles about the apex
direction (theta = 0 is the apex, as for DeformableShock); center is the Sun; height is the apex distance.
"""
import json

import numpy as np
from scipy.interpolate import RegularGridInterpolator

pi = np.pi


def _orthonormal_frame(e1):
    e1 = np.asarray(e1, float) / np.linalg.norm(e1)
    z = np.array([0.0, 0.0, 1.0]) if abs(e1[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    e2 = np.cross(z, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2, np.cross(e1, e2)


class RayFront:
    def __init__(self, theta, phi, R, V=None, apex=(1.0, 0.0, 0.0), time=None, meta=None):
        self.theta = np.asarray(theta, float)                       # colatitude grid [rad], ascending in [0, pi]
        self.phi = np.asarray(phi, float)                           # longitude grid [rad], ascending in [0, 2 pi)
        self.R = np.asarray(R, float)                               # (n_theta, n_phi)
        self.V = np.zeros_like(self.R) if V is None else np.asarray(V, float)
        self.frame = _orthonormal_frame(apex)
        self.center = np.zeros(3)
        self.time = time
        self.meta = dict(meta or {})
        self.half_width_deg = self.meta.get('half_width_deg')   # angular extent of the physical front (None: all)
        self.dents = []
        self.lmax = 0
        self.coeffs = []
        self._interp = {}

    # --- lookup on the ray grid (bilinear, periodic in longitude, NaN-aware)
    EDGE_WEIGHT = 0.5     # a point gets a value when at least this share of its bilinear weight lies on rays with one

    def _grid(self, name):
        if name not in self._interp:
            arr = np.asarray(getattr(self, name), float)
            self._interp[name] = (np.concatenate([self.phi, [self.phi[0] + 2 * pi]]),
                                  np.concatenate([arr, arr[:, :1]], axis=1))
        return self._interp[name]

    def _lookup(self, name, d):
        """Bilinear on the ray grid using only the corners that have a value (weights renormalised): a ray's own
        direction always returns its value, and the front ends halfway to the first ray without one -- the plain
        interpolation (0 x NaN = NaN) would lose the edge rays and, step by step, erode the front."""
        d = np.asarray(d, float)
        th = np.clip(np.arccos(np.clip(d[..., 2], -1, 1)), self.theta[0], self.theta[-1])
        P, A = self._grid(name)
        ph = np.mod(np.arctan2(d[..., 1], d[..., 0]) - P[0], 2 * pi) + P[0]
        T = self.theta
        i = np.clip(np.searchsorted(T, th) - 1, 0, len(T) - 2)
        j = np.clip(np.searchsorted(P, ph) - 1, 0, len(P) - 2)
        ft = np.clip((th - T[i]) / (T[i + 1] - T[i]), 0.0, 1.0)
        fp = np.clip((ph - P[j]) / (P[j + 1] - P[j]), 0.0, 1.0)
        num = np.zeros(np.shape(th))
        den = np.zeros(np.shape(th))
        for a, b, w in ((i, j, (1 - ft) * (1 - fp)), (i + 1, j, ft * (1 - fp)), (i, j + 1, (1 - ft) * fp), (i + 1, j + 1, ft * fp)):
            v = A[a, b]
            ok = np.isfinite(v)
            num = num + np.where(ok, w * np.where(ok, v, 0.0), 0.0)
            den = den + np.where(ok, w, 0.0)
        out = np.where(den >= self.EDGE_WEIGHT - 1e-12, num / np.where(den > 0, den, 1.0), np.nan)
        return out.reshape(d.shape[:-1]) if d.ndim > 1 else float(np.reshape(out, -1)[0])

    def R_at(self, d):
        """Front distance [R_sun] in the HGS direction(s) d (unit vectors)."""
        return self._lookup('R', d)

    def V_at(self, d):
        """Ray speed [km/s] in the HGS direction(s) d."""
        return self._lookup('V', d)

    def ray_speed(self, xyz):
        """Ray speed [km/s] at surface points xyz (by their direction)."""
        xyz = np.asarray(xyz, float)
        return self.V_at(xyz / np.maximum(np.linalg.norm(xyz, axis=-1, keepdims=True), 1e-12))

    # --- the DeformableShock interface (local angles about the apex direction)
    @staticmethod
    def grid(n_theta=25, n_phi=49):
        return np.meshgrid(np.linspace(0.0, pi, n_theta), np.linspace(0.0, 2 * pi, n_phi), indexing='ij')

    def direction(self, theta, phi):
        theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
        e1, e2, e3 = self.frame
        return (np.cos(theta)[..., None] * e1 + (np.sin(theta) * np.cos(phi))[..., None] * e2
                + (np.sin(theta) * np.sin(phi))[..., None] * e3)

    def radius(self, theta, phi):
        return self.R_at(self.direction(theta, phi))

    def position(self, theta, phi):
        d = self.direction(theta, phi)
        return np.asarray(self.R_at(d))[..., None] * d

    def normals(self, theta, phi, delta=None):
        """Outward unit normals from the gradient of F(y) = |y| - R(y/|y|) (no pole problem)."""
        return self.normals_at(self.position(theta, phi), delta)

    def normals_at(self, xyz, delta=None):
        xyz = np.asarray(xyz, float)
        r = np.linalg.norm(xyz, axis=-1, keepdims=True)
        h = (delta or 2e-3) * np.maximum(r, 1.0)
        grad = np.empty_like(xyz)
        for i in range(3):
            off = np.zeros(3)
            off[i] = 1.0
            yp, ym = xyz + h * off, xyz - h * off
            fp = np.linalg.norm(yp, axis=-1) - self.R_at(yp / np.linalg.norm(yp, axis=-1, keepdims=True))
            fm = np.linalg.norm(ym, axis=-1) - self.R_at(ym / np.linalg.norm(ym, axis=-1, keepdims=True))
            grad[..., i] = (fp - fm) / (2 * h[..., 0])
        n = grad / np.maximum(np.linalg.norm(grad, axis=-1, keepdims=True), 1e-300)
        flip = np.sum(n * xyz, -1) < 0
        return np.where(flip[..., None], -n, n)

    @property
    def height(self):
        return float(self.R_at(self.frame[0]))

    @property
    def speed(self):
        return float(self.V_at(self.frame[0]))

    @property
    def axes(self):
        """Compatibility with DeformableShock (a, b, c, centre distance): the front's extents along e1, e2, e3."""
        e1, e2, e3 = self.frame
        return (self.height, float(np.nanmax([self.R_at(e2), self.R_at(-e2)])),
                float(np.nanmax([self.R_at(e3), self.R_at(-e3)])), 0.0)

    @property
    def lat(self):
        return float(np.degrees(np.arcsin(self.frame[0][2])))

    @property
    def lon(self):
        return float(np.degrees(np.arctan2(self.frame[0][1], self.frame[0][0])))

    def to_dict(self):
        return {'model': 'rayfront', 'time': self.time, 'height': self.height, 'lat': self.lat, 'lon': self.lon,
                'tilt': 0.0, 'dents': [], 'lmax': 0, 'coeffs': [], 'n_theta': len(self.theta), 'n_phi': len(self.phi)}

    # --- files
    def save(self, path, arrays=None, **derived):
        extra = {k: v for k, v in (arrays or {}).items() if v is not None}
        np.savez_compressed(path, theta=self.theta, phi=self.phi, R=self.R, V=self.V, apex=self.frame[0],
                            time=str(self.time), meta=json.dumps({**self.meta, **derived}), **extra)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z['meta']))
        fr = cls(z['theta'], z['phi'], z['R'], z['V'], apex=z['apex'], time=str(z['time']), meta=meta)
        for k in z.files:
            if k not in ('theta', 'phi', 'R', 'V', 'apex', 'time', 'meta'):
                setattr(fr, k, z[k])
        return fr

    @classmethod
    def from_surface(cls, model, theta, phi, V=None, time=None):
        """Sample a star-shaped (about the Sun) DeformableShock onto rays: the outer crossing along each ray."""
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        d = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1)
        R = outer_crossing(model, d)
        R = np.where(R >= 1.0, R, np.nan)                        # a crossing inside the Sun is no front
        return cls(theta, phi, R, V, apex=model.frame[0], time=time or getattr(model, 'time', None))


def outer_crossing(model, d, n_coarse=400, chunk=4000):
    """Distance [R_sun] from the Sun to the outer crossing of `model` along unit directions d (NaN: no crossing)."""
    a, b, c, rc = model.axes
    s_max = np.linalg.norm(model.center) + 2.5 * max(a, b, c) + 1.0
    S = np.linspace(1e-3, s_max, n_coarse)
    e1, e2, e3 = model.frame
    shape = d.shape[:-1]
    d_all = d.reshape(-1, 3)
    out = np.full(len(d_all), np.nan)

    def offset(s, dd):                             # s: (N, M) distances along the rays dd (N, 3)
        y = s[..., None] * dd[:, None, :]
        g = y - model.center
        rr = np.linalg.norm(g, axis=-1)
        u = g / np.maximum(rr, 1e-12)[..., None]
        th = np.arccos(np.clip(u @ e1, -1, 1))
        ph = np.mod(np.arctan2(u @ e3, u @ e2), 2 * pi)
        return rr - model.radius(th, ph)
    for i0 in range(0, len(d_all), chunk):           # chunks keep the temporaries small
        dd = d_all[i0:i0 + chunk]
        F = offset(np.broadcast_to(S, (len(dd), n_coarse)), dd)
        change = np.sign(F[:, :-1]) != np.sign(F[:, 1:])
        has = change.any(1)
        j = np.where(has, n_coarse - 2 - np.argmax(change[:, ::-1], axis=1), 0)      # the outermost crossing
        lo, hi = S[j], S[j + 1]
        flo = offset(lo[:, None], dd)[:, 0]
        for _ in range(50):
            m = 0.5 * (lo + hi)
            fm = offset(m[:, None], dd)[:, 0]
            left = np.sign(fm) == np.sign(flo)
            lo, flo = np.where(left, m, lo), np.where(left, fm, flo)
            hi = np.where(left, hi, m)
        out[i0:i0 + chunk] = np.where(has, 0.5 * (lo + hi), np.nan)
    return out.reshape(shape)
