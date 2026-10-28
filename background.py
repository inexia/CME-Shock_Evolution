"""
background.py -- the background solar wind for the shock: one interface, interchangeable back-ends.

    bg = load_background()                      # the MAS (CORHEL) run(s) in event_config.BACKGROUND_DIRS
    s = bg.sample(xyz, time)                    # xyz: (N, 3) HGS Cartesian [R_sun]; time: ISO string or datetime
        s['n'] [cm^-3]  s['T'] [K]  s['p'] [Pa]  s['rho'] [kg/m^3]
        s['u'] [km/s, (N, 3) HGS, inertial frame]  s['B'] [nT, (N, 3) HGS]
        s['VA'], s['cs'] [km/s]  s['beta']
    line = bg.trace(start, time, direction=+1)  # magnetic field line through start: (M, 3) HGS [R_sun]
    up = upstream_on_shock(model, bg, time)     # the background on a DeformableShock surface (theta_Bn, VA, ...)

Back-ends
  MASBackground     a MAS/CORHEL run from PSI's MHDweb (https://www.predsci.com/mhdweb/data_access.php):
                    corona.zip (1 to ~30 R_sun) and helio.zip (~30 R_sun to beyond 1 AU) in BACKGROUND_DIR.
                    The first use unpacks them into BACKGROUND_DIR/corona and BACKGROUND_DIR/helio (the .hdf
                    files need pyhdf) and saves the data, converted to physical units, as .npy files in
                    BACKGROUND_DIR/cache; later runs read only the cache. B_SCALE and N_SCALE (event_config) are
                    applied when sampling, so changing them needs no new conversion.
  ParkerBackground  analytic Parker spiral (radial wind, Parker field, n ~ r^-2): exact answers for tests.

Frames. Positions and vectors go in and come out in HGS (Stonyhurst) Cartesian, like the shock surface. MAS is
in Carrington coordinates (theta = colatitude, phi = Carrington longitude); at time t a point's Carrington
longitude is its HGS longitude + L0(t). The steady solution is fixed in the corotating frame, so evaluating with
L0(t) at each time also turns the background with the Sun. Velocities are returned in the inertial frame: a part
computed in the corotating frame (detected from v_phi ~ -Omega r sin(theta) in its outer half) gets
Omega r sin(theta) added back.
Pressure: MAS's own p is used for beta and the sound speed; if a run has no p file, p = n k T in MAS's
normalisation. A density scale (N_SCALE) scales p too (temperature kept).

Run   python background.py   to unpack/convert, print a summary along the Sun-Earth line at TIME and draw
figure/background_<run>.png (B_r r^2, n r^2, v_r on spheres, with the sub-Earth point and the shock apex).
"""
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import event_config as cfg

RSUN_M = 6.957e8
RSUN_KM = 6.957e5
AU_RSUN = 215.032
MP = 1.67262192e-27                      # kg
KB = 1.380649e-23                        # J/K
MU0 = 4e-7 * np.pi
GAMMA = 5.0 / 3.0
OMEGA = 2 * np.pi / (25.38 * 86400.0)   # Carrington (sidereal) rotation rate [rad/s]

# MAS code units -> physical (the factors of PSI's psipy): B in nT, v in km/s, rho -> n in cm^-3, T in K, p in Pa
MAS_UNITS = {'br': 2.2068914e5, 'bt': 2.2068914e5, 'bp': 2.2068914e5,
             'vr': 481.37107, 'vt': 481.37107, 'vp': 481.37107,
             'rho': 1.0e8, 't': 2.8070667e7, 'p': 3.8757170e-2}
REQUIRED = ['br', 'bt', 'bp', 'vr', 'vt', 'vp', 'rho']
OPTIONAL = ['t', 'p']


def carrington_L0(time):
    """Carrington longitude of the sub-Earth point at `time` [rad]."""
    from sunpy.coordinates import sun
    return float(sun.L0(str(time).replace('T', ' ')).to_value('rad'))


def _spherical_basis(theta, lon):
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(lon), np.cos(lon)
    e_r = np.stack([st * cp, st * sp, ct], -1)
    e_t = np.stack([ct * cp, ct * sp, -st], -1)
    e_p = np.stack([-sp, cp, np.zeros_like(sp)], -1)
    return e_r, e_t, e_p


def _to_sph(xyz):
    xyz = np.atleast_2d(np.asarray(xyz, float))
    r = np.linalg.norm(xyz, axis=1)
    theta = np.arccos(np.clip(xyz[:, 2] / np.where(r > 0, r, 1.0), -1.0, 1.0))
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    return xyz, r, theta, lon


def _derived(n, T, p, B_nT, u):
    rho = n * 1e6 * MP
    with np.errstate(invalid='ignore', divide='ignore'):
        b_T = np.linalg.norm(B_nT, axis=-1) * 1e-9
        return {'n': n, 'T': T, 'p': p, 'rho': rho, 'u': u, 'B': B_nT,
                'VA': b_T / np.sqrt(MU0 * rho) / 1e3, 'cs': np.sqrt(GAMMA * p / rho) / 1e3,
                'beta': 2 * MU0 * p / b_T ** 2}


# ----------------------------------------------------------------------------- common part
class Background:
    name = 'background'
    r_range = (1.0, np.inf)

    def sample(self, xyz, time):
        xyz, r, theta, lon = _to_sph(xyz)
        return self._sample(r, theta, lon, self._offset(time))

    def _offset(self, time):
        return 0.0

    def _b_hat(self, x, offset):
        _, r, th, lon = _to_sph(x[None])
        B = self._field(r, th, lon, offset)[0]
        nb = np.linalg.norm(B)
        return B / nb if np.isfinite(nb) and nb > 0 else np.full(3, np.nan)

    def trace(self, start, time, direction=+1, step=0.01, step_max=1.0, r_min=None, r_max=None, max_steps=50000):
        """Field line through `start` (HGS [R_sun]) along +B (direction=+1) or -B; RK4, step = step * r."""
        offset = self._offset(time)
        r_min = self.r_range[0] if r_min is None else r_min
        r_max = min(self.r_range[1], 1e4) if r_max is None else r_max
        x = np.asarray(start, float)
        pts = [x.copy()]
        for _ in range(max_steps):
            h = direction * min(step_max, step * np.linalg.norm(x))
            k1 = self._b_hat(x, offset)
            k2 = self._b_hat(x + 0.5 * h * k1, offset)
            k3 = self._b_hat(x + 0.5 * h * k2, offset)
            k4 = self._b_hat(x + h * k3, offset)
            dx = h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            if not np.all(np.isfinite(dx)):
                break
            x = x + dx
            pts.append(x.copy())
            r = np.linalg.norm(x)
            if r <= r_min or r >= r_max:
                break
        return np.array(pts)


# ----------------------------------------------------------------------------- analytic Parker spiral
class ParkerBackground(Background):
    """Radial wind v_sw, Parker field B_r ~ r^-2 with B_phi = -B_r Omega r sin(theta) / v_sw, n ~ r^-2,
    isothermal T (p = 2 n k T). polarity: +1 / -1 everywhere, or 'flat_hcs' (+ north, - south)."""
    name = 'parker'

    def __init__(self, v_sw=400.0, n_1au=5.0, br_1au=3.5, T=1.0e5, polarity=+1):
        self.v_sw, self.n_1au, self.br_1au, self.T, self.polarity = v_sw, n_1au, br_1au, T, polarity

    def _sign(self, theta):
        if self.polarity == 'flat_hcs':
            return np.where(theta < np.pi / 2, 1.0, -1.0)
        return float(self.polarity) * np.ones_like(theta)

    def _field(self, r, theta, lon, offset):
        br = self._sign(theta) * self.br_1au * (AU_RSUN / r) ** 2
        bp = -br * OMEGA * r * RSUN_KM * np.sin(theta) / self.v_sw
        e_r, e_t, e_p = _spherical_basis(theta, lon)
        out = br[:, None] * e_r + bp[:, None] * e_p
        out[r < self.r_range[0]] = np.nan
        return out

    def _sample(self, r, theta, lon, offset):
        e_r, _, _ = _spherical_basis(theta, lon)
        n = self.n_1au * (AU_RSUN / r) ** 2
        T = np.full_like(r, self.T)
        p = 2 * n * 1e6 * KB * T
        return _derived(n, T, p, self._field(r, theta, lon, offset), self.v_sw * e_r)

    def theta_bn_radial(self, r, theta):
        """theta_Bn [deg] for a radial normal (a Sun-centred spherical shock): atan(Omega r sin(theta) / v)."""
        return np.degrees(np.arctan(OMEGA * r * RSUN_KM * np.sin(theta) / self.v_sw))


# ----------------------------------------------------------------------------- MAS / CORHEL
def _read_hdf(path):
    path = Path(path)
    if path.suffix == '.h5':
        import h5py
        with h5py.File(path, 'r') as f:
            ds = f['Data']
            data = np.array(ds)
            scales = [np.array(ds.dims[i][0]) for i in range(ds.ndim)][::-1]
        return data, scales
    try:
        from pyhdf.SD import SD, SDC
    except ImportError:
        raise ImportError('reading the MAS .hdf (HDF4) files needs pyhdf:  pip install pyhdf  (or  conda install '
                          '-c conda-forge pyhdf). It is needed once: the converted data are cached.') from None
    f = SD(str(path), SDC.READ)
    try:
        sds = f.select('Data-Set-2')
        data = np.array(sds.get())
        scales = [np.array(sds.dim(i).getscale()) for i in range(data.ndim)]
    finally:
        f.end()
    return data, scales


def _to_r_theta_phi(data, scales, name):
    """Put a MAS array in (r, theta, phi) order, telling the axes apart by their coordinate values."""
    def kind(s):
        s = np.asarray(s, float)
        return 'r' if s.min() > 0.5 else ('phi' if s.max() > 4.0 else 'theta')
    for order in (scales, scales[::-1]):                 # PSI order, or reversed (HDF5 written the other way)
        if all(len(s) == n for s, n in zip(order, data.shape)):
            kinds = [kind(s) for s in order]
            if sorted(kinds) == ['phi', 'r', 'theta']:
                perm = [kinds.index(k) for k in ('r', 'theta', 'phi')]
                return np.transpose(data, perm), [np.asarray(order[i], float) for i in perm]
    raise ValueError(f'{name}: cannot match the coordinate scales {[len(s) for s in scales]} to the data shape '
                     f'{data.shape}')


class _MASPart:
    """One MAS domain (corona or helio): its variables on their own (staggered) grids."""

    def __init__(self, folder, part, verbose=True):
        self.part = part
        cache = Path(folder) / 'cache' / part
        meta_path = cache / 'meta.json'
        if not meta_path.exists():
            self._convert(Path(folder), part, cache, verbose)
        self.meta = json.loads(meta_path.read_text())
        self.vars = self.meta['variables']
        self.grids = {v: dict(np.load(cache / f'{v}_grid.npz')) for v in self.vars}
        self.data = {v: np.load(cache / f'{v}.npy', mmap_mode='r') for v in self.vars}
        self.r_range = tuple(self.meta['r_range'])
        self.corotating = self.meta['corotating']
        self._interp = {}

    @staticmethod
    def _convert(folder, part, cache, verbose):
        src = folder / part
        archive = folder / f'{part}.zip'
        if not src.exists():
            if not archive.exists():
                raise FileNotFoundError(f'neither {src} nor {archive}')
            if verbose:
                print(f'unpacking {archive} -> {src}')
            with zipfile.ZipFile(archive) as z:
                z.extractall(src)
        files = {}
        for p in sorted(src.rglob('*')):
            m = re.match(r'^([a-z]+)(\d{3})\.(hdf|h5)$', p.name)
            if m and m.group(1) in REQUIRED + OPTIONAL:
                files[m.group(1)] = str(p)                  # sorted: the last (highest) step wins
        missing = [v for v in REQUIRED if v not in files]
        if missing:
            raise FileNotFoundError(f'{part}: no file for {missing} in {src}')
        cache.mkdir(parents=True, exist_ok=True)
        r_lo, r_hi = [], []
        for v, path in files.items():
            if verbose:
                print(f'  converting {part} {Path(path).name}')
            data, scales = _read_hdf(path)
            data, (r, th, ph) = _to_r_theta_phi(data, scales, Path(path).name)
            np.save(cache / f'{v}.npy', (np.asarray(data, np.float64) * MAS_UNITS[v]).astype(np.float32))
            np.savez(cache / f'{v}_grid.npz', r=r, theta=th, phi=ph)
            r_lo.append(r.min())
            r_hi.append(r.max())
        # velocity frame: corotating if v_phi ~ -Omega r sin(theta) in the outer half of the domain
        vp = np.load(cache / 'vp.npy', mmap_mode='r')
        g = dict(np.load(cache / 'vp_grid.npz'))
        i_r = int(np.searchsorted(g['r'], 0.5 * (min(r_hi) + max(r_lo)) + 0.4 * (min(r_hi) - max(r_lo))))
        i_r = min(max(i_r, 0), len(g['r']) - 1)
        band = (g['theta'] > np.radians(30)) & (g['theta'] < np.radians(150))
        rot = OMEGA * g['r'][i_r] * RSUN_KM * np.sin(g['theta'][band])
        ratio = float(np.median(np.asarray(vp[i_r][band], float) / rot[:, None]))
        meta = {'part': part, 'variables': list(files), 'files': files, 'r_range': [max(r_lo), min(r_hi)],
                'corotating': ratio < -0.5, 'vphi_over_omega_r_sin': ratio,
                'units': 'B nT, v km/s, rho -> n cm^-3, T K, p Pa'}
        (cache / 'meta.json').write_text(json.dumps(meta, indent=2))

    def interp(self, v, r, theta, phi):
        if v not in self._interp:
            g = self.grids[v]
            self._interp[v] = (RegularGridInterpolator((g['r'], g['theta'], g['phi']), self.data[v],
                                                       bounds_error=False, fill_value=np.nan), g)
        f, g = self._interp[v]
        ph = np.mod(phi, 2 * np.pi)
        ph = np.where(ph < g['phi'][0], ph + 2 * np.pi, ph)
        ph = np.where(ph > g['phi'][-1], ph - 2 * np.pi, ph)
        ph = np.clip(ph, g['phi'][0], g['phi'][-1])
        th = np.clip(theta, g['theta'][0], g['theta'][-1])
        return f(np.column_stack([r, th, ph]))


class MASBackground(Background):
    def __init__(self, folder=None, b_scale=None, n_scale=None, verbose=True):
        self.folder = Path(folder or cfg.BACKGROUND_DIR)
        self.name = f'mas_{self.folder.name}'
        self.b_scale = cfg.BACKGROUND_B_SCALE if b_scale is None else b_scale
        self.n_scale = cfg.BACKGROUND_N_SCALE if n_scale is None else n_scale
        self.parts = []
        for part in ('corona', 'helio'):
            if (self.folder / 'cache' / part / 'meta.json').exists() or (self.folder / part).exists() \
                    or (self.folder / f'{part}.zip').exists():
                self.parts.append(_MASPart(self.folder, part, verbose))
        if not self.parts:
            raise FileNotFoundError(f'no corona.zip / helio.zip (or unpacked folders) in {self.folder}')
        self.r_range = (self.parts[0].r_range[0], self.parts[-1].r_range[1])

    def _offset(self, time):
        return carrington_L0(time)

    def _which(self, r):
        """Index of the part to use at each r (corona up to its outer edge, then helio); -1 outside."""
        idx = np.full(r.shape, -1)
        for i in range(len(self.parts) - 1, -1, -1):
            lo, hi = self.parts[i].r_range
            idx[(r >= lo) & (r <= hi)] = i
        return idx

    def _field(self, r, theta, lon, offset):
        out = np.full((len(r), 3), np.nan)
        phi = lon + offset
        e_r, e_t, e_p = _spherical_basis(theta, lon)
        idx = self._which(r)
        for i, part in enumerate(self.parts):
            m = idx == i
            if m.any():
                br, bt, bp = (part.interp(v, r[m], theta[m], phi[m]) for v in ('br', 'bt', 'bp'))
                out[m] = self.b_scale * (br[:, None] * e_r[m] + bt[:, None] * e_t[m] + bp[:, None] * e_p[m])
        return out

    def _sample(self, r, theta, lon, offset):
        N = len(r)
        n, T, p = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)
        u = np.full((N, 3), np.nan)
        phi = lon + offset
        e_r, e_t, e_p = _spherical_basis(theta, lon)
        idx = self._which(r)
        for i, part in enumerate(self.parts):
            m = idx == i
            if not m.any():
                continue
            g = lambda v: part.interp(v, r[m], theta[m], phi[m])
            n[m] = self.n_scale * g('rho')
            if 't' in part.vars:
                T[m] = g('t')
            if 'p' in part.vars:
                p[m] = self.n_scale * g('p')
            else:
                p[m] = n[m] * 1e6 * KB * T[m]
            vr, vt, vp = g('vr'), g('vt'), g('vp')
            if part.corotating:
                vp = vp + OMEGA * r[m] * RSUN_KM * np.sin(theta[m])
            u[m] = vr[:, None] * e_r[m] + vt[:, None] * e_t[m] + vp[:, None] * e_p[m]
        return _derived(n, T, p, self._field(r, theta, lon, offset), u)

    def summary(self):
        lines = [f'MAS run in {self.folder} (B x {self.b_scale}, n x {self.n_scale})']
        for part in self.parts:
            g = part.grids['rho']
            lines.append(f'  {part.part:6s}: r {part.r_range[0]:.2f} - {part.r_range[1]:.2f} R_sun, grid (r, theta, phi) = '
                         f'{len(g["r"])} x {len(g["theta"])} x {len(g["phi"])}, variables {", ".join(part.vars)}; '
                         f'velocities {"corotating -> converted to inertial" if part.corotating else "inertial"} '
                         f'(v_phi / Omega r sin(theta) = {part.meta["vphi_over_omega_r_sin"]:+.2f})')
        return '\n'.join(lines)


def load_background(**kw):
    """The background of event_config: one MAS run, or several blended (BACKGROUND_DIRS)."""
    dirs = list(getattr(cfg, 'BACKGROUND_DIRS', None) or [cfg.BACKGROUND_DIR])
    if len(dirs) > 1 and 'folder' not in kw:
        return MultiMASBackground(dirs, **kw)
    if 'folder' not in kw:
        kw['folder'] = dirs[0]
    return MASBackground(**kw)


# ----------------------------------------------------------------------------- several rotations: the time view
DAY = 86400.0


def posix(t):
    """Seconds since 1970 for an ISO string or a naive (UTC) datetime."""
    import datetime as _dt
    if isinstance(t, (int, float, np.floating)):
        return float(t)
    if isinstance(t, str):
        t = _dt.datetime.fromisoformat(t.replace(' ', 'T')[:19])
    return (t - _dt.datetime(1970, 1, 1)).total_seconds()


def rotation_of(folder):
    """Carrington rotation number from a run folder named like 'cr2124' (case-insensitive)."""
    m = re.search(r'cr(\d{4})', Path(folder).name, re.I)
    if not m:
        raise ValueError(f'{folder}: name the run folder after its Carrington rotation, e.g. corhel/cr2124')
    return int(m.group(1))


def rotation_span(cr):
    """(start, end) of Carrington rotation cr as POSIX seconds."""
    from sunpy.coordinates import sun
    return (posix(sun.carrington_rotation_time(cr).to_datetime()), posix(sun.carrington_rotation_time(cr + 1).to_datetime()))


def observed_at(span, lam_deg):
    """When Carrington longitude lam (deg) faced Earth during the rotation with this span (POSIX seconds)."""
    t0, t1 = span
    return t0 + (360.0 - np.mod(lam_deg, 360.0)) / 360.0 * (t1 - t0)


class MultiMASBackground(Background):
    """
    Several MAS runs of consecutive rotations, blended point by point by data age: at Carrington longitude lam and
    lookup time t, run i (rotation span S_i) has the data age a_i = t_obs,i(lam) - t, the time between that
    longitude facing Earth in that rotation and t, and the weight w_i ~ 1 / (|a_i| / 1 day + BLEND_EPS_DAYS).
    With event_config.BACKGROUND_BLEND_TIME set, t is that fixed time (e.g. the eruption: only what a forecast
    would have weighted the most) instead of each lookup's own time. n, T, p and the B and u vectors are blended;
    V_A, c_s and beta follow from them. A run with no value at a point (outside its domain) drops out there.
    """
    BLEND_EPS_DAYS = 0.5

    def __init__(self, folders, b_scale=None, n_scale=None, verbose=True):
        self.components = [MASBackground(f, b_scale=b_scale, n_scale=n_scale, verbose=verbose) for f in folders]
        self.crs = [rotation_of(f) for f in folders]
        self.spans = [rotation_span(c) for c in self.crs]
        self.folder = Path(folders[0])
        self.name = 'mas_' + '+'.join(f'cr{c}' for c in self.crs)
        self.r_range = (max(c.r_range[0] for c in self.components), min(c.r_range[1] for c in self.components))
        fixed = getattr(cfg, 'BACKGROUND_BLEND_TIME', None)
        self.fixed_time = posix(fixed) if fixed else None

    def _offset(self, time):
        return carrington_L0(time), posix(time)

    def weights(self, phi_carrington, t):
        """(n_runs, N) weights at Carrington longitudes phi [rad] for the lookup time t (POSIX seconds)."""
        lam = np.degrees(np.asarray(phi_carrington, float))
        t_ref = self.fixed_time if self.fixed_time is not None else t
        w = np.array([1.0 / (np.abs(observed_at(s, lam) - t_ref) / DAY + self.BLEND_EPS_DAYS) for s in self.spans])
        return w / np.sum(w, axis=0, keepdims=True)

    def _blend(self, values, w):
        """Weighted sum over runs of arrays (n_runs, N, ...) ignoring runs with no value at a point."""
        v = np.asarray(values, float)
        ok = np.all(np.isfinite(v.reshape(v.shape[0], v.shape[1], -1)), axis=2)
        ww = np.where(ok, w, 0.0)
        norm = np.sum(ww, axis=0)
        ww = ww / np.where(norm > 0, norm, 1.0)
        out = np.sum(np.nan_to_num(v) * ww.reshape(ww.shape + (1,) * (v.ndim - 2)), axis=0)
        return np.where((norm > 0).reshape(norm.shape + (1,) * (v.ndim - 2)), out, np.nan)

    def _field(self, r, theta, lon, offset):
        L0, t = offset
        w = self.weights(lon + L0, t)
        return self._blend([c._field(r, theta, lon, L0) for c in self.components], w)

    def _sample(self, r, theta, lon, offset):
        L0, t = offset
        w = self.weights(lon + L0, t)
        parts = [c._sample(r, theta, lon, L0) for c in self.components]
        n, T, p = (self._blend([q[k] for q in parts], w) for k in ('n', 'T', 'p'))
        u, B = (self._blend([q[k] for q in parts], w) for k in ('u', 'B'))
        return _derived(n, T, p, B, u)

    def summary(self):
        return '\n'.join([f'{len(self.components)} MAS runs blended by data age: ' + ', '.join(f'CR {c}' for c in self.crs)]
                         + [c.summary() for c in self.components])


def time_view(bg, time, source_lon_hgs=None):
    """Lines describing the background's time axis for an event at `time`: the rotations, where the event falls,
    and the data age (and blend weights) at the Earth-facing longitude and 60 deg to either side. West longitudes
    (HGS +, where Earth's field lines usually root) faced Earth before the event, east ones after."""
    folders = [c.folder for c in bg.components] if hasattr(bg, 'components') else [getattr(bg, 'folder', None)]
    try:
        crs = [rotation_of(f) for f in folders]
    except (ValueError, TypeError):
        return ['(no rotation number in the run folder name: no time view)']
    import datetime as _dt
    fmt = lambda s: _dt.datetime.utcfromtimestamp(s).strftime('%Y-%m-%d %H:%M')
    t = posix(time)
    L0 = np.degrees(carrington_L0(time))
    lines = []
    for c in crs:
        s0, s1 = rotation_span(c)
        where = (f'{(t - s0) / DAY:.1f} days in ({(t - s0) / (s1 - s0) * 100:.0f}%)' if s0 <= t <= s1 else
                 (f'{(s0 - t) / DAY:.1f} days before its start' if t < s0 else f'{(t - s1) / DAY:.1f} days after its end'))
        lines.append(f'  CR {c}: {fmt(s0)} - {fmt(s1)} UT; the event is {where}')
    lines.append(f'  at {str(time).replace("T", " ")[:16]} UT the Carrington longitude facing Earth is {L0:.1f} deg')
    probes = [('Earth-facing', 0.0), ('60 deg west', 60.0), ('60 deg east', -60.0)]   # HGS longitude: + is west
    if source_lon_hgs is not None:
        probes.insert(0, ('source region', float(source_lon_hgs)))
    for name, dlon in probes:
        lam = L0 + dlon
        ages = [(observed_at(rotation_span(c), lam) - t) / DAY for c in crs]
        text = ', '.join(f'CR {c}: {a:+.1f} d' for c, a in zip(crs, ages))
        if hasattr(bg, 'weights'):
            w = bg.weights(np.radians([lam % 360.0]), t)[:, 0]
            text += '  -> weights ' + ', '.join(f'{x:.2f}' for x in w)
        lines.append(f'  {name:14s} (Carrington {lam % 360.0:5.1f} deg): data age {text}')
    return lines


# ----------------------------------------------------------------------------- the background on a shock surface
def upstream_on_shock(model, bg, time, n_theta=61, n_phi=121):
    """The (unshocked) background at the points of a DeformableShock surface, with theta_Bn from its normals."""
    th, ph = model.grid(n_theta, n_phi)
    xyz = model.position(th, ph)
    nrm = model.normals(th, ph)
    s = bg.sample(xyz.reshape(-1, 3), time)
    shape = th.shape
    B = s['B'].reshape(shape + (3,))
    u = s['u'].reshape(shape + (3,))
    with np.errstate(invalid='ignore'):
        cos_bn = np.abs(np.sum(B * nrm, -1)) / np.linalg.norm(B, axis=-1)
    out = {'theta': th, 'phi': ph, 'xyz': xyz, 'normal': nrm, 'B': B, 'u': u,
           'theta_bn': np.degrees(np.arccos(np.clip(cos_bn, 0.0, 1.0))), 'u_n': np.sum(u * nrm, -1)}
    for k in ('n', 'T', 'p', 'VA', 'cs', 'beta'):
        out[k] = s[k].reshape(shape)
    return out


# ----------------------------------------------------------------------------- quick look
def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from deformable_shock import DeformableShock

    bg = load_background()
    print(bg.summary())
    time = cfg.iso_time()
    print('\ntime view (data age: when a longitude faced Earth in that rotation, minus the event time):')
    print('\n'.join(time_view(bg, time)))
    L0 = carrington_L0(time)
    print(f'\nTIME {time}: Carrington longitude of the sub-Earth point L0 = {np.degrees(L0):.2f} deg')
    print('along the Sun-Earth line:\n   r [R_sun]   n [cm^-3]    T [K]   |B| [nT]   B_r [nT]   u_r [km/s]  VA [km/s]  cs [km/s]   beta')
    radii = [r for r in (1.5, 2.5, 5, 10, 21.5, 30, 50, 100, 150, AU_RSUN) if bg.r_range[0] <= r <= bg.r_range[1]]
    s = bg.sample(np.array([[r, 0.0, 0.0] for r in radii]), time)
    for i, r in enumerate(radii):
        print(f'   {r:8.1f}  {s["n"][i]:10.3g}  {s["T"][i]:8.3g}  {np.linalg.norm(s["B"][i]):9.3g}  {s["B"][i][0]:+9.3g}  '
              f'{s["u"][i][0]:9.0f}  {s["VA"][i]:9.0f}  {s["cs"][i]:9.0f}  {s["beta"][i]:7.3g}')

    apex = None
    p = cfg.results_path(f'shock_fit_{cfg.stamp()}.json')
    if not p.exists():
        p, _, _ = cfg.nearest_result(['shock_fit'])
    if p is not None:
        m = DeformableShock.from_dict(json.loads(p.read_text()))
        apex = (np.mod(m.lon + np.degrees(L0), 360.0), m.lat, p.name)
    shells = [r for r in (2.5, 21.5, AU_RSUN) if bg.r_range[0] <= r <= bg.r_range[1]]
    lon_c = np.linspace(0, 360, 241)
    lat = np.linspace(-89, 89, 121)
    LON, LAT = np.meshgrid(lon_c, lat)
    fig, axes = plt.subplots(len(shells), 3, figsize=(15, 3.6 * len(shells)), layout='constrained', squeeze=False)
    for row, r in zip(axes, shells):
        lon_s = np.radians(LON) - L0
        xyz = r * np.stack([np.cos(np.radians(LAT)) * np.cos(lon_s), np.cos(np.radians(LAT)) * np.sin(lon_s),
                            np.sin(np.radians(LAT))], -1).reshape(-1, 3)
        q = bg.sample(xyz, time)
        e_r = xyz / r
        br = np.sum(q['B'] * e_r, 1).reshape(LON.shape) * r ** 2
        nr2 = (q['n'] * r ** 2).reshape(LON.shape)
        ur = np.sum(q['u'] * e_r, 1).reshape(LON.shape)
        lim = np.nanpercentile(np.abs(br), 98)
        for a, val, title, kw in ((row[0], br, f'B$_r$ r$^2$ [nT R$_\\odot^2$]  at {r:g} R$_\\odot$',
                                   dict(cmap='RdBu_r', vmin=-lim, vmax=lim)),
                                  (row[1], np.log10(nr2), f'log$_{{10}}$ n r$^2$ [cm$^{{-3}}$ R$_\\odot^2$]  at {r:g} R$_\\odot$',
                                   dict(cmap='viridis')),
                                  (row[2], ur, f'u$_r$ [km/s]  at {r:g} R$_\\odot$',
                                   dict(cmap='plasma', vmin=min(np.nanmin(ur), np.nanmedian(ur) - 25),
                                        vmax=max(np.nanmax(ur), np.nanmedian(ur) + 25)))):
            im = a.pcolormesh(lon_c, lat, val, shading='auto', **kw)
            fig.colorbar(im, ax=a, shrink=0.85)
            if np.nanmin(br) < 0 < np.nanmax(br):
                a.contour(lon_c, lat, br, levels=[0], colors='k', linewidths=0.8)
            a.plot(np.degrees(L0), 0, 'w*', ms=14, mec='k', label='sub-Earth (L0)')
            if apex:
                a.plot(apex[0], apex[1], 'o', ms=9, mfc='none', mec='w', mew=2, label=f'shock apex ({apex[2]})')
            a.set_title(title, fontsize=10)
            a.set_xlim(0, 360)
            a.set_ylim(-90, 90)
            a.set_xlabel('Carrington longitude [deg]')
            a.set_ylabel('latitude [deg]')
        row[0].legend(fontsize=7, loc='lower left')
    fig.suptitle(f'{bg.name} at {time} UT (black: B$_r$ = 0, the current sheet)')
    png = cfg.figure_path(f'background_{bg.name}.png')
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f'\nsaved {png}')


if __name__ == '__main__':
    main()
