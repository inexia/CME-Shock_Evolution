"""
image_data.py -- running-difference images made from the data (FITS downloaded with sunpy's Fido), used by
gcs_shock_gui.py, trace_outline.py and view_fit.py when event_config.RD = True:

  SOHO/LASCO C2, C3      data/<event>/lasco/c2, lasco/c3
  SDO/AIA                data/<event>/aia/<wavelength>    (JSOC synoptic 1024-px level 1.5 files first, else the VSO)
  STEREO/SECCHI COR2     data/<event>/secchi_a/cor2, secchi_b/cor2       (total brightness only)
  STEREO/SECCHI EUVI     data/<event>/secchi_a/<wavelength>, secchi_b/<wavelength>

COR1 is not here: its images combine three polarisation states, so its running difference stays Helioviewer's.

An image = (frame at the event time) - (frame CORO_DIFF_MIN or EUV_DIFF_MIN earlier, the nearest available):
  1. each frame divided by its exposure time (DN/s);
  2. binned to about CORO_SIZE / EUV_SIZE pixels across; the earlier frame resampled if its size differs;
  3. the difference divided by its own noise, a robust sigma (1.4826 x the median absolute deviation, which the
     CME itself hardly changes): the image is in units of sigma, so the display ranges (lasco_vmm, cor2_vmm,
     euvi_vmm, aia_vmm in event_config.py) carry over from event to event;
  4. pixels without data in either frame (occulter, pylon, missing blocks) set to 0: neutral grey.
VSO searches and downloads are retried (TRIES, growing waits): its gateway times out at times. A harmless
"Exception ignored in BaseEventLoop.__del__ ... signal only works in main thread" may follow downloads (asyncio
clean-up noise from Fido's downloader). Nothing is downloaded twice: files already in the data folder are used first; Helioviewer images (RD = False, COR1)
are kept in data/<event>/helioviewer/ by request; SOHO's position is kept in data/<event>/soho_positions.json. Once
an event's images are on disk the GUI works offline. Otherwise Fido searches the VSO and
downloads the two frames needed. COR2 keeps only total-brightness frames (file names with 'd4c2', or POLAR = 1001
in the header), never single polarisation states. SOHO's position comes from JPL Horizons (LASCO files lack it).
"""
import datetime as dt
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.io import fits
from sunpy.map import Map

import event_config as cfg

MATCH_MIN = {'coronagraph': 12.0, 'euv': 1.0}      # a local frame this close to the wanted time [min] is used
L1_M = 1.5e9                                       # SOHO's distance sunward of Earth when Horizons fails [m]
TRIES = 3                                          # VSO search attempts (its gateway times out at times)
FRAME_TRIES = 2                                    # download attempts per frame before the next-nearest frame ...
CANDIDATES = 3                                     # ... is tried, up to this many frames per wanted time
RETRY_WAIT_S = 30                                  # wait before retry k: k x this [s] (as in the recipe that worked)
AIA_SYNOPTIC = True                                # AIA from JSOC's synoptic files (level 1.5, 1024 px, 2-min cadence,
                                                   # plain files: no export server); if that fails or this sunpy has no
                                                   # synoptic client, level 1 through the VSO (sdo7's export can stall)
SOCK_READ_S = 600                                  # downloads give up when no data arrive for this long [s]
                                                   # (Fido's default, 90 s, is too short for AIA's ~10 MB files)
SEARCH_PAD_MIN = {'coronagraph': 30.0, 'euv': 5.0}  # Fido search window beyond the two wanted times [min]


# ----------------------------------------------------------------------------- what to fetch for a view
def spec(kind, sc, detector):
    """Folder, Fido query (as plain tuples: sunpy.net is imported only when searching) and file filter of a view."""
    if kind == 'coronagraph':
        if sc == 'SOHO':
            det = detector.upper()
            return {'folder': cfg.DATA_DIR / 'lasco' / det.lower(), 'instrument': 'LASCO', 'kind': kind,
                    'query': [('Instrument', 'LASCO'), ('Detector', det)], 'name_filter': None, 'polar': None}
        if detector.upper() == 'COR2':
            ab = 'a' if sc == 'STA' else 'b'
            return {'folder': cfg.DATA_DIR / f'secchi_{ab}' / 'cor2', 'instrument': 'COR2', 'kind': kind,
                    'query': [('Instrument', 'SECCHI'), ('Detector', 'COR2'), ('Source', f'STEREO_{ab.upper()}')],
                    'name_filter': ('d4c2',), 'polar': 1001}
        raise ValueError(f'{sc} {detector}: no data running difference (COR1 uses Helioviewer)')
    obs, inst, wl = cfg.euv_view(sc)
    if inst == 'AIA':
        return {'folder': cfg.DATA_DIR / 'aia' / str(wl), 'instrument': 'AIA', 'kind': kind, 'wavelength': wl,
                'query': [('Instrument', 'AIA'), ('Wavelength', wl)], 'name_filter': None, 'polar': None}
    ab = 'a' if sc == 'STA' else 'b'
    return {'folder': cfg.DATA_DIR / f'secchi_{ab}' / str(wl), 'instrument': 'EUVI', 'kind': kind,
            'query': [('Instrument', 'SECCHI'), ('Detector', 'EUVI'), ('Wavelength', wl),
                      ('Source', f'STEREO_{ab.upper()}')], 'name_filter': None, 'polar': None}


def fido_attrs(query):
    from sunpy.net import attrs as a
    return [a.Wavelength(v * u.angstrom) if k == 'Wavelength' else getattr(a, k)(v) for k, v in query]


# ----------------------------------------------------------------------------- local files
def _header_time(h):
    """Observation time from a FITS header (SECCHI/AIA: DATE-OBS with time; LASCO: DATE-OBS + TIME-OBS)."""
    d = str(h.get('DATE-OBS') or h.get('DATE_OBS') or '').replace('/', '-').strip()
    if 'T' not in d and ' ' not in d and h.get('TIME-OBS'):
        d = f'{d}T{str(h["TIME-OBS"]).strip()}'
    d = d.replace(' ', 'T')[:19]
    return dt.datetime.fromisoformat(d) if d else None


def local_frames(sp):
    """[(time, path)] of usable frames in the view's data folder."""
    out = []
    folder = Path(sp['folder'])
    if not folder.is_dir():
        return out
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in ('.fts', '.fits', '.fit') and not p.name.lower().endswith('.fits.gz'):
            continue
        try:
            with fits.open(p) as hdul:
                h = next((x.header for x in hdul if x.header.get('NAXIS', 0) >= 2), hdul[0].header)
            if sp['polar'] is not None and 'POLAR' in h and int(h['POLAR']) != sp['polar']:
                continue                                         # COR2: a single polarisation state
            t = _header_time(h)
        except Exception:
            continue
        if t is not None:
            out.append((t, p))
    return out


def pick(frames, when, lag_min, tol_min):
    """(current, earlier) from [(time, x)]: the frame nearest `when` within tol, and the frame nearest
    current - lag that is at least a minute earlier (within lag/2 + tol). None where missing."""
    if not frames:
        return None, None
    cur = min(frames, key=lambda f: abs((f[0] - when).total_seconds()))
    if abs((cur[0] - when).total_seconds()) > tol_min * 60:
        return None, None
    target = cur[0] - dt.timedelta(minutes=lag_min)
    earlier = [f for f in frames if (cur[0] - f[0]).total_seconds() >= 60]
    if not earlier:
        return cur, None
    prev = min(earlier, key=lambda f: abs((f[0] - target).total_seconds()))
    if abs((prev[0] - target).total_seconds()) > (lag_min / 2 + tol_min) * 60:
        return cur, None
    return cur, prev


# ----------------------------------------------------------------------------- Fido
def _row_time(v):
    """A search result's start time as a datetime, or None. VSO result columns are masked when any record lacks a
    value, and a masked Time converts to a masked array, not a datetime: unmask first."""
    from astropy.time import Time
    try:
        tv = v if isinstance(v, Time) else Time(v)
        if getattr(tv, 'masked', False):
            if np.any(np.asarray(tv.mask)):
                return None
            tv = tv.unmasked
        return dt.datetime.fromisoformat(str(tv.isot)[:19])
    except Exception:
        try:
            return dt.datetime.fromisoformat(str(v).replace(' ', 'T')[:19])
        except Exception:
            return None


def _row_name(table, i):
    if 'fileid' not in table.colnames:
        return ''
    v = table['fileid'][i]
    return '' if np.ma.is_masked(v) or str(v) == '--' else str(v).lower()


def _reason(e):
    msg = ' '.join(str(e).split())
    return f'{type(e).__name__}: {msg[:300]}' if msg else type(e).__name__


def _retry(what, call, tries=None):
    """call() with up to `tries` attempts and growing waits (VSO's gateway times out at times: 504 / TransportError)."""
    import time
    tries = tries or TRIES
    for k in range(1, tries + 1):
        try:
            return call()
        except Exception as e:
            if k == tries:
                raise
            print(f'  {what} failed ({_reason(e)}); retry {k} of {tries - 1} in {k * RETRY_WAIT_S} s')
            time.sleep(k * RETRY_WAIT_S)


def _downloader():
    """A one-file downloader with a long read timeout (None: Fido's default)."""
    try:
        import aiohttp
        from parfive import Downloader, SessionConfig
        return Downloader(max_conn=1, progress=False, overwrite=False,
                          config=SessionConfig(timeouts=aiohttp.ClientTimeout(total=0, sock_read=SOCK_READ_S)))
    except Exception:
        return None


def view_label(kind, sc, detector):
    """'SOHO C3', 'STA COR2', 'SDO AIA 193', 'STA EUVI 195': the observatory that takes the image."""
    if kind == 'euv':
        obs, inst, wl = cfg.euv_view(sc)
        return f'{obs} {inst} {wl}'
    return f'{sc} {detector}'


def _aia_synoptic_search(t0, t1, wl):
    """JSOC's AIA synoptic files (sunpy >= 7: AIASynopsisClient). Its dispatch check rejects a wavelength in some
    sunpy versions, so the client is asked directly; the result is wrapped like a Fido result. None: no client."""
    from sunpy.net import Fido, attrs as a
    from sunpy.net.fido_factory import UnifiedResponse
    client = next((c for c in Fido.registry if c.__name__ == 'AIASynopsisClient'), None)
    if client is None:
        return None
    res = client().search(a.Time(t0, t1), a.Instrument('AIA'), a.Level('1.5s'), a.Wavelength(wl * u.angstrom))
    return UnifiedResponse(res) if len(res) else None


def _routes(sp):
    """[(name, search(t0, t1))] in order of preference."""
    from sunpy.net import Fido, attrs as a
    vso = ('VSO', lambda t0, t1: Fido.search(a.Time(t0, t1), *fido_attrs(sp['query'])))
    if sp['instrument'] == 'AIA' and AIA_SYNOPTIC:
        return [('JSOC synoptic (level 1.5, 1024 px)', lambda t0, t1: _aia_synoptic_search(t0, t1, sp['wavelength'])),
                vso]
    return [vso]


def fetch(sp, when, lag_min):
    """Download the two frames: each route in turn (for AIA the synoptic files, then the VSO) until one works."""
    pad = SEARCH_PAD_MIN[sp['kind']]
    t0 = when - dt.timedelta(minutes=lag_min + pad)
    t1 = when + dt.timedelta(minutes=pad)
    routes = _routes(sp)
    for n, (name, search) in enumerate(routes):
        try:
            res = _retry(f'{name} search', lambda: search(t0, t1))
            if res is None:
                raise RuntimeError('no client or no files')
            print(f'  from {name}')
            _fetch_from(sp, res, when, lag_min, t0, t1)
            return
        except Exception as e:
            if n == len(routes) - 1:
                raise
            print(f'  {name}: {_reason(e)}; trying {routes[n + 1][0]}')


def _fetch_from(sp, res, when, lag_min, t0, t1):
    from sunpy.net import Fido
    pad = SEARCH_PAD_MIN[sp['kind']]
    rows = []
    for k, table in enumerate(res):
        for i in range(len(table)):
            fid = _row_name(table, i)
            if sp['name_filter'] and fid and not any(f in fid for f in sp['name_filter']):
                continue
            t = _row_time(table['Start Time'][i])
            if t is not None:
                rows.append((t, (k, i)))
    if not rows:
        raise RuntimeError(f'no {sp["instrument"]} frames between {t0:%Y-%m-%d %H:%M} and {t1:%H:%M}')
    cur, prev = pick(rows, when, lag_min, MATCH_MIN[sp['kind']] + pad)
    if cur is None:
        raise RuntimeError(f'no {sp["instrument"]} frame near {when:%Y-%m-%d %H:%M} in the search results')
    Path(sp['folder']).mkdir(parents=True, exist_ok=True)

    def download(row):
        k, i = row

        def one():                                   # a slice of the whole result, res[k, i:i+1], as in the recipe
            got = Fido.fetch(res[k, i:i + 1], path=str(Path(sp['folder']) / '{file}'),            # that worked
                             downloader=_downloader())
            errs = list(getattr(got, 'errors', None) or [])
            if errs:
                raise RuntimeError('; '.join(_reason(getattr(x, 'exception', x)) for x in errs[:2]))
            if hasattr(got, '__len__') and not len(got):
                raise RuntimeError('nothing downloaded')
            return got
        return one

    targets = [('current', cur[0])] + ([('earlier', prev[0])] if prev is not None else [])
    for role, t_want in targets:                     # the wanted frame, else the next-nearest ones
        order = sorted(rows, key=lambda r: abs((r[0] - t_want).total_seconds()))
        if role == 'earlier':
            order = [r for r in order if (cur[0] - r[0]).total_seconds() >= 60]
        for n, (t_row, row) in enumerate(order[:CANDIDATES]):
            try:
                _retry(f'download of the {t_row:%H:%M:%S} frame', download(row), tries=FRAME_TRIES)
                break
            except Exception as e:
                last = n == min(CANDIDATES, len(order)) - 1
                print(f'  {t_row:%H:%M:%S} frame not downloaded ({_reason(e)}); '
                      f'{"the last candidate" if last else "trying the next-nearest frame"}')
                if last:
                    raise


# ----------------------------------------------------------------------------- the difference image
def _per_second(m):
    exp = m.exposure_time.to_value(u.s) if getattr(m, 'exposure_time', None) is not None else 1.0
    return np.asarray(m.data, float) / max(float(exp), 1e-6)


def _binned(m, size):
    f = int(round(m.data.shape[0] / size))
    return m.superpixel(u.Quantity([f, f], u.pix), func=np.mean) if f > 1 else m


def difference(cur, prev, size, instrument):
    """The running difference of two maps in units of its robust sigma (see the module docstring)."""
    cur_b, prev_b = _binned(cur, size), _binned(prev, size)
    if prev_b.data.shape != cur_b.data.shape:
        prev_b = prev_b.resample(u.Quantity(cur_b.dimensions))
    a, b = _per_second(cur_b), _per_second(prev_b)
    ok = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    d = np.where(ok, a - b, np.nan)
    med = np.nanmedian(d)
    sigma = 1.4826 * np.nanmedian(np.abs(d - med)) if ok.any() else 1.0
    out = np.where(ok, d / (sigma if sigma > 0 else 1.0), 0.0)
    meta = dict(cur_b.meta)
    meta.update({'rdsource': 'data', 'rdinstr': instrument, 'rdprev': prev.date.isot[:19], 'bunit': 'sigma',
                 'rdsigma': float(sigma)})
    return Map(out, meta)


def soho_position(when):
    """(HGLN_OBS [deg], HGLT_OBS [deg], DSUN_OBS [m]) of SOHO at `when`, from JPL Horizons once, then from
    DATA_DIR/soho_positions.json (LASCO files and images lack it; the cache also lets the GUI start offline)."""
    import json
    key = when.isot[:19] if hasattr(when, 'isot') else when.isoformat()[:19]
    path = Path(cfg.DATA_DIR) / 'soho_positions.json'
    cache = json.loads(path.read_text()) if path.exists() else {}
    if key not in cache:
        try:
            from sunpy.coordinates import get_horizons_coord
            c = get_horizons_coord('SOHO', when)
        except Exception as e:                    # Horizons down or failing (server errors happen): SOHO at L1
            from sunpy.coordinates import get_earth
            earth = get_earth(when)
            print(f'  JPL Horizons unavailable ({_reason(e)[:120]}); SOHO taken at L1, 1.5e6 km sunward of Earth '
                  f'(its halo orbit is within ~0.3 deg of that); not cached, a later run asks Horizons again')
            return (earth.lon.to_value(u.deg), earth.lat.to_value(u.deg), earth.radius.to_value(u.m) - L1_M)
        cache[key] = [c.lon.to_value(u.deg), c.lat.to_value(u.deg), c.radius.to_value(u.m)]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=1))
    return tuple(cache[key])


def _with_soho_position(m):
    m.meta['HGLN_OBS'], m.meta['HGLT_OBS'], m.meta['DSUN_OBS'] = soho_position(m.date)
    return m


def cached_download(key, download):
    """A Helioviewer image kept in DATA_DIR/helioviewer/<key>.<ext>: downloaded once, then read from disk.
    `download()` returns the downloaded file's path (or its bytes)."""
    import shutil
    folder = Path(cfg.DATA_DIR) / 'helioviewer'
    have = sorted(folder.glob(key + '.*')) if folder.is_dir() else []
    if have:
        return have[0]
    got = download()
    folder.mkdir(parents=True, exist_ok=True)
    if isinstance(got, (bytes, bytearray)):
        path = folder / (key + '.jp2')
        path.write_bytes(got)
    else:
        path = folder / (key + ''.join(Path(str(got)).suffixes or ['.jp2']))
        shutil.copy(str(got), path)
    return path


def load_rd(kind, sc, detector, date):
    """The running-difference map of a view from the data: local frames first, else downloaded."""
    sp = spec(kind, sc, detector)
    lag = cfg.CORO_DIFF_MIN if kind == 'coronagraph' else cfg.EUV_DIFF_MIN
    cur, prev = pick(local_frames(sp), date, lag, MATCH_MIN[kind])
    if cur is None or prev is None:
        print(f'{view_label(kind, sc, detector)}: downloading frames with Fido into {sp["folder"]} ...')
        fetch(sp, date, lag)
        cur, prev = pick(local_frames(sp), date, lag, MATCH_MIN[kind])
        if cur is None or prev is None:
            raise RuntimeError(f'no pair of frames {lag} min apart near {date:%Y-%m-%d %H:%M}')
    m_cur, m_prev = Map(cur[1]), Map(prev[1])
    if sc == 'SOHO' and kind == 'coronagraph':
        m_cur = _with_soho_position(m_cur)
    size = cfg.CORO_SIZE if kind == 'coronagraph' else cfg.EUV_SIZE
    return difference(m_cur, m_prev, size, sp['instrument'])


def vmm(instrument):
    """The display range of a data running difference [sigma] from event_config.py."""
    return {'LASCO': cfg.lasco_vmm, 'COR2': cfg.cor2_vmm, 'EUVI': cfg.euvi_vmm, 'AIA': cfg.aia_vmm}[instrument]
