"""
Tests for image_data.py (data running differences) and the GUI's dispatch. No network: Fido and Horizons mocked.
Run:  python test_image_data.py
"""
import contextlib
import datetime as dt
import io
import shutil
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

import event_config as cfg
import image_data as idt

T = dt.datetime(2021, 10, 28, 16, 54)
N = 256


def header(kind, when, exp, polar=None):
    h = {'CTYPE1': 'HPLN-TAN', 'CTYPE2': 'HPLT-TAN', 'CUNIT1': 'arcsec', 'CUNIT2': 'arcsec',
         'CDELT1': 120.0, 'CDELT2': 120.0, 'CRPIX1': (N + 1) / 2, 'CRPIX2': (N + 1) / 2, 'CRVAL1': 0.0, 'CRVAL2': 0.0,
         'EXPTIME': exp, 'HGLN_OBS': 0.0, 'HGLT_OBS': 0.0, 'DSUN_OBS': 1.496e11, 'RSUN_REF': 6.957e8}
    if kind == 'lasco':                                       # LASCO level 0.5: date and time in two keywords
        h.update({'DATE-OBS': when.strftime('%Y/%m/%d'), 'TIME-OBS': when.strftime('%H:%M:%S.000'),
                  'INSTRUME': 'LASCO', 'DETECTOR': 'C3', 'TELESCOP': 'SOHO'})
    elif kind == 'cor2':
        h.update({'DATE-OBS': when.isoformat(), 'INSTRUME': 'SECCHI', 'DETECTOR': 'COR2', 'OBSRVTRY': 'STEREO_A',
                  'POLAR': polar if polar is not None else 1001})
    else:
        h.update({'DATE-OBS': when.isoformat(), 'INSTRUME': 'AIA_4', 'TELESCOP': 'SDO/AIA', 'WAVELNTH': 193,
                  'WAVEUNIT': 'angstrom'})
    return fits.Header(h)


def scene(when, with_front):
    """A corona falling off with radius, an occulter (zeros), and optionally a bright ring (the shock front)."""
    y, x = np.mgrid[:N, :N] - (N - 1) / 2
    r = np.hypot(x, y)
    img = 5000.0 / (1 + (r / 20.0) ** 2)
    if with_front:
        img = img * (1 + 0.15 * np.exp(-((r - 80.0) / 4.0) ** 2))
    img[r < 25] = 0.0
    return img


def write(folder, name, kind, when, exp, with_front, polar=None, seed=0):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    data = scene(when, with_front) * exp + rng.normal(0, 2.0, (N, N)) * (scene(when, False) > 0)
    fits.writeto(folder / name, data.astype('float32'), header(kind, when, exp, polar), overwrite=True)


def setup():
    tmp = Path(tempfile.mkdtemp())
    cfg.DATA_DIR, cfg.CORO_SIZE, cfg.EUV_SIZE, cfg.CORO_DIFF_MIN, cfg.EUV_DIFF_MIN = tmp / 'data', N, N, 12, 10
    return tmp


def test_local_frames_are_found_and_paired():
    tmp = setup()
    sp = idt.spec('coronagraph', 'STA', 'COR2')
    f = Path(sp['folder'])
    write(f, '20211028_164000_d4c2A.fts', 'cor2', T - dt.timedelta(minutes=14), 4.0, False)
    write(f, '20211028_164500_n4c2A.fts', 'cor2', T - dt.timedelta(minutes=9), 4.0, False, polar=240)   # polarised
    write(f, '20211028_165400_d4c2A.fts', 'cor2', T, 4.0, True)
    frames = idt.local_frames(sp)
    assert len(frames) == 2                                                     # the polarised one is skipped
    cur, prev = idt.pick(frames, T + dt.timedelta(minutes=3), 12, idt.MATCH_MIN['coronagraph'])
    assert cur[0] == T and prev[0] == T - dt.timedelta(minutes=14)
    sp = idt.spec('coronagraph', 'SOHO', 'C3')                                  # LASCO: DATE-OBS + TIME-OBS
    write(Path(sp['folder']), 'c3.fts', 'lasco', T, 18.0, True)
    assert idt.local_frames(sp)[0][0] == T
    shutil.rmtree(tmp)
    return 'COR2 total-brightness frames paired (polarised frame skipped); LASCO date + time parsed'


def test_difference_is_exposure_normalised_in_sigma():
    tmp = setup()
    sp = idt.spec('euv', 'SOHO', 'AIA193')
    f = Path(sp['folder'])
    write(f, 'a.fits', 'aia', T - dt.timedelta(minutes=10), 2.0, False, seed=1)
    write(f, 'b.fits', 'aia', T, 3.0, True, seed=2)                             # different exposure, a front
    with contextlib.redirect_stdout(io.StringIO()):
        m = idt.load_rd('euv', 'SOHO', 'AIA193', T)
    y, x = np.mgrid[:N, :N] - (N - 1) / 2
    r = np.hypot(x, y)
    far = (r > 110) & (r < 120)                                                 # no front: noise only
    ring = np.abs(r - 80) < 1
    assert m.meta['rdsource'] == 'data' and m.meta['bunit'] == 'sigma' and m.meta['rdinstr'] == 'AIA'
    assert abs(np.median(m.data[far])) < 0.3 and 0.6 < np.std(m.data[far]) < 1.6       # ~ unit noise, no exposure step
    assert np.median(m.data[ring]) > 10                                                  # the front stands out
    assert np.all(m.data[r < 24] == 0)                                                   # occulter: neutral
    shutil.rmtree(tmp)
    return f'noise {np.std(m.data[far]):.2f} sigma, front {np.median(m.data[ring]):.0f} sigma, occulter 0'


class _Unified:
    """A stand-in for Fido's UnifiedResponse: iterates over its tables and is sliced as res[k, i:j]."""
    def __init__(self, tables):
        self.tables = tables

    def __iter__(self):
        return iter(self.tables)

    def __len__(self):
        return len(self.tables)

    def __getitem__(self, key):
        k, sl = key
        return self.tables[k][sl]


class _Table:
    """A stand-in for a Fido result table: columns, len, and row slices."""
    def __init__(self, rows):
        self.rows, self.colnames = rows, ['Start Time', 'fileid']

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        if isinstance(k, str):
            return [r[0] if k == 'Start Time' else r[1] for r in self.rows]
        return _Table(self.rows[k])


def test_fido_route_downloads_the_two_frames_it_needs():
    import sunpy.net
    tmp = setup()
    calls = {'fetched': []}
    when = [T - dt.timedelta(minutes=m) for m in (24, 15, 12, 0)]
    rows = [(when[0], 'cor2/20211028_163000_d4c2A.fts'), (when[1], 'cor2/20211028_163900_n4c2A.fts'),
            (when[2], 'cor2/20211028_164200_d4c2A.fts'), (when[3], 'cor2/20211028_165400_d4c2A.fts')]

    class FakeFido:
        @staticmethod
        def search(*args):
            return _Unified([_Table(rows)])

        @staticmethod
        def fetch(table, path, progress=False, downloader=None):
            t, fid = table.rows[0]
            name = fid.split('/')[-1]
            write(Path(path).parent, name, 'cor2', t, 4.0, t == T, polar=1001 if 'd4c2' in fid else 0)
            calls['fetched'].append(name)
            return type('R', (), {'errors': []})()
    real = sunpy.net.Fido
    sunpy.net.Fido = FakeFido
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            m = idt.load_rd('coronagraph', 'STA', 'COR2', T)
    finally:
        sunpy.net.Fido = real
    assert calls['fetched'] == ['20211028_165400_d4c2A.fts', '20211028_164200_d4c2A.fts'], calls
    assert m.meta['rdprev'].endswith('16:42:00') and m.meta['rdinstr'] == 'COR2'
    shutil.rmtree(tmp)
    return 'only the two total-brightness frames 12 min apart downloaded; the polarised one never'


def test_gui_dispatch():
    from PyQt5 import QtWidgets                                  # the GUI module needs a Qt application
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import matplotlib.cbook
    matplotlib.cbook._get_running_interactive_framework = lambda: None   # also without a display
    import gcs_shock_gui as G
    seen = []
    real_img, real_euv, real_rd = G.load_image, G.load_euv, idt.load_rd
    G.load_image = lambda sc, det, date, rd: seen.append(('helioviewer', det)) or 'hv'
    G.load_euv = lambda sc, date, rd: seen.append(('helioviewer', 'euv')) or 'hv'
    try:
        cfg.RD = True
        idt.load_rd = lambda *a: seen.append(('data', a[2])) or 'data'
        assert G.load_view('coronagraph', 'STA', 'COR2', T) == 'data'
        assert G.load_view('coronagraph', 'STA', 'COR1', T) == 'hv'                     # COR1: Helioviewer
        idt.load_rd = lambda *a: (_ for _ in ()).throw(RuntimeError('no network'))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            assert G.load_view('euv', 'SOHO', 'AIA193', T) == 'hv'                       # falls back
        assert 'using Helioviewer' in out.getvalue()
        cfg.RD = False
        idt.load_rd = lambda *a: seen.append(('data', a[2])) or 'data'
        assert G.load_view('coronagraph', 'SOHO', 'C3', T) == 'hv'                       # RD off: plain Helioviewer
    finally:
        G.load_image, G.load_euv, idt.load_rd = real_img, real_euv, real_rd
        cfg.RD = True
    from sunpy.map import Map
    m = Map(np.zeros((4, 4)), {**dict(header('lasco', T, 1.0)), 'rdsource': 'data', 'rdinstr': 'LASCO'})
    nrm = G.image_norm(m, 'coronagraph')
    assert (nrm.vmin, nrm.vmax) == tuple(cfg.lasco_vmm)
    return 'COR2 -> data; COR1 -> Helioviewer; failure -> Helioviewer; RD off -> Helioviewer; range from lasco_vmm'


def test_nothing_is_downloaded_twice():
    import json
    import sunpy.coordinates
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    tmp = setup()
    calls = {'horizons': 0, 'hv': 0}
    real = sunpy.coordinates.get_horizons_coord

    def fake_horizons(body, time):
        calls['horizons'] += 1
        return SkyCoord(0.15 * u.deg, 4.65 * u.deg, 0.9828 * u.AU, frame=sunpy.coordinates.frames.HeliographicStonyhurst,
                        obstime=time)
    sunpy.coordinates.get_horizons_coord = fake_horizons
    try:
        a = idt.soho_position(dt.datetime(2021, 10, 28, 16, 54, 7))
        b = idt.soho_position(dt.datetime(2021, 10, 28, 16, 54, 7))
    finally:
        sunpy.coordinates.get_horizons_coord = real
    assert calls['horizons'] == 1 and a == b and (cfg.DATA_DIR / 'soho_positions.json').exists()
    src = tmp / 'downloaded.fts'                                  # what a Helioviewer client would hand back
    write(tmp, 'downloaded.fts', 'cor2', T, 1.0, False)

    def download():
        calls['hv'] += 1
        return str(src)
    p1 = idt.cached_download('STEREO_A_SECCHI_COR1_20211028_165400', download)
    p2 = idt.cached_download('STEREO_A_SECCHI_COR1_20211028_165400', download)
    assert calls['hv'] == 1 and p1 == p2 and p1.parent.name == 'helioviewer' and p1.suffix == '.fts'
    shutil.rmtree(tmp)
    return 'SOHO position: Horizons queried once; Helioviewer image: downloaded once, then from disk'


def test_masked_vso_columns_and_retries():
    """A VSO result table whose 'Start Time' column is masked (one record without a time) and whose 'fileid' is
    masked for a record; the first download attempt times out. The frames are still found and fetched."""
    import sunpy.net
    import time as _time
    from astropy.time import Time
    from astropy.utils.masked import Masked
    tmp = setup()
    times = [T - dt.timedelta(minutes=24), T - dt.timedelta(minutes=12), T, T + dt.timedelta(minutes=12)]
    col = Time(Masked(Time(times).jd, mask=[False, False, False, True]), format='jd')   # the last: no time
    names = ['lasco/a.fts', '--', 'lasco/c.fts', 'lasco/d.fts']                        # the 2nd: masked name

    class Tab:
        colnames = ['Start Time', 'fileid']

        def __init__(self, idx):
            self.idx = idx

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, k):
            if k == 'Start Time':
                return col[self.idx]
            if k == 'fileid':
                return np.ma.masked_array(np.array(names, dtype=object)[self.idx],
                                          mask=[n == '--' for n in np.array(names)[self.idx]])
            return Tab(self.idx[k])
    calls = {'fetch': 0}

    class FakeFido:
        @staticmethod
        def search(*args):
            return _Unified([Tab(list(range(4)))])

        @staticmethod
        def fetch(table, path, progress=False, downloader=None):
            calls['fetch'] += 1
            if calls['fetch'] == 1:
                raise ConnectionError('504 Gateway Timeout')                           # VSO hiccup
            k = table.idx[0]
            write(Path(path).parent, f'f{k}.fts', 'lasco', times[k], 18.0, times[k] == T)
            return type('R', (), {'errors': []})()
    real_fido, real_sleep, real_pos = sunpy.net.Fido, _time.sleep, idt._with_soho_position
    sunpy.net.Fido, _time.sleep, idt._with_soho_position = FakeFido, (lambda s: None), (lambda m: m)
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            m = idt.load_rd('coronagraph', 'SOHO', 'C3', T)
    finally:
        sunpy.net.Fido, _time.sleep, idt._with_soho_position = real_fido, real_sleep, real_pos
    assert 'retry 1 of 1' in out.getvalue() and '504 Gateway Timeout' in out.getvalue()  # the reason is shown
    assert calls['fetch'] == 3                                                         # 1 failed + 2 frames
    assert m.meta['rdprev'].endswith('16:42:00') and m.meta['rdinstr'] == 'LASCO'
    shutil.rmtree(tmp)
    return 'masked start times and names handled; a timed-out download retried; frames 16:54 - 16:42'


def test_the_downloader_waits_long_enough():
    d = idt._downloader()
    assert d is not None and d.config.timeouts.sock_read == idt.SOCK_READ_S == 600
    assert idt.view_label('euv', 'SOHO', 'AIA193') == 'SDO AIA 193' and idt.view_label('coronagraph', 'STA', 'COR2') == 'STA COR2'
    return f'read timeout {d.config.timeouts.sock_read:.0f} s (Fido default 90 s); labels SDO AIA 193, STA COR2'


def test_a_failing_frame_is_replaced_by_the_next_nearest():
    """AIA-like: frames every 12 s; the wanted frame never downloads (an unavailable record) -> the next-nearest."""
    import sunpy.net
    import time as _time
    tmp = setup()
    times = [T - dt.timedelta(minutes=10, seconds=12 * k) for k in range(-2, 3)] + \
            [T + dt.timedelta(seconds=12 * k) for k in range(-2, 3)]
    rows = [(t, f'aia/{t:%H%M%S}.fits') for t in times]
    calls = []

    class FakeFido:
        @staticmethod
        def search(*args):
            return _Unified([_Table(rows)])

        @staticmethod
        def fetch(table, path, progress=False, downloader=None):
            t, fid = table.rows[0]
            calls.append(t)
            if t == T:                                                    # this record is never available
                return type('R', (), {'errors': [RuntimeError('File not found on the server')],
                                      '__len__': lambda self: 0})()
            write(Path(path).parent, fid.split('/')[-1], 'aia', t, 2.0, abs((t - T).total_seconds()) < 30)
            return type('R', (), {'errors': [], '__len__': lambda self: 1})()
    real_fido, real_sleep, real_syn = sunpy.net.Fido, _time.sleep, idt.AIA_SYNOPTIC
    sunpy.net.Fido, _time.sleep, idt.AIA_SYNOPTIC = FakeFido, (lambda s: None), False      # the VSO route
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            m = idt.load_rd('euv', 'SOHO', 'AIA193', T)
    finally:
        sunpy.net.Fido, _time.sleep, idt.AIA_SYNOPTIC = real_fido, real_sleep, real_syn
    assert 'File not found on the server' in out.getvalue() and 'trying the next-nearest frame' in out.getvalue()
    assert 'SDO AIA 193: downloading' in out.getvalue()                                  # the observatory, not SOHO
    assert calls[:2] == [T, T] and abs((calls[2] - T).total_seconds()) == 12            # 2 tries, then 12 s away
    assert m.meta['rdinstr'] == 'AIA'
    shutil.rmtree(tmp)
    return f'the 16:54:00 record failed twice (reason shown); the frame {calls[2]:%H:%M:%S} used instead'


def _aia_rows():
    times = [T - dt.timedelta(minutes=2 * k) for k in range(8)]
    return [(t, f'aia/AIA{t:%Y%m%d_%H%M}_0193.fits') for t in times]


def _run_aia(synoptic_search):
    """load_rd for AIA with a stand-in synoptic search and a stand-in Fido; returns (output, VSO searches, map)."""
    import sunpy.net
    import time as _time
    calls = {'vso': 0}

    class FakeFido:
        @staticmethod
        def search(*args):
            calls['vso'] += 1
            return _Unified([_Table(_aia_rows())])

        @staticmethod
        def fetch(table, path, progress=False, downloader=None):
            t, fid = table.rows[0]
            write(Path(path).parent, fid.split('/')[-1], 'aia', t, 2.0, t == T)
            return type('R', (), {'errors': [], '__len__': lambda self: 1})()
    real = sunpy.net.Fido, _time.sleep, idt._aia_synoptic_search, idt.AIA_SYNOPTIC
    sunpy.net.Fido, _time.sleep, idt._aia_synoptic_search, idt.AIA_SYNOPTIC = FakeFido, (lambda s: None), synoptic_search, True
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            m = idt.load_rd('euv', 'SOHO', 'AIA193', T)
    finally:
        sunpy.net.Fido, _time.sleep, idt._aia_synoptic_search, idt.AIA_SYNOPTIC = real
    return out.getvalue(), calls['vso'], m


def test_aia_comes_from_the_synoptic_files():
    tmp = setup()
    out, vso, m = _run_aia(lambda t0, t1, wl: _Unified([_Table(_aia_rows())]))
    assert 'from JSOC synoptic' in out and vso == 0 and m.meta['rdinstr'] == 'AIA', (out, vso)
    assert m.meta['rdprev'].endswith('16:44:00')                                       # 10 min earlier
    shutil.rmtree(tmp)
    return 'synoptic files used, the VSO never asked; frames 16:54 - 16:44'


def test_aia_falls_back_to_the_vso():
    tmp = setup()

    def broken(t0, t1, wl):
        raise ConnectionError('jsoc1 unreachable')
    out, vso, m = _run_aia(broken)
    assert 'jsoc1 unreachable' in out and 'trying VSO' in out and 'from VSO' in out and vso == 1, out
    shutil.rmtree(tmp)
    return 'synoptic route failed (reason shown) -> the VSO route'


def test_soho_position_without_horizons():
    import json
    import sunpy.coordinates
    from sunpy.coordinates import get_earth
    tmp = setup()
    real = sunpy.coordinates.get_horizons_coord

    def failing(body, time):
        raise ValueError('wldini(): missing required file LTKERNL')                    # JPL's error of 2026-10-26
    sunpy.coordinates.get_horizons_coord = failing
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            lon, lat, d = idt.soho_position(dt.datetime(2021, 10, 28, 15, 50))
    finally:
        sunpy.coordinates.get_horizons_coord = real
    e = get_earth('2021-10-28 15:50')
    assert abs(lon - e.lon.to_value('deg')) < 1e-9 and abs(lat - e.lat.to_value('deg')) < 1e-9
    assert abs(d - (e.radius.to_value('m') - 1.5e9)) < 1.0 and 'SOHO taken at L1' in out.getvalue()
    cache = cfg.DATA_DIR / 'soho_positions.json'
    assert not cache.exists() or '2021-10-28T15:50:00' not in json.loads(cache.read_text())    # not cached
    shutil.rmtree(tmp)
    return f'Horizons failing -> SOHO at L1 ({d / 1.496e11:.4f} AU, lat {lat:+.2f} deg), not cached'


if __name__ == '__main__':
    tests = [('local frames found and paired (COR2 polarisation, LASCO date/time)', test_local_frames_are_found_and_paired),
             ('difference: exposure-normalised, in sigma, occulter neutral', test_difference_is_exposure_normalised_in_sigma),
             ('Fido route: downloads only the two frames it needs', test_fido_route_downloads_the_two_frames_it_needs),
             ('GUI dispatch and display range', test_gui_dispatch),
             ('nothing downloaded twice (Horizons, Helioviewer)', test_nothing_is_downloaded_twice),
             ('masked VSO columns and a timed-out download', test_masked_vso_columns_and_retries),
             ('a failing frame is replaced by the next-nearest', test_a_failing_frame_is_replaced_by_the_next_nearest),
             ('the downloader waits long enough; labels', test_the_downloader_waits_long_enough),
             ('AIA from the JSOC synoptic files', test_aia_comes_from_the_synoptic_files),
             ('AIA falls back to the VSO', test_aia_falls_back_to_the_vso),
             ('SOHO position without Horizons', test_soho_position_without_horizons)]
    passed = 0
    for name, fn in tests:
        try:
            out = fn()
            passed += 1
            print(f'PASS  {name}' + (f'   {out}' if out is not None else ''))
        except Exception as e:
            import traceback
            print(f'FAIL  {name}: {type(e).__name__}: {e}')
            traceback.print_exc()
    print(f'{passed}/{len(tests)} passed')
