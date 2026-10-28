"""
event_config.py -- settings shared by gcs_shock_gui.py, trace_outline.py, fit_event.py and view_fit.py.
Edit here (not in the scripts) when you move to another time or event, so all four stay consistent.
"""
import datetime as dt
import re
from pathlib import Path

# ================================ EVENT ================================
EVENT_ID = '20211028-shock'                 # every result and figure of this event goes into <folder>/<EVENT_ID>/
SC = 'soho, sta'            # spacecraft columns, any of: soho, sta, stb (e.g. 'soho, sta' for 2021)
TIME = '2021-10-28 16:36'        # 'yyyy-mm-dd HH:MM' UT; Helioviewer returns the closest image per view
RD = True                        # False: Helioviewer images as they are. True: running difference (image minus the one
                                 # CORO_DIFF_MIN / EUV_DIFF_MIN earlier), made from downloaded FITS data for LASCO
                                 # C2/C3, COR2, EUVI and AIA (image_data.py), from Helioviewer images for COR1
CORO_DIFF_MIN = 90
CORO_SIZE = 1024                 # data running differences: coronagraph images binned to about this many pixels
# display range of the data running differences, in units of each image's noise (robust sigma)
lasco_vmm = [-6.0, 6.0]
cor2_vmm = [-2.0, 4.0]
euvi_vmm = [-4.0, 4.0]
aia_vmm = [-4.0, 4.0]
SOHO = 'c3'                      # 'c2' or 'c3'
STEREO = 'cor2'                  # 'cor1' or 'cor2'
EUV_ON = False                    # True: EUV row (AIA for soho, EUVI for sta/stb) in the GUI, EUV tracing and
                                 #       EUV footprint in the fit; False: no EUV data downloaded or used anywhere
AIA_WAVELENGTH = 193             # AIA channel
EUVI_WAVELENGTH = 195            # EUVI channel
EUV_DIFF_MIN = 10                # EUV running difference: image minus the one EUV_DIFF_MIN earlier
EUV_SIZE = 1024                  # EUV images are binned to about this many pixels across (AIA is 4096)
FOOTPRINT_R = 1.05               # EUV-wave footprint sphere [R_sun]: on-disk EUV clicks are placed on it
FIT_CME = True                   # gcs_shock_gui.py: also fit the CME flux rope (GCS); False: the shock only
GUI_START = 'last'               # gcs_shock_gui.py starts from: 'last' = the GUI save written most recently (any time;
                                 # shock_<stamp>.json, gcs_<stamp>.json; fits ignored), 'nearest' = this time's file,
                                 # else the nearest fitted time, 'defaults' = fresh. Saves are files in fitted results/<id>/semi

# ================================ FOLDERS ================================
RESULTS_DIR = Path('fitted results') / EVENT_ID / 'semi'    # traces (csv), GUI saves and fits (json), fit summary
FINAL_DIR = Path('fitted results') / EVENT_ID / 'final'     # derived series, shock properties, cobpoints
FIGURE_DIR = Path('figure') / EVENT_ID                 # all figures
DATA_DIR = Path('data') / EVENT_ID                     # FITS for the data running differences: lasco/c2, lasco/c3,
                                                       # aia/193, secchi_a/cor2, secchi_a/195, secchi_b/...
# (the background below is per Carrington rotation, shared by events, so it stays outside the event folders;
#  move_to_event_folder.py moves results made before EVENT_ID existed into these folders)

# ----- background solar wind (background.py)
BACKGROUND_DIRS = [Path('corhel/cr2250')]   # MAS (CORHEL) run folders named after their rotation (crNNNN); two or
                                            # more (e.g. cr2124, cr2125 for an eruption near a rotation's end) are
                                            # blended point by point by data age (background.MultiMASBackground)
BACKGROUND_BLEND_TIME = None                # None: weigh by each lookup's own time (the background evolves); a time
                                            # like '2012-06-20 12:00': weigh by that fixed time (forecast test)
BACKGROUND_DIR = BACKGROUND_DIRS[0]         # MAS (CORHEL) run: corona.zip / helio.zip from MHDweb; unpacked and
                                            # converted data are saved in this same folder
BACKGROUND_B_SCALE = 1.0                    # multiply the model B by this (Kouloumvakos et al. 2019: x1.6-2.4)
BACKGROUND_N_SCALE = 1.0                    # multiply the model density by this (Kouloumvakos et al. 2019: x0.3-0.63)

# ----- in-situ shock arrival (drag_fit.py): ONE drag for the three shock-evolution versions, fitted along the
#       spacecraft direction with the real background; None = no observation (a default drag is used)
ARRIVAL_TIME = '2021-10-31 09:32'           # 'YYYY-MM-DD HH:MM' (UT)
ARRIVAL_SPEED = 450                         # in-situ shock speed [km/s]
SC_DISTANCE_AU = 1.00                       # heliocentric distance of that spacecraft [AU]
SC_LON = 0.0                               # its Stonyhurst longitude [deg] (0 = Earth / L1)
SC_LAT = 0.0                                # its latitude [deg]; 0 is the usual assumption
# ==========================================================================

# Helioviewer source ids (identical in hvpy's DataSource)
EUV_SOURCE = {('SDO', 'AIA', 193): 11, ('SDO', 'AIA', 171): 10, ('SDO', 'AIA', 211): 12,
              ('STA', 'EUVI', 195): 21, ('STA', 'EUVI', 171): 20,
              ('STB', 'EUVI', 195): 25, ('STB', 'EUVI', 171): 24}


def parse_spacecraft(setting=None):
    names = {'soho': 'SOHO', 'sta': 'STA', 'stb': 'STB'}
    out = []
    for token in (SC if setting is None else setting).split(','):
        t = token.strip().lower()
        if not t:
            continue
        if t not in names:
            raise ValueError(f"unknown spacecraft '{token.strip()}' in SC; use soho, sta, stb")
        if names[t] not in out:
            out.append(names[t])
    if not out:
        raise ValueError('SC must name at least one spacecraft')
    return out


def date():
    return dt.datetime.strptime(TIME.strip(), '%Y-%m-%d %H:%M')


def stamp(time_str=None):
    """File-name time stamp, e.g. 20120307_0054."""
    d = date() if time_str is None else dt.datetime.strptime(time_str.strip(), '%Y-%m-%d %H:%M')
    return d.strftime('%Y%m%d_%H%M')


def iso_time():
    return date().strftime('%Y-%m-%dT%H:%M:%S')


def coronagraph(sc):
    return SOHO.upper() if sc == 'SOHO' else STEREO.upper()


def euv_view(sc):
    """(observatory, instrument, wavelength) of the EUV imager for a spacecraft column."""
    return ('SDO', 'AIA', AIA_WAVELENGTH) if sc == 'SOHO' else (sc, 'EUVI', EUVI_WAVELENGTH)


def views():
    """All (kind, sc, detector) views of the event: coronagraphs first, then EUV."""
    out = [('coronagraph', sc, coronagraph(sc)) for sc in parse_spacecraft()]
    if EUV_ON:
        for sc in parse_spacecraft():
            obs, inst, wl = euv_view(sc)
            out.append(('euv', sc, f'{inst}{wl}'))
    return out


def results_path(name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name


def final_path(name):
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    return FINAL_DIR / name


def figure_path(name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURE_DIR / name


def stamp_date(s):
    return dt.datetime.strptime(s, '%Y%m%d_%H%M')


def nearest_result(kinds):
    """
    The saved result nearest in time to TIME, earlier or later (this TIME itself excluded):
    fitted results/<id>/semi/<kind>_<stamp>.json for kind in `kinds`, listed in order of preference when two
    kinds exist for the same time. On an exact tie in time the earlier one is used.
    Returns (path, stamp, minutes from TIME, + = later) or (None, None, None).
    """
    now, cur, best = date(), stamp(), None
    for p in RESULTS_DIR.glob('*.json'):
        m = re.match(r'^(.+)_(\d{8}_\d{4})\.json$', p.name)
        if not m or m.group(1) not in kinds or m.group(2) == cur:
            continue
        minutes = (stamp_date(m.group(2)) - now).total_seconds() / 60.0
        key = (abs(minutes), minutes > 0, kinds.index(m.group(1)))
        if best is None or key < best[0]:
            best = (key, p, m.group(2), minutes)
    return (best[1], best[2], best[3]) if best else (None, None, None)


def trace_path(sc, detector):
    return results_path(f'trace_{stamp()}_{sc.upper()}_{detector.upper()}.csv')
