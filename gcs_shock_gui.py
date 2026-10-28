"""
GCS flux rope (optional: FIT_CME in event_config.py) + shock (ellipsoid with dents/bulges) on EUV and
coronagraph images.

Run:  python gcs_shock_gui.py            (event settings: event_config.py)

Layout: one column per spacecraft (SC in event_config.py). With EUV_ON = True there are two rows:
    top    EUV  (AIA for the Earth view, EUVI for STEREO), with the shock footprint on the low corona
           (the ellipsoid crossing the r = FOOTPRINT_R sphere, red) for comparison with the EUV wave
    bottom coronagraphs (LASCO C2/C3, COR1/COR2)
Parts of a front behind the solar disk, or inside the Sun, are not drawn.

Fronts
    live flux rope (GCS, royal blue)   sliders in the "Flux rope" tab
    live shock (orange)                sliders in the "Shock" tab: PyThea's ellipsoid -- apex height h,
                                       kappa, epsilon, alpha, direction, tilt. Moving h with kappa, epsilon,
                                       alpha fixed is PyThea's self-similar expansion. No dents here: dents
                                       and bulges are added only by fit_event.py.
    static fronts                      "Load": any flux-rope or shock JSON, in the colour typed above it
Start values: this TIME's own files (whichever was saved last), else the nearest time with results, earlier
or later (for the shock its final fit, else the GUI save), else gcs_params.json / shock_params.json, else defaults.
"Save" writes  fitted results/<id>/semi/shock_<stamp>.json  (and gcs_<stamp>.json with FIT_CME; the time inside)
and figure/<id>/gui_<stamp>.png, then closes. With FIT_CME = False the window is shock-only: no flux-rope tab,
mesh, start-value search or file.
"""
import datetime as dt
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import astropy.units as u
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QLabel, QComboBox, QCheckBox, QTabWidget, QScrollArea
from astropy.coordinates import SkyCoord, concatenate
from matplotlib import colors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from sunpy import log
from sunpy.coordinates import get_horizons_coord, frames, sun
from sunpy.map import Map

import event_config as cfg
import shock_fit as sf
from geometry import gcs_mesh_sunpy, apex_radius
from deformable_shock import DeformableShock
from utils.helioviewer import get_helioviewer_client
from utils.widgets import SliderAndTextbox
import utils.widgets as _widgets_module

if not hasattr(SliderAndTextbox, '_to_steps'):
    raise ImportError(f'{_widgets_module.__file__} is the original gcs_python widgets.py, whose range check snaps sliders with a positive minimum. '
                      'Replace it with the current utils/widgets.py.')

matplotlib.use('Qt5Agg')

hv = get_helioviewer_client()

straight_vertices, front_vertices, circle_vertices = 10, 10, 20   # GCS mesh resolution
shock_grid = {'grid': (19, 37), 'point cloud': (25, 49)}            # shock mesh (n_theta, n_phi) per draw mode
draw_modes = ['off', 'point cloud', 'grid']
live_colors = {'gcs': 'royalblue', 'shock': 'orange'}
FOOTPRINT_COLOR = 'red'
RSUN_M = sun.constants.radius.to_value(u.m)

log.setLevel('ERROR')


# ----------------------------------------------------------------------------- images
def running_difference(a, b):
    return Map(b.data * 1.0 - a.data * 1.0, b.meta)


def download_helioviewer(date, observatory, instrument, detector):
    """A Helioviewer image, downloaded once per request and then read from data/<event>/helioviewer/."""
    import image_data
    key = f'{observatory}_{instrument}_{detector}_{date:%Y%m%d_%H%M%S}'
    f = Map(image_data.cached_download(key, lambda: hv.download_jp2(date, observatory=observatory,
                                                                      instrument=instrument, detector=detector)))
    if observatory == 'SOHO':                                       # LASCO JP2s lack the observer position
        f = image_data._with_soho_position(f)
    return f


def load_image(spacecraft, detector, date, runndiff):
    """Coronagraph image (LASCO C2/C3, COR1/COR2), optionally running difference."""
    table = {'STA': ('STEREO_A', 'SECCHI', ('COR1', 'COR2')), 'STB': ('STEREO_B', 'SECCHI', ('COR1', 'COR2')),
             'SOHO': ('SOHO', 'LASCO', ('C2', 'C3'))}
    if spacecraft not in table:
        raise ValueError(f'unknown spacecraft: {spacecraft}')
    observatory, instrument, allowed = table[spacecraft]
    if detector not in allowed:
        raise ValueError(f'unknown detector {detector} for spacecraft {spacecraft}.')
    f = download_helioviewer(date, observatory, instrument, detector)
    if runndiff:
        f2 = download_helioviewer(date - dt.timedelta(minutes=cfg.CORO_DIFF_MIN), observatory, instrument, detector)
        return running_difference(f2, f)
    return f


def _bin(smap):
    f = int(round(smap.data.shape[0] / cfg.EUV_SIZE))
    return smap.superpixel(u.Quantity([f, f], u.pix), func=np.mean) if f > 1 else smap


def load_euv(spacecraft, date, runndiff):
    """EUV image for a spacecraft column: AIA for SOHO (Earth view), EUVI for STA/STB; binned to ~EUV_SIZE."""
    key = cfg.euv_view(spacecraft)
    if key not in cfg.EUV_SOURCE:
        raise ValueError(f'no Helioviewer source configured for {key}')
    sid = cfg.EUV_SOURCE[key]
    import image_data

    def get(when):                                                  # downloaded once per request, then from disk
        return _bin(Map(image_data.cached_download(f'source{sid}_{when:%Y%m%d_%H%M%S}',
                                                   lambda: hv.download_jp2(when, source_id=sid))))
    m = get(date)
    if runndiff:
        m0 = get(date - dt.timedelta(minutes=cfg.EUV_DIFF_MIN))
        if m0.data.shape == m.data.shape:
            return running_difference(m0, m)
        print(f'{key}: earlier EUV image has a different size, showing the plain image')
    return m


def load_view(kind, sc, detector, date):
    """RD = False: the Helioviewer image. RD = True: the running difference made from downloaded FITS data
    (image_data.py) -- except COR1, whose difference stays Helioviewer's; if the data route fails, Helioviewer's."""
    if cfg.RD and not (kind == 'coronagraph' and detector.upper() == 'COR1'):
        try:
            import image_data
            return image_data.load_rd(kind, sc, detector, date)
        except Exception as e:
            print(f'{image_data.view_label(kind, sc, detector)}: data running difference failed '
                  f'({type(e).__name__}: {e}); using Helioviewer')
    return load_euv(sc, date, cfg.RD) if kind == 'euv' else load_image(sc, detector, date, cfg.RD)


def image_norm(image, kind):
    if not cfg.RD:
        return None
    if image.meta.get('rdsource') == 'data':                     # from the data: event_config's range [sigma]
        import image_data
        lo, hi = image_data.vmm(image.meta['rdinstr'])
        return colors.Normalize(vmin=lo, vmax=hi)
    if kind == 'euv':
        lim = float(np.nanpercentile(np.abs(image.data), 98)) or 1.0
        return colors.Normalize(vmin=-lim, vmax=lim)
    return colors.Normalize(vmin=-30, vmax=30)


def panel_title(image, kind, sc, detector):
    name = (getattr(image, 'nickname', None) or f'{sc} {detector}').strip()
    if kind == 'euv':
        name += f' {detector[-3:]}' if detector[-3:] not in name else ''
    if cfg.RD:
        name += '  (running diff., data)' if image.meta.get('rdsource') == 'data' else '  (running diff.)'
    return f'{name}\n{image.date.isot[:19].replace("T", " ")} UT'


def clean_axes(ax):
    for coord in (ax.coords[0], ax.coords[1]):
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)
        coord.set_axislabel('')


def plot_xyz(ax, xyz, observer, date, style, **kw):
    """Plot HGS points [R_sun] on a map axis, hiding those behind the solar disk or inside the Sun."""
    pts = np.array(xyz, float).reshape(-1, 3)
    pts[sf.occulted(pts, observer)] = np.nan
    c = SkyCoord(*(pts.T * sun.constants.radius), frame=frames.HeliographicStonyhurst, obstime=date,
                 representation_type='cartesian')
    return ax.plot_coord(c, style, scalex=False, scaley=False, **kw)[0]


# ----------------------------------------------------------------------------- parameter files
DEFAULT_GCS = {'model': 'gcs', 'half_angle': 25, 'height': 10, 'kappa': 0.25, 'lat': 0, 'lon': 0, 'tilt': 0}
DEFAULT_SHOCK = {'model': 'shock', 'height': 13.0, 'kappa': 0.6, 'epsilon': 0.0, 'alpha': 1.0, 'lat': 15.0,
                 'lon': 325.0, 'tilt': 0.0, 'lmax': 0, 'coeffs': [], 'dents': []}
LEGACY = {'gcs': Path('gcs_params.json'), 'shock': Path('shock_params.json')}


def model_type(params):
    """'gcs' or 'shock'; files written before the shock existed carry no 'model' key."""
    if params.get('model') in ('gcs', 'shock'):
        return params['model']
    if 'half_angle' in params:
        return 'gcs'
    if 'a' in params or ('epsilon' in params and 'alpha' in params):
        return 'shock'
    raise ValueError('cannot tell whether this JSON holds flux-rope or shock parameters')


def find_start(kind):
    """
    Start parameters for 'gcs' or 'shock' and where they came from (event_config.GUI_START):
    'last'      the GUI save written most recently, whatever its time (shock_<stamp>.json / gcs_<stamp>.json;
                fits are ignored) -- you carry on from where you stopped, at any time stamp
    'nearest'   this TIME's own files (the one saved last, GUI save or final fit), else the nearest time's
    'defaults'  fresh
    then gcs_params.json / shock_params.json, then defaults.
    """
    how = getattr(cfg, 'GUI_START', 'last')
    if how == 'last':
        saves = [p for p in cfg.RESULTS_DIR.glob(f'{kind}_*.json') if re.match(rf'^{kind}_\d{{8}}_\d{{4}}\.json$', p.name)]
        if saves:
            p = max(saves, key=lambda q: q.stat().st_mtime)
            when = dt.datetime.fromtimestamp(p.stat().st_mtime)
            return json.loads(p.read_text()), f'{p.name} (your last save, {when:%Y-%m-%d %H:%M})'
    elif how == 'nearest':
        kinds = ['shock_fit', 'shock'] if kind == 'shock' else ['gcs']
        here = [p for p in (cfg.results_path(f'{k}_{cfg.stamp()}.json') for k in kinds) if p.exists()]
        if here:
            p = max(here, key=lambda q: q.stat().st_mtime)
            return json.loads(p.read_text()), f'{p.name} (this time)'
        p, st, minutes = cfg.nearest_result(kinds)
        if p is not None:
            return json.loads(p.read_text()), f'{p.name} (nearest fitted time, {minutes:+.0f} min)'
    if how != 'defaults' and LEGACY[kind].exists():
        return json.loads(LEGACY[kind].read_text()), LEGACY[kind].name
    return dict(DEFAULT_GCS if kind == 'gcs' else DEFAULT_SHOCK), 'defaults'


def save_json(path, params):
    with open(path, 'w') as fh:
        json.dump(params, fh, indent=2)


# ----------------------------------------------------------------------------- GUI
class GCSGui(QtWidgets.QMainWindow):
    def __init__(self, date, spacecraft):
        super().__init__()
        self._spacecraft = spacecraft
        self._date = date
        self._rows = (['euv'] if cfg.EUV_ON else []) + ['coronagraph']
        self.setWindowTitle(f'{"GCS + shock" if cfg.FIT_CME else "Shock"}  |  {date:%Y-%m-%d %H:%M}  |  {", ".join(spacecraft)}')

        self._root = QtWidgets.QWidget()
        self.setCentralWidget(self._root)
        self._mainlayout = QtWidgets.QHBoxLayout(self._root)
        # the panels are tiled by _layout: square, packed with small gaps, centred, redone on every resize
        self._figure = Figure(figsize=(4.3 * len(spacecraft), 4.6 * len(self._rows)))
        canvas = FigureCanvas(self._figure)
        canvas.mpl_connect('resize_event', self._layout)
        self._mainlayout.addWidget(canvas, stretch=5)
        self.addToolBar(NavigationToolbar(canvas, self))
        self._current_draw_mode = None
        self._loaded_fronts = []                   # [{'params', 'color', 'artists'}]
        self._live_artists = {'gcs': [], 'shock': []}
        self.create_widgets()
        self._last_saved = dict(zip(('gcs', 'shock'), self._snapshot()))   # to ask before closing about changes
        self.make_plot()
        self.show()

    # ----- widgets
    def _tab(self, widgets):
        layout = QtWidgets.QVBoxLayout()
        for w in widgets:
            for lab in w.findChildren(QLabel):                     # slider titles wrap instead of widening the tab
                lab.setWordWrap(True)
            layout.addWidget(w)
        layout.addStretch(1)
        page = QtWidgets.QWidget()
        page.setLayout(layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)   # never wider than the panel: long text
        scroll.setWidget(page)                                             # wraps, the number boxes stay in view
        return scroll

    def create_widgets(self):
        if cfg.FIT_CME:
            gcs, gcs_src = find_start('gcs')
            gcs = {**DEFAULT_GCS, **gcs}
        else:                                                     # shock only: no flux rope anywhere
            gcs, gcs_src = dict(DEFAULT_GCS), 'not fitted (FIT_CME = False)'
        shock_raw, shock_src = find_start('shock')
        try:
            start = DeformableShock.from_dict(shock_raw)          # PyThea or semi-axes files
        except (KeyError, ValueError):
            start, shock_src = DeformableShock.from_dict(DEFAULT_SHOCK), 'defaults'
        shock = start.to_dict()                                   # -> PyThea parameters for the sliders
        print(f'start values: shock from {shock_src}' + (f', flux rope from {gcs_src}' if cfg.FIT_CME else ''))
        self._src = {'gcs': gcs_src, 'shock': shock_src}
        if cfg.FIT_CME and gcs_src in ('defaults', LEGACY['gcs'].name) and shock_src not in ('defaults', LEGACY['shock'].name):
            print('  no flux rope saved near this time: set it in the Flux rope tab and press Save '
                  '(the flux rope is stored only by Save; fit_event.py stores only the shock)')
        if start.dents:
            print(f'  ({len(start.dents)} dent(s)/bulge(s) in {shock_src} not shown: the GUI sets the ellipsoid, '
                  f'dents and bulges come from fit_event.py)')

        self._s_half_angle = SliderAndTextbox('Half angle, \u03B1 [°]', 0, 90, gcs['half_angle'])
        self._s_height = SliderAndTextbox('Apex Height, R\u2090\u209A\u2091\u2093 [Rs]', 0, 24, gcs['height'])
        self._s_kappa = SliderAndTextbox('Aspect Ratio, \u03BA', 0, 1, gcs['kappa'])
        self._s_lat = SliderAndTextbox('Heliographic Latitude, \u03B8 [°]', -90, 90, gcs['lat'])
        self._s_lon = SliderAndTextbox('Stonyhurst Longitude, \u03C6 [°]', 0, 360, gcs['lon'])
        self._s_tilt = SliderAndTextbox('Tilt angle, \u03B3 [°]', -90, 90, gcs['tilt'])
        self._cb_show_gcs = QCheckBox('show flux rope')
        self._cb_show_gcs.setChecked(cfg.FIT_CME)
        self._l_gcs = QLabel()
        self._l_gcs.setWordWrap(True)
        gcs_widgets = [self._cb_show_gcs, self._s_half_angle, self._s_height, self._s_kappa,
                       self._s_lat, self._s_lon, self._s_tilt, self._l_gcs]

        self._s_sh_height = SliderAndTextbox('Apex height, h [Rs]', 1.05, 30, shock['height'])
        self._s_sh_kappa = SliderAndTextbox('Self-similar const., \u03BA = b/(h\u22121)', 0.01, 2, shock['kappa'])
        self._s_sh_eps = SliderAndTextbox('Eccentricity, \u03B5 (PyThea def.)', -0.95, 0.95, shock['epsilon'])
        self._s_sh_alpha = SliderAndTextbox('Aspect ratio, \u03B1 = b/c', 0.2, 5, shock['alpha'])
        self._s_sh_lat = SliderAndTextbox('Apex latitude, \u03B8 [°]', -90, 90, shock['lat'])
        self._s_sh_lon = SliderAndTextbox('Apex Stonyhurst longitude, \u03C6 [°]', 0, 360, shock['lon'])
        self._s_sh_tilt = SliderAndTextbox('Tilt angle, \u03B3 [°]', -90, 90, shock['tilt'])
        self._cb_show_shock = QCheckBox('show shock')
        self._cb_show_shock.setChecked(True)
        self._l_shock = QLabel()
        self._l_shock.setWordWrap(True)
        shock_widgets = [self._cb_show_shock, self._s_sh_height, self._s_sh_kappa, self._s_sh_eps,
                         self._s_sh_alpha, self._s_sh_lat, self._s_sh_lon, self._s_sh_tilt, self._l_shock]

        for w in gcs_widgets + shock_widgets:
            if isinstance(w, SliderAndTextbox):
                w.valueChanged.connect(self.plot_mesh)
            elif isinstance(w, QCheckBox):
                w.stateChanged.connect(self.plot_mesh)

        tabs = QTabWidget()
        tabs.addTab(self._tab(shock_widgets), 'Shock')
        if cfg.FIT_CME:
            tabs.addTab(self._tab(gcs_widgets), 'Flux rope')
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(tabs, stretch=1)
        layout.addWidget(QLabel('Display mode'))
        self._cb_mode = QComboBox()
        for mode in draw_modes:
            self._cb_mode.addItem(mode)
        self._cb_mode.setCurrentIndex(2)
        layout.addWidget(self._cb_mode)
        self._cb_mode.currentIndexChanged.connect(self.plot_mesh)
        layout.addWidget(QLabel('Color for loaded front'))
        self._t_color = QtWidgets.QLineEdit('crimson')
        layout.addWidget(self._t_color)
        b_load = QtWidgets.QPushButton('Load')
        b_load.clicked.connect(self.load_front)
        layout.addWidget(b_load)
        row = QtWidgets.QHBoxLayout()
        for label, parts, close, tip in (('Save a', ('gcs', 'shock'), True, 'all: shock and flux rope, then close'),
                                         ('Save s', ('shock',), True, 'the shock only, then close'),
                                         ('Save m', ('gcs',), True, 'the flux rope (MFR) only, then close')):
            if 'gcs' in parts and not cfg.FIT_CME and len(parts) == 1:
                continue                                         # no flux rope fitted: no "Save m"
            b = QtWidgets.QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, p=parts, c=close: self.save(p, c))
            row.addWidget(b)
        layout.addLayout(row)
        self._l_saved = QLabel('not saved yet')                 # what was saved last, and when
        self._l_saved.setStyleSheet('color: gray')
        layout.addWidget(self._l_saved)
        panel = QtWidgets.QWidget()
        panel.setLayout(layout)
        panel.setFixedWidth(300)
        self._mainlayout.addWidget(panel)

    # ----- images
    def make_plot(self):
        fig = self._figure
        spec = GridSpec(ncols=len(self._spacecraft), nrows=len(self._rows), figure=fig)
        self._axes, self._images, self._observers, self._kinds = [], [], [], []
        for r, kind in enumerate(self._rows):
            for i, sc in enumerate(self._spacecraft):
                detector = cfg.coronagraph(sc) if kind == 'coronagraph' else ''.join(map(str, cfg.euv_view(sc)[1:]))
                image = load_view(kind, sc, detector, self._date)
                ax = fig.add_subplot(spec[r, i], projection=image)
                image.plot(axes=ax, cmap='Greys_r', norm=image_norm(image, kind))
                clean_axes(ax)
                ax.set_title(panel_title(image, kind, sc, detector), fontsize=10)
                self._axes.append(ax)
                self._images.append(image)
                self._observers.append(sf.observer_from_map(image, sc))
                self._kinds.append(kind)
        self._layout()
        self.plot_mesh()

    def _layout(self, event=None):
        """Tile the image panels: the largest square that fits, small gaps, room for the two-line titles, centred."""
        fig = self._figure
        if not getattr(self, '_axes', None):
            return
        W, H = fig.get_size_inches() * fig.dpi
        nr, nc = len(self._rows), len(self._spacecraft)
        title = 2 * 10 * 1.4 / 72 * fig.dpi + 4                    # two lines of 10-pt text [px]
        gap, margin = 0.02 * min(W, H), 0.01 * min(W, H)
        s = min((W - 2 * margin - (nc - 1) * gap) / nc, (H - 2 * margin - nr * title - (nr - 1) * gap) / nr)
        if s <= 10:
            return
        x0 = (W - (nc * s + (nc - 1) * gap)) / 2
        y0 = (H - (nr * (s + title) + (nr - 1) * gap)) / 2
        for k, ax in enumerate(self._axes):
            r, c = divmod(k, nc)
            left = x0 + c * (s + gap)
            bottom = y0 + (nr - 1 - r) * (s + title + gap)
            ax.set_position([left / W, bottom / H, s / W, s / H])
        if event is not None:
            fig.canvas.draw_idle()
        fig.canvas.draw()

    # ----- meshes
    def _front_xyz(self, params, draw_mode):
        """HGS points [R_sun] of a front in drawing order, and the shock model (None for a flux rope)."""
        if model_type(params) == 'gcs':
            mesh = gcs_mesh_sunpy(self._date, np.radians(params['half_angle']), params['height'],
                                  straight_vertices, front_vertices, circle_vertices,
                                  params['kappa'], np.radians(params['lat']), np.radians(params['lon']),
                                  np.radians(params['tilt']))
            if draw_mode == 'grid':
                mesh2 = mesh.reshape((front_vertices + straight_vertices) * 2 - 3, circle_vertices).T.flatten()
                mesh = concatenate([mesh, mesh2])
            c = mesh.cartesian
            return np.stack([c.x.to_value(u.m), c.y.to_value(u.m), c.z.to_value(u.m)], axis=-1) / RSUN_M, None
        model = DeformableShock.from_dict(params)
        n_theta, n_phi = shock_grid[draw_mode]
        return model.mesh(n_theta, n_phi, 'grid' if draw_mode == 'grid' else 'points'), model

    def _draw_front(self, params, color, draw_mode):
        if draw_mode == 'off':
            return []
        xyz, model = self._front_xyz(params, draw_mode)
        style = {'grid': '-', 'point cloud': '.'}[draw_mode]
        kw = {'grid': dict(lw=0.5), 'point cloud': dict(ms=2)}[draw_mode]
        footprint = sf.ellipsoid_footprint(model, cfg.FOOTPRINT_R, n_theta=181, n_phi=121) \
            if model is not None and 'euv' in self._kinds else []
        artists = []
        for ax, obs, kind in zip(self._axes, self._observers, self._kinds):
            artists.append(plot_xyz(ax, xyz, obs, self._date, style, color=color, **kw))
            if kind == 'euv':
                for line in footprint:
                    artists.append(plot_xyz(ax, line, obs, self._date, '-', color=FOOTPRINT_COLOR, lw=1.6))
        return artists

    @staticmethod
    def _remove(artists):
        for a in artists:
            a.remove()

    def load_front(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load front parameters', str(cfg.RESULTS_DIR),
                                                        'JSON files (*.json);;All files (*)')
        if not path:
            return
        try:
            with open(path) as fh:
                params = json.load(fh)
            if model_type(params) == 'shock':
                params = DeformableShock.from_dict(params).to_dict()
        except (OSError, json.JSONDecodeError, ValueError, KeyError) as e:
            QtWidgets.QMessageBox.warning(self, 'Load failed', str(e))
            return
        color = self._t_color.text().strip() or 'crimson'
        if not colors.is_color_like(color):
            QtWidgets.QMessageBox.warning(self, 'Invalid color', f'"{color}" is not a valid color, using crimson.')
            color = 'crimson'
        draw_mode = draw_modes[self._cb_mode.currentIndex()]
        self._loaded_fronts.append({'params': params, 'color': color,
                                    'artists': self._draw_front(params, color, draw_mode)})
        self._figure.canvas.draw()

    def plot_mesh(self, *args):
        draw_mode = draw_modes[self._cb_mode.currentIndex()]
        if draw_mode != self._current_draw_mode:
            for front in self._loaded_fronts:
                self._remove(front['artists'])
                front['artists'] = self._draw_front(front['params'], front['color'], draw_mode)
            self._current_draw_mode = draw_mode
        gcs, shock = self.get_gcs_params(), self.get_shock_params()
        self._update_labels(gcs, shock)
        for key, params, show in (('gcs', gcs, cfg.FIT_CME and self._cb_show_gcs.isChecked()),
                                  ('shock', shock, self._cb_show_shock.isChecked())):
            self._remove(self._live_artists[key])
            try:
                self._live_artists[key] = self._draw_front(params, live_colors[key], draw_mode) if show else []
            except ValueError:
                self._live_artists[key] = []
        self._figure.canvas.draw()

    def _update_labels(self, gcs, shock):
        self._l_gcs.setText('Apex cross-section radius: {:.2f} Rs\nstart: {}'.format(
            apex_radius(gcs['height'], gcs['kappa']), self._src['gcs']))
        try:
            s = DeformableShock.from_dict(shock)
        except ValueError as e:
            self._l_shock.setText(f'invalid shock parameters: {e}')
            return
        a, b, c, rc = s.axes
        rear = rc - a
        self._l_shock.setText(f'semi-axes a, b, c = {a:.2f}, {b:.2f}, {c:.2f} Rs\n'
                              f'centre at {rc:.2f} Rs, rear at {rear:.2f} Rs'
                              + (' -- BEHIND THE SUN: lower |\u03b5| or \u03ba' if rear < -1.0 else '') + '\n'
                              f'Moving h with \u03BA, \u03B5, \u03B1 fixed = self-similar expansion '
                              f'(PyThea). Dents/bulges: fit_event.py.\nstart: {self._src["shock"]}')

    # ----- parameters
    def get_gcs_params(self):
        return {'model': 'gcs', 'time': cfg.iso_time(), 'half_angle': self._s_half_angle.val,
                'height': self._s_height.val, 'kappa': self._s_kappa.val, 'lat': self._s_lat.val,
                'lon': self._s_lon.val, 'tilt': self._s_tilt.val}

    def get_shock_params(self):
        return {'model': 'shock', 'time': cfg.iso_time(), 'height': self._s_sh_height.val,
                'kappa': self._s_sh_kappa.val, 'epsilon': self._s_sh_eps.val, 'alpha': self._s_sh_alpha.val,
                'lat': self._s_sh_lat.val, 'lon': self._s_sh_lon.val, 'tilt': self._s_sh_tilt.val,
                'lmax': 0, 'coeffs': [], 'dents': []}

    def _snapshot(self):
        strip = lambda d: {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items() if k != 'time'}
        return (strip(self.get_gcs_params()) if cfg.FIT_CME else None), strip(self.get_shock_params())

    def _write(self, parts=('gcs', 'shock')):
        """Save the given parts ('gcs': the flux rope, 'shock') and the figure."""
        saved = []
        now = dict(zip(('gcs', 'shock'), self._snapshot()))
        if 'gcs' in parts and cfg.FIT_CME:
            g = cfg.results_path(f'gcs_{cfg.stamp()}.json')
            save_json(g, self.get_gcs_params())
            saved.append(str(g))
            self._last_saved['gcs'] = now['gcs']
        if 'shock' in parts:
            s = cfg.results_path(f'shock_{cfg.stamp()}.json')
            save_json(s, self.get_shock_params())
            saved.append(str(s))
            self._last_saved['shock'] = now['shock']
        png = cfg.figure_path(f'gui_{cfg.stamp()}.png')
        self._figure.savefig(png, dpi=130, bbox_inches='tight')
        print(f'saved {", ".join(saved + [str(png)])}')
        what = ' and '.join({'gcs': 'flux rope', 'shock': 'shock'}[k] for k in ('gcs', 'shock')
                            if k in parts and (k != 'gcs' or cfg.FIT_CME))
        self._l_saved.setText(f'saved {what} at {dt.datetime.now():%H:%M:%S}')
        self._l_saved.setStyleSheet('color: green')
        self.statusBar().showMessage('saved ' + ', '.join(Path(p).name for p in saved + [str(png)]), 15000)

    def _unsaved(self):
        now = dict(zip(('gcs', 'shock'), self._snapshot()))
        return [k for k in ('gcs', 'shock') if now[k] is not None and now[k] != self._last_saved[k]]

    def save(self, parts=('gcs', 'shock'), close=True):
        self._write(parts)
        if close:
            self._closing_by_save = True                         # a deliberate choice: no question about the rest
            self.close()

    def closeEvent(self, event):
        """Closing the window without Save after changing anything: ask, so a flux rope set here is not lost."""
        unsaved = self._unsaved()
        if not unsaved or getattr(self, '_closing_by_save', False):
            event.accept()
            return
        what = ' and '.join({'gcs': 'the flux rope', 'shock': 'the shock'}[k] for k in unsaved)
        box = QtWidgets.QMessageBox
        answer = box.question(self, 'Save before closing?',
                              f'{what[0].upper() + what[1:]} for {cfg.TIME} UT changed since last saved.\n'
                              f'Save to "{cfg.RESULTS_DIR}" before closing?',
                              box.Save | box.Discard | box.Cancel, box.Save)
        if answer == box.Save:
            self._write(unsaved)
            event.accept()
        elif answer == box.Discard:
            event.accept()
        else:
            event.ignore()


def main():
    spacecraft = cfg.parse_spacecraft()
    if cfg.SOHO.upper() not in ('C2', 'C3'):
        raise ValueError(f"SOHO must be 'c2' or 'c3', got '{cfg.SOHO}'")
    if cfg.STEREO.upper() not in ('COR1', 'COR2'):
        raise ValueError(f"STEREO must be 'cor1' or 'cor2', got '{cfg.STEREO}'")
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    qapp = QtWidgets.QApplication(sys.argv)
    app = GCSGui(cfg.date(), spacecraft)
    app.show()
    qapp.exec_()


if __name__ == '__main__':
    main()
