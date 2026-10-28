# Irregular CME-shock reconstruction and evolution

Reconstructs the 3-D shock front of a coronal mass ejection from coronagraph (and EUV) images as an ellipsoid with
dents and bulges, follows it from the low corona to 1 AU in a MAS (CORHEL) background solar wind, and derives the
shock parameters (V_n, θ_Bn, M_A, M_f, compression ratio X) over the whole front, at the points magnetically
connected to a spacecraft (cobpoints), and at the spacecraft themselves.

The interplanetary evolution comes in three versions, all with one drag parameter fitted to the observed in-situ
arrival along the spacecraft direction:

| version | front | script |
|---|---|---|
| SSE | self-similar expansion of the coronal shape | `shock_properties_sse.py` |
| DBM | every ray of the front decelerates in its local wind (drag-based model) | `shock_properties_dbm.py` |
| GSD | DBM plus geometrical shock dynamics (shock strength along ray tubes) | `shock_properties_gsd.py` |

---

## Requirements

Python 3.11 is recommended. The packages of [gcs_python](https://github.com/johan12345/gcs_python), plus two readers
for the MAS background files:

```
astroquery
matplotlib
numpy
scipy>=1.2.0
sunpy[net,jpeg2000]>=2.1.0,<5.2.0
PyQt5
h5py
pyhdf
```

Two files from gcs_python are used directly: `geometry.py` (the GCS flux-rope mesh) and the `utils/` folder
(Helioviewer client and slider widgets).

> With sunpy ≥ 7, AIA images are downloaded from JSOC's synoptic files (level 1.5, 1024 px); with older sunpy
> they come through the VSO. Everything else works with either.

## Installation

```bash
conda create -n imgcs python=3.11
conda activate imgcs
pip install astroquery matplotlib numpy "scipy>=1.2.0" "sunpy[net,jpeg2000]>=2.1.0,<5.2.0" PyQt5 h5py
conda install -c conda-forge pyhdf        # HDF4 reader for MAS files; easier from conda-forge than pip

git clone https://github.com/johan12345/gcs_python
```

Then either copy `gcs_python/geometry.py` and `gcs_python/utils/` into this folder, or add gcs_python to the path:

```bash
export PYTHONPATH=/path/to/gcs_python:$PYTHONPATH
```

## Background solar wind (MAS / CORHEL)

1. Go to PSI's MHDweb (https://www.predsci.com/mhdweb/), open *Data Access*, and enter the eruption's date and time
   (UT). This selects the Carrington rotation whose run is used.
2. Download the ZIP with *Coronal* selected, then again with *Heliospheric* selected.
3. Put them in a folder named after the rotation: `corhel/cr2250/corona.zip` and `corhel/cr2250/helio.zip`.
4. Set `BACKGROUND_DIRS = [Path('corhel/cr2250')]` in `event_config.py`.

The files are unpacked and converted on first use; the converted data are kept in the same folder. For an eruption
near the start or end of a rotation, list two consecutive rotations (e.g. `cr2249`, `cr2250`): they are blended
point by point by how recently each longitude was observed.

Check the result with `python background.py` (a time view of the rotations and maps of the background) and
`python check_background_interface.py` (the hand-over between the coronal and heliospheric runs at 30 R☉).

## Configuration

All settings shared by the scripts are in **`event_config.py`**; edit it (not the scripts) when you move to another
time or event. The main ones:

| setting | meaning |
|---|---|
| `EVENT_ID` | name of the event; all results and figures go into folders with this name |
| `SC` | spacecraft columns, e.g. `'soho, sta'` |
| `TIME` | `'YYYY-MM-DD HH:MM'` (UT), the time worked on in the GUI and the tracing |
| `SOHO`, `STEREO` | coronagraphs: `'c2'`/`'c3'`, `'cor1'`/`'cor2'` |
| `EUV_ON` | show the EUV row (AIA / EUVI) in the GUI |
| `RD` | `True`: running-difference images (made from downloaded FITS data for LASCO, COR2, EUVI, AIA); `False`: Helioviewer images as they are |
| `CORO_DIFF_MIN`, `EUV_DIFF_MIN` | time between the two images of a running difference [min] |
| `lasco_vmm`, `cor2_vmm`, `euvi_vmm`, `aia_vmm` | display range of the running differences, in units of the image noise (σ) |
| `FIT_CME` | also fit the GCS flux rope in the GUI |
| `GUI_START` | the GUI starts from your `'last'` save, the `'nearest'` fit, or `'defaults'` |
| `BACKGROUND_DIRS` | MAS run folder(s), see above |
| `ARRIVAL_TIME`, `ARRIVAL_SPEED`, `SC_DISTANCE_AU`, `SC_LON`, `SC_LAT` | the observed in-situ shock arrival, used to fit the drag |

## Running the model

For each event, in this order:

### 1. Fit the shock by hand — `gcs_shock_gui.py`
For each time: set `TIME` in `event_config.py`, run

```bash
python gcs_shock_gui.py
```

and adjust the shock (and, with `FIT_CME = True`, the flux rope) over the images. Fitting order: latitude, then
longitude and height, then the shape (κ, ε, α) and tilt. **Save a** saves both and closes, **Save s** the shock only,
**Save m** the flux rope only. Images are downloaded once and kept in `data/<EVENT_ID>/`.

### 2. Trace the shock — `trace_outline.py`
For each time: click along the shock's outer edge in every coronagraph view (left click: add, right click: remove
the last point, Enter: finish). Guide lines show the position angle.

```bash
python trace_outline.py
```

A view that already has a trace asks whether to skip it, add points, or replace it (the old file is kept in
`trace_backup/`).

### 3. Fit all traced times — `fit_event.py`

```bash
python fit_event.py
```

Fits every traced time in chronological order: an ellipsoid anchored on your GUI save (each parameter moves only as
far as the traces justify), then dents and bulges. Useful settings: `FIX` (e.g. `['kappa', 'epsilon', 'alpha']`),
`MAX_DENTS`, `FEATURE_MAX`. Check the fits with

```bash
python check_fits.py      # physical checks: rear point inside the Sun, base ring, speeds, direction jumps, flux rope
python view_fit.py        # the fit over the images and in 3-D
```

### 4. Corona series — `derive_para_corona.py`

```bash
python derive_para_corona.py
```

The shock from 1.5 R☉ to `H_END` (e.g. 30 R☉) in 1 R☉ steps, by several methods: `constant` (linear kinematics,
mean parameters), `varied` (2nd-order kinematics, trends), `observed` (between the fits) and `nearest` (the fitted
front carried self-similarly beyond the fits). Poorly constrained fits are left out automatically.

### 5–7. Interplanetary evolution and shock properties

```bash
python shock_properties_sse.py    # drag fit (drag_fit.py) + SSE fronts (derive_para_ip.py) + corona and SSE properties
python shock_properties_dbm.py    # DBM fronts and properties (uses the drag fit of the previous step)
python shock_properties_gsd.py    # GSD fronts and properties (refits the drag for its own kinematics)
```

Run them in this order: the DBM and GSD versions reuse the drag fitted by the SSE script. Set `METHODS` in each to
the corona methods you want (e.g. `['constant', 'varied', 'nearest']`).

### 8. Cobpoints, shock width and the shock at the spacecraft

```bash
python evolution_cobpoint.py      # field lines from the targets to the shock; properties at the cobpoints
python evolution_front_width.py   # how much of the front is still a shock (M_f, M_A or X above a threshold)
```

Both read the fronts made in steps 5–7 (they do not propagate anything) and end with a table of the shock's arrival
and local properties at each target. Targets are set with `TARGETS`, e.g. `'L1, STA'` (Earth, L1 and planets from
sunpy's ephemeris; spacecraft from JPL Horizons). Use `L1` when the observed arrival comes from Wind, ACE or OMNI.

Rerun steps 5–8 whenever something upstream changes (fits, corona series, background, arrival settings).

## Outputs

```
data/<EVENT_ID>/                    downloaded FITS and Helioviewer images (reused, never downloaded twice)
fitted results/<EVENT_ID>/semi/     GUI saves, traces, fits, fit summary, check_fits.csv
fitted results/<EVENT_ID>/final/    corona and interplanetary series, drag fit, shock properties, cobpoints
figure/<EVENT_ID>/                  all figures:
    shock_evo/                      the corona (shared by the three versions)
    shock_evo_sse/, _dbm/, _gsd/    the interplanetary versions
    cobpoint/, front_width/         cobpoint and shock-width figures
corhel/crNNNN/                      the MAS background (shared by events)
```

## Tests

Each module has a test file that checks it against exact answers (synthetic data; no network needed):

```bash
python test_deformable_shock.py
python test_shock_fit.py
python test_background.py
python test_image_data.py
python test_check_fits.py
python test_derive_para_corona.py
python test_drag_fit.py
python test_shock_properties_sse.py
python test_shock_properties_dbm.py
python test_shock_properties_gsd.py
python test_evolution_cobpoint.py
python test_evolution_front_width.py
```

## Troubleshooting

- **JPL Horizons errors** (needed for SOHO's position): the code falls back to L1, 1.5 million km sunward of Earth
  (within ~0.3° of SOHO), and asks Horizons again on a later run.
- **VSO download timeouts**: searches and downloads are retried with growing waits, and a failing frame is replaced by
  the next-nearest one; if a view still fails, it falls back to the Helioviewer running difference.
- **`Exception ignored in BaseEventLoop.__del__ … signal only works in main thread`** after downloads: harmless
  clean-up message from sunpy's downloader.
- **A method is skipped by the cobpoint script**: the message gives the reason recorded by the propagation script
  (e.g. the spacecraft direction lies outside that method's front).
- **Trace files edited in a spreadsheet**: they are read correctly (trailing commas, Windows line endings).

## Acknowledgements

- [gcs_python](https://github.com/johan12345/gcs_python) for the GCS model and the GUI's image handling
- the ellipsoid shock parameterisation follows PyThea (Kouloumvakos et al.)
- the background solar wind: MAS / CORHEL runs by Predictive Science Inc. (MHDweb)
- data access through SunPy (Fido, VSO, JSOC) and Helioviewer
