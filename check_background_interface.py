"""
check_background_interface.py -- does the background jump where the MAS coronal run hands over to the heliospheric
one (30 R_sun), and why? Both runs are sampled on that sphere (within LAT_BAND of the equator, where the shock and
the spacecraft are) and compared:

  level ratio     helio / corona for n r^2, v_r and |B_r| r^2 (median over the band): != 1 means the runs differ in
                  level at the interface -- the simulation itself (the heliospheric run is a separate, polytropic run
                  whose inner boundary is derived from the coronal solution)
  longitude shift the shift of the heliospheric run that best lines up its structures with the coronal ones
                  (correlation of log n over the band): != 0 means the two runs' longitudes differ -- a frame or
                  convention mismatch, which background.py would have to correct

Run:  python check_background_interface.py      (the background of event_config.py)
Writes figure/<id>/background_interface.png: the two runs on the 30 R_sun sphere and their ratio.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import event_config as cfg
import background as bk

# ================================ SETTINGS ================================
LAT_BAND = 40.0                      # compare within this many degrees of the equator
D_DEG = 1.0                          # longitude / latitude step [deg]
# ==========================================================================


def shell(part, r, theta, phi):
    """n r^2 [cm^-3 at 1 AU], v_r [km/s] and |B_r| r^2 [nT at 1 AU] of one part on the sphere r."""
    T, P = np.meshgrid(theta, phi, indexing='ij')
    rr = np.full(T.size, r)
    g = lambda v: part.interp(v, rr, T.ravel(), P.ravel()).reshape(T.shape)
    s = (r / bk.AU_RSUN) ** 2
    return {'n r^2': g('rho') * s, 'v_r': g('vr'), '|B_r| r^2': np.abs(g('br')) * s}


def best_shift(a, b):
    """The longitude shift [deg, in steps of D_DEG] of b that best correlates log b with log a, and the correlation."""
    la, lb = np.log10(a), np.log10(b)
    la, lb = la - np.nanmean(la), lb - np.nanmean(lb)
    ok = np.isfinite(la) & np.isfinite(lb)
    la, lb = np.where(ok, la, 0.0), np.where(ok, lb, 0.0)
    corr = [float(np.sum(la * np.roll(lb, k, axis=1))) for k in range(la.shape[1])]
    norm = np.sqrt(np.sum(la ** 2) * np.sum(lb ** 2))
    k = int(np.argmax(corr))
    shift = k * D_DEG if k * D_DEG <= 180 else k * D_DEG - 360
    return shift, corr[k] / norm, corr[0] / norm


def compare(run, r_int):
    theta = np.radians(np.arange(90.0 - LAT_BAND, 90.0 + LAT_BAND + 1e-9, D_DEG))
    phi = np.radians(np.arange(0.0, 360.0, D_DEG))
    corona, helio = run.parts[0], run.parts[1]
    c = shell(corona, min(r_int, corona.r_range[1]), theta, phi)
    h = shell(helio, max(r_int, helio.r_range[0]), theta, phi)
    shift, c_best, c_zero = best_shift(c['n r^2'], h['n r^2'])
    k = int(round(shift / D_DEG))
    ratios = {q: (float(np.nanmedian(h[q] / c[q])), float(np.nanmedian(np.roll(h[q], k, axis=1) / c[q]))) for q in c}
    return theta, phi, c, h, shift, c_best, c_zero, ratios


def figure(theta, phi, c, h, shift, path):
    lat, lon = 90.0 - np.degrees(theta), np.degrees(phi)
    fig, ax = plt.subplots(3, 3, figsize=(16, 9), layout='constrained')
    for i, q in enumerate(c):
        a, b = c[q], h[q]
        lo, hi = np.nanpercentile(np.r_[a.ravel(), b.ravel()], [2, 98])
        for j, (data, title) in enumerate(((a, 'corona (r \u2264 30)'), (b, 'heliosphere (r \u2265 30)'))):
            im = ax[i, j].pcolormesh(lon, lat, data, shading='auto', vmin=lo, vmax=hi, cmap='viridis')
            ax[i, j].set_title(f'{q}: {title}', fontsize=10)
            fig.colorbar(im, ax=ax[i, j], shrink=0.8)
        im = ax[i, 2].pcolormesh(lon, lat, b / a, shading='auto', cmap='RdBu_r', vmin=0.5, vmax=1.5)
        ax[i, 2].set_title(f'{q}: heliosphere / corona', fontsize=10)
        fig.colorbar(im, ax=ax[i, 2], shrink=0.8)
    for a in ax.flat:
        a.set_xlabel('Carrington longitude [deg]')
        a.set_ylabel('latitude [deg]')
    fig.suptitle(f'the two MAS runs on the 30 R$_\\odot$ sphere (best-aligning shift of the heliospheric run: '
                 f'{shift:+.0f} deg)')
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    bg = bk.load_background(verbose=False)
    runs = bg.components if hasattr(bg, 'components') else [bg]
    for run in runs:
        if len(getattr(run, 'parts', [])) < 2:
            print(f'{run.folder}: one part only, no interface')
            continue
        r_int = run.parts[1].r_range[0]
        theta, phi, c, h, shift, c_best, c_zero, ratios = compare(run, r_int)
        print(f'\n{run.folder}: coronal run up to {run.parts[0].r_range[1]:.2f} R_sun, heliospheric from '
              f'{r_int:.2f} R_sun; compared within {LAT_BAND:g} deg of the equator')
        print(f'  longitude shift that best aligns the heliospheric structures: {shift:+.0f} deg '
              f'(correlation {c_best:.2f}; unshifted {c_zero:.2f})')
        for q, (r0, r1) in ratios.items():
            print(f'  {q:10s} heliosphere / corona: {r0:.2f} as they are' + (f', {r1:.2f} after the shift' if shift else ''))
        if abs(shift) <= 2 * D_DEG:
            verdict = ('the structures line up: our longitudes agree. Any jump is the simulation itself -- the two runs '
                       'differ in level at the interface' if any(abs(r0 - 1) > 0.1 for r0, _ in ratios.values())
                       else 'the two runs agree at the interface: no jump from the background')
        else:
            verdict = (f'the structures are rotated by {shift:+.0f} deg between the runs: a longitude mismatch, to be '
                       f'corrected in background.py')
        print(f'  -> {verdict}')
        path = cfg.figure_path('background_interface.png')
        figure(theta, phi, c, h, shift, path)
        print(f'  saved {path}')


if __name__ == '__main__':
    main()
