#!/usr/bin/env python3
"""
Plot timing circles and sky-localization credible regions for one injection.

Edit the two lines at the top, then run:
    python plot_timing_circles.py
"""

# ── configuration ────────────────────────────────────────────────────────────
INJECTION_NUMBER = 102
BASE_PATH = (
    "/Users/maxmelching/Documents/PhD/research/dsa-2000"
    "/early_warning_dsa/sky_localization/results"
    "/H1_O5_L1_O5_V1_O5_many"
)
# ─────────────────────────────────────────────────────────────────────────────

import ast
import os
import sys

import lal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import healpy as hp
from matplotlib import rc_context
from matplotlib.lines import Line2D
from matplotlib.path import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
import ligo.skymap.plot
from ligo.skymap.io import read_sky_map
from ligo.skymap.plot import outline_text
from ligo.skymap.plot.backdrop import coastlines
from ligo.skymap.postprocess import find_greedy_credible_levels

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interferometer import interferometer


# ── detector setup ────────────────────────────────────────────────────────────
IFO_NAMES = ['H1', 'L1', 'V1']
detectors = {ifo: interferometer(ifo) for ifo in IFO_NAMES}

_LAL_IFO = {
    'H1': lal.LALDetectorIndexLHODIFF,
    'L1': lal.LALDetectorIndexLLODIFF,
    'V1': lal.LALDetectorIndexVIRGODIFF,
}
DETECTOR_POSITION = {}
for _name in IFO_NAMES:
    _det = lal.CachedDetectors[_LAL_IFO[_name]]
    DETECTOR_POSITION[_name] = (
        np.rad2deg(_det.frDetector.vertexLongitudeRadians),
        np.rad2deg(_det.frDetector.vertexLatitudeRadians),
    )


# ── helper functions ──────────────────────────────────────────────────────────
def get_ring_w_coloring(det1, det2, ra, dec, t_event):
    time_delay = detectors[det1].time_delay(
        detectors[det2].vertex, ra=ra, dec=dec, t_event=t_event)
    possible_ras, possible_decs = detectors[det1].sky_location(
        detectors[det2].vertex, time_delay=time_delay, t_event=t_event)
    possible_ras = possible_ras % (2 * np.pi)

    F1, F2 = [], []
    for r, d in zip(possible_ras, possible_decs):
        def _antenna(det, r=r, d=d):
            ep = detectors[det].get_polarization_tensor(r, d, t_event, 0, 'plus')
            ec = detectors[det].get_polarization_tensor(r, d, t_event, 0, 'cross')
            Fp = np.einsum('ij,ij', detectors[det].detector_tensor, ep)
            Fc = np.einsum('ij,ij', detectors[det].detector_tensor, ec)
            return np.sqrt(Fp**2 + Fc**2)
        F1.append(_antenna(det1))
        F2.append(_antenna(det2))
    return possible_ras, possible_decs, np.array(F1), np.array(F2)


def plot_continents_icrs(ax, gmst):
    segs = coastlines()
    for lon_arr, lat_arr in zip(segs[::2], segs[1::2]):
        ra  = np.rad2deg((np.deg2rad(np.array(lon_arr, float)) + gmst) % (2 * np.pi))
        dec = np.array(lat_arr, float)
        jumps = np.where(np.abs(np.diff(ra)) > 180)[0] + 1
        ra  = np.insert(ra,  jumps, np.nan)
        dec = np.insert(dec, jumps, np.nan)
        ax.plot(ra, dec, color='0.5', linewidth=0.5,
                transform=ax.get_transform('world'))


def _rect(p0, p1, hw):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0; d /= np.hypot(*d)
    n = np.array([-d[1], d[0]]) * hw
    c = [p0+n, p1+n, p1-n, p0-n]
    return Path(np.array(c + [c[0]]),
                [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])

def _merge(paths):
    return Path(np.vstack([p.vertices for p in paths]),
                np.concatenate([p.codes for p in paths]))

def _pad(p, g):
    return Path(np.vstack([p.vertices, [[g, g], [-g, -g]]]),
                np.concatenate([p.codes, [Path.MOVETO, Path.MOVETO]]))

_ARM, _IN, _OUT, _HW, _OW, _M, _R = 1.0, 0.60, 0.55, 0.045, 0.045, 0.24, 0.16
_arc = Path.arc(180, 360)
_pd  = Path(np.vstack([_arc.vertices * _R + [0.0, -_OUT], [0.0, -_OUT]]),
            list(_arc.codes) + [Path.CLOSEPOLY])
IFO_BEAMS   = _merge([_rect((-_IN, 0), (_ARM, 0), _HW),
                      _rect((0, -_OUT), (0, _ARM), _HW)])
IFO_OPTICS  = _merge([_rect((-0.20, -0.20), (0.20, 0.20), _OW),
                      _rect((_ARM, -_M), (_ARM, _M), _OW),
                      _rect((-_M, _ARM), (_M, _ARM), _OW), _pd])
_G = max(np.max(np.abs(p.vertices)) for p in (IFO_BEAMS, IFO_OPTICS))
IFO_BEAMS, IFO_OPTICS = _pad(IFO_BEAMS, _G), _pad(IFO_OPTICS, _G)

def plot_ifo(ax, lon, lat, size=46, beam_color='red', optic_color='k', **kw):
    common = dict(markersize=size, linestyle='none', markeredgewidth=0, **kw)
    ax.plot(lon, lat, marker=IFO_BEAMS,  markerfacecolor=beam_color,  **common)
    ax.plot(lon, lat, marker=IFO_OPTICS, markerfacecolor=optic_color, **common)


# ── load injection parameters ─────────────────────────────────────────────────
stats = pd.read_csv(
    os.path.join(BASE_PATH, 'stats', 'combined_stats.dat'),
    sep='\t', index_col=0,
)
row = stats[stats['simulation_id'] == INJECTION_NUMBER].iloc[0]
true_ra, true_dec = ast.literal_eval(row['ra_dec'])   # radians
true_obstime      = float(row['time'])                 # GPS seconds
true_gmst         = lal.GreenwichMeanSiderealTime(true_obstime) % (2 * np.pi)

print(f"Injection {INJECTION_NUMBER}: "
      f"RA={np.rad2deg(true_ra):.2f}°  Dec={np.rad2deg(true_dec):.2f}°  "
      f"GPS={true_obstime:.0f}")


# ── load available skymaps per frequency ──────────────────────────────────────
freq_dirs = sorted(
    [d for d in os.listdir(BASE_PATH) if d.isdigit()], key=int)
skymaps = {}
for freq_str in freq_dirs:
    fits_path = os.path.join(
        BASE_PATH, freq_str, 'fits', f'sim_id_{INJECTION_NUMBER}.fits')
    if os.path.exists(fits_path):
        skymap, _ = read_sky_map(fits_path, nest=False)
        skymaps[int(freq_str)] = skymap

if not skymaps:
    raise FileNotFoundError(
        f"No FITS files found for injection {INJECTION_NUMBER} in {BASE_PATH}")
print(f"Skymaps found at frequencies (Hz): {sorted(skymaps)}")


# ── precompute timing circles ─────────────────────────────────────────────────
RING_PAIRS = [('H1', 'V1'), ('H1', 'L1'), ('L1', 'V1')]
rings = {
    (d1, d2): get_ring_w_coloring(d1, d2, true_ra, true_dec, true_obstime)
    for d1, d2 in RING_PAIRS
}

# ── RA/Dec grid for reprojecting HEALPix skymaps ─────────────────────────────
_ra_g  = np.linspace(0, 360, 720)
_dec_g = np.linspace(-90, 90, 360)
_RA, _DEC = np.meshgrid(_ra_g, _dec_g)
_theta_g, _phi_g = np.deg2rad(90 - _DEC), np.deg2rad(_RA)


# ── plot ──────────────────────────────────────────────────────────────────────
PROJECTION = 'astro degrees mollweide'
TEXTSIZE   = 16
freq_colors = cm.plasma(np.linspace(0.1, 0.9, len(skymaps)))

with rc_context({'xtick.labelsize': 14, 'ytick.labelsize': 14, 'lines.linewidth': 3}):
    fig = plt.figure(figsize=(14, 7))
    ax  = fig.add_subplot(projection=PROJECTION)
    ax.invert_xaxis()

    PLT_ARGS = dict(transform=ax.get_transform('world'))

    ax.set_aspect('equal')
    ax.set_facecolor(plt.get_cmap('cylon')(0.0))
    ax.grid()

    # continents
    plot_continents_icrs(ax, true_gmst)

    # skymap credible regions (50 % and 90 %)
    for (freq, skymap), color in zip(sorted(skymaps.items()), freq_colors):
        nside   = hp.npix2nside(len(skymap))
        pix     = hp.ang2pix(nside, _theta_g, _phi_g, nest=False)
        cls     = find_greedy_credible_levels(skymap)
        cls_grid = cls[pix]
        ax.contour(
            _RA, _DEC, cls_grid,
            levels=[0.5, 0.9],
            colors=[color],
            linewidths=1.5,
            **PLT_ARGS,
        )

    # timing circles
    ring_color = plt.get_cmap('viridis')(0.0)
    for (d1, d2), (ras, decs, _, _) in rings.items():
        label = f'{d1}-{d2}'
        ax.scatter(np.rad2deg(ras), np.rad2deg(decs),
                   s=1, color=ring_color, zorder=10, **PLT_ARGS)
        idx  = 60  if label == 'L1-V1' else 420
        ha   = 'right' if label == 'L1-V1' else 'left'
        va   = 'top'   if label == 'L1-V1' else 'bottom'
        ax.text(np.rad2deg(ras)[idx], np.rad2deg(decs)[idx], label,
                transform=ax.get_transform('world'),
                fontsize=TEXTSIZE, horizontalalignment=ha, verticalalignment=va)

    # true source location
    ax.plot_coord(
        SkyCoord(np.rad2deg(true_ra), np.rad2deg(true_dec), unit=u.deg),
        '*', markerfacecolor='white', markeredgecolor='black',
        markersize=12, zorder=100,
    )

    # detector locations
    LABEL_ARGS = dict(ha='right', va='bottom', fontsize=8, **PLT_ARGS)
    for name in IFO_NAMES:
        lon_deg, lat_deg = DETECTOR_POSITION[name]
        ra_det  = np.rad2deg((np.deg2rad(lon_deg) + true_gmst) % (2 * np.pi))
        dec_det = lat_deg
        plot_ifo(ax, ra_det, dec_det, size=24, **PLT_ARGS)
        ax.text(ra_det, dec_det, name, **LABEL_ARGS)

    outline_text(ax)

    ax.set_title(
        f"Injection {INJECTION_NUMBER}  |  "
        f"RA = {np.rad2deg(true_ra):.1f}°   Dec = {np.rad2deg(true_dec):.1f}°   "
        f"GPS = {true_obstime:.0f}",
        fontsize=12,
    )

    legend_elements = [
        Line2D([0], [0], color=color, lw=2, label=f'{freq} Hz')
        for (freq, _), color in zip(sorted(skymaps.items()), freq_colors)
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
              framealpha=0.7)

    plt.tight_layout()
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'timing_circle_inj{INJECTION_NUMBER}.png',
    )
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {out_path}")
    plt.show()
