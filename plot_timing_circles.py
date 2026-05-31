#!/usr/bin/env python3
"""
Plot timing circles and sky-localization credible regions for one injection.

Edit the two lines at the top, then run:
    python plot_timing_circles.py
"""

# ── configuration ────────────────────────────────────────────────────────────
INJECTION_NUMBER = 4
PLOT_FREQ = 56  # Hz — must exist in BASE_PATH/{PLOT_FREQ}/fits/
GEO = False  # True → 'geo globe' (geographic lon/lat); False → 'astro degrees mollweide' (RA/Dec)
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
from matplotlib import rc_context

# from matplotlib.colors import LinearSegmentedColormap
# from matplotlib.lines import Line2D
from matplotlib.path import Path
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
import ligo.skymap.plot
from ligo.skymap.io import fits as skymap_fits
from ligo.skymap import moc as lsm_moc
from ligo.skymap import postprocess as lsm_postprocess
from ligo.skymap.plot import outline_text
from ligo.skymap.plot.backdrop import coastlines

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interferometer import interferometer


# ── detector setup ────────────────────────────────────────────────────────────
IFO_NAMES = ["H1", "L1", "V1", "K1", "I1"]
detectors = {ifo: interferometer(ifo) for ifo in IFO_NAMES}

_LAL_IFO = {
    "H1": lal.LALDetectorIndexLHODIFF,
    "L1": lal.LALDetectorIndexLLODIFF,
    "V1": lal.LALDetectorIndexVIRGODIFF,
    "K1": lal.LALDetectorIndexKAGRADIFF,
    "I1": lal.LALDetectorIndexLIODIFF,
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
        detectors[det2].vertex, ra=ra, dec=dec, t_event=t_event
    )
    possible_ras, possible_decs = detectors[det1].sky_location(
        detectors[det2].vertex, time_delay=time_delay, t_event=t_event
    )
    possible_ras = possible_ras % (2 * np.pi)

    F1, F2 = [], []
    for r, d in zip(possible_ras, possible_decs):

        def _antenna(det, r=r, d=d):
            ep = detectors[det].get_polarization_tensor(r, d, t_event, 0, "plus")
            ec = detectors[det].get_polarization_tensor(r, d, t_event, 0, "cross")
            Fp = np.einsum("ij,ij", detectors[det].detector_tensor, ep)
            Fc = np.einsum("ij,ij", detectors[det].detector_tensor, ec)
            return np.sqrt(Fp ** 2 + Fc ** 2)

        F1.append(_antenna(det1))
        F2.append(_antenna(det2))
    return possible_ras, possible_decs, np.array(F1), np.array(F2)


def _format_area(area):
    if area <= 100:
        return np.format_float_positional(area, precision=3, fractional=False, trim="-")
    else:
        return f"{np.round(area).astype(int):,d}"


def plot_continents_icrs(ax, gmst):
    segs = coastlines()
    for lon_arr, lat_arr in zip(segs[::2], segs[1::2]):
        ra = np.rad2deg((np.deg2rad(np.array(lon_arr, float)) + gmst) % (2 * np.pi))
        dec = np.array(lat_arr, float)
        jumps = np.where(np.abs(np.diff(ra)) > 180)[0] + 1
        ra = np.insert(ra, jumps, np.nan)
        dec = np.insert(dec, jumps, np.nan)
        ax.plot(
            ra, dec, color="0.5", linewidth=0.5, transform=ax.get_transform("world")
        )


def _rect(p0, p1, hw):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    d /= np.hypot(*d)
    n = np.array([-d[1], d[0]]) * hw
    c = [p0 + n, p1 + n, p1 - n, p0 - n]
    return Path(
        np.array(c + [c[0]]),
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
    )


def _merge(paths):
    return Path(
        np.vstack([p.vertices for p in paths]), np.concatenate([p.codes for p in paths])
    )


def _pad(p, g):
    return Path(
        np.vstack([p.vertices, [[g, g], [-g, -g]]]),
        np.concatenate([p.codes, [Path.MOVETO, Path.MOVETO]]),
    )


_ARM, _IN, _OUT, _HW, _OW, _M, _R = 1.0, 0.60, 0.55, 0.045, 0.045, 0.24, 0.16
_arc = Path.arc(180, 360)
_pd = Path(
    np.vstack([_arc.vertices * _R + [0.0, -_OUT], [0.0, -_OUT]]),
    list(_arc.codes) + [Path.CLOSEPOLY],
)
IFO_BEAMS = _merge(
    [_rect((-_IN, 0), (_ARM, 0), _HW), _rect((0, -_OUT), (0, _ARM), _HW)]
)
IFO_OPTICS = _merge(
    [
        _rect((-0.20, -0.20), (0.20, 0.20), _OW),
        _rect((_ARM, -_M), (_ARM, _M), _OW),
        _rect((-_M, _ARM), (_M, _ARM), _OW),
        _pd,
    ]
)
_G = max(np.max(np.abs(p.vertices)) for p in (IFO_BEAMS, IFO_OPTICS))
IFO_BEAMS, IFO_OPTICS = _pad(IFO_BEAMS, _G), _pad(IFO_OPTICS, _G)


def plot_ifo(ax, lon, lat, size=46, beam_color="red", optic_color="k", **kw):
    common = dict(markersize=size, linestyle="none", markeredgewidth=0, **kw)
    ax.plot(lon, lat, marker=IFO_BEAMS, markerfacecolor=beam_color, **common)
    ax.plot(lon, lat, marker=IFO_OPTICS, markerfacecolor=optic_color, **common)


# ── load injection parameters ─────────────────────────────────────────────────
stats = pd.read_csv(
    os.path.join(BASE_PATH, "stats", "combined_stats.dat"),
    sep="\t",
    index_col=0,
)
row = stats[stats["simulation_id"] == INJECTION_NUMBER].iloc[0]
true_ra, true_dec = ast.literal_eval(row["ra_dec"])  # radians
true_obstime = float(row["time"])  # GPS seconds
true_gmst = lal.GreenwichMeanSiderealTime(true_obstime) % (2 * np.pi)

print(
    f"Injection {INJECTION_NUMBER}: "
    f"RA={np.rad2deg(true_ra):.2f}°  Dec={np.rad2deg(true_dec):.2f}°  "
    f"GPS={true_obstime:.0f}"
)


# ── load available skymaps per frequency ──────────────────────────────────────
freq_dirs = sorted([d for d in os.listdir(BASE_PATH) if d.isdigit()], key=int)
skymaps = {}
for freq_str in freq_dirs:
    fits_path = os.path.join(
        BASE_PATH, freq_str, "fits", f"sim_id_{INJECTION_NUMBER}.fits"
    )
    if os.path.exists(fits_path):
        skymaps[int(freq_str)] = skymap_fits.read_sky_map(fits_path, moc=True)

if not skymaps:
    raise FileNotFoundError(
        f"No FITS files found for injection {INJECTION_NUMBER} in {BASE_PATH}"
    )
print(f"Skymaps found at frequencies (Hz): {sorted(skymaps)}")


# ── precompute timing circles ─────────────────────────────────────────────────
RING_PAIRS = [
    ("L1", "H1"),  # LIGO only
    ("L1", "V1"), ("H1", "V1"), # Add VIRGO
    # ("L1", "K1"), ("H1", "K1"), ("V1", "K1"),  # Add KAGRA
]
rings = {
    (d1, d2): get_ring_w_coloring(d1, d2, true_ra, true_dec, true_obstime)
    for d1, d2 in RING_PAIRS
}

# ── plot ──────────────────────────────────────────────────────────────────────
PROJECTION = "geo globe" if GEO else "astro degrees mollweide"
TEXTSIZE = 16
freq_colors = cm.plasma(np.linspace(0.1, 0.9, len(skymaps)))

# plt.style.use('../../../plot_stylesheet.sty')


with rc_context({"xtick.labelsize": 14, "ytick.labelsize": 14, "lines.linewidth": 3}):
    fig = plt.figure(figsize=(9, 9) if GEO else (14, 7))
    if GEO:
        # geo globe needs obstime so its WCS can convert ICRS ↔ ITRS
        ax = fig.add_subplot(
            projection=PROJECTION,
            obstime=Time(true_obstime, format="gps"),
            # center takes lon, lat in degrees
            # center="-90d +23d",
            center=f"{np.rad2deg((true_ra - true_gmst) % (2 * np.pi))}d +23d",
            # Not replacing 23 degrees because I kinda like view on Northern hemisphere
        )
        # Geo.__init__ already calls invert_xaxis(); calling it again would un-flip
    else:
        ax = fig.add_subplot(projection=PROJECTION)
        ax.invert_xaxis()

    PLT_ARGS = dict(transform=ax.get_transform("world"))
    # For astro: 'world' == ICRS (RA, Dec).  For geo: 'world' == ITRS (lon, lat).

    if not GEO:
        ax.set_aspect("equal")
    ax.set_facecolor(plt.get_cmap("cylon")(0.0))
    ax.grid()

    # continents
    if GEO:
        # coastlines() returns raw geographic (lon, lat) — pass directly to 'world' (ITRS)
        ax.plot(*coastlines(), color="0.5", linewidth=0.5, **PLT_ARGS)
    else:
        plot_continents_icrs(ax, true_gmst)

    # skymap credible regions (50 % and 90 %) for PLOT_FREQ only
    skymap = skymaps[PLOT_FREQ]
    dA = lsm_moc.uniq2pixarea(skymap["UNIQ"])
    dP = skymap["PROBDENSITY"] * dA
    cls = 100 * lsm_postprocess.find_greedy_credible_levels(dP, skymap["PROBDENSITY"])
    CONTOUR_LEVELS = [50, 90]
    cs = ax.contour_hpx(
        (Table({"UNIQ": skymap["UNIQ"], "CLS": cls}), "ICRS"),
        colors="k",
        linewidths=0.5,
        levels=CONTOUR_LEVELS,
        order="nearest-neighbor",
    )
    fmt = r"%g\%%"
    plt.clabel(cs, fmt=fmt, fontsize=6, inline=True)

    # Annotate credible-region areas (mirrors ligo-skymap-plot --annotate).
    sr_to_deg2 = u.sr.to(u.deg ** 2)
    _sort_idx = np.flipud(np.argsort(skymap["PROBDENSITY"]))
    _areas = lsm_postprocess.interp_greedy_credible_levels(
        CONTOUR_LEVELS, cls[_sort_idx], np.cumsum(dA[_sort_idx]), right=4 * np.pi
    )
    _ann_lines = [
        f"{int(np.round(p))}% area: {_format_area(a * sr_to_deg2)} deg²"
        for p, a in zip(CONTOUR_LEVELS, _areas)
    ]
    ax.text(
        0.99,
        0.99,
        "\n".join(_ann_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )

    # Convert PROBDENSITY from per-sr to per-deg² so the dynamic range is
    # finite and imshow scales the colormap correctly (matches ligo_skymap_plot).
    skymap["PROBDENSITY"] = skymap["PROBDENSITY"] / sr_to_deg2

    img = ax.imshow_hpx(
        (skymap, "ICRS"), vmin=0, cmap="cylon", order="nearest-neighbor"
    )

    # Build a cylon variant whose alpha ramps from 0 (transparent at vmin) to
    # 1 (opaque at vmax), so zero-probability pixels are invisible and the
    # timing rings remain visible beneath the map at any zorder.
    # _rgba = plt.get_cmap("cylon")(np.linspace(0, 1, 256))
    # _rgba[:, 3] = np.linspace(0, 1, 256)
    # _cylon_alpha = LinearSegmentedColormap.from_list("cylon_alpha", _rgba)

    # img = ax.imshow_hpx(
    #     (skymap, "ICRS"), vmin=0, cmap=_cylon_alpha,
    #     order="nearest-neighbor", zorder=200
    # )

    # timing circles — one color per ring pair, inline label along the curve
    # _ring_colors = plt.get_cmap('tab10')(np.linspace(0, 0.3, len(RING_PAIRS)))
    _ring_colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(RING_PAIRS)))
    _label_fracs = 4 * [0.85, 0.50, 0.75]  # fractional position along ring for each label
    for (d1, d2), color, frac in zip(RING_PAIRS, _ring_colors, _label_fracs):
        ras, decs, _, _ = rings[(d1, d2)]
        label = f"{d1}-{d2}"
        # Convert ICRS (RA/Dec radians) → native projection coordinates (degrees)
        if GEO:
            lons = np.rad2deg((ras - true_gmst) % (2 * np.pi))
        else:
            lons = np.rad2deg(ras)
        lats = np.rad2deg(decs)
        ax.scatter(lons, lats, s=1, color=color, zorder=10, **PLT_ARGS)
        # Inline label: rotate text to match the local tangent direction
        idx = int(frac * len(ras)) % len(ras)
        step = max(3, len(ras) // 60)
        i0, i1 = (idx - step) % len(ras), (idx + step) % len(ras)
        dlon = lons[i1] - lons[i0]
        dlat = lats[i1] - lats[i0]
        # wrap to (-180, 180] to handle 0/360 boundary
        dlon = (dlon + 180) % 360 - 180
        # both projections invert the x-axis, so the angle formula is the same
        angle = np.rad2deg(np.arctan2(dlat, dlon))
        # Both projections invert the x-axis, so increasing lon moves LEFT on screen.
        # Negate dlon so the angle is in screen space: arctan2(screen_dy, screen_dx).
        # angle = np.rad2deg(np.arctan2(dlat, -dlon))
        # angle = np.rad2deg(np.arctan2(dlat, dlon+90))
        ax.text(
            lons[idx],
            lats[idx],
            f" {label} ",
            transform=ax.get_transform("world"),
            fontsize=9,
            ha="center",
            va="center",
            rotation=angle,  # TODO: decide whether to comment or not -> leave, is fine for most cases
            rotation_mode="anchor",
            color=color,
            fontweight="bold",
            # bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none', alpha=0.85),
            zorder=20,
        )

    # true source location
    _src_lon = (
        np.rad2deg((true_ra - true_gmst) % (2 * np.pi)) if GEO else np.rad2deg(true_ra)
    )
    ax.plot(
        _src_lon,
        np.rad2deg(true_dec),
        "*",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=12,
        zorder=100,
        **PLT_ARGS,
    )

    # detector locations
    LABEL_ARGS = dict(ha="right", va="bottom", fontsize=8, **PLT_ARGS)
    for name in IFO_NAMES:
        geo_lon, geo_lat = DETECTOR_POSITION[name]  # always geographic
        if GEO:
            plot_lon, plot_lat = geo_lon, geo_lat
        else:
            plot_lon = np.rad2deg((np.deg2rad(geo_lon) + true_gmst) % (2 * np.pi))
            plot_lat = geo_lat
        plot_ifo(ax, plot_lon, plot_lat, size=24, **PLT_ARGS)
        ax.text(plot_lon, plot_lat, name, **LABEL_ARGS)

    outline_text(ax)

    ax.set_title(
        f"Injection {INJECTION_NUMBER}  |  "
        f"RA = {np.rad2deg(true_ra):.1f}°   Dec = {np.rad2deg(true_dec):.1f}°   "
        f"GPS = {true_obstime:.0f}",
        fontsize=12,
    )

    plt.tight_layout()
    det_label = "".join([n[0] for n in sorted(set(np.array(RING_PAIRS).flatten()))]).lower()
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"timing_circle_inj{INJECTION_NUMBER}_f{PLOT_FREQ}_{det_label}" + ("_geo" if GEO else "") + ".png",
    )
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved → {out_path}")
    plt.show()
