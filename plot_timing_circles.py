#!/usr/bin/env python3
# Copyright (C) 2026 Max Melching
"""
Plot timing circles and sky-localization credible regions for one injection.

Basic usage (injection number with explicit files):
    python plot_timing_circles.py -n 4 \\
        --stats-file /path/to/stats/combined_stats.dat \\
        --skymap-file /path/to/56/fits/sim_id_4.fits

All detectors from a network, with auto-computed timing annuli:
    python plot_timing_circles.py -n 4 \\
        --stats-file /path/to/stats/combined_stats.dat \\
        --skymap-file /path/to/56/fits/sim_id_4.fits \\
        --detectors H1 L1 V1 \\
        --timing-uncertainty \\
        --n-annulus 80

Override timing uncertainty with an explicit sigma:
    python plot_timing_circles.py -n 4 \\
        --stats-file /path/to/stats/combined_stats.dat \\
        --skymap-file /path/to/56/fits/sim_id_4.fits \\
        --ring-pairs L1-H1 L1-V1 H1-V1 \\
        --timing-uncertainty --timing-sigma-ms 0.42 \\
        --n-annulus 50

Provide sky position directly (no injection lookup):
    python plot_timing_circles.py \\
        --sky-pos 45.0 -30.0 1234567890 \\
        --ring-pairs L1-H1 L1-V1 \\
        --timing-uncertainty --timing-sigma-ms 0.5

Sky position with a skymap overlay:
    python plot_timing_circles.py \\
        --sky-pos 45.0 -30.0 1234567890 \\
        --skymap-file /path/to/skymap.fits \\
        --ring-pairs L1-H1 L1-V1 \\
        --timing-uncertainty --timing-sigma-ms 0.5

Full example with geo projection and custom output directory:
    python plot_timing_circles.py \\
        --stats-file /path/to/stats/combined_stats.dat \\
        --skymap-file /path/to/56/fits/sim_id_4.fits \\
        --injection-number 4 \\
        --plot-freq 56 \\
        --ring-pairs L1-H1 L1-V1 H1-V1 \\
        --timing-uncertainty \\
        --contour-levels 50 90 \\
        --geo \\
        --outdir /tmp/plots

Bilby result (posterior KDE skymap, auto-computed timing uncertainty):
    python plot_timing_circles.py \\
        --bilby-json /path/to/result.json \\
        --detectors H1 V1 \\
        --timing-uncertainty

Bilby result with custom smoothing and explicit timing sigma:
    python plot_timing_circles.py \\
        --bilby-json /path/to/result.json \\
        --detectors H1 V1 \\
        --timing-uncertainty --timing-sigma-ms 0.5 \\
        --posterior-smooth-deg 2.0 \\
        --contour-levels 50 90
"""

import argparse
import ast
import itertools
import os
import sys

import healpy as hp
import lal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc_context
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
import ligo.skymap.plot  # Needed to register projections etc
from ligo.skymap.io import fits as skymap_fits
from ligo.skymap import moc as lsm_moc
from ligo.skymap import postprocess as lsm_postprocess
from ligo.skymap.plot import outline_text
from ligo.skymap.plot.backdrop import coastlines

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interferometer import interferometer


# ── argument parsing ──────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src_group = p.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--injection-number",
        "-n",
        type=int,
        default=None,
        dest="injection_number",
        metavar="ID",
        help="Simulation ID; loads true sky position and GPS time from the stats file.",
    )
    src_group.add_argument(
        "--sky-pos",
        nargs=3,
        type=float,
        default=None,
        metavar=("RA_DEG", "DEC_DEG", "GPS"),
        help="True sky position and GPS time directly: RA [deg], Dec [deg], GPS [s]. "
        "Skips stats-file lookup. Pass --skymap-file to overlay a skymap.",
    )
    src_group.add_argument(
        "--bilby-json",
        default=None,
        metavar="PATH",
        help="Path to a bilby result JSON file. Reads injection parameters (true sky "
        "position, GPS time, per-detector SNRs) and posterior samples (ra, dec) "
        "directly. Replaces --injection-number + --stats-file.",
    )
    p.add_argument(
        "--stats-file",
        default=None,
        metavar="PATH",
        help="Path to the stats CSV. The injection number selects the correct row.",
    )
    p.add_argument(
        "--skymap-file",
        default=None,
        metavar="PATH",
        help="Path to a FITS skymap to overlay.",
    )
    p.add_argument(
        "--plot-freq",
        type=int,
        default=None,
        metavar="HZ",
        help="Frequency [Hz] label for the output filename (optional).",
    )
    pair_group = p.add_mutually_exclusive_group()
    pair_group.add_argument(
        "--ring-pairs",
        nargs="+",
        default=None,
        metavar="D1-D2",
        help="Explicit detector pairs for timing circles, e.g. L1-H1 L1-V1 H1-V1 H1-K1. "
        "Mutually exclusive with --detectors.",
    )
    pair_group.add_argument(
        "--detectors",
        nargs="+",
        default=None,
        metavar="DET",
        help="List of detectors; all pairwise combinations are used as ring pairs, "
        "e.g. --detectors H1 L1 V1 K1 produces six pairs. "
        "Mutually exclusive with --ring-pairs.",
    )
    geo_group = p.add_mutually_exclusive_group()
    geo_group.add_argument(
        "--geo",
        action="store_true",
        help="Use 'geo globe' projection instead of 'astro degrees mollweide'",
    )
    geo_group.add_argument(
        "--globe",
        action="store_true",
        help="Like --geo but omits continents, uses an opaque globe surface, "
        "and shows only grid lines (no coastlines).",
    )
    p.add_argument(
        "--geo-center",
        default="auto",
        metavar="'LONd LATd'",
        help="Center for --geo / --globe: 'auto' centres on the source longitude; "
        "or pass any string accepted by SkyCoord, e.g. '-90d +23d'",
    )
    p.add_argument(
        "--timing-uncertainty",
        action="store_true",
        help="Draw annuli around each timing circle by sampling τ ~ N(τ_true, σ²). "
        "σ is taken from --timing-sigma-ms if provided, otherwise auto-computed "
        "per pair from per-detector SNR and hardcoded effective bandwidth.",
    )
    p.add_argument(
        "--timing-sigma-ms",
        type=float,
        default=None,
        metavar="MS",
        help="Override timing uncertainty σ [ms] for all pairs (requires --timing-uncertainty). "
        "If omitted, σ is auto-computed from SNR and TIMING_BANDWIDTH_HZ.",
    )
    p.add_argument(
        "--n-annulus",
        type=int,
        default=50,
        metavar="N",
        help="Number of sampled rings per pair when --timing-uncertainty is set",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for annulus sampling (ensures reproducible figures)",
    )
    p.add_argument(
        "--posterior-smooth-deg",
        type=float,
        default=1.5,
        metavar="DEG",
        help="Gaussian smoothing width [deg] applied to the bilby posterior HEALPix map "
        "(nside=128). Set to 0 to disable smoothing.",
    )
    p.add_argument(
        "--contour-levels",
        nargs="+",
        type=float,
        default=[50, 90],
        metavar="PCT",
        help="Credible-region contour levels in percent",
    )
    p.add_argument(
        "--output",
        "-o",
        default=None,
        help="Full path for the saved PNG (overrides --outdir and auto-naming)",
    )
    p.add_argument(
        "--outdir",
        default=None,
        metavar="DIR",
        help="Directory for the auto-named PNG (default: script directory)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for the saved figure",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the figure to disk",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window (useful for batch runs)",
    )
    return p


# ── detector setup (module-level constants) ───────────────────────────────────
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
DETECTOR_ARM_AZ = {}
for _name in IFO_NAMES:
    _det = lal.CachedDetectors[_LAL_IFO[_name]]
    DETECTOR_POSITION[_name] = (
        np.rad2deg(_det.frDetector.vertexLongitudeRadians),
        np.rad2deg(_det.frDetector.vertexLatitudeRadians),
    )
    DETECTOR_ARM_AZ[_name] = _det.frDetector.xArmAzimuthRadians

# Effective RMS bandwidth [Hz] per detector for timing-uncertainty estimation.
# σ_τ = 1 / (2π · ρ · f_rms); values approximate O5 sensitivity from 20 Hz lower cutoff.
TIMING_BANDWIDTH_HZ = {
    "H1": 35.0,
    "L1": 35.0,
    "V1": 30.0,
    "K1": 25.0,
    "I1": 25.0,
}


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


def _network_snr(row):
    """Extract network SNR from a stats row, trying common column names."""
    for col in ("snr", "network_snr", "rho", "SNR"):
        if col in row.index:
            print(row[col])
            return float(row[col])
    raise KeyError(
        f"Cannot find network SNR column. "
        f"Tried: snr, network_snr, rho, SNR. "
        f"Available columns: {list(row.index)}"
    )


def compute_pair_sigma_ms(d1, d2, row, n_det):
    """Timing uncertainty [ms] for a detector pair.

    Approximates per-detector SNR as ρ_net / √n_det (equal-SNR assumption),
    then combines in quadrature: σ_τ = √(σ_1² + σ_2²),
    where σ_i = 1 / (2π · ρ_i · f_rms_i).
    """
    rho_per_det = _network_snr(row) / np.sqrt(n_det)
    s1 = 1.0 / (2 * np.pi * rho_per_det * TIMING_BANDWIDTH_HZ[d1])
    s2 = 1.0 / (2 * np.pi * rho_per_det * TIMING_BANDWIDTH_HZ[d2])
    return 1000.0 * np.sqrt(s1 ** 2 + s2 ** 2)


def _plot_skymap(ax, sm, contour_levels, plot_freq, show_annotation):
    """Render a MOC sky map (UNIQ + PROBDENSITY table) onto ax."""
    sr_to_deg2 = u.sr.to(u.deg ** 2)
    dA = lsm_moc.uniq2pixarea(sm["UNIQ"])
    dP = sm["PROBDENSITY"] * dA
    cls = 100 * lsm_postprocess.find_greedy_credible_levels(dP, sm["PROBDENSITY"])
    cs = ax.contour_hpx(
        (Table({"UNIQ": sm["UNIQ"], "CLS": cls}), "ICRS"),
        colors="k",
        linewidths=0.5,
        levels=contour_levels,
        order="nearest-neighbor",
    )
    plt.clabel(cs, fmt=r"%g%%", fontsize=10, inline=True)

    if show_annotation:
        _sort_idx = np.flipud(np.argsort(sm["PROBDENSITY"]))
        _areas = lsm_postprocess.interp_greedy_credible_levels(
            contour_levels,
            cls[_sort_idx],
            np.cumsum(dA[_sort_idx]),
            right=4 * np.pi,
        )
        _ann_lines = ([rf"$f$ = {plot_freq} Hz"] if plot_freq is not None else []) + [
            f"{int(np.round(p))}% area: {_format_area(a * sr_to_deg2)} deg²"
            for p, a in zip(contour_levels, _areas)
        ]
        ax.text(
            0.88,
            1.0,
            "\n".join(_ann_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=20,
            bbox=dict(boxstyle="round,pad=0.4", fc="white"),
        )

    sm["PROBDENSITY"] = sm["PROBDENSITY"] / sr_to_deg2
    ax.imshow_hpx((sm, "ICRS"), vmin=0, cmap="cylon", order="nearest-neighbor")


def _posterior_to_skymap(ra, dec, smooth_deg=1.5, nside=128):
    """Bin posterior (ra, dec) samples [rad] into a HEALPix PROBDENSITY Table.

    Uses a Gaussian beam of width smooth_deg applied via healpy.smoothing so
    that the result can be fed directly to ax.imshow_hpx / ax.contour_hpx.
    UNIQ indices follow the nested-order MOC convention: UNIQ = 4*nside^2 + ipix.
    """
    npix = hp.nside2npix(nside)
    ipix = hp.ang2pix(nside, np.pi / 2 - dec, ra, nest=True)
    counts = np.bincount(ipix, minlength=npix).astype(float)
    if smooth_deg > 0:
        counts = hp.reorder(counts, n2r=True)
        counts = hp.smoothing(counts, sigma=np.deg2rad(smooth_deg))
        counts = hp.reorder(np.maximum(counts, 0.0), r2n=True)
    prob = counts / counts.sum()
    prob_density = prob / hp.nside2pixarea(nside)
    uniq = (4 * nside ** 2 + np.arange(npix)).astype(np.int64)
    return Table({"UNIQ": uniq, "PROBDENSITY": prob_density})


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
        np.vstack([p.vertices for p in paths]),
        np.concatenate([p.codes for p in paths]),
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


def rotate_path(path, angle_deg):
    return path.transformed(Affine2D().rotate_deg(angle_deg))


def arm_screen_angle(ax, plot_lon, plot_lat, arm_az_rad, geo_lat_deg, epsilon=0.3):
    """Screen rotation angle (CCW from right, degrees) for a detector arm.

    Works for both Mollweide and geo-globe projections by finite-differencing
    the projected positions of the arm endpoint and the detector vertex.
    `arm_az_rad` is the x-arm azimuth (clockwise from geographic North).
    `geo_lat_deg` is the geographic latitude of the detector.
    """
    lat_rad = np.deg2rad(geo_lat_deg)
    dlat = epsilon * np.cos(arm_az_rad)
    dlon = epsilon * np.sin(arm_az_rad) / np.cos(lat_rad)
    t = ax.get_transform("world")
    p0 = t.transform([[plot_lon, plot_lat]])
    p1 = t.transform([[plot_lon + dlon, plot_lat + dlat]])
    return np.rad2deg(np.arctan2(p1[0, 1] - p0[0, 1], p1[0, 0] - p0[0, 0]))


def plot_ifo(
    ax, lon, lat, size=46, beam_color="red", optic_color="k", rotation=0.0, **kw
):
    beams = rotate_path(IFO_BEAMS, rotation)
    optics = rotate_path(IFO_OPTICS, rotation)
    common = dict(markersize=size, linestyle="none", markeredgewidth=0, **kw)
    ax.plot(lon, lat, marker=beams, markerfacecolor=beam_color, **common)
    ax.plot(lon, lat, marker=optics, markerfacecolor=optic_color, **common)


# ── main ──────────────────────────────────────────────────────────────────────
def main(argv=None):
    args = build_parser().parse_args(argv)
    use_geo = args.geo or args.globe  # either mode uses the geo globe projection

    # ── resolve ring pairs ────────────────────────────────────────────────────
    if args.detectors is not None:
        for det in args.detectors:
            if det not in IFO_NAMES:
                raise ValueError(f"Unknown detector '{det}'. Known: {IFO_NAMES}")
        ring_pairs = list(itertools.combinations(args.detectors, 2))
    else:
        tokens = (
            args.ring_pairs
            if args.ring_pairs is not None
            else ["L1-H1", "L1-V1", "H1-V1"]
        )
        ring_pairs = []
        for token in tokens:
            parts = token.split("-")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid ring pair '{token}'; expected format 'D1-D2'"
                )
            d1, d2 = parts
            if d1 not in IFO_NAMES or d2 not in IFO_NAMES:
                raise ValueError(
                    f"Unknown detector in pair '{token}'. Known: {IFO_NAMES}"
                )
            ring_pairs.append((d1, d2))

    n_det = len({det for pair in ring_pairs for det in pair})

    # ── load injection parameters ─────────────────────────────────────────────
    row = None
    snr_n_det = n_det  # number of detectors that contributed to rho_net
    posterior_ra = posterior_dec = None
    bilby_label = None
    if args.injection_number is not None:
        if args.stats_file is None:
            raise ValueError("--injection-number requires --stats-file")
        stats_path = args.stats_file
        stats = pd.read_csv(stats_path, sep="\t", index_col=0)
        row = stats[stats["simulation_id"] == args.injection_number].iloc[0]
        true_ra, true_dec = ast.literal_eval(row["ra_dec"])  # radians
        true_obstime = float(row["time"])  # GPS seconds
        print(
            f"Injection {args.injection_number}: "
            f"RA={np.rad2deg(true_ra):.2f}°  Dec={np.rad2deg(true_dec):.2f}°  "
            f"GPS={true_obstime:.0f}"
        )
    elif args.bilby_json is not None:
        import bilby

        result = bilby.core.result.read_in_result(args.bilby_json)
        ip = result.injection_parameters
        true_ra = float(ip["ra"])
        true_dec = float(ip["dec"])
        true_obstime = float(ip["geocent_time"])
        # Build a row-like Series so compute_pair_sigma_ms works unchanged.
        # Network SNR is computed from whichever {det}_optimal_snr keys exist.
        det_snrs = [v for k, v in ip.items() if k.endswith("_optimal_snr")]
        if det_snrs:
            rho_net = float(np.sqrt(sum(s ** 2 for s in det_snrs)))
            row = pd.Series({"snr": rho_net})
            snr_n_det = len(det_snrs)
        posterior_ra = result.posterior["ra"].to_numpy()
        posterior_dec = result.posterior["dec"].to_numpy()
        bilby_label = (
            result.label or os.path.splitext(os.path.basename(args.bilby_json))[0]
        )
        print(
            f"Bilby result '{bilby_label}': "
            f"RA={np.rad2deg(true_ra):.2f}°  Dec={np.rad2deg(true_dec):.2f}°  "
            f"GPS={true_obstime:.0f}  N_posterior={len(posterior_ra)}"
        )
    else:
        ra_deg, dec_deg, true_obstime = args.sky_pos
        true_ra = np.deg2rad(ra_deg)
        true_dec = np.deg2rad(dec_deg)
        print(
            f"Sky position: RA={ra_deg:.2f}°  Dec={dec_deg:.2f}°  GPS={true_obstime:.0f}"
        )

    true_gmst = lal.GreenwichMeanSiderealTime(true_obstime) % (2 * np.pi)

    # ── load skymap ───────────────────────────────────────────────────────────
    skymap = None
    if args.skymap_file is not None:
        skymap = skymap_fits.read_sky_map(args.skymap_file, moc=True)
        print(f"Skymap loaded from {args.skymap_file}")

    # ── precompute timing circles ─────────────────────────────────────────────
    rings = {
        (d1, d2): get_ring_w_coloring(d1, d2, true_ra, true_dec, true_obstime)
        for d1, d2 in ring_pairs
    }
    activated_ifos = set(np.array(ring_pairs).flatten())

    # ── plot ──────────────────────────────────────────────────────────────────
    projection = "geo globe" if use_geo else "astro degrees mollweide"

    with rc_context(
        {
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "lines.linewidth": 3,
            "font.family": "sans-serif",
            "font.sans-serif": ["Georgia", "DejaVu Sans"],  # With fallback
            "mathtext.fontset": "cm",
        }
    ):
        fig = plt.figure(figsize=(9, 9) if use_geo else (14, 7))

        back_ax = None
        BACK_PLT_ARGS = {}
        _globe_n_view = None  # unit vector toward viewer, set only in --globe mode

        if use_geo:
            if args.geo_center == "auto":
                src_lon_deg = np.rad2deg((true_ra - true_gmst) % (2 * np.pi))
                geo_center = f"{src_lon_deg:.2f}d +23d"
            else:
                geo_center = args.geo_center

            # Unit vector toward the viewer — used to restrict labels to the
            # front hemisphere in both --geo and --globe modes.
            _parts = geo_center.replace('d', '').split()
            _lon_c, _lat_c = float(_parts[0]), float(_parts[1])
            _globe_n_view = np.array([
                np.cos(np.deg2rad(_lat_c)) * np.cos(np.deg2rad(_lon_c)),
                np.cos(np.deg2rad(_lat_c)) * np.sin(np.deg2rad(_lon_c)),
                np.sin(np.deg2rad(_lat_c)),
            ])

            if args.globe:
                # Antipodal center: looking from the opposite side of the globe.
                # invert_xaxis() corrects the left-right mirror that arises from
                # switching hemispheres, so back-side arcs align with front-side arcs
                # at the limb (as they would on a real transparent sphere).
                back_center = f"{(_lon_c + 180.0) % 360.0:.2f}d {-_lat_c:+.2f}d"

                _rect = [0.05, 0.02, 0.90, 0.90]
                back_ax = fig.add_axes(
                    _rect,
                    projection="geo globe",
                    obstime=Time(true_obstime, format="gps"),
                    center=back_center,
                )
                back_ax.set_facecolor("none")
                back_ax.invert_xaxis()
                for _c in back_ax.coords:
                    _c.set_ticklabel_visible(False)
                    _c.set_ticks_visible(False)
                back_ax.grid(color="gray", alpha=0.35, linewidth=0.5)
                BACK_PLT_ARGS = dict(transform=back_ax.get_transform("world"))

                ax = fig.add_axes(
                    _rect,
                    projection="geo globe",
                    obstime=Time(true_obstime, format="gps"),
                    center=geo_center,
                )
            else:
                ax = fig.add_subplot(
                    projection=projection,
                    obstime=Time(true_obstime, format="gps"),
                    center=geo_center,
                )
        else:
            ax = fig.add_subplot(projection=projection)
            ax.invert_xaxis()
            # Need inversion because Mollweide basically assumes that we look
            # from Earth out to the sky. But what we plot is view _onto_ Earth,
            # which flips the orientation along the vertical axis.

        PLT_ARGS = dict(transform=ax.get_transform("world"))

        if not use_geo:
            ax.set_aspect("equal")
        if args.globe:
            ax.set_facecolor("none")
        else:
            ax.set_facecolor(plt.get_cmap("cylon")(0.0))
        ax.grid()

        # continents
        if args.geo:
            ax.plot(*coastlines(), color="0.5", linewidth=0.5, **PLT_ARGS)
        elif not args.globe:
            plot_continents_icrs(ax, true_gmst)

        # skymap
        if skymap is not None:
            _plot_skymap(
                ax, skymap, args.contour_levels, args.plot_freq, show_annotation=True
            )

        # bilby posterior KDE skymap
        if posterior_ra is not None:
            post_skymap = _posterior_to_skymap(
                posterior_ra,
                posterior_dec,
                smooth_deg=args.posterior_smooth_deg,
            )
            _plot_skymap(
                ax,
                post_skymap,
                args.contour_levels,
                args.plot_freq,
                show_annotation=(skymap is None),
            )

        # timing circles
        _ring_colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(ring_pairs)))
        _label_fracs = (4 * [0.85, 0.50, 0.75])[: len(ring_pairs)]
        np.random.seed(args.seed)

        for (d1, d2), color, frac in zip(ring_pairs, _ring_colors, _label_fracs):
            ras, decs, _, _ = rings[(d1, d2)]
            label = f"{d1}-{d2}"

            if args.timing_uncertainty:
                if args.timing_sigma_ms is not None:
                    sigma_ms = args.timing_sigma_ms
                elif row is not None:
                    sigma_ms = compute_pair_sigma_ms(d1, d2, row, snr_n_det)
                    print(f"  {d1}-{d2}: auto σ_τ = {sigma_ms:.3f} ms (ρ_net/√{snr_n_det})")
                else:
                    raise ValueError(
                        f"Cannot auto-compute timing uncertainty for {d1}-{d2}: "
                        "no stats row available when using --sky-pos. "
                        "Provide --timing-sigma-ms explicitly."
                    )
                sigma_tau = sigma_ms * 1e-3
                true_tau = detectors[d1].time_delay(
                    detectors[d2].vertex, ra=true_ra, dec=true_dec, t_event=true_obstime
                )
                D = np.linalg.norm(detectors[d2].vertex - detectors[d1].vertex)
                max_tau = D / 3e8
                tau_samples = np.clip(
                    np.random.normal(true_tau, sigma_tau, args.n_annulus),
                    -max_tau + 1e-9,
                    max_tau - 1e-9,
                )
                for tau in tau_samples:
                    s_ras, s_decs = detectors[d1].sky_location(
                        detectors[d2].vertex, time_delay=tau, t_event=true_obstime
                    )
                    s_ras = s_ras % (2 * np.pi)
                    s_lons = (
                        np.rad2deg((s_ras - true_gmst) % (2 * np.pi))
                        if use_geo
                        else np.rad2deg(s_ras)
                    )

                    s_decs_deg = np.rad2deg(s_decs)
                    jumps = np.where(np.abs(np.diff(s_lons)) > 180)[0] + 1
                    s_lons_p = np.insert(s_lons.astype(float), jumps, np.nan)
                    s_decs_p = np.insert(s_decs_deg.astype(float), jumps, np.nan)
                    ax.plot(
                        s_lons_p,
                        s_decs_p,
                        color=color,
                        linewidth=10,
                        alpha=1 / args.n_annulus,
                        # alpha=2 / args.n_annulus,
                        zorder=9,
                        **PLT_ARGS,
                    )
                    if back_ax is not None:
                        back_ax.plot(
                            s_lons_p,
                            s_decs_p,
                            color=color,
                            linewidth=10,
                            alpha=0.5 / args.n_annulus,
                            zorder=9,
                            **BACK_PLT_ARGS,
                        )

            lons = (
                np.rad2deg((ras - true_gmst) % (2 * np.pi))
                if use_geo
                else np.rad2deg(ras)
            )
            lats = np.rad2deg(decs)

            jumps = np.where(np.abs(np.diff(lons)) > 180)[0] + 1
            lons_p = np.insert(lons.astype(float), jumps, np.nan)
            lats_p = np.insert(lats.astype(float), jumps, np.nan)
            ax.plot(lons_p, lats_p, linewidth=2, color=color, zorder=10, **PLT_ARGS)
            if back_ax is not None:
                back_ax.plot(
                    lons_p, lats_p, linewidth=2, color=color, alpha=0.4, zorder=10,
                    **BACK_PLT_ARGS,
                )

            # inline label — in geo/globe mode restrict to front-hemisphere points
            idx = int(frac * len(ras)) % len(ras)
            step = max(3, len(ras) // 60)
            i0 = (idx - step) % len(ras)
            i1 = (idx + step) % len(ras)
            if _globe_n_view is not None:
                _lr = np.deg2rad(lons)
                _br = np.deg2rad(lats)
                _dot = (np.cos(_br) * np.cos(_lr) * _globe_n_view[0]
                        + np.cos(_br) * np.sin(_lr) * _globe_n_view[1]
                        + np.sin(_br) * _globe_n_view[2])
                _front = np.where(_dot > 0)[0]
                if _front.size > 0:
                    # Build the ordered front arc. When the arc wraps around the
                    # ring-array boundary (two disjoint index ranges), a single gap
                    # appears in _front; we rejoin the pieces in ring order so that
                    # i0/i1 neighbours are always adjacent on the great circle.
                    _diffs = np.diff(_front)
                    _gaps = np.where(_diffs > 1)[0]
                    if len(_gaps) == 0:
                        _arc = _front
                    elif len(_gaps) == 1:
                        _arc = np.concatenate([_front[_gaps[0]+1:], _front[:_gaps[0]+1]])
                    else:
                        _segs = np.split(_front, _gaps + 1)
                        _arc = max(_segs, key=len)
                    _step_a = max(1, len(_arc) // 60)
                    _fi = int(frac * len(_arc)) % len(_arc)
                    idx = _arc[_fi]
                    i0 = _arc[(_fi - _step_a) % len(_arc)]
                    i1 = _arc[(_fi + _step_a) % len(_arc)]

            # Angle computation varies depending on the transform used (needed
            # because orthographic "geo globe" projection distorts scales).
            if use_geo:
                _pts = ax.get_transform("world").transform(
                    np.array([[lons[i0], lats[i0]], [lons[i1], lats[i1]]])
                )
                if np.all(np.isfinite(_pts)):
                    angle = np.rad2deg(np.arctan2(
                        _pts[1, 1] - _pts[0, 1], _pts[1, 0] - _pts[0, 0]
                    ))
                else:
                    dlon = (lons[i1] - lons[i0] + 180) % 360 - 180
                    dlat = lats[i1] - lats[i0]
                    angle = np.rad2deg(np.arctan2(dlat, dlon))
            else:
                dlon = (lons[i1] - lons[i0] + 180) % 360 - 180
                dlat = lats[i1] - lats[i0]
                angle = np.rad2deg(np.arctan2(dlat, dlon))

            import matplotlib.colors as mcolors

            darker = tuple(c * 0.5 for c in mcolors.to_rgba(color)[:3]) + (1.0,)

            ax.text(
                lons[idx],
                lats[idx],
                f" {label} ",
                transform=ax.get_transform("world"),
                fontsize=20,
                ha="center",
                va="center",
                rotation=angle,
                rotation_mode="anchor",
                # color=color,
                # color="black",
                color=darker,
                fontweight="bold",
                zorder=20,
            )

        # true source
        src_lon = (
            np.rad2deg((true_ra - true_gmst) % (2 * np.pi))
            if use_geo
            else np.rad2deg(true_ra)
        )
        ax.plot(
            src_lon,
            np.rad2deg(true_dec),
            "*",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=12,
            zorder=100,
            **PLT_ARGS,
        )

        # detector markers
        LABEL_ARGS = dict(ha="right", va="bottom", fontsize=12, **PLT_ARGS)
        for name in activated_ifos:
            geo_lon, geo_lat = DETECTOR_POSITION[name]
            if use_geo:
                plot_lon, plot_lat = geo_lon, geo_lat
            else:
                plot_lon = np.rad2deg((np.deg2rad(geo_lon) + true_gmst) % (2 * np.pi))
                plot_lat = geo_lat
            arm_rot = arm_screen_angle(
                ax, plot_lon, plot_lat, DETECTOR_ARM_AZ[name], geo_lat
            )
            plot_ifo(ax, plot_lon, plot_lat, size=24, rotation=arm_rot, **PLT_ARGS)
            ax.text(plot_lon, plot_lat, name, **LABEL_ARGS)

        outline_text(ax)

        if args.bilby_json is not None:
            title_prefix = bilby_label
        elif args.injection_number is not None:
            title_prefix = f"Injection {args.injection_number}"
        else:
            title_prefix = "Manual"
        ax.set_title(
            f"{title_prefix}  |  "
            f"RA = {np.rad2deg(true_ra):.1f}°   Dec = {np.rad2deg(true_dec):.1f}°   "
            f"GPS = {true_obstime:.0f}",
            fontsize=26,
            pad=20,
        )

        plt.tight_layout()

        if not args.no_save:
            if args.output:
                out_path = args.output
            else:
                det_label = "".join(sorted({n[0] for n in activated_ifos})).lower()
                if args.injection_number is not None:
                    inj_tag = f"inj{args.injection_number}"
                elif args.bilby_json is not None:
                    inj_tag = bilby_label
                else:
                    inj_tag = "manual"
                out_dir = (
                    args.outdir
                    if args.outdir is not None
                    else os.path.dirname(os.path.abspath(__file__))
                )
                freq_tag = f"_f{args.plot_freq}" if args.plot_freq is not None else ""
                out_path = os.path.join(
                    out_dir,
                    f"timing_circle_{inj_tag}{freq_tag}_{det_label}"
                    + ("_annulus" if args.timing_uncertainty else "")
                    + ("_geo" if args.geo else "_globe" if args.globe else "")
                    + ".png",
                )
            plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
            print(f"Saved → {out_path}")

        if not args.no_show:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    main()
