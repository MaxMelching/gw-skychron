#!/usr/bin/env python3
# Copyright (C) 2026 Max Melching
"""
Plot timing circles and sky-localization credible regions for one injection.

Basic usage (injection number + base path):
    python plot_timing_circles.py -n 4 \\
        --base-path ../../../sky_localization/results/H1_O5_L1_O5_V1_O5_many

Separate stats and skymap files (no base path):
    python plot_timing_circles.py -n 4 \\
        --stats-file /path/to/combined_stats.dat \\
        --skymap-file /path/to/sim_id_4.fits

All detectors from a network, with auto-computed timing annuli:
    python plot_timing_circles.py -n 4 \\
        --base-path ../../../sky_localization/results/H1_O5_L1_O5_V1_O5_many \\
        --detectors H1 L1 V1 \\
        --timing-uncertainty \\
        --n-annulus 80

Override timing uncertainty with an explicit sigma:
    python plot_timing_circles.py -n 4 \\
        --base-path ../../../sky_localization/results/H1_O5_L1_O5_V1_O5_many \\
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
    python plot_timing_circles.py -n 4 \\
        --base-path ../../../sky_localization/results/H1_O5_L1_O5_V1_O5_many \\
        --plot-freq 56 \\
        --ring-pairs L1-H1 L1-V1 H1-V1 \\
        --timing-uncertainty \\
        --contour-levels 50 90 \\
        --geo \\
        --outdir /tmp/plots
"""

import argparse
import ast
import itertools
import os
import sys

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
    p.add_argument(
        "--base-path",
        default=None,
        metavar="DIR",
        help="Results directory containing stats/ and {freq}/fits/ subdirs. "
        "When omitted, use --stats-file and/or --skymap-file instead.",
    )
    p.add_argument(
        "--stats-file",
        default=None,
        metavar="PATH",
        help="Path to the stats CSV directly (alternative to --base-path for injection lookup). "
        "The injection number is still used to select the correct row.",
    )
    p.add_argument(
        "--skymap-file",
        default=None,
        metavar="PATH",
        help="Path to a single FITS skymap (alternative to --base-path skymap discovery). "
        "Skips frequency-directory scanning.",
    )
    p.add_argument(
        "--plot-freq",
        type=int,
        default=56,
        metavar="HZ",
        help="Frequency [Hz] of the skymap to display (must exist in BASE_PATH/HZ/fits/)",
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
    p.add_argument(
        "--geo",
        action="store_true",
        help="Use 'geo globe' projection instead of 'astro degrees mollweide'",
    )
    p.add_argument(
        "--geo-center",
        default="auto",
        metavar="'LONd LATd'",
        help="Center for geo globe: 'auto' centres on the source longitude; "
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
        "--contour-levels",
        nargs="+",
        type=float,
        default=[50, 90],
        metavar="PCT",
        help="Credible-region contour levels in percent",
    )
    p.add_argument(
        "--no-skymap",
        action="store_true",
        help="Skip loading and plotting the HEALPix skymap entirely",
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


def plot_ifo(ax, lon, lat, size=46, beam_color="red", optic_color="k", rotation=0.0, **kw):
    beams = rotate_path(IFO_BEAMS, rotation)
    optics = rotate_path(IFO_OPTICS, rotation)
    common = dict(markersize=size, linestyle="none", markeredgewidth=0, **kw)
    ax.plot(lon, lat, marker=beams, markerfacecolor=beam_color, **common)
    ax.plot(lon, lat, marker=optics, markerfacecolor=optic_color, **common)


# ── main ──────────────────────────────────────────────────────────────────────
def main(argv=None):
    args = build_parser().parse_args(argv)

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
    if args.injection_number is not None:
        if args.stats_file is not None:
            stats_path = args.stats_file
        elif args.base_path is not None:
            stats_path = os.path.join(args.base_path, "stats", "combined_stats.dat")
        else:
            raise ValueError(
                "--injection-number requires either --base-path or --stats-file"
            )
        stats = pd.read_csv(stats_path, sep="\t", index_col=0)
        row = stats[stats["simulation_id"] == args.injection_number].iloc[0]
        true_ra, true_dec = ast.literal_eval(row["ra_dec"])  # radians
        true_obstime = float(row["time"])  # GPS seconds
        print(
            f"Injection {args.injection_number}: "
            f"RA={np.rad2deg(true_ra):.2f}°  Dec={np.rad2deg(true_dec):.2f}°  "
            f"GPS={true_obstime:.0f}"
        )
    else:
        ra_deg, dec_deg, true_obstime = args.sky_pos
        true_ra = np.deg2rad(ra_deg)
        true_dec = np.deg2rad(dec_deg)
        print(
            f"Sky position: RA={ra_deg:.2f}°  Dec={dec_deg:.2f}°  GPS={true_obstime:.0f}"
        )

    true_gmst = lal.GreenwichMeanSiderealTime(true_obstime) % (2 * np.pi)

    # ── load skymaps ──────────────────────────────────────────────────────────
    skymaps = {}
    if not args.no_skymap:
        if args.skymap_file is not None:
            skymaps[args.plot_freq] = skymap_fits.read_sky_map(
                args.skymap_file, moc=True
            )
            print(f"Skymap loaded from {args.skymap_file}")
        elif args.base_path is not None and args.injection_number is not None:
            freq_dirs = sorted(
                [d for d in os.listdir(args.base_path) if d.isdigit()], key=int
            )
            for freq_str in freq_dirs:
                fits_path = os.path.join(
                    args.base_path,
                    freq_str,
                    "fits",
                    f"sim_id_{args.injection_number}.fits",
                )
                if os.path.exists(fits_path):
                    skymaps[int(freq_str)] = skymap_fits.read_sky_map(
                        fits_path, moc=True
                    )

            if not skymaps:
                raise FileNotFoundError(
                    f"No FITS files found for injection {args.injection_number} "
                    f"in {args.base_path}"
                )
            if args.plot_freq not in skymaps:
                raise KeyError(
                    f"Requested --plot-freq {args.plot_freq} Hz not found. "
                    f"Available: {sorted(skymaps)}"
                )
            print(f"Skymaps found at frequencies (Hz): {sorted(skymaps)}")

    # ── precompute timing circles ─────────────────────────────────────────────
    rings = {
        (d1, d2): get_ring_w_coloring(d1, d2, true_ra, true_dec, true_obstime)
        for d1, d2 in ring_pairs
    }
    activated_ifos = set(np.array(ring_pairs).flatten())

    # ── plot ──────────────────────────────────────────────────────────────────
    projection = "geo globe" if args.geo else "astro degrees mollweide"

    with rc_context(
        {
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "lines.linewidth": 3,
            "font.family": "sans-serif",
            "font.sans-serif": ["Georgia", "DejaVu Sans"],  # With fallback
        }
    ):
        fig = plt.figure(figsize=(9, 9) if args.geo else (14, 7))

        if args.geo:
            if args.geo_center == "auto":
                src_lon_deg = np.rad2deg((true_ra - true_gmst) % (2 * np.pi))
                geo_center = f"{src_lon_deg:.2f}d +23d"
            else:
                geo_center = args.geo_center
            ax = fig.add_subplot(
                projection=projection,
                obstime=Time(true_obstime, format="gps"),
                center=geo_center,
            )
        else:
            ax = fig.add_subplot(projection=projection)
            ax.invert_xaxis()

        PLT_ARGS = dict(transform=ax.get_transform("world"))

        if not args.geo:
            ax.set_aspect("equal")
        ax.set_facecolor(plt.get_cmap("cylon")(0.0))
        ax.grid()

        # continents
        if args.geo:
            ax.plot(*coastlines(), color="0.5", linewidth=0.5, **PLT_ARGS)
        else:
            plot_continents_icrs(ax, true_gmst)

        # skymap
        if skymaps:
            skymap = skymaps[args.plot_freq]
            dA = lsm_moc.uniq2pixarea(skymap["UNIQ"])
            dP = skymap["PROBDENSITY"] * dA
            cls = 100 * lsm_postprocess.find_greedy_credible_levels(
                dP, skymap["PROBDENSITY"]
            )
            cs = ax.contour_hpx(
                (Table({"UNIQ": skymap["UNIQ"], "CLS": cls}), "ICRS"),
                colors="k",
                linewidths=0.5,
                levels=args.contour_levels,
                order="nearest-neighbor",
            )
            fmt = r"%g\%%"
            plt.clabel(cs, fmt=fmt, fontsize=10, inline=True)

            sr_to_deg2 = u.sr.to(u.deg ** 2)
            _sort_idx = np.flipud(np.argsort(skymap["PROBDENSITY"]))
            _areas = lsm_postprocess.interp_greedy_credible_levels(
                args.contour_levels,
                cls[_sort_idx],
                np.cumsum(dA[_sort_idx]),
                right=4 * np.pi,
            )
            _ann_lines = [f"$f = {freq_str} \, \mathrm{{Hz}}$"] + [
                f"{int(np.round(p))}% area: {_format_area(a * sr_to_deg2)} deg²"
                for p, a in zip(args.contour_levels, _areas)
            ]
            ax.text(
                0.99,
                0.99,
                "\n".join(_ann_lines),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=20,
                bbox=dict(boxstyle="round,pad=0.4", fc="white"),
            )

            skymap["PROBDENSITY"] = skymap["PROBDENSITY"] / sr_to_deg2
            ax.imshow_hpx(
                (skymap, "ICRS"), vmin=0, cmap="cylon", order="nearest-neighbor"
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
                    sigma_ms = compute_pair_sigma_ms(d1, d2, row, n_det)
                    print(f"  {d1}-{d2}: auto σ_τ = {sigma_ms:.3f} ms (ρ_net/√{n_det})")
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
                        if args.geo
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

            lons = (
                np.rad2deg((ras - true_gmst) % (2 * np.pi))
                if args.geo
                else np.rad2deg(ras)
            )
            lats = np.rad2deg(decs)

            jumps = np.where(np.abs(np.diff(lons)) > 180)[0] + 1
            lons_p = np.insert(lons.astype(float), jumps, np.nan)
            lats_p = np.insert(lats.astype(float), jumps, np.nan)
            ax.plot(lons_p, lats_p, linewidth=2, color=color, zorder=10, **PLT_ARGS)

            # inline label
            idx = int(frac * len(ras)) % len(ras)
            step = max(3, len(ras) // 60)
            i0, i1 = (idx - step) % len(ras), (idx + step) % len(ras)
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
            if args.geo
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
            if args.geo:
                plot_lon, plot_lat = geo_lon, geo_lat
            else:
                plot_lon = np.rad2deg((np.deg2rad(geo_lon) + true_gmst) % (2 * np.pi))
                plot_lat = geo_lat
            arm_rot = arm_screen_angle(ax, plot_lon, plot_lat, DETECTOR_ARM_AZ[name], geo_lat)
            plot_ifo(ax, plot_lon, plot_lat, size=24, rotation=arm_rot, **PLT_ARGS)
            ax.text(plot_lon, plot_lat, name, **LABEL_ARGS)

        outline_text(ax)

        ax.set_title(
            f"Injection {args.injection_number}  |  "
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
                inj_tag = (
                    f"inj{args.injection_number}"
                    if args.injection_number is not None
                    else "manual"
                )
                out_dir = (
                    args.outdir
                    if args.outdir is not None
                    else os.path.dirname(os.path.abspath(__file__))
                )
                out_path = os.path.join(
                    out_dir,
                    f"timing_circle_{inj_tag}_f{args.plot_freq}_{det_label}"
                    + ("_annulus" if args.timing_uncertainty else "")
                    + ("_geo" if args.geo else "")
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
