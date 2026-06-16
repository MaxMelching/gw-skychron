#!/usr/bin/env python3
# Copyright (C) 2026 Max Melching
"""
Plot timing circles and sky-localization credible regions for one injection.

Injection from a stats file with explicit timing sigma:
    gw-skychron \\
        --stats-file /path/to/stats/combined_stats.dat \\
        --skymap-file /path/to/56/fits/sim_id_4.fits \\
        --injection-number 4 \\
        --ring-pairs L1-H1 L1-V1 H1-V1 \\
        --timing-uncertainty --timing-sigma-ms 0.42 \\
        --contour-levels 50 90

All detectors, auto-computed timing annuli:
    gw-skychron -n 4 \\
        --stats-file /path/to/stats/combined_stats.dat \\
        --skymap-file /path/to/56/fits/sim_id_4.fits \\
        --detectors H1 L1 V1 \\
        --timing-uncertainty --n-annulus 80

Sky position provided directly, with an optional skymap overlay:
    gw-skychron \\
        --sky-pos 45.0 -30.0 1234567890 \\
        --skymap-file /path/to/skymap.fits \\
        --ring-pairs L1-H1 L1-V1 \\
        --timing-uncertainty --timing-sigma-ms 0.5

Geo projection with custom output directory:
    gw-skychron \\
        --stats-file /path/to/stats/combined_stats.dat \\
        --injection-number 4 --plot-freq 56 \\
        --ring-pairs L1-H1 L1-V1 H1-V1 \\
        --timing-uncertainty --contour-levels 50 90 \\
        --geo --outdir /tmp/plots

Bilby result with custom smoothing:
    gw-skychron \\
        --bilby-json /path/to/result.json \\
        --detectors H1 V1 \\
        --timing-uncertainty --timing-sigma-ms 0.5 \\
        --posterior-smooth-deg 2.0 --contour-levels 50 90

Color circles by antenna response (globe projection):
    gw-skychron \\
        --stats-file /path/to/stats/combined_stats.dat \\
        --injection-number 4 --plot-freq 56 \\
        --ring-pairs L1-H1 L1-V1 H1-V1 \\
        --resp-func --globe

Override label positions (single value for all pairs, or one per pair —
the fraction is measured clockwise from the true source):
    gw-skychron -n 4 \\
        --stats-file /path/to/stats/combined_stats.dat \\
        --ring-pairs L1-H1 L1-V1 H1-V1 \\
        --label-frac 0.25

    gw-skychron -n 4 \\
        --stats-file /path/to/stats/combined_stats.dat \\
        --ring-pairs L1-H1 L1-V1 H1-V1 \\
        --label-frac 0.15 0.30 0.45
"""

import argparse
import ast
import itertools
import os

import lal
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import rc_context
from matplotlib.collections import LineCollection
from astropy.time import Time
import astropy.units as u
import ligo.skymap.plot  # registers projections
from ligo.skymap.io import fits as skymap_fits
from ligo.skymap.plot import outline_text
from ligo.skymap.plot.backdrop import coastlines

from .detectors import (
    IFO_NAMES,
    detectors,
    DETECTOR_POSITION,
    DETECTOR_ARM_AZ,
    compute_antenna_response,
    get_ring_w_coloring,
    compute_pair_sigma_ms,
)
from .ifo_marker import arm_screen_angle, plot_ifo
from .skymap import plot_skymap, posterior_to_skymap, plot_continents_icrs


# -- Argument parsing ---------------------------------------------------------
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
        "--label-frac",
        nargs="+",
        type=float,
        default=None,
        metavar="F",
        help="Override automatic label placement: fraction of the full circle "
        "traversed clockwise from the true source (0 = at source, "
        "0.5 = halfway around, 1 = back at source). "
        "One value applies to all pairs; one value per pair is also accepted "
        "(requires --ring-pairs so the per-pair mapping is unambiguous). "
        "Default: label placed at the highest visible point on the ring.",
    )
    p.add_argument(
        "--resp-func",
        action="store_true",
        help="Color each timing circle by the combined antenna response "
        "sqrt(F1² + F2²) at each point along the ring, using a single "
        "normalized color scale across all pairs. Annulus rings (if any) "
        "keep their per-pair color.",
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


# -- Other Axes Helpers -------------------------------------------------------
def _setup_axes(fig, args, use_geo, projection, true_obstime, true_ra, true_gmst):
    """Create and configure the main (and optionally back) axes.

    Returns (ax, back_ax, PLT_ARGS, BACK_PLT_ARGS, _globe_n_view).
    back_ax and _globe_n_view are None when not in globe/geo mode.
    """
    back_ax = None
    BACK_PLT_ARGS = {}
    _globe_n_view = None

    if use_geo:
        if args.geo_center == "auto":
            src_lon_deg = np.rad2deg((true_ra - true_gmst) % (2 * np.pi))
            geo_center = f"{src_lon_deg:.2f}d +23d"
        else:
            geo_center = args.geo_center

        # Unit vector toward the viewer — used to restrict labels to the
        # front hemisphere in both --geo and --globe modes.
        _parts = geo_center.replace("d", "").split()
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

            # 10 % left/right margins so WCSAxes latitude labels (which
            # render outside the axes rectangle on the left) are not
            # clipped at the figure edge.  --geo uses add_subplot() which
            # provides ~12.5 % automatically; we need to replicate that.
            _rect = [0.10, 0.02, 0.80, 0.90]
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
        # add_axes() + WCSAxes auto-tick heuristic produces far fewer ticks
        # than add_subplot(); pin the spacing explicitly to match --geo.
        for _c in ax.coords:
            _c.set_ticks(spacing=30 * u.deg)
            _c.set_ticklabel_visible(True)
            _c.set_ticks_visible(True)
    else:
        ax.set_facecolor(plt.get_cmap("cylon")(0.0))
    ax.grid()

    return ax, back_ax, PLT_ARGS, BACK_PLT_ARGS, _globe_n_view


def _find_label_idx(lons, lats, jumps, label_frac, globe_n_view, ax,
                    use_geo, true_ra, true_dec, true_gmst):
    """Return (idx, i0, i1): label point and its two step-neighbours for angle.

    idx  — index into lons/lats where the label will be placed
    i0/i1 — indices one step before/after idx (used to compute tangent angle)
    """
    _n = len(lons)
    step = max(3, _n // 60)
    _lr = np.deg2rad(lons)
    _br = np.deg2rad(lats)

    if label_frac is not None:
        # User-specified: place at fraction label_frac of the full circle,
        # measured clockwise from the true source as seen on screen.
        # CW on screen (y-down physically) = negative signed area in
        # matplotlib display coords (y-up).
        _all_screen = ax.get_transform("world").transform(
            np.column_stack([lons, lats])
        )
        _ok = np.all(np.isfinite(_all_screen), axis=1)
        _sx, _sy = _all_screen[_ok, 0], _all_screen[_ok, 1]
        _sa = 0.5 * float(np.sum(_sx[:-1] * _sy[1:] - _sx[1:] * _sy[:-1]))
        _cw = 1 if _sa < 0 else -1  # +1 if +index direction is CW on screen
        _tl = (true_ra - true_gmst) % (2 * np.pi) if use_geo else true_ra
        _cos_sep = (
            np.sin(_br) * np.sin(true_dec)
            + np.cos(_br) * np.cos(true_dec) * np.cos(_lr - _tl)
        )
        _idx_src = int(np.argmax(_cos_sep))
        idx = int((_idx_src + _cw * int(label_frac * _n)) % _n)
        i0 = (idx - step) % _n
        i1 = (idx + step) % _n

    elif globe_n_view is not None:
        # geo/globe default: highest visible point on the front arc.
        _dot = (
            np.cos(_br) * np.cos(_lr) * globe_n_view[0]
            + np.cos(_br) * np.sin(_lr) * globe_n_view[1]
            + np.sin(_br) * globe_n_view[2]
        )
        _front = np.where(_dot > 0)[0]
        if _front.size > 0:
            _diffs = np.diff(_front)
            _gaps = np.where(_diffs > 1)[0]
            if len(_gaps) == 0:
                _arc = _front
            elif len(_gaps) == 1:
                _arc = np.concatenate([_front[_gaps[0] + 1:], _front[:_gaps[0] + 1]])
            else:
                _arc = max(np.split(_front, _gaps + 1), key=len)
            _screen = ax.get_transform("world").transform(
                np.column_stack([lons[_arc], lats[_arc]])
            )
            _fi = int(np.argmax(_screen[:, 1]))
            idx = int(_arc[_fi])
            _na = len(_arc)
            _step_a = max(1, _na // 60)
            i0 = int(_arc[(_fi - _step_a) % _na])
            i1 = int(_arc[(_fi + _step_a) % _na])
        else:
            idx = 0
            i0 = (idx - step) % _n
            i1 = (idx + step) % _n

    else:
        # Mollweide default: highest visible point in source's segment
        # (or the full ring when there are no longitude-wrap jumps).
        if len(jumps) == 0:
            _visible = np.arange(_n)
        else:
            _tl = true_ra
            _cos_sep = (
                np.sin(_br) * np.sin(true_dec)
                + np.cos(_br) * np.cos(true_dec) * np.cos(_lr - _tl)
            )
            _idx_src = int(np.argmax(_cos_sep))
            _bounds = np.concatenate([[0], jumps, [_n]])
            _si = int(np.searchsorted(_bounds, _idx_src, side="right")) - 1
            _visible = np.arange(int(_bounds[_si]), int(_bounds[_si + 1]))
        _screen = ax.get_transform("world").transform(
            np.column_stack([lons[_visible], lats[_visible]])
        )
        _vi = int(np.argmax(_screen[:, 1]))
        idx = int(_visible[_vi])
        i0 = (idx - step) % _n
        i1 = (idx + step) % _n

    return idx, i0, i1


def main(argv=None):
    _p = build_parser()
    args = _p.parse_args(argv)
    use_geo = args.geo or args.globe

    # ─ Resolve ring pairs
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

    # -- Resolve per-pair label fractions
    if args.label_frac is not None:
        if len(args.label_frac) > 1:
            if args.ring_pairs is None:
                _p.error(
                    "--label-frac with multiple values requires explicit "
                    "--ring-pairs so the per-pair mapping is unambiguous"
                )
            if len(args.label_frac) != len(ring_pairs):
                _p.error(
                    f"--label-frac has {len(args.label_frac)} values "
                    f"but {len(ring_pairs)} ring pairs were specified"
                )
        label_fracs = (
            args.label_frac * len(ring_pairs)
            if len(args.label_frac) == 1
            else list(args.label_frac)
        )
    else:
        label_fracs = [None] * len(ring_pairs)

    # -- Load source parameters
    row = None
    snr_n_det = n_det
    posterior_ra = posterior_dec = None
    bilby_label = None

    if args.injection_number is not None:
        if args.stats_file is None:
            raise ValueError("--injection-number requires --stats-file")
        stats = pd.read_csv(args.stats_file, sep="\t", index_col=0)
        row = stats[stats["simulation_id"] == args.injection_number].iloc[0]
        true_ra, true_dec = ast.literal_eval(row["ra_dec"])  # radians
        true_obstime = float(row["time"])
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

    # -- Load skymap
    skymap = None
    if args.skymap_file is not None:
        skymap = skymap_fits.read_sky_map(args.skymap_file, moc=True)
        print(f"Skymap loaded from {args.skymap_file}")

    # -- Precompute timing circles
    rings = {
        (d1, d2): get_ring_w_coloring(
            d1, d2, true_ra, true_dec, true_obstime, resp_func=args.resp_func
        )
        for d1, d2 in ring_pairs
    }
    activated_ifos = set(np.array(ring_pairs).flatten())

    # -- Finally, the actual ploting
    projection = "geo globe" if use_geo else "astro degrees mollweide"

    with rc_context(
        {
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "lines.linewidth": 3,
            "font.family": "sans-serif",
            "font.sans-serif": ["Georgia", "DejaVu Sans"],
            "mathtext.fontset": "cm",
        }
    ):
        fig = plt.figure(figsize=(9, 9) if use_geo else (14, 7))
        ax, back_ax, PLT_ARGS, BACK_PLT_ARGS, _globe_n_view = _setup_axes(
            fig, args, use_geo, projection, true_obstime, true_ra, true_gmst
        )

        # -- Continents
        if args.geo:
            ax.plot(*coastlines(), color="0.5", linewidth=0.5, **PLT_ARGS)
        elif not args.globe:
            plot_continents_icrs(ax, true_gmst)

        # -- Skymap overlay (FITS and/or bilby posterior)
        if skymap is not None:
            plot_skymap(ax, skymap, args.contour_levels, args.plot_freq, show_annotation=True)

        if posterior_ra is not None:
            post_skymap = posterior_to_skymap(
                posterior_ra, posterior_dec, smooth_deg=args.posterior_smooth_deg
            )
            plot_skymap(
                ax, post_skymap, args.contour_levels, args.plot_freq,
                show_annotation=(skymap is None),
            )

        # -- Timing circles
        _ring_colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(ring_pairs)))
        np.random.seed(args.seed)

        if args.resp_func:
            _resp_vals_all = [
                np.sqrt(rings[(d1, d2)][2] ** 2 + rings[(d1, d2)][3] ** 2)
                for d1, d2 in ring_pairs
            ]
            _resp_norm = plt.Normalize(
                min(v.min() for v in _resp_vals_all),
                max(v.max() for v in _resp_vals_all),
            )
            _resp_cmap = plt.get_cmap("viridis")

        for (d1, d2), color, _lf in zip(ring_pairs, _ring_colors, label_fracs):
            ras, decs, F1, F2 = rings[(d1, d2)]
            label = f"{d1}-{d2}"

            # -- Uncertainty band ---------------------------------------------
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
                    jumps_s = np.where(np.abs(np.diff(s_lons)) > 180)[0] + 1

                    if args.resp_func:
                        F1_s, F2_s = compute_antenna_response(
                            d1, d2, s_ras, s_decs, true_obstime
                        )
                        resp_s = np.sqrt(F1_s ** 2 + F2_s ** 2)
                        _jump_set_s = set(jumps_s.tolist())
                        _segs_s, _svals_s = [], []
                        for _i in range(len(s_lons) - 1):
                            if (_i + 1) in _jump_set_s:
                                continue
                            _segs_s.append([
                                (s_lons[_i], s_decs_deg[_i]),
                                (s_lons[_i + 1], s_decs_deg[_i + 1]),
                            ])
                            _svals_s.append((resp_s[_i] + resp_s[_i + 1]) / 2)
                        _svals_s = np.asarray(_svals_s)
                        _lc_s = LineCollection(
                            _segs_s, linewidth=10,
                            alpha=1 / args.n_annulus, zorder=9,
                            transform=ax.get_transform("world"),
                        )
                        _lc_s.set_array(_svals_s)
                        _lc_s.set_cmap(_resp_cmap)
                        _lc_s.set_norm(_resp_norm)
                        ax.add_collection(_lc_s)
                        if back_ax is not None:
                            _lc_s_back = LineCollection(
                                _segs_s, linewidth=10,
                                alpha=0.5 / args.n_annulus, zorder=9,
                                transform=back_ax.get_transform("world"),
                            )
                            _lc_s_back.set_array(_svals_s)
                            _lc_s_back.set_cmap(_resp_cmap)
                            _lc_s_back.set_norm(_resp_norm)
                            back_ax.add_collection(_lc_s_back)
                    else:
                        s_lons_p = np.insert(s_lons.astype(float), jumps_s, np.nan)
                        s_decs_p = np.insert(s_decs_deg.astype(float), jumps_s, np.nan)
                        ax.plot(
                            s_lons_p, s_decs_p,
                            color=color, linewidth=10,
                            alpha=1 / args.n_annulus, zorder=9, **PLT_ARGS,
                        )
                        if back_ax is not None:
                            back_ax.plot(
                                s_lons_p, s_decs_p,
                                color=color, linewidth=10,
                                alpha=0.5 / args.n_annulus, zorder=9, **BACK_PLT_ARGS,
                            )

            # -- Main circle --------------------------------------------------
            lons = (
                np.rad2deg((ras - true_gmst) % (2 * np.pi))
                if use_geo
                else np.rad2deg(ras)
            )
            lats = np.rad2deg(decs)
            jumps = np.where(np.abs(np.diff(lons)) > 180)[0] + 1
            lons_p = np.insert(lons.astype(float), jumps, np.nan)
            lats_p = np.insert(lats.astype(float), jumps, np.nan)

            # -- Check if with or without coloring from response functions
            if args.resp_func:
                resp_vals = np.sqrt(F1 ** 2 + F2 ** 2)
                _jump_set = set(jumps.tolist())
                _segs, _svals = [], []
                for _i in range(len(lons) - 1):
                    if (_i + 1) in _jump_set:
                        continue
                    _segs.append([(lons[_i], lats[_i]), (lons[_i + 1], lats[_i + 1])])
                    _svals.append((resp_vals[_i] + resp_vals[_i + 1]) / 2)
                _svals = np.asarray(_svals)
                _lc = LineCollection(
                    _segs, linewidth=2, zorder=10, transform=ax.get_transform("world")
                )
                _lc.set_array(_svals)
                _lc.set_cmap(_resp_cmap)
                _lc.set_norm(_resp_norm)
                ax.add_collection(_lc)
                if back_ax is not None:
                    _lc_back = LineCollection(
                        _segs, linewidth=2, alpha=0.4, zorder=10,
                        transform=back_ax.get_transform("world"),
                    )
                    _lc_back.set_array(_svals)
                    _lc_back.set_cmap(_resp_cmap)
                    _lc_back.set_norm(_resp_norm)
                    back_ax.add_collection(_lc_back)
            else:
                ax.plot(lons_p, lats_p, linewidth=2, color=color, zorder=10, **PLT_ARGS)
                if back_ax is not None:
                    back_ax.plot(
                        lons_p, lats_p, linewidth=2, color=color,
                        alpha=0.4, zorder=10, **BACK_PLT_ARGS,
                    )

            # -- Inline label
            idx, i0, i1 = _find_label_idx(
                lons, lats, jumps, _lf, _globe_n_view, ax,
                use_geo, true_ra, true_dec, true_gmst,
            )

            # Always compute angle in display space so the rotation matches the
            # rendered ring tangent, regardless of projection distortions or
            # axis inversions (Mollweide inverts x, which flips dlon in geo space).
            _pts = ax.get_transform("world").transform(
                np.array([[lons[i0], lats[i0]], [lons[i1], lats[i1]]])
            )
            if np.all(np.isfinite(_pts)):
                angle = np.rad2deg(
                    np.arctan2(_pts[1, 1] - _pts[0, 1], _pts[1, 0] - _pts[0, 0])
                )
            else:
                dlon = (lons[i1] - lons[i0] + 180) % 360 - 180
                angle = np.rad2deg(np.arctan2(lats[i1] - lats[i0], dlon))

            darker = (
                tuple(c * 0.5 for c in mcolors.to_rgba(color)[:3]) + (1.0,)
                if not args.resp_func
                # else _resp_cmap(0.0)
                else "black"
            )
            ax.text(
                lons[idx], lats[idx], f" {label} ",
                transform=ax.get_transform("world"),
                fontsize=20, ha="center", va="center",
                rotation=angle, rotation_mode="anchor",
                color=darker, fontweight="bold", zorder=20,
            )

        # -- Colorbar for resp-func
        if args.resp_func:
            _sm = plt.cm.ScalarMappable(cmap=_resp_cmap, norm=_resp_norm)
            _sm.set_array([])
            _cb = plt.colorbar(_sm, ax=ax, location="bottom", fraction=0.046, pad=0.04)
            _cb.set_label(r"$\sqrt{F_1^2 + F_2^2}$", fontsize=22)
            # In globe mode ax and back_ax share the same explicit rect; after
            # the colorbar steals space from ax, keep back_ax in sync.
            if back_ax is not None:
                back_ax.set_position(ax.get_position())

        # -- Special markers (true source and detectors)
        src_lon = (
            np.rad2deg((true_ra - true_gmst) % (2 * np.pi))
            if use_geo
            else np.rad2deg(true_ra)
        )
        ax.plot(
            src_lon, np.rad2deg(true_dec), "*",
            markerfacecolor="white", markeredgecolor="black",
            markersize=12, zorder=100, **PLT_ARGS,
        )

        LABEL_ARGS = dict(ha="right", va="bottom", fontsize=12, **PLT_ARGS)
        for name in activated_ifos:
            geo_lon, geo_lat = DETECTOR_POSITION[name]
            if use_geo:
                plot_lon, plot_lat = geo_lon, geo_lat
            else:
                plot_lon = np.rad2deg((np.deg2rad(geo_lon) + true_gmst) % (2 * np.pi))
                plot_lat = geo_lat
            arm_rot = arm_screen_angle(ax, plot_lon, plot_lat, DETECTOR_ARM_AZ[name], geo_lat)
            plot_ifo(ax, plot_lon, plot_lat, size=24, rotation=arm_rot, **PLT_ARGS)
            ax.text(plot_lon, plot_lat, name, **LABEL_ARGS)

        # -- Title
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

        # -- Final layout stuff
        outline_text(ax)
        plt.tight_layout()

        # -- Show and/or save
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
                    + ("_resp" if args.resp_func else "")
                    + ("_geo" if args.geo else "_globe" if args.globe else "")
                    + ".png",
                )
            plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
            print(f"Saved → {out_path}")

        if not args.no_show:
            plt.show()
        else:
            plt.close()
