"""Sky-map and posterior rendering utilities."""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from astropy.table import Table
import astropy.units as u
from ligo.skymap import moc as lsm_moc
from ligo.skymap import postprocess as lsm_postprocess
from ligo.skymap.plot.backdrop import coastlines


def _format_area(area):
    if area <= 100:
        return np.format_float_positional(area, precision=3, fractional=False, trim="-")
    else:
        return f"{np.round(area).astype(int):,d}"


def plot_skymap(ax, sm, contour_levels, plot_freq, show_annotation):
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
            0.89,
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


def posterior_to_skymap(ra, dec, smooth_deg=1.5, nside=128):
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


def plot_continents_icrs(ax, gmst):
    """Draw geographic coastlines transformed into ICRS (Mollweide) frame."""
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
