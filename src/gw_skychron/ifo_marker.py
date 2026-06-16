#!/usr/bin/env python3
# Copyright (C) 2026 Max Melching
"""Interferometer marker geometry and screen-space rendering."""

import numpy as np
from matplotlib.path import Path
from matplotlib.transforms import Affine2D


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
