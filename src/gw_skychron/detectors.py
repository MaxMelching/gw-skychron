#!/usr/bin/env python3
# Copyright (C) 2026 Max Melching
"""Detector configuration, timing-circle computation, and timing uncertainty."""

import lal
import numpy as np

from .interferometer import interferometer


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


def compute_antenna_response(det1, det2, ras, decs, t_event):
    """Return (F1, F2) arrays — sqrt(F+²+Fx²) at each (ra, dec) [radians].

    Fully vectorized over N sky positions (no Python loop).  Assumes psi=0,
    which is the only value ever used here.

    Wave-frame vectors for psi=0 (arXiv:gr-qc/0008066, appendix B):
        X = [ sin φ,  -cos φ,   0       ]
        Y = [-cos φ sin δ, -sin φ sin δ, cos δ]
    where φ = ra - gmst, δ = dec.

    Antenna patterns:
        F+ = D:( XX - YY )   →   (XD*X - YD*Y).sum(axis=1)
        F× = D:( XY + YX )   →   2*(XD*Y).sum(axis=1)   (D symmetric)
    """
    gmst = lal.GreenwichMeanSiderealTime(t_event) % (2 * np.pi)
    phi = np.asarray(ras) - gmst        # shape (N,)
    dec = np.asarray(decs)              # shape (N,)

    # Wave-frame basis vectors, shape (N, 3)
    X = np.column_stack([np.sin(phi), -np.cos(phi), np.zeros(len(phi))])
    Y = np.column_stack([-np.cos(phi) * np.sin(dec),
                         -np.sin(phi) * np.sin(dec),
                          np.cos(dec)])

    def _F(det):
        D = detectors[det].detector_tensor          # (3, 3)
        XD = X @ D                                  # (N, 3)
        YD = Y @ D                                  # (N, 3)
        Fp = (XD * X).sum(axis=1) - (YD * Y).sum(axis=1)
        Fc = 2.0 * (XD * Y).sum(axis=1)
        return np.sqrt(Fp ** 2 + Fc ** 2)

    return _F(det1), _F(det2)


def get_ring_w_coloring(det1, det2, ra, dec, t_event, resp_func=False):
    """Return (ras, decs, F1, F2) for the timing circle of a detector pair.

    When resp_func=False (default), F1 and F2 are None and the antenna
    response calculation is skipped entirely.

    When resp_func=True, antenna response is evaluated at psi=0 (GW
    polarization angle).  We do not average over psi because the ring shape
    is polarization-independent and the coloring is intended as a qualitative
    orientation guide, not a detection statistic.
    """
    time_delay = detectors[det1].time_delay(
        detectors[det2].vertex, ra=ra, dec=dec, t_event=t_event
    )
    possible_ras, possible_decs = detectors[det1].sky_location(
        detectors[det2].vertex, time_delay=time_delay, t_event=t_event
    )
    possible_ras = possible_ras % (2 * np.pi)
    if resp_func:
        F1, F2 = compute_antenna_response(det1, det2, possible_ras, possible_decs, t_event)
    else:
        F1 = F2 = None
    return possible_ras, possible_decs, F1, F2


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
