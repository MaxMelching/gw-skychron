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


def get_ring_w_coloring(det1, det2, ra, dec, t_event):
    """Return (ras, decs, F1, F2) for the timing circle of a detector pair."""
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
