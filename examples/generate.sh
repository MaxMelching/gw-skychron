#!/bin/bash

cd "$(dirname "$0")"


# -- sim_id_4. First Mollweide, then geo, then globe.
gw-skychron \
    --skymap-file sim_id_4.fits \
    --stats-file combined_stats.dat \
    --injection-number 4 \
    --plot-freq 56 \
    --ring-pairs L1-H1 L1-V1 H1-V1 \
    --contour-levels 50 90 \
    --resp-func \
    --outdir . \
    --no-show

gw-skychron \
    --skymap-file sim_id_4.fits \
    --stats-file combined_stats.dat \
    --injection-number 4 \
    --plot-freq 56 \
    --ring-pairs L1-H1 L1-V1 H1-V1 \
    --contour-levels 50 90 \
    --resp-func \
    --geo \
    --outdir . \
    --no-show

gw-skychron \
    --stats-file combined_stats.dat \
    --injection-number 4 \
    --plot-freq 56 \
    --ring-pairs L1-H1 L1-V1 H1-V1 \
    --resp-func \
    --timing-uncertainty --timing-sigma-ms 0.25 \
    --globe \
    --outdir . \
    --no-show

# Still sim_id_4, now with timing uncertainty
gw-skychron \
    --skymap-file sim_id_4.fits \
    --stats-file combined_stats.dat \
    --injection-number 4 \
    --plot-freq 56 \
    --ring-pairs L1-H1 L1-V1 H1-V1 \
    --timing-uncertainty \
    --timing-sigma-ms 0.25 \
    --n-annulus 67 \
    --contour-levels 50 90 \
    --outdir . \
    --no-show

gw-skychron \
    --skymap-file sim_id_4.fits \
    --stats-file combined_stats.dat \
    --injection-number 4 \
    --plot-freq 56 \
    --ring-pairs L1-H1 L1-V1 H1-V1 \
    --timing-uncertainty \
    --timing-sigma-ms 0.25 \
    --n-annulus 96 \
    --contour-levels 50 90 \
    --geo \
    --outdir . \
    --no-show

# -- sim_id_25. Very similar idea to sim_id_4 plots.
# -- Note that this might be an event mainly seen by two of the detectors,
# -- hence the long circle (for three, it would be more circular).
gw-skychron \
    --skymap-file sim_id_25.fits \
    --stats-file combined_stats.dat \
    --injection-number 25 \
    --plot-freq 1024 \
    --ring-pairs L1-H1 L1-V1 H1-V1 \
    --label-frac 0.34 0.8 0.8 \
    --timing-uncertainty \
    --timing-sigma-ms 0.25 \
    --n-annulus 67 \
    --contour-levels 50 90 \
    --outdir . \
    --no-show

gw-skychron \
    --skymap-file sim_id_25.fits \
    --stats-file combined_stats.dat \
    --injection-number 25 \
    --plot-freq 1024 \
    --ring-pairs L1-H1 L1-V1 H1-V1 \
    --label-frac 0.34 0.8 0.2 \
    --timing-uncertainty \
    --timing-sigma-ms 0.25 \
    --n-annulus 96 \
    --contour-levels 50 90 \
    --geo \
    --outdir . \
    --no-show


# -- bilby file
gw-skychron \
    --bilby-json hv_true_snr_fixed_spins_nohom_zero_noise_result.json \
    --ring-pairs H1-V1 \
    --contour-levels 50 90 \
    --resp-func \
    --outdir . \
    --no-show


# -- Manual specification of true location
gw-skychron \
    --sky-pos 97.2 -35.7 1187008882 \
    --ring-pairs L1-H1 L1-V1 H1-V1 \
    --label-frac 0.42 0.8 0.8 \
    --geo \
    --outdir . \
    --no-show
