#!/bin/bash

cd "$(dirname "$0")"


python ../plot_timing_circles.py \
    --skymap-file sim_id_4.fits \
    --stats-file combined_stats.dat \
    --injection-number 4 \
    --plot-freq 56 \
    --ring-pairs L1-H1 L1-V1 H1-V1 \
    --contour-levels 50 90 \
    --outdir . \
    --no-show

python ../plot_timing_circles.py \
    --skymap-file sim_id_4.fits \
    --stats-file combined_stats.dat \
    --injection-number 4 \
    --plot-freq 56 \
    --ring-pairs L1-H1 L1-V1 H1-V1 \
    --contour-levels 50 90 \
    --geo \
    --outdir . \
    --no-show

python ../plot_timing_circles.py \
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

python ../plot_timing_circles.py \
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

python ../plot_timing_circles.py \
    --skymap-file sim_id_25.fits \
    --stats-file combined_stats.dat \
    --injection-number 25 \
    --plot-freq 1024 \
    --ring-pairs L1-H1 L1-V1 H1-V1 \
    --timing-uncertainty \
    --timing-sigma-ms 0.25 \
    --n-annulus 67 \
    --contour-levels 50 90 \
    --outdir . \
    --no-show

python ../plot_timing_circles.py \
    --skymap-file sim_id_25.fits \
    --stats-file combined_stats.dat \
    --injection-number 25 \
    --plot-freq 1024 \
    --ring-pairs L1-H1 L1-V1 H1-V1 \
    --timing-uncertainty \
    --timing-sigma-ms 0.25 \
    --n-annulus 96 \
    --contour-levels 50 90 \
    --geo \
    --outdir . \
    --no-show

# -- Might have mainly been a two-detector injection


python ../plot_timing_circles.py \
    --bilby-json hv_true_snr_fixed_spins_nohom_zero_noise_result.json \
    --ring-pairs L1-H1 L1-V1 H1-V1 \
    --contour-levels 50 90 \
    --timing-uncertainty --n-annulus 96 \
    --outdir . \
    --no-show


python ../plot_timing_circles.py \
    --sky-pos 97.2 -35.7 1187008882 \
    --ring-pairs L1-H1 L1-V1 H1-V1 \
    --geo \
    --outdir . \
