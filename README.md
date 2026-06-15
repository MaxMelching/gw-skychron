# gw-skychron

Visualize gravitational-wave sky localization by overlaying timing circles on a sky map. Given a true (or injected) sky position and a detector network, the script draws the iso-delay ring for each detector pair — the locus of sky positions consistent with the measured arrival-time difference. Timing uncertainty annuli and credible-region contours from a HEALPix skymap or a bilby posterior can be overlaid.

## Dependencies

All required packages are available in the [IGWN conda distribution](https://computing.docs.ligo.org/conda/):

```bash
conda create -n igwn-py311 python=3.11
conda activate igwn-py311
conda install -c conda-forge lalsuite ligo.skymap bilby healpy
```

Core runtime dependencies: `lal`, `ligo.skymap`, `healpy`, `numpy`, `scipy`, `pandas`, `matplotlib`, `astropy`. `bilby` is only required when using `--bilby-json`.

## Quick start

**Injection from a stats file + FITS skymap:**

```bash
python plot_timing_circles.py \
    --skymap-file examples/sim_id_4.fits \
    --stats-file examples/combined_stats.dat \
    --injection-number 4 \
    --ring-pairs L1-H1 L1-V1 H1-V1 \
    --timing-uncertainty --timing-sigma-ms 0.25 \
    --contour-levels 50 90
```

![Injection 4 with annuli](examples/timing_circle_inj4_f56_hlv_annulus.png)

**Bilby posterior (KDE skymap, auto-computed timing uncertainty):**

```bash
python plot_timing_circles.py \
    --bilby-json examples/hv_true_snr_fixed_spins_nohom_zero_noise_result.json \
    --detectors H1 V1 \
    --timing-uncertainty \
    --contour-levels 50 90 \
    --timing-uncertainty --n-annulus 96 \
```

![Bilby result](examples/timing_circle_hv_true_snr_fixed_spins_nohom_zero_noise_hlv_annulus.png)

**Globe projection (transparent globe with back-hemisphere circles):**

```bash
python plot_timing_circles.py \
    --stats-file examples/combined_stats.dat \
    --injection-number 4 \
    --plot-freq 56 \
    --ring-pairs L1-H1 L1-V1 H1-V1 \
    --timing-uncertainty --timing-sigma-ms 0.25 \
    --globe
```

![Injection 4 on globe](examples/timing_circle_inj4_f56_hlv_globe.png)

**Direct sky position (no injection lookup):**

```bash
python plot_timing_circles.py \
    --sky-pos 97.2 -35.7 1187008882 \
    --ring-pairs L1-H1 L1-V1 H1-V1 \
    --label-frac 0.42 0.8 0.8 \
    --geo
```

![Manual plot on geo](examples/timing_circle_manual_hlv_geo.png)

## Input formats

| Source flag | Required companion | What it provides |
| --- | --- | --- |
| `--injection-number N` | `--stats-file PATH` | True RA, Dec, GPS from a tab-separated stats CSV; optionally pass `--skymap-file` for a FITS credible-region overlay |
| `--bilby-json PATH` | — | True sky position and GPS from `injection_parameters`; posterior samples rendered as a smoothed HEALPix KDE skymap; per-detector SNRs used for timing uncertainty |
| `--sky-pos RA DEC GPS` | — | Explicit sky position in degrees and GPS seconds; optionally pass `--skymap-file` |

## Key options

| Flag | Default | Description |
| --- | --- | --- |
| `--detectors D1 D2 …` | — | All pairwise combinations become ring pairs |
| `--ring-pairs D1-D2 …` | `L1-H1 L1-V1 H1-V1` | Explicit detector pairs |
| `--timing-uncertainty` | off | Draw uncertainty annuli around each ring |
| `--timing-sigma-ms MS` | auto | Override timing σ [ms]; auto uses SNR + effective bandwidth |
| `--n-annulus N` | 50 | Number of sampled rings per pair |
| `--posterior-smooth-deg DEG` | 1.5 | Gaussian smoothing width for bilby KDE skymap; set to 0 to disable |
| `--contour-levels PCT …` | `50 90` | Credible-region contour levels [%] |
| `--geo` | off | Orthographic geo-globe projection centred on the source |
| `--globe` | off | Like `--geo` but with a transparent globe surface and back-hemisphere circles shown through it |
| `--label-frac F [F …]` | auto | Override label position: fraction of circle clockwise from source (0 = source, 0.5 = opposite side). One value for all pairs or one per pair (requires `--ring-pairs`) |
| `--outdir DIR` | script dir | Output directory for auto-named PNG |
| `--output PATH` | — | Explicit output path (overrides auto-naming) |
| `--no-show` | off | Skip interactive window (useful for batch runs) |

Run `python plot_timing_circles.py --help` for the full reference.
