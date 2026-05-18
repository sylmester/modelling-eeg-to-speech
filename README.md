# Modelling eeg responses to speech stimuli

A bachelor thesis repository for EEG-to-speech modelling using temporal response functions (TRFs) and auditory attention decoding (AAD). The work covers three datasets: Alice, Fuglsang, and SNHL, with analyses of within-dataset performance, cross-dataset generalisation, and cross-population generalisation.

---

## Overview

- `alice/`: data loading, predictor computation, TRF estimation, and decoding analysis for the Alice dataset.
- `fuglsang/`: predictor generation, TRF estimation, and result visualisation for the Fuglsang dataset.
- `beyond/`: cross-dataset and cross-population generalisation analyses using combined outputs from Alice and Fuglsang.
- Root modules (`dependencies.py`, `constants.py`, `paths.py`, `helper_functions.py`) provide shared settings, paths, imports, and utilities.

---

## Repository structure

- `dependencies.py`: shared imports and the `AveragedTRF` class.
- `constants.py`: experiment constants, filter settings, and dataset-specific parameters.
- `paths.py`: dataset paths rooted at `~/Data/modelling-eeg-to-speech/`.
- `helper_functions.py`: common helper functions for loading subjects, predictors, and TRFs.

- `alice/`
  - `eeg-data/`: EEG loading, preprocessing, and trial concatenation.
  - `load_data/`: computation of gammatone spectrograms, envelopes, and onset predictors.
  - `analysis/`: forward and backward TRF estimation and universal decoder creation.
  - `figures/`: decoding statistics and result figures.

- `fuglsang/`
  - `load_data/`: predictor computation for Fuglsang audio.
  - `analysis/`: TRF estimation notebooks for attended/ignored conditions.
  - `figures/`: visualization and statistics notebooks.

- `beyond/`
  - `within-fuglsang/`: pooled and average universal TRF estimation plus analysis.
  - `across-datasets/`
    - `alice-in-fuglsang/`: channel alignment, cross-dataset application, and figures.
    - `fuglsang-in-snhl/`: channel alignment for SNHL and population-level statistics.

---

## Data structure

Data is expected under `~/Data/modelling-eeg-to-speech/`. `paths.py` defines all repository paths and creates directories as needed.

- `alice/`
  - `stimuli/`: raw audio trials.
  - `predictors/`: concatenated predictor objects.
  - `processed-predictors/`: filtered and resampled predictor objects.
  - `eeg/<subject>/`: raw and processed EEG files.
  - `TRFs/`: per-subject TRFs and universal decoder TRFs.

- `fuglsang/`
  - `data_raw/`: original downloaded EEG archives and metadata in expinfo.csv.
  - `data_preprocessed/`: preprocessed EEG files.
  - `stimuli/`: Fuglsang audio files.
  - `eeg/<subject>/`: trimmed EEG data.
  - `TRFs/`: self-computed and original `.mat` predictor TRFs.
  - `predictors/`: self-computed, mat-file-based, and concatenated predictors.
  - `aad-results/`: decoding accuracy outputs.

- `ds-eeg-snhl/`
  - `stimuli/`: audio stimuli.
  - `derivatives/`: gammatone features, envelopes, and onset predictors.
  - `sub-001`: subject EEG
  - ...

- `beyond/`
  - `aad-results/`: cross-population decoding results.
  -

- `figures/`
  - dataset-specific figures for Alice, Fuglsang, SNHL, and beyond analyses.

---

## Setup

Python environment: `3.13`.

Install key dependencies with conda or pip:

```bash
pip install eelbrain mne numpy scipy pandas matplotlib seaborn tqdm
```

Notebooks import shared modules from the repository root with:

```python
import sys
sys.path.insert(0, '../..')
from dependencies import *
from constants import *
from paths import *
import helper_functions
```

---

## Typical workflow

1. Compute predictors from audio.
2. Load and preprocess EEG.
3. Estimate TRFs for each subject.
4. Build universal TRFs.
5. Evaluate decoding and create figures.

### Alice workflow

- `alice/load_data/self-computed-predictors.ipynb`
- `alice/eeg-data/load-data.ipynb`
- `alice/analysis/estimate-trf-encoder.ipynb`
- `alice/analysis/estimate-trf-decoder.ipynb`
- `alice/analysis/decoder-universal-trf.ipynb`
- `alice/figures/statistics-decoder.ipynb`

### Fuglsang workflow

- `fuglsang/load_data/self-computed-predictors.ipynb`
- TRF estimation notebooks for attended/ignored conditions.
- Figures and statistics notebooks.

### Beyond workflows

- `beyond/within-fuglsang/estimate-pooled-universal-trf.ipynb`
- `beyond/within-fuglsang/figures/averaged-trfs-analysis.ipynb`
- `beyond/across-datasets/alice-in-fuglsang/align-alice-trfs.ipynb`
- `beyond/across-datasets/alice-in-fuglsang/analysis.ipynb`
- `beyond/across-datasets/alice-in-fuglsang/figures.ipynb`
- `beyond/across-datasets/fuglsang-in-snhl/align-data.ipynb`
- `beyond/across-datasets/fuglsang-in-snhl/statistics.ipynb`

---

## Dataset sources

- Fuglsang audio and preprocessed EEG are available from Zenodo.
- Alice EEG is from the original Brodbeck et al. study.
- SNHL data is available on Zenodo and should be placed in `ds-eeg-snhl/`.

---

## Notes

- Intermediate results are saved as `.pickle` objects and notebooks generally skip recomputation when files exist.
- Alice TRFs require channel remapping when applied to Fuglsang EEG.
- SNHL data uses TP7/TP8 as the closest available mastoid reference.
- `beyond/` depends on processed outputs from both Alice and Fuglsang analyses.
