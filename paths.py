from dependencies import *

# =============================================================================
# ROOTS
# =============================================================================

DATA_ROOT       = Path("~").expanduser() / 'Data' / 'modelling-eeg-to-speech'

ALICE_ROOT      = DATA_ROOT / 'alice'
FUGLSANG_ROOT   = DATA_ROOT / 'fuglsang'
SNHL_ROOT       = DATA_ROOT / 'ds-eeg-snhl'
BEYOND_ROOT      = DATA_ROOT / 'beyond'


# =============================================================================
# COMMON DIRS
# =============================================================================
FIGURES_DIR             = DATA_ROOT     / 'figures'
ALICE_FIGURES_DIR       = FIGURES_DIR   / 'alice'
FUGLSANG_FIGURES_DIR    = FIGURES_DIR   / 'fuglsang'
SNHL_FIGURES_DIR        = FIGURES_DIR   / 'snhl'
BEYOND_FIGURES_DIR      = FIGURES_DIR   / 'beyond'



# =============================================================================
# ALICE
# =============================================================================

ALICE_STIMULUS_DIR              = ALICE_ROOT    / 'stimuli'
ALICE_ENVELOPES_DIR             = ALICE_ROOT    / 'envelopes'
ALICE_PREDICTOR_DIR             = ALICE_ROOT    / 'predictors'
ALICE_PROCESSED_PREDICTOR_DIR   = ALICE_ROOT    / 'processed-predictors'
ALICE_EEG_DIR                   = ALICE_ROOT    / 'eeg'
ALICE_TRF_DIR                   = ALICE_ROOT    / 'TRFs'
ALICE_GENERAL_TRF_DIR           = ALICE_TRF_DIR / 'generalised'

# =============================================================================
# FUGLSANG
# =============================================================================

FUGLSANG_DATA_RAW               = FUGLSANG_ROOT / 'data_raw'
FUGLSANG_DATA_PREPROC           = FUGLSANG_ROOT / 'data_preprocessed'
FUGLSANG_STIMULUS_DIR           = FUGLSANG_ROOT / 'stimuli'
FUGLSANG_ENVELOPES_DIR          = FUGLSANG_ROOT / 'envelopes'
FUGLSANG_EEG_DIR                = FUGLSANG_ROOT / 'eeg'

FUGLSANG_TRF_DIR                = FUGLSANG_ROOT / 'TRFs'
FUGLSANG_TRF_SELF_DIR           =   FUGLSANG_TRF_DIR / 'self_computed'
FUGLSANG_TRF_GENERAL_DIR        =     FUGLSANG_TRF_SELF_DIR / 'generalised'
FUGLSANG_TRF_MAT_DIR            =   FUGLSANG_TRF_DIR / 'mat_file'

FUGLSANG_PREDICTOR_DIR          = FUGLSANG_ROOT / 'predictors'
FUGLSANG_PRED_SELF_DIR          =   FUGLSANG_PREDICTOR_DIR / 'self_computed'
FUGLSANG_PRED_MAT_DIR           =   FUGLSANG_PREDICTOR_DIR / 'mat_file'
FUGLSANG_PRED_CONCAT_DIR        =   FUGLSANG_PREDICTOR_DIR / 'concatenated'
FUGLSANG_PRED_CONCAT_SELF_DIR   =     FUGLSANG_PRED_CONCAT_DIR / 'self_computed'
FUGLSANG_PRED_CONCAT_MAT_DIR    =     FUGLSANG_PRED_CONCAT_DIR / 'mat_file'

FUGLSANG_AAD_RESULTS               = FUGLSANG_ROOT / 'aad-results'


# =============================================================================
# SNHL
# =============================================================================

SNHL_STIMULUS_DIR               = SNHL_ROOT / 'derivatives' / 'stimuli'


# =============================================================================
# BEYOND
# =============================================================================

BEYOND_AAD_RESULTS               = BEYOND_ROOT / 'aad-results'
BEYOND_FUGLSANG_IN_SNHL_RESULTS  = BEYOND_AAD_RESULTS / 'fuglsang_in_snhl'

# =============================================================================
# MKDIR
# =============================================================================

for _dir in [
    # Common
    FIGURES_DIR, ALICE_FIGURES_DIR, FUGLSANG_FIGURES_DIR, SNHL_FIGURES_DIR, BEYOND_FIGURES_DIR,
    # Alice
    ALICE_STIMULUS_DIR, ALICE_PREDICTOR_DIR, ALICE_PROCESSED_PREDICTOR_DIR,
    ALICE_EEG_DIR, ALICE_TRF_DIR, ALICE_GENERAL_TRF_DIR,
    # Fuglsang
    FUGLSANG_DATA_PREPROC, FUGLSANG_STIMULUS_DIR, FUGLSANG_ENVELOPES_DIR, FUGLSANG_EEG_DIR,
    FUGLSANG_TRF_SELF_DIR, FUGLSANG_TRF_GENERAL_DIR, FUGLSANG_TRF_MAT_DIR,
    FUGLSANG_PRED_SELF_DIR, FUGLSANG_PRED_MAT_DIR, FUGLSANG_PRED_CONCAT_SELF_DIR, FUGLSANG_PRED_CONCAT_MAT_DIR,
    # SNHL
    SNHL_STIMULUS_DIR,
    # BEYOND
    BEYOND_AAD_RESULTS, BEYOND_FUGLSANG_IN_SNHL_RESULTS
]:
    _dir.mkdir(parents=True, exist_ok=True)