from dependencies import *

# =============================================================================
# SIGNAL PROCESSING
# =============================================================================

GAMMATONE_SPECTROGRAM_BANDS      = 128
GAMMATONE_FREQUENCY_RANGE_LOW    = 80
GAMMATONE_FREQUENCY_RANGE_HIGH   = 15000

EEG_SAMPLING_RATE                = 64
BANDPASS_FILTER_LOW          = 0.5
BANDPASS_FILTER_HIGH         = 20

PADDING_ONSET                    = 0.1
PADDING_OFFSET                   = 1.0

# =============================================================================
# TRF
# =============================================================================

TRF_LAG_START                    = -0.100
TRF_LAG_END                      =  1.000
BASIS_FUNCTION_WIDTH             =  0.050




# =============================================================================
# ENUM CLASSES
# =============================================================================

class PREDICTOR_TYPE(Enum):
    ENVELOPE = "envelope"
    ENVELOPE_ONSET = "envelope_onset"

class ATTENTION_TYPE(Enum):
    ATTENDED = "attended"
    IGNORED  = "ignored"
    SINGLE = ""

class MODEL_TYPE(Enum):
    FORWARD  = "forward"
    BACKWARD = "backward"

class GENERALISATION_TYPE(Enum):
    POOLED = "pooled"
    AVERAGE = "averaged"
    INDIVIDUAL = "individual"

class CROSS_VALIDATION_TYPE(Enum):
    HOLD_OUT = "hold_out"
    LOO = "loocv"

class DATASET_TYPE(Enum):
    FUGLSANG = "fuglsang"
    ALICE = "alice"
    SNHL = "snhl"

class AAD_APPROACH(Enum):
    SINGLE = "single"
    DOUBLE = "double"



# =============================================================================
# DOWNLOAD URLS
# =============================================================================

# FUGLSANG
AUDIO_URL        = 'https://zenodo.org/record/1199011/files/AUDIO.zip'
DATA_PREPROC_URL = 'https://zenodo.org/record/1199011/files/DATA_preproc.zip'

# ALICE

# SNHL


