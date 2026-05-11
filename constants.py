from dependencies import *

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


