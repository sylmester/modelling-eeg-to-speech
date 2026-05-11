# IMPORT LIBRARIES
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import eelbrain
import mne
import re
import os
import pandas as pd
import scipy
import copy

# IMPORT MODULES
from pathlib import Path
from matplotlib.patches import ConnectionPatch, Patch
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec
from matplotlib import colors
from tqdm import tqdm


from scipy import stats
from scipy.optimize import linear_sum_assignment
from enum import Enum



# =============================================================================
# SPECIAL CLASSES
# =============================================================================
class AveragedTRF:
    """
        Wraps only the averaged h_scaled, blocking access to stale BoostingResult fields.
        Used to keep averaged TRFs in a class like BoostingResult but without fields which
        would have stale data.
    """

    def __init__(self, h_scaled):
        self.h_scaled = h_scaled

    def __getattr__(self, name):
        raise AttributeError(
            f"'{name}' is not available on AveragedTRF — only 'h_scaled' is valid."
        )


