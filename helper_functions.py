from dependencies import *
from constants import *
from paths import *


# =============================================================================
# GLOBAL FUNCTIONS
# =============================================================================

def load_trfs(dataset, subjects, checks, trf_dir):
    """
    Load TRFs for all subjects and checks.

    checks: list of tuples (predictor, attention, model_type) 
            or (predictor, attention, model_type, padded)
            padded defaults to False if not provided.

    Returns:
        trf_data:   dict keyed by model_name -> list of TRFs (one per subject)
        n_subjects: number of subjects successfully loaded
    """

    def unpack_check(check):
        """Unpack check tuple, defaulting padded to False if not provided."""
        if len(check) == 4:
            p, a, m, padded = check
        else:
            p, a, m = check
            padded = False
        return p, a, m, padded

    # unpack all checks upfront
    unpacked = [unpack_check(c) for c in checks]

    g = GENERALISATION_TYPE.INDIVIDUAL
    trf_data = {get_trf_model_name(dataset, p, a, m, g, padded): [] 
                for p, a, m, padded in unpacked}
    skipped  = []
    loaded   = []

    for subject in subjects:
        missing = []
        for p, a, m, padded in unpacked:
            name = get_trf_model_name(dataset, p, a, m, g, padded)
            if dataset == DATASET_TYPE.FUGLSANG:
                path = trf_dir / subject / f"{subject}_{name}_trf.pickle"
            elif dataset == DATASET_TYPE.ALICE:
                path = trf_dir / subject / f"{subject} 64hz-{name}.pickle"
            if not path.exists():
                missing.append(name)

        if missing:
            print(f"  ✗ {subject}: skipping — missing {len(missing)} TRF(s):")
            for name in missing:
                print(f"      - {name}")
            skipped.append(subject)
            continue

        for p, a, m, padded in unpacked:
            name = get_trf_model_name(dataset, p, a, m, g, padded)
            if dataset == DATASET_TYPE.FUGLSANG:
                path = trf_dir / subject / f"{subject}_{name}_trf.pickle"
            elif dataset == DATASET_TYPE.ALICE:
                path = trf_dir / subject / f"{subject} 64hz-{name}.pickle"
            trf_data[name].append(eelbrain.load.unpickle(path))

        loaded.append(subject)
        print(f"  ✓ {subject}")

    print(f"\nLoaded: {len(loaded)} subjects | Skipped: {len(skipped)} subjects")
    if skipped:
        print(f"  Skipped: {skipped}")

    return trf_data, len(loaded)  


def get_predictor_name(predictors, padded=False) -> str:
    """
    Format:
        <predictor1+predictor2+...>[ _padded ]

    Example:
        envelope+envelope_onset_padded
    """
    if isinstance(predictors, PREDICTOR_TYPE):
        predictors = [predictors]

    predictors = sorted(predictors, key=lambda p: p.value)
    name = "+".join(p.value for p in predictors)

    if padded:
        name += "_padded"

    return name

def get_attentional_predictor_name(predictors, attention: ATTENTION_TYPE, padded=False) -> str:
    """
    Format:
        <attention_type>_<predictor_combination>

    Example:
        attended_envelope+envelope_onset_padded
    """
    return f"{attention.value}_{get_predictor_name(predictors, padded)}"

    

def get_trf_model_name(
    dataset: DATASET_TYPE,
    predictors: PREDICTOR_TYPE | list[PREDICTOR_TYPE],
    attention: ATTENTION_TYPE,
    model: MODEL_TYPE,
    generalised: GENERALISATION_TYPE = GENERALISATION_TYPE.INDIVIDUAL,
    padded: bool = False
):
    """
    Generate standardized TRF model names.

    Format:
        [<generalisation_type>]_<model_type>_<trf_type>_<predictor1+predictor2>[ _padded ]

    Example:
        backward_attended_envelope+envelope_onset_padded
    """
    if isinstance(predictors, PREDICTOR_TYPE):
        predictors = [predictors]

    predictors = sorted(predictors, key=lambda p: p.value)
    predictor_names = "+".join(
        map_predictor_name(p, dataset) for p in predictors
    )

    if dataset == DATASET_TYPE.FUGLSANG:
        name = f"{model.value}_{attention.value}_{predictor_names}"
        if padded:
            name += "_padded"

    elif dataset == DATASET_TYPE.ALICE:
        parts = []
        if model == MODEL_TYPE.BACKWARD:
            parts.append("decoder")
        parts.append(predictor_names)
        name = "-".join(parts)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Prepend generalisation prefix if not individual
    if generalised != GENERALISATION_TYPE.INDIVIDUAL:
        name = f"{generalised.value}_{name}"

    return name



def map_predictor_name(predictor: PREDICTOR_TYPE, dataset: DATASET_TYPE):
    if dataset == DATASET_TYPE.FUGLSANG:
        return predictor.value

    elif dataset == DATASET_TYPE.ALICE:
        mapping = {
            PREDICTOR_TYPE.ENVELOPE: "envelope_log",
            PREDICTOR_TYPE.ENVELOPE_ONSET: "envelope_onset",
        }
        return mapping[predictor]

    else:
        raise ValueError(f"Unknown dataset: {dataset}")






# =============================================================================
# AAD FUNCTIONS
# =============================================================================


def aad_single_classifier(eeg, true_att, true_ign, trf):
    """
    One generic TRF: reconstruct once, correlate against both stimuli.
    Returns True if reconstruction correlates more with attended stimulus.
    """
    pred = eelbrain.convolve(trf.h_scaled, eeg).x

    att = true_att.x if hasattr(true_att, 'x') else np.asarray(true_att)
    ign = true_ign.x if hasattr(true_ign, 'x') else np.asarray(true_ign)

    # convolve can produce pred that is 1 sample shorter than the envelopes
    n    = min(pred.shape[-1], att.shape[-1], ign.shape[-1])
    pred = pred[..., :n]
    att  = att[...,  :n]
    ign  = ign[...,  :n]

    r_att = np.abs(np.corrcoef(pred, att)[0, 1])
    r_ign = np.abs(np.corrcoef(pred, ign)[0, 1])
    #r_att = (np.corrcoef(pred, att)[0, 1])
    #r_ign = (np.corrcoef(pred, ign)[0, 1])
    return r_att > r_ign, r_att, r_ign


def aad_double_classifier(eeg, true_att, true_ign, att_trf, ign_trf):
    """
    Two condition-specific TRFs: reconstruct attended and ignored separately,
    correlate each with its respective true stimulus.
    Returns True if attended reconstruction wins.
    """
    pred_att = eelbrain.convolve(att_trf.h_scaled, eeg).x
    pred_ign = eelbrain.convolve(ign_trf.h_scaled, eeg).x

    att = true_att.x if hasattr(true_att, 'x') else np.asarray(true_att)
    ign = true_ign.x if hasattr(true_ign, 'x') else np.asarray(true_ign)

    # each convolve call can drift independently, so trim each pair separately
    n_att = min(pred_att.shape[-1], att.shape[-1])
    n_ign = min(pred_ign.shape[-1], ign.shape[-1])

    r_att = np.abs(np.corrcoef(pred_att[..., :n_att], att[..., :n_att])[0, 1])
    r_ign = np.abs(np.corrcoef(pred_ign[..., :n_ign], ign[..., :n_ign])[0, 1])

    return r_att > r_ign, r_att, r_ign



def aad_single_classifier_forward(att_pred, ign_pred, eeg, trf):
    """
    Single TRF forward AAD classifier.
    Convolves TRF with attended and ignored predictors to reconstruct EEG,
    correlates each reconstruction with true EEG per sensor, then averages.

    Parameters
    ----------
    att_pred : NDVar
        Attended predictor for this trial.
    ign_pred : NDVar
        Ignored predictor for this trial.
    eeg : NDVar
        True EEG for this trial (sensors x time).
    trf : BoostingResult
        Estimated forward TRF.

    Returns
    -------
    correct : bool
    r_att : float
    r_ign : float
    """
    rec_att = eelbrain.convolve(trf.h_scaled, att_pred).x
    rec_ign = eelbrain.convolve(trf.h_scaled, ign_pred).x

    eeg_data = eeg.get_data(('sensor', 'time'))  # (n_sensors, n_time)

    # align lengths
    n = min(eeg_data.shape[1], rec_att.shape[-1], rec_ign.shape[-1])
    eeg_data = eeg_data[:, :n]
    rec_att  = rec_att[..., :n]
    rec_ign  = rec_ign[..., :n]

    # correlate per sensor then average
    r_att = float(np.mean([
        np.abs(np.corrcoef(eeg_data[i], rec_att[i])[0, 1])
        for i in range(eeg_data.shape[0])
    ]))
    r_ign = float(np.mean([
        np.abs(np.corrcoef(eeg_data[i], rec_ign[i])[0, 1])
        for i in range(eeg_data.shape[0])
    ]))

    correct = r_att > r_ign
    return correct, r_att, r_ign


def aad_double_classifier_forward(att_pred, ign_pred, eeg, att_trf, ign_trf):
    """
    Works for both single and multivariate forward TRFs.
    att_pred and ign_pred can be NDVar or list of NDVar.
    """
    # eelbrain.convolve accepts a list of NDVars for multivariate TRFs
    rec_att = eelbrain.convolve(att_trf.h_scaled, att_pred).x
    rec_ign = eelbrain.convolve(ign_trf.h_scaled, ign_pred).x

    eeg_data = eeg.get_data(('sensor', 'time'))

    n = min(eeg_data.shape[1], rec_att.shape[-1], rec_ign.shape[-1])
    eeg_data = eeg_data[:, :n]
    rec_att  = rec_att[..., :n]
    rec_ign  = rec_ign[..., :n]

    r_att = float(np.mean([
        np.abs(np.corrcoef(eeg_data[i], rec_att[i])[0, 1])
        for i in range(eeg_data.shape[0])
    ]))
    r_ign = float(np.mean([
        np.abs(np.corrcoef(eeg_data[i], rec_ign[i])[0, 1])
        for i in range(eeg_data.shape[0])
    ]))

    correct = r_att > r_ign
    return correct, r_att, r_ign





def aad_classifier(predictors, subjects, 
                   generalised=GENERALISATION_TYPE.AVERAGE, 
                   cv=CROSS_VALIDATION_TYPE.HOLD_OUT, aad_type = AAD_APPROACH.SINGLE):

    att_predictor_name = get_attentional_predictor_name(predictors, ATTENTION_TYPE.ATTENDED)
    ign_predictor_name = get_attentional_predictor_name(predictors, ATTENTION_TYPE.IGNORED)

    att_trf_name = get_trf_model_name(DATASET_TYPE.FUGLSANG, predictors, ATTENTION_TYPE.ATTENDED, MODEL_TYPE.BACKWARD, generalised=generalised)

    if (aad_type == AAD_APPROACH.DOUBLE):
        ign_trf_name = get_trf_model_name(DATASET_TYPE.FUGLSANG, predictors, ATTENTION_TYPE.IGNORED,  MODEL_TYPE.BACKWARD, generalised=generalised)

    # For hold-out, one TRF for all subjects — load once
    if cv == CROSS_VALIDATION_TYPE.HOLD_OUT:
        trf_att = eelbrain.load.unpickle(FUGLSANG_TRF_GENERAL_DIR / f'hold_out_{att_trf_name}.pickle')
        if (aad_type == AAD_APPROACH.DOUBLE):
            trf_ign = eelbrain.load.unpickle(FUGLSANG_TRF_GENERAL_DIR / f'hold_out_{ign_trf_name}.pickle')

    decisions = {}
    r_atts    = {}
    r_igns    = {}

    for subject in subjects:

        # For LOO, load the TRF that excluded this subject
        if cv == CROSS_VALIDATION_TYPE.LOO:
            trf_att = eelbrain.load.unpickle(FUGLSANG_TRF_GENERAL_DIR / f'loocv_{subject}_{att_trf_name}.pickle')
            if (aad_type == AAD_APPROACH.DOUBLE):
                trf_ign = eelbrain.load.unpickle(FUGLSANG_TRF_GENERAL_DIR / f'loocv_{subject}_{ign_trf_name}.pickle')

        eeg = eelbrain.load.unpickle(
            FUGLSANG_EEG_DIR / subject / f'{subject}_eeg.pickle'
        )
        true_att = eelbrain.load.unpickle(
            FUGLSANG_PRED_CONCAT_SELF_DIR / subject / f'{att_predictor_name}_concat.pickle'
        ).x
        true_ign = eelbrain.load.unpickle(
            FUGLSANG_PRED_CONCAT_SELF_DIR / subject / f'{ign_predictor_name}_concat.pickle'
        ).x

        if (aad_type == AAD_APPROACH.DOUBLE):
            decision, r_att, r_ign = aad_double_classifier(eeg, true_att, true_ign, trf_att, trf_ign)
        else:
            decision, r_att, r_ign = aad_single_classifier(eeg, true_att, true_ign, trf_att)
        decisions[subject] = decision
        r_atts[subject]    = r_att
        r_igns[subject]    = r_ign
        print(f"{subject}: r_att={r_att:.3f}, r_ign={r_ign:.3f}")

    acc = sum(decisions.values()) / len(subjects)
    print(f"\n✅ Classification rate ({predictors}): {acc:.2%}")
    print('\n' + '─' * 60 + '\n')

    return acc, decisions, r_atts, r_igns






# =============================================================================
# FUGLSANG FUNCTIONS
# =============================================================================

# Utility function to get subject list 
def fuglsang_get_subjects():
    subjects = [path.stem.split("_")[0] for path in FUGLSANG_DATA_PREPROC.glob("*.mat")]
    subjects = sorted(subjects, key=lambda x: int(re.search(r'S(\d+)', x).group(1)))
    return subjects

def get_subject_data_file(subject: str):
    return FUGLSANG_DATA_PREPROC / f"{subject}_data_preproc.mat"

# Utility function to get stimulus paths --------------------------------------------------
def get_stimuli_paths():
    return [stimulus.stem for stimulus in FUGLSANG_STIMULUS_DIR.glob("*.wav")]

def get_trials(subject):
    csv_path = os.path.join(FUGLSANG_DATA_RAW, f'{subject}_expinfo.csv')
    
    if not os.path.exists(csv_path):
        print(f"Missing expinfo CSV for {subject}: {csv_path}")
        return {}

    expinfo = pd.read_csv(csv_path)
    trials  = {}
    trial_idx = 0

    for _, row in expinfo.iterrows():
        if row['n_speakers'] == 1:
            continue

        if row['attend_mf'] == 1:
            attended_wavfile = Path(row['wavfile_male']).stem.strip("'\"")
            ignored_wavfile  = Path(row['wavfile_female']).stem.strip("'\"")
        else:
            attended_wavfile = Path(row['wavfile_female']).stem.strip("'\"")
            ignored_wavfile  = Path(row['wavfile_male']).stem.strip("'\"")

        trials[trial_idx] = {
            'attended': attended_wavfile,
            'ignored':  ignored_wavfile
        }
        trial_idx += 1

    print(f"{subject}: loaded {len(trials)} trials")
    return trials

# =============================================================================
# ALICE FUNCTIONS
# =============================================================================

# Utility function to get subject list
def alice_get_subjects():
    subjects = [path.name for path in ALICE_EEG_DIR.iterdir() if path.is_dir()]
    subjects = sorted(subjects, key=lambda x: int(re.search(r'S(\d+)', x).group(1)))
    return subjects

def alice_get_durations(envelope, STIMULI):
    durations = [gt.time.tmax for stimulus, gt in zip(STIMULI, envelope)]
    return durations


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def set_plot_style():
    FONT      = 'Arial'
    FONT_SIZE = 8
    RC = {
        'figure.dpi':          100,
        'savefig.dpi':         300,
        'savefig.transparent': True,
        'font.family':         'sans-serif',
        'font.sans-serif':     FONT,
        'font.size':           FONT_SIZE,
        'figure.labelsize':    FONT_SIZE,
        'figure.titlesize':    FONT_SIZE,
        'axes.labelsize':      FONT_SIZE,
        'axes.titlesize':      FONT_SIZE,
        'xtick.labelsize':     FONT_SIZE,
        'ytick.labelsize':     FONT_SIZE,
        'legend.fontsize':     FONT_SIZE,
    }
    plt.rcParams.update(RC)


# Utility function to get significance marker based on p-value -----------
def sig_marker(p):
        if p < 0.001:  return '***'
        elif p < 0.01: return '**'
        elif p < 0.05: return '*'
        else:          return 'n.s.'

# Utility function to add significance lines to plots -------
def add_sig_line(ax, x1, x2, y, text, color='k'):
    ax.plot([x1, x1, x2, x2], [y, y + 0.0005, y + 0.0005, y], color=color, linewidth=1)
    ax.text((x1 + x2) / 2, y + 0.0005, text, ha='center', va='bottom', fontsize=10)

def lighten_color(color, amount=0.7):
    """
    Lighten a matplotlib color.

    Parameters
    ----------
    color : str or tuple
        Any matplotlib-compatible color.
    amount : float
        0 → original color
        1 → white

    Returns
    -------
    tuple
        Lightened RGB color
    """
    c = colors.to_rgb(color)
    return tuple(1 - (1 - x) * (1 - amount) for x in c)



# =============================================================================
# STATISTICAL TESTING FUNCTIONS
# =============================================================================

def fisher_z(r):
    """Transform Pearson r to Fisher z."""
    return np.arctanh(r)


def fisher_r(z):
    """Transform Fisher z back to Pearson r."""
    return np.tanh(z)


def one_sample_ttest(r_subjects, alternative='greater'):
    """
    One-sample t-test against zero on Fisher-transformed correlations.
    Tests whether mean correlation across subjects is significantly above zero.

    Parameters
    ----------
    r_subjects : array-like
        Pearson correlation values, one per subject.
    alternative : str
        'greater' (default), 'less', or 'two-sided'.

    Returns
    -------
    dict with t-statistic, p-value, and mean r.
    """
    z = fisher_z(np.array(r_subjects))
    t, p = stats.ttest_1samp(z, popmean=0, alternative=alternative)
    return {
        't': t,
        'p': p,
        'mean_r': fisher_r(np.mean(z)),
    }


def paired_ttest(r_a, r_b, alternative='two-sided'):
    """
    Paired-samples t-test on Fisher-transformed correlations.
    Tests whether two sets of correlations differ significantly across subjects.

    Parameters
    ----------
    r_a : array-like
        Pearson correlations for condition/model A, one per subject.
    r_b : array-like
        Pearson correlations for condition/model B, one per subject.
    alternative : str
        'two-sided' (default), 'greater', or 'less'.

    Returns
    -------
    dict with t-statistic, p-value, and mean difference in r.
    """
    z_a = fisher_z(np.array(r_a))
    z_b = fisher_z(np.array(r_b))
    t, p = stats.ttest_rel(z_a, z_b, alternative=alternative)
    return {
        't': t,
        'p': p,
        'mean_r_a': fisher_r(np.mean(z_a)),
        'mean_r_b': fisher_r(np.mean(z_b)),
        'mean_diff_r': fisher_r(np.mean(z_a)) - fisher_r(np.mean(z_b)),
    }


def sign_flip_permutation(r_subjects, n_permutations=10000, alternative='greater',
                           seed=42):
    """
    Non-parametric sign-flip permutation test against zero.
    Generates empirical null distribution by randomly flipping signs of
    Fisher-transformed correlations across subjects.

    Parameters
    ----------
    r_subjects : array-like
        Pearson correlation values, one per subject.
    n_permutations : int
        Number of permutations (default 10000).
    alternative : str
        'greater' (default), 'less', or 'two-sided'.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with observed mean r, null distribution, and p-value.
    """
    rng = np.random.default_rng(seed)
    z = fisher_z(np.array(r_subjects))
    observed_mean = np.mean(z)

    null_distribution = np.array([
        np.mean(z * rng.choice([-1, 1], size=len(z)))
        for _ in range(n_permutations)
    ])

    if alternative == 'greater':
        p = np.mean(null_distribution >= observed_mean)
    elif alternative == 'less':
        p = np.mean(null_distribution <= observed_mean)
    elif alternative == 'two-sided':
        p = np.mean(np.abs(null_distribution) >= np.abs(observed_mean))
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")

    return {
        'observed_mean_r': fisher_r(observed_mean),
        'null_distribution': null_distribution,
        'p': p,
    }


def binomial_test(n_correct, n_total, alternative='greater'):
    """
    Exact one-tailed binomial test against chance level (p=0.5).
    Tests whether classification accuracy exceeds chance.

    Parameters
    ----------
    n_correct : int
        Number of correctly classified subjects/trials.
    n_total : int
        Total number of classifications.
    alternative : str
        'greater' (default), 'less', or 'two-sided'.

    Returns
    -------
    dict with accuracy, p-value.
    """
    result = stats.binomtest(n_correct, n_total, p=0.5, alternative=alternative) # use one tail since we only care about above-chance accuracy
    return {
        'accuracy': n_correct / n_total,
        'p': result.pvalue,
        'n_correct': n_correct,
        'n_total': n_total,
    }


def mcnemar_test(correct_a, correct_b, exact='auto'):
    """
    McNemar's test for paired binary classification outcomes.
    Implemented entirely via scipy.stats:
      - Uses scipy.stats.binomtest for exact test
      - Uses scipy.stats.chi2 for chi-squared approximation

    Parameters
    ----------
    correct_a : array-like of bool
        Whether each subject/trial was correctly classified by model A.
    correct_b : array-like of bool
        Whether each subject/trial was correctly classified by model B.
    exact : bool or 'auto'
        If True, always use exact binomial test.
        If False, always use chi-squared approximation.
        If 'auto' (default), use exact when n_discordant < 25.

    Returns
    -------
    dict with discordant counts, p-value, and which test was used.
    """
    correct_a = np.array(correct_a, dtype=bool)
    correct_b = np.array(correct_b, dtype=bool)

    n10 = np.sum(correct_a & ~correct_b)   # A correct, B wrong
    n01 = np.sum(~correct_a & correct_b)   # A wrong, B correct
    n_discordant = n10 + n01

    if n_discordant == 0:
        return {
            'n10': 0,
            'n01': 0,
            'n_discordant': 0,
            'p': 1.0,
            'test_used': 'none (no discordant pairs)',
        }

    # determine whether to use exact test
    if exact == 'auto':
        use_exact = n_discordant < 25
    elif isinstance(exact, bool):
        use_exact = exact
    else:
        raise ValueError("exact must be True, False, or 'auto'")

    if use_exact:
        result = stats.binomtest(n10, n_discordant, p=0.5, alternative='two-sided')
        p = result.pvalue
        test_used = f'exact binomial (n_discordant={n_discordant})'
    else:
        chi2_stat = (n10 - n01) ** 2 / (n10 + n01)
        p = stats.chi2.sf(chi2_stat, df=1)
        test_used = f'chi-squared approximation (n_discordant={n_discordant})'

    return {
        'n10': int(n10),
        'n01': int(n01),
        'n_discordant': int(n_discordant),
        'p': p,
        'test_used': test_used,
    }