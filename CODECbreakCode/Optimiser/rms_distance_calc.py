
import numpy as np
import Optimiser.config as config
from Optimiser.config import CONFIG_FullTrack, CONFIG_SingleAudio


def per_dim_normalize(vec: np.ndarray,
                      a_min: np.ndarray,
                      a_max: np.ndarray) -> np.ndarray:
    """
    Affine-normalize each dimension independently to [0,1].

    vec[i] = (vec[i] - a_min[i]) / (a_max[i] - a_min[i])

    We also clip to [0,1] just in case.
    """
    vec   = np.asarray(vec,   dtype=np.float32)
    a_min = np.asarray(a_min, dtype=np.float32)
    a_max = np.asarray(a_max, dtype=np.float32)

    denom = (a_max - a_min)
    # avoid divide by zero for any frozen parameter
    denom = np.where(denom == 0.0, 1.0, denom)

    norm = (vec - a_min) / denom
    norm = np.clip(norm, 0.0, 1.0)
    return norm

def rms_distance_normalized(vec: np.ndarray,
                             target: np.ndarray,
                             a_min: np.ndarray,
                             a_max: np.ndarray) -> float:
    """
    RMS distance between vec and target,
    measured AFTER per-dimension [0,1] normalization.

    Result range:
        0.0  -> vec == target (perfect match)
        ~1.0 -> vec is as far as possible in *every* dim
                (target at one extreme, vec at the other)
    """
    vec_n     = per_dim_normalize(vec,     a_min, a_max)
    target_n  = per_dim_normalize(target,  a_min, a_max)

    diff_sq = np.square(vec_n - target_n)
    return float(np.sqrt(np.mean(diff_sq)))

def similarity_score_balanced_custom(vec: np.ndarray,
                               target: np.ndarray,
                               a_min: np.ndarray,
                               a_max: np.ndarray) -> float:
    """
    Returns a score in [0,1]:
        1.0 -> vec matches target exactly
        0.0 -> vec is maximally far in every dim
    """
    dist01 = rms_distance_normalized(vec, target, a_min, a_max)
    return 1.0 - dist01

def similarity_score_balanced(vec: np.ndarray,
                              target: np.ndarray = None) -> float:
    """
    Dimension-balanced similarity score in [0,1].

    We:
    - normalise vec and target per-dim using that mode's act_min/act_max
    - compute RMS distance in that normalised space
    - invert so that higher is better (1.0 = perfect match)

    Handles both FullTrack and SingleAudio based on action length.
    """
    vec = np.asarray(vec, dtype=np.float32)

    if len(vec) == 7:
        a_min   = np.asarray(CONFIG_SingleAudio["env"]["act_min"], dtype=np.float32)
        a_max   = np.asarray(CONFIG_SingleAudio["env"]["act_max"], dtype=np.float32)
        default_target = np.asarray(CONFIG_SingleAudio["env"]["minimum_inter"], dtype=np.float32)
    else:
        a_min   = np.asarray(CONFIG_FullTrack["env"]["act_min"], dtype=np.float32)
        a_max   = np.asarray(CONFIG_FullTrack["env"]["act_max"], dtype=np.float32)
        default_target = np.asarray(CONFIG_FullTrack["env"]["minimum_inter"], dtype=np.float32)

    if target is None:
        target = default_target
    else:
        target = np.asarray(target, dtype=np.float32)

    # --- per-dimension [0,1] normalisation ---
    vec_n    = per_dim_normalize(vec,    a_min, a_max)
    target_n = per_dim_normalize(target, a_min, a_max)

    diff_sq = np.square(vec_n - target_n)
    dist01  = float(np.sqrt(np.mean(diff_sq)))  # in [0,1]

    score = 1.0 - dist01  # higher = closer to target
    return float(np.clip(score, 0.0, 1.0))