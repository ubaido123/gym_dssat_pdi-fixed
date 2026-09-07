import numpy as np
import torch
import dssat_configuration


# ── State normalization ──────────────────────────────────────────────────────
# BUG FIX (same root cause as ACTOR-CRITIC/crop_utils.py): raw DSSAT
# observations here are mixed-scale and unbounded across a season --
# `totir` (cumulative irrigation) and `crain` (cumulative rainfall) both
# grow into the hundreds or thousands of mm as the season progresses.
# Fed unnormalized into PolicyNet, this saturates the final softmax from
# episode 1 (a large input dot small random init weights still produces a
# large logit spread), collapsing the policy toward a single action
# (observed: ~40mm/day irrigation applied essentially every day,
# regardless of state) and killing the gradient signal needed to ever
# learn away from that collapsed policy.
#
# Fix: online running mean/std normalization (Welford's algorithm),
# identical approach to ACTOR-CRITIC's RunningNormalizer. main.py is
# responsible for calling STATE_NORMALIZER.update(raw_vector) once per
# step, mirroring how ACTOR-CRITIC/main.py drives its STATE_NORMALIZER.
class RunningNormalizer:
    def __init__(self, size, epsilon=1e-4, clip=5.0):
        self.mean  = np.zeros(size, dtype=np.float64)
        self.var   = np.ones(size,  dtype=np.float64)
        self.count = epsilon
        self.clip  = clip

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        delta     = x - self.mean
        new_count = self.count + 1
        new_mean  = self.mean + delta / new_count
        self.var  = (self.var * self.count + delta * (x - new_mean)) / new_count
        self.mean = new_mean
        self.count = new_count

    def normalize(self, x):
        x    = np.asarray(x, dtype=np.float64)
        std  = np.sqrt(np.maximum(self.var, 1e-8))
        norm = (x - self.mean) / std
        norm = np.clip(norm, -self.clip, self.clip)
        norm = np.nan_to_num(norm, nan=0.0, posinf=self.clip, neginf=-self.clip)
        return norm.astype(np.float32)

    def save(self, path):
        np.savez(path, mean=self.mean, var=self.var, count=self.count)

    def load(self, path):
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        self.mean  = data["mean"]
        self.var   = data["var"]
        self.count = float(data["count"])


# Module-level singleton, same pattern as ACTOR-CRITIC's STATE_NORMALIZER,
# so main.py and this module share one running statistic across training.
STATE_NORMALIZER = RunningNormalizer(dssat_configuration.INPUT_DIM)


def flatten_obs(obs):
    """Build the raw (unnormalized) 9-dim state vector -- same layout as before."""
    if obs is None:
        return np.zeros(dssat_configuration.INPUT_DIM, dtype=np.float32)

    stage   = obs['istage']
    lai     = obs['xlai']
    sw_top5 = obs['sw'][:5]
    cu_rain = obs.get('crain', 0.0)
    cu_irr  = obs['totir']

    flat = np.concatenate(([stage, lai], sw_top5, [cu_rain, cu_irr])).astype(np.float32)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    return flat


def extract_state(obs, normalize=True):
    """
    9-dimensional state vector matching the paper's exact specification:
    1. Crop Stage
    2. Leaf Area Index (LAI)
    3-7. Extractable Soil Water (Top 5 layers)
    8. Cumulative Rainfall
    9. Cumulative Irrigation

    Now normalized via STATE_NORMALIZER before being handed to PolicyNet,
    matching ACTOR-CRITIC's fix for the identical softmax-saturation bug.
    """
    flat = flatten_obs(obs)
    if normalize:
        flat = STATE_NORMALIZER.normalize(flat)
    return torch.FloatTensor(flat)