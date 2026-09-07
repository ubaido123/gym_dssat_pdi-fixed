import numpy as np

CROPS           = ["maize"]
CROP_TO_IDX     = {crop: idx for idx, crop in enumerate(CROPS)}
NUM_CROPS       = len(CROPS)
RAW_OBS_SIZE    = 25
INPUT_DIM       = RAW_OBS_SIZE + NUM_CROPS + 1  # 25 + NUM_CROPS + 1
                                                  # +1 = nitrogen slot flag
                                                  # NOTE: INPUT_DIM depends on len(CROPS).
                                                  # Checkpoints are only valid for the CROPS
                                                  # list (and therefore INPUT_DIM) they were
                                                  # trained under -- changing CROPS invalidates
                                                  # old checkpoints (state_dict shape mismatch).

WATER_AMOUNTS      = [0, 10, 20, 30, 40]
NITROGEN_AMOUNTS   = [0, 5, 10, 20, 40]
N_WATER_ACTIONS    = len(WATER_AMOUNTS)
N_NITROGEN_ACTIONS = len(NITROGEN_AMOUNTS)


def crop_one_hot(crop_name):
    vec = np.zeros(NUM_CROPS, dtype=np.float32)
    vec[CROP_TO_IDX[crop_name]] = 1.0
    return vec


def flatten_obs(obs):
    if isinstance(obs, dict):
        parts = [np.atleast_1d(np.array(v, dtype=np.float32)) for v in obs.values()]
        flat  = np.concatenate(parts)
    else:
        flat = np.atleast_1d(np.array(obs, dtype=np.float32))
    if len(flat) < RAW_OBS_SIZE:
        flat = np.pad(flat, (0, RAW_OBS_SIZE - len(flat)))
    elif len(flat) > RAW_OBS_SIZE:
        flat = flat[:RAW_OBS_SIZE]

    # BUG FIX (NaN/Inf poisoning): gym_dssat_pdi can emit NaN or Inf for
    # derived fields on certain days (e.g. a stress ratio computed as
    # available_water / potential_uptake becomes 0/0 right after harvest).
    # Left unguarded, a single such value permanently poisons
    # STATE_NORMALIZER's running mean/var, which then emits NaN for every
    # state for the rest of training -> NaN actor probs -> crash.
    # Sanitize at the source: replace non-finite raw values with 0.0
    # before they ever reach the normalizer.
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    return flat


# ── State normalization ─────────────────────────────────────────────────────
# Raw DSSAT observations (topwt in thousands of kg/ha, dap 0-150, esw in mm,
# etc.) are mixed-scale and, fed unnormalized into the network, saturate
# the final softmax logits -- causing entropy collapse / a near-deterministic
# policy from episode 1. Fix: online running mean/std normalization
# (Welford's algorithm), same approach as OpenAI Baselines' VecNormalize.
# main.py is responsible for calling STATE_NORMALIZER.update(...) once per
# step; this module only normalizes.
class RunningNormalizer:
    def __init__(self, size, epsilon=1e-4, clip=5.0):
        self.mean  = np.zeros(size, dtype=np.float64)
        self.var   = np.ones(size,  dtype=np.float64)
        self.count = epsilon
        self.clip  = clip

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)

        # BUG FIX (NaN/Inf poisoning): guard against a non-finite value
        # permanently poisoning self.mean/self.var. flatten_obs() already
        # sanitizes raw obs; this is defense-in-depth for any other
        # caller (e.g. advantage/return normalizers reusing this class).
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        delta      = x - self.mean
        new_count  = self.count + 1
        new_mean   = self.mean + delta / new_count
        self.var   = (self.var * self.count + delta * (x - new_mean)) / new_count
        self.mean  = new_mean
        self.count = new_count

    def normalize(self, x):
        x    = np.asarray(x, dtype=np.float64)
        std  = np.sqrt(np.maximum(self.var, 1e-8))
        norm = (x - self.mean) / std
        norm = np.clip(norm, -self.clip, self.clip)

        # BUG FIX (NaN/Inf poisoning): np.clip does NOT clip NaN back to a
        # finite value (comparisons against NaN are always False). Final
        # safety net in case a checkpoint predating this fix is loaded.
        norm = np.nan_to_num(norm, nan=0.0, posinf=self.clip, neginf=-self.clip)
        return norm.astype(np.float32)

    def save(self, path):
        np.savez(path, mean=self.mean, var=self.var, count=self.count)

    def load(self, path):
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        self.mean  = data["mean"]
        self.var   = data["var"]
        self.count = float(data["count"])


# Module-level singleton so main.py/agent share one running statistic
# across the whole training run. Only the raw RAW_OBS_SIZE part is
# normalized -- the crop one-hot and nitrogen-slot flag are already
# bounded in [0, 1] and should NOT be normalized.
STATE_NORMALIZER = RunningNormalizer(RAW_OBS_SIZE)


def augment_state(obs, crop_name, is_nitrogen_slot=False, normalize=True):
    flat = flatten_obs(obs)
    if normalize:
        flat = STATE_NORMALIZER.normalize(flat)
    slot_flag = np.array([1.0 if is_nitrogen_slot else 0.0], dtype=np.float32)
    return np.concatenate([flat, crop_one_hot(crop_name), slot_flag])


def action_index_to_amounts(water_idx, nitrogen_idx):
    return WATER_AMOUNTS[water_idx], NITROGEN_AMOUNTS[nitrogen_idx]