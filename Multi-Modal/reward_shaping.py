"""
reward_shaping.py
-----------------
All checkpoint outputs shown as percentages:
  topwt   : % of crop maximum biomass (maize 15000, cotton 10000 kg/ha)
  cp_bonus: % of maximum checkpoint bonus (0-100%)
  reward  : normalised reward × 100 (%)
"""

import numpy as np

CHECKPOINT_DAPS = [30, 60, 90]
XLAI_THRESHOLD  = {"maize": 1.8, "cotton": 2.5}
LAI_TARGETS     = {"maize": {30: 0.20, 60: 1.9, 90: 1.5},
                   "cotton": {30: 0.13, 60: 2.0, 90: 3.0}}

# Maximum expected above-ground biomass per crop (kg/ha) — for % display
TOPWT_MAX = {"maize": 15000.0, "cotton": 10000.0}

LAI_GROWTH_BONUS    =  0.5
TOPWT_GROWTH_BONUS  =  0.0001
CHECKPOINT_SCALE    =  1.0
N_EARLY_BONUS       =  0.001
N_GATE_PENALTY      = -0.3
WATER_SAVING_BONUS  =  0.05
BASE_NORM           = 300.0

RANDOM_BASELINE_NORM = -29000.0 / BASE_NORM
BEST_POSSIBLE_NORM   =  15000.0 / BASE_NORM


def to_percent(cumulative_norm):
    return ((cumulative_norm - RANDOM_BASELINE_NORM) /
            (BEST_POSSIBLE_NORM - RANDOM_BASELINE_NORM)) * 100


class SeasonalRewardShaper:

    def __init__(self, crop_name: str, verbose: bool = True):
        self.crop_name   = crop_name
        self.xlai_thresh = XLAI_THRESHOLD.get(crop_name, 1.8)
        self.verbose     = verbose
        self._triggered  = set()
        self._episode    = 0
        self._cumulative = 0.0
        self._prev_xlai  = 0.0
        self._prev_topwt = 0.0

    def reset(self, crop_name: str, episode: int = 0):
        self.crop_name   = crop_name
        self.xlai_thresh = XLAI_THRESHOLD.get(crop_name, 1.8)
        self._triggered  = set()
        self._episode    = episode
        self._cumulative = 0.0
        self._prev_xlai  = 0.0
        self._prev_topwt = 0.0

    def step(self, obs_dict, raw_reward, done,
             last_water_action=0.0, last_n_action=0.0):

        if raw_reward is None:
            base = 0.0
        elif isinstance(raw_reward, (list, np.ndarray)):
            base = float(raw_reward[0]) + float(raw_reward[1])
        else:
            base = float(raw_reward)

        base_norm = base / BASE_NORM
        self._cumulative += base_norm

        dap  = int(obs_dict.get("dap",  0))
        xlai = float(obs_dict.get("xlai", 0.0))

        if dap == 0 and not done:
            return base_norm

        if done:
            pct = to_percent(self._cumulative)
            if self.verbose:
                print(f"    [EP {self._episode:5d}] [{self.crop_name:8s}] "
                      f"CHECKPOINT 4 | END OF SEASON  | "
                      f"season score = {pct:+.1f}%\n")
            return base_norm

        topwt   = float(obs_dict.get("topwt", 0.0))
        shaping = 0.0

        lai_growth   = max(0.0, xlai  - self._prev_xlai)
        topwt_growth = max(0.0, topwt - self._prev_topwt)
        shaping += LAI_GROWTH_BONUS   * lai_growth
        shaping += TOPWT_GROWTH_BONUS * topwt_growth

        canopy_closed = xlai >= self.xlai_thresh

        if last_n_action > 0:
            if not canopy_closed:
                shaping += N_EARLY_BONUS * last_n_action
            else:
                shaping += N_GATE_PENALTY

        if canopy_closed and last_n_action == 0 and last_water_action <= 10:
            shaping += WATER_SAVING_BONUS

        self._prev_xlai  = xlai
        self._prev_topwt = topwt

        checkpoint_bonus = 0.0
        crop_targets = LAI_TARGETS.get(self.crop_name, {30: 0.20, 60: 1.9, 90: 1.5})

        for cp_num, cp_dap in enumerate(CHECKPOINT_DAPS, start=1):
            if dap >= cp_dap and cp_dap not in self._triggered:
                self._triggered.add(cp_dap)
                target    = crop_targets.get(cp_dap, 1.0)
                lai_ratio = min(xlai / target, 1.0) if target > 0 else 0.0
                checkpoint_bonus = CHECKPOINT_SCALE * lai_ratio

                if self.verbose:
                    hit    = "✓" if xlai >= target else "✗"
                    status = "CANOPY CLOSED — N-gate active" if canopy_closed else "canopy open"

                    # ── percentage conversions ──────────────────────────
                    top_max    = TOPWT_MAX.get(self.crop_name, 15000.0)
                    topwt_pct  = min(topwt / top_max * 100, 100.0)
                    bonus_pct  = checkpoint_bonus * 100.0
                    reward_pct = (base_norm + shaping + checkpoint_bonus) * 100.0

                    print(f"    [EP {self._episode:5d}] [{self.crop_name:8s}] "
                          f"CHECKPOINT {cp_num} | DAP {dap:3d} | "
                          f"xlai={xlai:.2f} (target {target}) {hit}  "
                          f"topwt={topwt_pct:4.1f}%  "
                          f"cp_bonus={bonus_pct:5.1f}%  "
                          f"reward={reward_pct:+6.1f}%  "
                          f"[{status}]")
                break

        return base_norm + shaping + checkpoint_bonus
