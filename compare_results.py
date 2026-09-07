"""
compare_results.py

Full results-section analysis for AC vs REINFORCE (and optionally a third
baseline-policy log), covering:

  - Per-crop summary table (return, water, nitrogen, yield, profit)
  - RL vs. baseline comparison (using the per-episode vs_baseline_pct
    column already logged by both agents)
  - Welch's t-test on profit and water use (AC vs REINFORCE)
  - 95% confidence intervals for average return, per agent
  - Cohen's d effect size for profit/yield/water/nitrogen (AC vs REINFORCE)
  - Optional one-way ANOVA (permutation-based, no scipy needed) if a third
    log (e.g. an explicit baseline-policy run) is supplied via --baseline

No scipy dependency (environment pinned to numpy==1.24.1). Uses math.erf
for the normal-CDF approximation, and a permutation test (shuffling group
labels) for the ANOVA p-value rather than the F-distribution CDF, which
avoids needing scipy.stats.f entirely.

Usage (run from the project root, the folder containing ACTOR-CRITIC/,
REINFORCE/, figures/, results_output/):
    python compare_results.py \
        --ac ACTOR-CRITIC/training_log.csv \
        --reinforce REINFORCE/training_log.csv \
        --window 100
        [--baseline BASELINE/training_log.csv]

Output tables are written to results_output/ by default (see --out).
"""

import argparse
import math
import os

import numpy as np
import pandas as pd

RNG_SEED = 42


# ---- Statistics helpers (no scipy) ----------------------------------------

def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def welch_t_test(a, b):
    """Welch's t-test (unequal variance), two-sided, p-value via normal
    approximation (valid once each sample has >~30 observations)."""
    n1, n2 = len(a), len(b)
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)

    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return 0.0, 1.0

    t_stat = (m1 - m2) / se
    p_value = 2 * (1 - normal_cdf(abs(t_stat)))
    return t_stat, p_value


def cohens_d(a, b):
    """Cohen's d using pooled standard deviation. Convention:
    d ~ 0.2 small, ~0.5 medium, ~0.8 large."""
    n1, n2 = len(a), len(b)
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled_std


def confidence_interval_95(series):
    """95% CI for the mean via normal approximation (z=1.96)."""
    n = len(series)
    mean = series.mean()
    se = series.std(ddof=1) / math.sqrt(n)
    margin = 1.96 * se
    return mean, mean - margin, mean + margin


def permutation_anova(groups, n_permutations=10000, seed=RNG_SEED):
    """One-way ANOVA F-statistic with a permutation-test p-value instead
    of the F-distribution CDF (avoids needing scipy.stats.f). Valid for
    any number of groups >= 2.

    groups: list of 1-D array-likes (e.g. [ac_profit, reinforce_profit,
    baseline_profit]).
    """
    rng = np.random.default_rng(seed)
    all_values = np.concatenate([np.asarray(g, dtype=np.float64) for g in groups])
    sizes = [len(g) for g in groups]

    def f_stat(values, sizes):
        splits = np.split(values, np.cumsum(sizes)[:-1])
        grand_mean = values.mean()
        k = len(splits)
        n_total = len(values)

        ss_between = sum(len(s) * (s.mean() - grand_mean) ** 2 for s in splits)
        ss_within = sum(((s - s.mean()) ** 2).sum() for s in splits)

        df_between = k - 1
        df_within = n_total - k
        if ss_within == 0 or df_within <= 0:
            return 0.0

        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        return ms_between / ms_within

    observed_f = f_stat(all_values, sizes)

    count_extreme = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(all_values)
        if f_stat(shuffled, sizes) >= observed_f:
            count_extreme += 1
    p_value = count_extreme / n_permutations

    return observed_f, p_value


# ---- Data loading & summaries ----------------------------------------------

def load_log(path):
    return pd.read_csv(path)


def per_crop_summary(df, label, window):
    """Per-crop table: return, water, nitrogen, yield, profit -- averaged
    over the last `window` episodes for each crop present in the log."""
    tail = df.groupby("crop", group_keys=False).apply(
        lambda g: g.tail(window)
    ).reset_index(drop=True)

    summary = tail.groupby("crop").agg(
        n_episodes=("episode", "count"),
        mean_return=("season_return", "mean"),
        mean_water_mm=("total_water_mm", "mean"),
        mean_nitrogen_kgha=("total_nitrogen_kgha", "mean"),
        mean_yield_kgha=("grain_yield_kgha", "mean"),
        mean_profit_usd_ha=("profit_usd_ha", "mean"),
        std_profit_usd_ha=("profit_usd_ha", "std"),
    ).reset_index()
    summary.insert(0, "agent", label)
    return summary


def baseline_comparison(df, label, window):
    """RL vs. baseline using the vs_baseline_pct column already logged
    per episode (positive = better than the random/baseline policy)."""
    tail = df.tail(window)
    mean, lo, hi = confidence_interval_95(tail["vs_baseline_pct"])
    return {
        "agent": label,
        "mean_vs_baseline_pct": mean,
        "ci95_lower": lo,
        "ci95_upper": hi,
    }


def return_confidence_interval(df, label, window):
    tail = df.tail(window)
    mean, lo, hi = confidence_interval_95(tail["season_return"])
    return {
        "agent": label,
        "mean_return": mean,
        "ci95_lower": lo,
        "ci95_upper": hi,
    }


def main():
    parser = argparse.ArgumentParser(description="Full AC vs REINFORCE results-section analysis")
    parser.add_argument("--ac", default="ACTOR-CRITIC/training_log.csv")
    parser.add_argument("--reinforce", default="REINFORCE/training_log.csv")
    parser.add_argument("--baseline", default=None,
                         help="Optional third log (e.g. explicit baseline-policy run) for ANOVA")
    parser.add_argument("--window", type=int, default=100,
                         help="Number of final episodes to summarize/compare over")
    parser.add_argument("--out", default="results_output/comparison_summary.csv",
                         help="Base path for output tables (default: results_output/ folder)")
    args = parser.parse_args()

    # Make sure the output directory exists (results_output/ should already
    # be there per the project layout, but this guards against running from
    # a fresh checkout or a custom --out path with a new subfolder).
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ac_df = load_log(args.ac)
    rf_df = load_log(args.reinforce)

    if len(ac_df) < args.window or len(rf_df) < args.window:
        print(f"WARNING: --window={args.window} but AC has {len(ac_df)} episodes "
              f"and REINFORCE has {len(rf_df)}.")

    # 1. Per-crop summary tables
    print("=" * 70)
    print("1. PER-CROP SUMMARY (last {} episodes)".format(args.window))
    print("=" * 70)
    ac_crop_summary = per_crop_summary(ac_df, "Actor-Critic", args.window)
    rf_crop_summary = per_crop_summary(rf_df, "REINFORCE", args.window)
    crop_summary = pd.concat([ac_crop_summary, rf_crop_summary], ignore_index=True)
    print(crop_summary.to_string(index=False))

    # 2. RL vs. baseline comparison (using logged vs_baseline_pct)
    print("\n" + "=" * 70)
    print("2. RL vs. BASELINE (vs_baseline_pct, last {} episodes)".format(args.window))
    print("=" * 70)
    ac_baseline = baseline_comparison(ac_df, "Actor-Critic", args.window)
    rf_baseline = baseline_comparison(rf_df, "REINFORCE", args.window)
    baseline_df = pd.DataFrame([ac_baseline, rf_baseline])
    print(baseline_df.to_string(index=False))

    # 3. Confidence intervals for average return
    print("\n" + "=" * 70)
    print("3. 95% CONFIDENCE INTERVALS -- AVERAGE RETURN (last {} episodes)".format(args.window))
    print("=" * 70)
    ac_ci = return_confidence_interval(ac_df, "Actor-Critic", args.window)
    rf_ci = return_confidence_interval(rf_df, "REINFORCE", args.window)
    ci_df = pd.DataFrame([ac_ci, rf_ci])
    print(ci_df.to_string(index=False))

    # 4. Welch's t-test + Cohen's d on profit and water use
    print("\n" + "=" * 70)
    print("4. AC vs REINFORCE -- SIGNIFICANCE & EFFECT SIZE (last {} episodes)".format(args.window))
    print("=" * 70)
    ac_tail = ac_df.tail(args.window)
    rf_tail = rf_df.tail(args.window)

    for metric, col in [("Profit (USD/ha)", "profit_usd_ha"),
                         ("Grain yield (kg/ha)", "grain_yield_kgha"),
                         ("Water use (mm)", "total_water_mm"),
                         ("Nitrogen use (kg/ha)", "total_nitrogen_kgha")]:
        a, b = ac_tail[col], rf_tail[col]
        t_stat, p_value = welch_t_test(a, b)
        d = cohens_d(a, b)
        sig = "significant (p<0.05)" if p_value < 0.05 else "not significant"
        print(f"{metric:22s} | AC mean={a.mean():10.2f}  REINFORCE mean={b.mean():10.2f}  "
              f"t={t_stat:7.3f}  p={p_value:.4f} ({sig})  Cohen's d={d:6.3f}")

    # 5. Optional: ANOVA if a third (baseline) log is supplied
    if args.baseline:
        print("\n" + "=" * 70)
        print("5. ONE-WAY ANOVA (permutation test) -- AC vs REINFORCE vs BASELINE")
        print("=" * 70)
        baseline_df_raw = load_log(args.baseline)
        bl_tail = baseline_df_raw.tail(args.window)
        for metric, col in [("Profit (USD/ha)", "profit_usd_ha"),
                             ("Grain yield (kg/ha)", "grain_yield_kgha")]:
            groups = [ac_tail[col].values, rf_tail[col].values, bl_tail[col].values]
            f_stat, p_value = permutation_anova(groups)
            sig = "significant (p<0.05)" if p_value < 0.05 else "not significant"
            print(f"{metric:22s} | F={f_stat:7.3f}  p={p_value:.4f} ({sig})")
    else:
        print("\n(No --baseline log supplied -- skipping 3-group ANOVA. "
              "The vs_baseline_pct comparison in section 2 above already "
              "captures RL-vs-baseline using the per-episode logged value.)")

    crop_summary.to_csv(args.out.replace(".csv", "_per_crop.csv"), index=False)
    baseline_df.to_csv(args.out.replace(".csv", "_baseline.csv"), index=False)
    ci_df.to_csv(args.out.replace(".csv", "_ci.csv"), index=False)
    print(f"\nTables saved: {args.out.replace('.csv', '_per_crop.csv')}, "
          f"{args.out.replace('.csv', '_baseline.csv')}, "
          f"{args.out.replace('.csv', '_ci.csv')}")


if __name__ == "__main__":
    main()