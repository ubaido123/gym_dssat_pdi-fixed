"""
plot_learning_curves.py

Generates the "Figures" items from the results section:
  - Learning curve: reward (avg100_return) vs. episode, AC vs REINFORCE
  - Profit trend: avg100_profit_usd_ha vs. episode, AC vs REINFORCE
  - Actor loss convergence, AC vs REINFORCE
  - Critic loss convergence (Actor-Critic only -- REINFORCE has no critic)

Saves PNGs to --outdir (default: ./figures/). No scipy dependency.

Usage:
    python plot_learning_curves.py \
        --ac ACTOR-CRITIC/training_log.csv \
        --reinforce REINFORCE/training_log.csv \
        --outdir figures
"""

import argparse
import os
import warnings

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display in a headless container
import matplotlib.pyplot as plt

# NOTE: gym-dssat-pdi hard-pins matplotlib==3.3.4 as an actual dependency
# (not just a bundled leftover -- `pip check` fails without it). That old
# version's savefig() throws MatplotlibDeprecationWarning spam for the
# dpi/facecolor/edgecolor/orientation/bbox_inches_restore kwargs on every
# call, even though the files still save correctly. Upgrading matplotlib
# to silence this isn't safe here (it drags in a newer numpy that breaks
# pandas==1.5.3's compiled extensions -- see Dockerfile comments). So we
# suppress the warning at the source instead of fighting the pinned
# dependency.
warnings.filterwarnings("ignore", message=".*savefig.*no longer supported.*")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def smooth(series, window=50):
    """Rolling mean for readability on noisy per-episode data."""
    return series.rolling(window=window, min_periods=1).mean()


def plot_learning_curve(ac_df, rf_df, outdir):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ac_df["episode"], ac_df["avg100_return"], label="Actor-Critic", color="tab:blue", linewidth=1.5)
    ax.plot(rf_df["episode"], rf_df["avg100_return"], label="REINFORCE", color="tab:orange", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average return (100-episode rolling)")
    ax.set_title("Learning Curve: Reward vs. Episode")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "learning_curve_reward.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_profit_trend(ac_df, rf_df, outdir):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ac_df["episode"], ac_df["avg100_profit_usd_ha"], label="Actor-Critic", color="tab:blue", linewidth=1.5)
    ax.plot(rf_df["episode"], rf_df["avg100_profit_usd_ha"], label="REINFORCE", color="tab:orange", linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average profit, USD/ha (100-episode rolling)")
    ax.set_title("Economic Profit Trend: AC vs REINFORCE")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "profit_trend.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_actor_loss(ac_df, rf_df, outdir):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ac_df["episode"], smooth(ac_df["actor_loss"]), label="Actor-Critic", color="tab:blue", linewidth=1.5)
    ax.plot(rf_df["episode"], smooth(rf_df["actor_loss"]), label="REINFORCE", color="tab:orange", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Actor loss (50-episode rolling mean)")
    ax.set_title("Actor Loss Convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "actor_loss_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_critic_loss(ac_df, outdir):
    """Critic loss exists only for Actor-Critic -- REINFORCE has no critic."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ac_df["episode"], smooth(ac_df["critic_loss"]), color="tab:green", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Critic loss (50-episode rolling mean)")
    ax.set_title("Critic Loss Convergence (Actor-Critic only)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "critic_loss_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_entropy(ac_df, outdir):
    """Entropy trend -- should decline from ~ln(5)=1.609 as the policy
    sharpens. Flat at 1.609 for the whole run would indicate the policy
    never learned to differentiate actions (a live issue to catch, not
    the earlier NaN bug)."""
    if "entropy_water" not in ac_df.columns or "entropy_nitrogen" not in ac_df.columns:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ac_df["episode"], smooth(ac_df["entropy_water"]), label="Water head entropy", color="tab:blue")
    ax.plot(ac_df["episode"], smooth(ac_df["entropy_nitrogen"]), label="Nitrogen head entropy", color="tab:red")
    ax.axhline(1.609, color="gray", linestyle="--", linewidth=1, label="Max entropy (ln 5)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Entropy (nats, 50-episode rolling mean)")
    ax.set_title("Actor-Critic Policy Entropy Over Training")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "entropy_trend.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate learning-curve and loss-convergence figures")
    parser.add_argument("--ac", default="ACTOR-CRITIC/training_log.csv")
    parser.add_argument("--reinforce", default="REINFORCE/training_log.csv")
    parser.add_argument("--outdir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ac_df = pd.read_csv(args.ac)
    rf_df = pd.read_csv(args.reinforce)

    saved = []
    saved.append(plot_learning_curve(ac_df, rf_df, args.outdir))
    saved.append(plot_profit_trend(ac_df, rf_df, args.outdir))
    saved.append(plot_actor_loss(ac_df, rf_df, args.outdir))
    saved.append(plot_critic_loss(ac_df, args.outdir))
    entropy_path = plot_entropy(ac_df, args.outdir)
    if entropy_path:
        saved.append(entropy_path)

    print("Figures saved:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()