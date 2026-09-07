import gym
import gym_dssat_pdi
import torch
import torch.optim as optim
from torch.distributions import Categorical
import json, os, csv

import dssat_configuration
from model import PolicyNet
from state_processor import extract_state, STATE_NORMALIZER
from reward import compute_reward

# ── Profit potential ceiling (display only) ────────────────────────────────
PROFIT_POTENTIAL_PATH = "profit_potential.json"
if os.path.exists(PROFIT_POTENTIAL_PATH):
    with open(PROFIT_POTENTIAL_PATH) as f:
        PROFIT_POTENTIAL = json.load(f)["profit_potential"]
    print(f"Loaded profit potential ceiling: {PROFIT_POTENTIAL:.2f}")
else:
    PROFIT_POTENTIAL = None
    print(f"[WARNING] {PROFIT_POTENTIAL_PATH} not found — percentage logging disabled.")

def fmt_pct(value):
    if PROFIT_POTENTIAL is None or PROFIT_POTENTIAL == 0:
        return "  n/a "
    return f"{(value / PROFIT_POTENTIAL) * 100:6.1f}%"

def pct_value(value):
    """Same percentage as fmt_pct, but as a plain float for CSV columns
    (no % sign, no padding) — keeps this column numeric to match
    ACTOR-CRITIC/training_log.csv's vs_baseline_pct column."""
    if PROFIT_POTENTIAL is None or PROFIT_POTENTIAL == 0:
        return ""
    return round((value / PROFIT_POTENTIAL) * 100, 2)

def main():
    env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', mode='irrigation',
                run_dssat_location='/opt/dssat_pdi/run_dssat')

    policy = PolicyNet(dssat_configuration.INPUT_DIM, dssat_configuration.NUM_ACTIONS)
    optimizer = optim.Adam(policy.parameters(), lr=dssat_configuration.LR)

    profit_history  = []
    best_avg_profit = -float('inf')

    # ── CSV logging setup ──────────────────────────────────────────────────
    # Column names/order match ACTOR-CRITIC/training_log.csv so the two logs
    # can be concatenated directly for the results-section comparison.
    CSV_PATH   = "./training_log.csv"
    csv_file   = open(CSV_PATH, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "episode", "crop", "season_return", "avg100_return",
        "actor_loss", "critic_loss",
        "total_water_mm", "total_nitrogen_kgha",
        "vs_baseline_pct",
        "grain_yield_kgha", "profit_usd_ha", "avg100_profit_usd_ha"
    ])
    print(f"CSV logging → {CSV_PATH}\n")

    # ── Per-DAP detail logging (for LAI trajectory / application-pattern figures) ──
    DAILY_LOG_INTERVAL = 500
    DAILY_LOG_PATH = "./daily_log.csv"
    daily_csv_file = open(DAILY_LOG_PATH, "w", newline="")
    daily_csv_writer = csv.writer(daily_csv_file)
    daily_csv_writer.writerow([
        "episode", "crop", "dap", "xlai", "topwt", "water_mm", "nitrogen_kg"
    ])
    print(f"Daily (per-DAP) logging → {DAILY_LOG_PATH} "
          f"(episode 1, then every {DAILY_LOG_INTERVAL} episodes)\n")

    def should_log_daily(ep):
        return ep == 1 or ep % DAILY_LOG_INTERVAL == 0

    print("Orchestration pipeline active. Commencing training epochs...")

    for episode in range(1, dssat_configuration.EPISODES + 1):
        raw_obs = env.reset()
        state   = extract_state(raw_obs)
        done    = False

        log_probs      = []
        rewards        = []
        last_valid_obs = raw_obs
        season_water_mm = 0.0

        while not done:
            probs      = policy(state)
            m          = Categorical(probs)
            action_idx = m.sample()
            log_probs.append(m.log_prob(action_idx))

            irrigation_depth = dssat_configuration.ACTION_MAP[action_idx.item()]
            season_water_mm += irrigation_depth
            next_raw_obs, _, done, _ = env.step({'amir': irrigation_depth})

            if should_log_daily(episode):
                obs_for_log = next_raw_obs if isinstance(next_raw_obs, dict) else raw_obs
                dap_now = int(obs_for_log.get("dap", 0)) if isinstance(obs_for_log, dict) else 0
                daily_csv_writer.writerow([
                    episode, "maize", dap_now,
                    round(float(obs_for_log.get("xlai", 0.0)), 4) if isinstance(obs_for_log, dict) else 0.0,
                    round(float(obs_for_log.get("topwt", 0.0)), 2) if isinstance(obs_for_log, dict) else 0.0,
                    round(irrigation_depth, 1), 0.0
                ])

            reward = compute_reward(raw_obs, next_raw_obs, irrigation_depth, done)
            rewards.append(reward)

            if next_raw_obs is not None:
                last_valid_obs = next_raw_obs
                state   = extract_state(next_raw_obs)
                raw_obs = next_raw_obs

        # Return computation — no gamma, matches paper exactly
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        for log_prob, G_t in zip(log_probs, returns):
            policy_loss.append(-log_prob * G_t)

        optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

        # Raw profit tracking — this reward IS already the $/ha profit
        # (see reward.py: -WATER_COST*irrigation + YIELD_PRICE*yield at done)
        episode_profit = sum(rewards)
        profit_history.append(episode_profit)

        window     = profit_history[-100:]
        avg_profit = sum(window) / len(window)

        if avg_profit > best_avg_profit:
            best_avg_profit = avg_profit
            torch.save(policy.state_dict(), "dssat_policy_model_best.pth")
            # Save the normalizer's running mean/var alongside the best
            # checkpoint. Without this, resuming from
            # dssat_policy_model_best.pth later would restart the
            # normalizer from zero -- a fresh, unwarmed normalizer
            # computes different normalized values than the one these
            # weights were actually trained against, silently corrupting
            # anything built on top of a resumed model.
            STATE_NORMALIZER.save("state_normalizer_best")

        final_yield = last_valid_obs['grnwt']

        # ── Write one CSV row per episode ──────────────────────────────────
        # REINFORCE never applies nitrogen, so total_nitrogen_kgha is always 0
        # (kept as an explicit column so both logs share one schema).
        # actor_loss/critic_loss have no REINFORCE equivalent (single policy
        # network, no separate value/critic loss) — left blank.
        csv_writer.writerow([
            episode, "maize",
            round(episode_profit, 4),
            round(avg_profit, 4),
            "", "",
            round(season_water_mm, 1),
            0.0,
            pct_value(episode_profit),
            round(final_yield, 1),
            round(episode_profit, 2),
            round(avg_profit, 2)
        ])
        if episode % 100 == 0:
            csv_file.flush()
            daily_csv_file.flush()

        if episode % 10 == 0 or episode == 1:
            print(
                f"Season {episode:05d} | "
                f"Profit: {episode_profit:8.2f} ({fmt_pct(episode_profit)}) | "
                f"MovAvg(100): {avg_profit:8.2f} ({fmt_pct(avg_profit)}) | "
                f"Best MovAvg: {best_avg_profit:8.2f} ({fmt_pct(best_avg_profit)}) | "
                f"Final Yield: {final_yield:.1f} kg/ha"
            )

    csv_file.close()
    daily_csv_file.close()
    print("\nTraining complete.")
    print(f"Best 100-episode moving average profit : {best_avg_profit:.2f} ({fmt_pct(best_avg_profit)})")
    print(f"Training log saved                     : {CSV_PATH}")
    torch.save(policy.state_dict(), "dssat_policy_model.pth")
    STATE_NORMALIZER.save("state_normalizer_final")
    print("Final weights  → dssat_policy_model.pth")
    print("Best weights   → dssat_policy_model_best.pth")
    print("Normalizer stats (final) → state_normalizer_final.npz")
    print("Normalizer stats (best)  → state_normalizer_best.npz")

if __name__ == "__main__":
    main()