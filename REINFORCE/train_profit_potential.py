"""
train_profit_potential.py
--------------------------
Establishes the profit ceiling for percentage-based reporting.

Mirrors the paper's "profit potential" methodology (Section 2.4.3):
the same REINFORCE algorithm is run on a SINGLE FIXED weather pattern
(random_weather=False, fixed seed) instead of stochastic weather.
With no weather randomness, the optimisation problem is easier and
the resulting best moving-average profit approximates the maximum
achievable profit under this DSSAT maize setup.

This ceiling is then used to express all REINFORCE training output
as a percentage of profit potential, matching the paper's reporting style.
"""

import gym
import gym_dssat_pdi
import torch
import torch.optim as optim
from torch.distributions import Categorical
import json

import dssat_configuration
from model import PolicyNet
from state_processor import extract_state
from reward import compute_reward

FIXED_SEED = 123456
EPISODES   = 20000   # same budget as main training, deterministic env converges similarly

def main():
    env = gym.make(
        'gym_dssat_pdi:GymDssatPdi-v0',
        mode='irrigation',
        random_weather=False,
        seed=FIXED_SEED,
    )

    policy = PolicyNet(dssat_configuration.INPUT_DIM, dssat_configuration.NUM_ACTIONS)
    optimizer = optim.Adam(policy.parameters(), lr=dssat_configuration.LR)

    profit_history = []
    best_avg_profit = -float('inf')

    print("Profit potential run — deterministic weather, fixed seed.")
    print(f"Seed: {FIXED_SEED} | Episodes: {EPISODES}\n")

    for episode in range(1, EPISODES + 1):
        raw_obs = env.reset()
        state = extract_state(raw_obs)
        done = False

        log_probs = []
        rewards = []
        last_valid_obs = raw_obs

        while not done:
            probs = policy(state)
            m = Categorical(probs)
            action_idx = m.sample()
            log_probs.append(m.log_prob(action_idx))

            irrigation_depth = dssat_configuration.ACTION_MAP[action_idx.item()]
            next_raw_obs, _, done, _ = env.step({'amir': irrigation_depth})

            reward = compute_reward(raw_obs, next_raw_obs, irrigation_depth, done)
            rewards.append(reward)

            if next_raw_obs is not None:
                last_valid_obs = next_raw_obs
                state = extract_state(next_raw_obs)
                raw_obs = next_raw_obs

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

        episode_profit = sum(rewards)
        profit_history.append(episode_profit)

        window = profit_history[-100:]
        avg_profit = sum(window) / len(window)

        if avg_profit > best_avg_profit:
            best_avg_profit = avg_profit

        if episode % 100 == 0 or episode == 1:
            print(
                f"Season {episode:05d} | "
                f"Profit: {episode_profit:.2f} | "
                f"MovAvg(100): {avg_profit:.2f} | "
                f"Best MovAvg: {best_avg_profit:.2f} | "
                f"Final Yield: {last_valid_obs['grnwt']:.1f} kg/ha"
            )

    env.close()

    print(f"\nProfit potential established: {best_avg_profit:.2f}")

    with open("profit_potential.json", "w") as f:
        json.dump({
            "profit_potential": best_avg_profit,
            "seed": FIXED_SEED,
            "episodes": EPISODES,
        }, f, indent=2)

    print("Saved to profit_potential.json")

if __name__ == "__main__":
    main()
